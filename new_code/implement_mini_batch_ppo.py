import argparse
import json
import os
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    TrainerCallback,
    set_seed
)
from accelerate import PartialState

# --- 1. 核心数学运算：提取 Log Probs & token length ---
def get_batch_log_probs(logits, labels):
    """提取给定 label 对应的 token logit，并返回总对数概率和有效 token 长度"""
    # logits shape: (batch, seq_len, vocab_size)
    # labels shape: (batch, seq_len)
    
    # 语言模型的 logits 需要错位对齐
    shifted_logits = logits[..., :-1, :]
    shifted_labels = labels[..., 1:].contiguous()
    
    log_probs = F.log_softmax(shifted_logits, dim=-1)
    loss_mask = shifted_labels != -100
    
    safe_labels = shifted_labels.clone()
    safe_labels[~loss_mask] = 0
    
    per_token_log_probs = torch.gather(log_probs, dim=2, index=safe_labels.unsqueeze(2)).squeeze(2)
    total_log_probs = (per_token_log_probs * loss_mask).sum(dim=-1)   # 每个样本的总对数概率
    valid_lengths = loss_mask.sum(dim=-1)                             # 每个样本的有效 token 数
    valid_lengths = torch.clamp(valid_lengths, min=1.0)           # 避免除以零的情况，最小长度设为 1
    return total_log_probs, valid_lengths

# --- 新增：每个token的log概率函数 ---
def get_per_token_log_probs(logits, labels):
    """返回每个token的log概率和对应的mask（有效token为1）"""
    # logits: (batch, seq_len, vocab_size)
    # labels: (batch, seq_len)
    shifted_logits = logits[..., :-1, :]
    shifted_labels = labels[..., 1:].contiguous()
    
    log_probs = F.log_softmax(shifted_logits, dim=-1)
    loss_mask = (shifted_labels != -100).float()  # (batch, seq_len-1)
    
    # 将 -100 替换为0，避免 gather 越界
    safe_labels = shifted_labels.clone()
    safe_labels[shifted_labels == -100] = 0
    per_token_log_probs = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    per_token_log_probs = per_token_log_probs * loss_mask  # 无效位置置0
    return per_token_log_probs, loss_mask

# --- 2. 独立实现的日志回调 ---
class ResidualLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and logs and args.output_dir:
            # 将 logs 中所有 Tensor 转换为标量
            logs = {k: (v.item() if torch.is_tensor(v) else v) for k, v in logs.items()}
            os.makedirs(args.output_dir, exist_ok=True)
            log_file = os.path.join(args.output_dir, "residual_metrics.jsonl")
            entry = {"step": state.global_step, **logs}
            with open(log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")

# --- 3. 全新重构的 Trainer ---
class ResidualRLHFTrainer(Trainer):
    def __init__(self, ref_model=None, beta=0.5, clip_eps=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        if ref_model is not None:
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
        self.beta = beta
        self.clip_eps = clip_eps
        self._batch_metrics = {}
        self._step_logs_accumulator = {}
        self._step_counter = 0
        self._eval_logs_accumulator = {}
        self._eval_step_counter = 0

    # --- 修改后的 compute_loss（使用 per-token）---
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Policy model forward
        outputs_c = model(input_ids=inputs["chosen_input_ids"], attention_mask=inputs["chosen_attention_mask"])
        pi_logp_c_per_token, pi_mask_c = get_per_token_log_probs(outputs_c.logits, inputs["chosen_labels"])

        dummy_outputs_c = outputs_c if return_outputs else None
        if not return_outputs:
            del outputs_c  # 释放内存

        outputs_r = model(input_ids=inputs["rejected_input_ids"], attention_mask=inputs["rejected_attention_mask"])
        pi_logp_r_per_token, pi_mask_r = get_per_token_log_probs(outputs_r.logits, inputs["rejected_labels"])
        del outputs_r  # 释放内存

        with torch.no_grad():
            ref_outputs_c = self.ref_model(input_ids=inputs["chosen_input_ids"], attention_mask=inputs["chosen_attention_mask"])
            ref_logp_c_per_token, _ = get_per_token_log_probs(ref_outputs_c.logits, inputs["chosen_labels"])
            del ref_outputs_c  # 释放内存
            
            ref_outputs_r = self.ref_model(input_ids=inputs["rejected_input_ids"], attention_mask=inputs["rejected_attention_mask"])
            ref_logp_r_per_token, _ = get_per_token_log_probs(ref_outputs_r.logits, inputs["rejected_labels"])
            del ref_outputs_r  # 释放内存

        f_Y_c = inputs["f_Y_chosen"]
        f_Y_r = inputs["f_Y_rejected"]
        residual_c = 1.0 - f_Y_r 
        residual_r = f_Y_c

        # 对数比例（逐token）
        log_ratio_c_per_token = torch.clamp(pi_logp_c_per_token - ref_logp_c_per_token, -3.0, 3.0)
        log_ratio_r_per_token = torch.clamp(pi_logp_r_per_token - ref_logp_r_per_token, -3.0, 3.0)

        ratio_c_per_token = torch.exp(log_ratio_c_per_token)
        ratio_r_per_token = torch.exp(log_ratio_r_per_token)

        # 采用 DeepSeek/GRPO 的严格非负 KL 估计量 (k3 形式)
        # 公式: (pi_ref / pi_theta) - log(pi_ref / pi_theta) - 1
        # 代换后: exp(-log_ratio) + log_ratio - 1
        # kl_penalty_c_per_token = torch.exp(-log_ratio_c_per_token) + log_ratio_c_per_token - 1.0
        # kl_penalty_r_per_token = torch.exp(-log_ratio_r_per_token) + log_ratio_r_per_token - 1.0

        # k2
        kl_penalty_c_per_token = 0.5 * (log_ratio_c_per_token ** 2)
        kl_penalty_r_per_token = 0.5 * (log_ratio_r_per_token ** 2)

        weight_c_per_token = residual_c.unsqueeze(-1)  # (batch, 1)
        weight_r_per_token = residual_r.unsqueeze(-1)

        # PPO-style clipping [1-eps, 1+eps] = [0.8, 1.2]
        clipped_ratio_c = torch.clamp(ratio_c_per_token, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
        clipped_ratio_r = torch.clamp(ratio_r_per_token, 1.0 - self.clip_eps, 1.0 + self.clip_eps)

        # Chosen (want to maximize term_c): 取 min → 悲观估计，防止 ratio 过高时获得虚假收益
        surr1_c = ratio_c_per_token * weight_c_per_token
        surr2_c = clipped_ratio_c * weight_c_per_token
        term_c_per_token = torch.min(surr1_c, surr2_c)

        # Rejected (want to minimize term_r): 取 max → 悲观估计，防止 ratio 过低时获得虚假惩罚
        surr1_r = ratio_r_per_token * weight_r_per_token
        surr2_r = clipped_ratio_r * weight_r_per_token
        term_r_per_token = torch.max(surr1_r, surr2_r)

        
        # 计算有效长度，避免除以 0
        pi_len_c = torch.clamp(pi_mask_c.sum(dim=-1), min=1.0)  # (batch,)
        pi_len_r = torch.clamp(pi_mask_r.sum(dim=-1), min=1.0)

        # 对每个序列的有效token求和
        term_c_sum = (term_c_per_token * pi_mask_c).sum(dim=-1) / pi_len_c  # (batch,)
        term_r_sum = (term_r_per_token * pi_mask_r).sum(dim=-1) / pi_len_r

        kl_c_sum = (kl_penalty_c_per_token * pi_mask_c).sum(dim=-1) / pi_len_c
        kl_r_sum = (kl_penalty_r_per_token * pi_mask_r).sum(dim=-1) / pi_len_r

        total_kl_penalty = (kl_c_sum + kl_r_sum) / 2.0

        j_sym = term_c_sum - term_r_sum - self.beta * total_kl_penalty
        loss = -j_sym.mean()

        # 记录指标（示例）
        with torch.no_grad():
            valid_ratio_c = ratio_c_per_token[pi_mask_c == 1]
            valid_ratio_r = ratio_r_per_token[pi_mask_r == 1]

            if valid_ratio_c.numel() > 0:
                ratio_c_mean = valid_ratio_c.mean().item()
                ratio_c_median = valid_ratio_c.median().item()
                ratio_c_max = valid_ratio_c.max().item()
                ratio_c_min = valid_ratio_c.min().item()
            else:
                ratio_c_mean = ratio_c_median = ratio_c_max = ratio_c_min = 0.0

            if valid_ratio_r.numel() > 0:
                ratio_r_mean = valid_ratio_r.mean().item()
                ratio_r_median = valid_ratio_r.median().item()
                ratio_r_max = valid_ratio_r.max().item()
                ratio_r_min = valid_ratio_r.min().item()
            else:
                ratio_r_mean = ratio_r_median = ratio_r_max = ratio_r_min = 0.0  

            micro_logs = {
                "loss": loss.item(),
                "j_sym": j_sym.mean().item(),
                "kl_penalty": total_kl_penalty.mean().item(),
                "ratio_c_mean": ratio_c_mean.item() if torch.is_tensor(ratio_c_mean) else ratio_c_mean,
                "ratio_r_mean": ratio_r_mean.item() if torch.is_tensor(ratio_r_mean) else ratio_r_mean,
                "ratio_c_median": ratio_c_median.item() if torch.is_tensor(ratio_c_median) else ratio_c_median,
                "ratio_r_median": ratio_r_median.item() if torch.is_tensor(ratio_r_median) else ratio_r_median,
                "ratio_c_max": ratio_c_max.item() if torch.is_tensor(ratio_c_max) else ratio_c_max,
                "ratio_r_max": ratio_r_max.item() if torch.is_tensor(ratio_r_max) else ratio_r_max,
                "ratio_c_min": ratio_c_min.item() if torch.is_tensor(ratio_c_min) else ratio_c_min,
                "ratio_r_min": ratio_r_min.item() if torch.is_tensor(ratio_r_min) else ratio_r_min,
                "f_Y_c": f_Y_c.mean().item(),
                "f_Y_r": f_Y_r.mean().item(),
            }
            # 分别累加到 train / eval accumulator
            if model.training:
                for key, value in micro_logs.items():
                    if key not in self._step_logs_accumulator:
                        self._step_logs_accumulator[key] = value
                    else:
                        self._step_logs_accumulator[key] += value
                self._step_counter += 1
            else:
                for key, value in micro_logs.items():
                    if key not in self._eval_logs_accumulator:
                        self._eval_logs_accumulator[key] = value
                    else:
                        self._eval_logs_accumulator[key] += value
                self._eval_step_counter += 1

        return (loss, dummy_outputs_c) if return_outputs else loss
    
    def log(self, logs: dict, *args, **kwargs) -> None:
        if getattr(self, '_step_counter', 0) > 0:
            avg_logs = {
                k: v / self._step_counter 
                for k, v in self._step_logs_accumulator.items()
            }
            # 将所有 Tensor 转换为 Python 标量
            avg_logs = {k: (v.item() if torch.is_tensor(v) else v) for k, v in avg_logs.items()}
            logs.update(avg_logs)
            self._step_logs_accumulator = {}
            self._step_counter = 0
        super().log(logs, *args, **kwargs)

    def evaluate(self, *args, **kwargs):
        """重写 evaluate，将 eval 阶段的自定义指标（KL、ratio 等）注入结果"""
        # 清空 eval accumulator
        self._eval_logs_accumulator = {}
        self._eval_step_counter = 0
        # 调用父类 evaluate（内部会多次调用 compute_loss，eval 指标会累积）
        result = super().evaluate(*args, **kwargs)
        # 将累积的 eval 自定义指标取平均后注入
        if self._eval_step_counter > 0:
            for k, v in self._eval_logs_accumulator.items():
                result[f"eval_{k}"] = v / self._eval_step_counter
        # 清空
        self._eval_logs_accumulator = {}
        self._eval_step_counter = 0
        return result

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """重写 prediction_step，接管评估阶段的前向传播"""
        # 1. 确保数据被放到正确的 GPU 上 (FSDP 环境必须)
        inputs = self._prepare_inputs(inputs)
        
        # 2. 强制使用我们写好的 compute_loss 来计算指标
        with torch.no_grad():
            # 这里依赖你 compute_loss 返回的 (loss, outputs_c)
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.detach()
            
        if prediction_loss_only:
            return (loss, None, None)
            
        # 3. 提取 logits 和 labels 塞给 HF 的底层评估引擎，防止其报错
        logits = outputs.logits if outputs is not None else None
        labels = inputs.get("chosen_labels", None)
        
        return (loss, logits, labels)

# --- 4. 独立实现的 Data Collator (含 Prompt Masking) ---
class ResidualDataCollator:
    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        batch = {}
        # 1. 基础标量特征提取
        batch["ref_logp_chosen"] = torch.tensor([f["ref_logp_chosen"] for f in features], dtype=torch.float32)
        batch["ref_logp_rejected"] = torch.tensor([f["ref_logp_rejected"] for f in features], dtype=torch.float32)
        batch["f_Y_chosen"] = torch.tensor([f["f_Y_chosen"] for f in features], dtype=torch.float32)
        batch["f_Y_rejected"] = torch.tensor([f["f_Y_rejected"] for f in features], dtype=torch.float32)
        
        prompts = [f["prompt"] for f in features]
        chosen_texts = [f["prompt"] + f["chosen"] for f in features]
        rejected_texts = [f["prompt"] + f["rejected"] for f in features]
        
        # 2. 仅对完整序列进行 Tokenize (保持统一的 padding_side="right")
        enc_c = self.tokenizer(chosen_texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt", add_special_tokens=False)
        enc_r = self.tokenizer(rejected_texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt", add_special_tokens=False)
        
        labels_c = enc_c["input_ids"].clone()
        labels_r = enc_r["input_ids"].clone()
        
        # 3. 屏蔽 Padding 部分 (设为 -100)
        labels_c[enc_c["attention_mask"] == 0] = -100
        labels_r[enc_r["attention_mask"] == 0] = -100
        
        # 4. 精准屏蔽 Prompt 部分 (前缀发散匹配法)
        for i, prompt_text in enumerate(prompts):
            # 单独 tokenize 当前的 prompt
            prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            
            # --- 处理 Chosen 边界 ---
            full_c_ids = enc_c["input_ids"][i].tolist()
            match_len_c = 0
            for p_id, f_id in zip(prompt_ids, full_c_ids):
                if p_id == f_id:
                    match_len_c += 1
                else:
                    break  # 遇到第一个不同的 Token，说明到达合并边界，立刻停止
            # 将完美匹配的前缀部分（纯 Prompt）设为 -100
            labels_c[i, :match_len_c] = -100
            
            # --- 处理 Rejected 边界 ---
            full_r_ids = enc_r["input_ids"][i].tolist()
            match_len_r = 0
            for p_id, f_id in zip(prompt_ids, full_r_ids):
                if p_id == f_id:
                    match_len_r += 1
                else:
                    break
            # 将完美匹配的前缀部分（纯 Prompt）设为 -100
            labels_r[i, :match_len_r] = -100
            
        batch["chosen_input_ids"] = enc_c["input_ids"]
        batch["chosen_attention_mask"] = enc_c["attention_mask"]
        batch["chosen_labels"] = labels_c
        
        batch["rejected_input_ids"] = enc_r["input_ids"]
        batch["rejected_attention_mask"] = enc_r["attention_mask"]
        batch["rejected_labels"] = labels_r
        
        return batch

# --- 5. 训练主流程 ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="/root/autodl-fs/train/Llama-3-8B-Instruct", help="Path to the SFT base model.")
    parser.add_argument("--data_path", type=str, default="/root/autodl-fs/train/ds_with_metrics", help="Path to the precomputed dataset.")
    parser.add_argument("--output_dir", default="/root/autodl-fs/train/output_model/Llama-3-8B-Instruct", help="Directory to save checkpoints and final model.")
    
    # --- 算法核心参数 ---
    parser.add_argument("--beta", type=float, default=0.5, help="KL penalty coefficient.")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clip epsilon, range = [1-eps, 1+eps] = [0.8, 1.2]")
    parser.add_argument("--max_length", type=int, default=2048, help="Max length for log prob calculation.")
    
    # --- 训练硬件与基础配置 ---
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=10)
    
    # --- 新增：标准的优化器与调度器超参 ---
    parser.add_argument("--optim", type=str, default="adamw_torch", help="The optimizer to use.")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Peak learning rate (usually 1e-6 to 5e-7 for RLHF/DPO).")
    # parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to prevent overfitting.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate schedule.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Percentage of total steps used for linear warmup.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping threshold.")
    
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # 强制设备挂载，保障集群多卡通信
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 统一使用 right padding，最符合自回归 CausalLM 的默认掩码习惯
    tokenizer.padding_side = "right" 
    
    print("Loading dataset...")
    dataset = load_from_disk(args.data_path)
    
    # 清洗逻辑：必须确保四项浮点特征均存在
    def is_valid(ex):
        keys = ["ref_logp_chosen", "ref_logp_rejected", "f_Y_chosen", "f_Y_rejected"]
        return all(ex.get(k) is not None for k in keys)

    # ================= 修复多卡文件踩踏 =================
    distributed_state = PartialState()
    with distributed_state.main_process_first():
        dataset = dataset.filter(is_valid)
    # ====================================================
    
    dataset = dataset.filter(is_valid)
    
    # 取前8个样本作为一个 mini batch 用于过拟合训练
    mini_batch_dataset = dataset.select(range(min(8, len(dataset))))
    # 取剩下的数据作为验证集（如果全量太大跑 eval 太慢，可以限制一个大小，比如 1000 条）
    eval_dataset = dataset.select(range(8, min(5000 + 8, len(dataset))))
    print(f"Mini-batch training dataset size: {len(mini_batch_dataset)}")
    print(f"Full evaluation dataset size: {len(eval_dataset)}")
    
    print("Loading policy model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        attn_implementation="flash_attention_2"  # <-- 新增
    )

    print("Loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        attn_implementation="flash_attention_2"  # <-- 新增
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=100, # 设置为 100 步用于 mini batch 的过拟合测试
        bf16=True,
        gradient_checkpointing=False, # 如果显存不够（OOM），把它改成 True
        # gradient_checkpointing_kwargs={"use_reentrant": False}, # <--- 加上这一行极其关键！它能解决 FSDP 下的报错并强制释放激活值显存
        logging_strategy="steps",
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=10,
        
        # 👇 加上这一行：彻底禁用 tqdm 进度条
        disable_tqdm=True,

        report_to=["tensorboard"],
        save_strategy="no",
        # save_steps=500,
        # save_total_limit=2,
        # save_only_model=True,
        # save_safetensors=True,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        
        # --- 引入新增的优化器与调度超参 ---
        optim=args.optim,
        learning_rate=args.learning_rate,
        # weight_decay=args.weight_decay,
        lr_scheduler_type="constant", # 测试过拟合时，建议固定学习率，排除 warmup/decay 的干扰
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
    )
    
    data_collator = ResidualDataCollator(tokenizer=tokenizer, max_length=args.max_length)
    
    trainer = ResidualRLHFTrainer(
        model=model,
        ref_model=ref_model,
        beta=args.beta,
        clip_eps=args.clip_eps,
        args=training_args,
        train_dataset=mini_batch_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[ResidualLoggingCallback()]
    )
    
    # ========== 关键修复：准备参考模型以适应 FSDP 分布式训练 ==========
    if trainer.accelerator.distributed_type != "NO":  # 如果使用了分布式（FSDP/DDP）
        # 用 accelerator.prepare 包装 ref_model，它会将模型分片并移动到当前设备
        trainer.ref_model = trainer.accelerator.prepare(trainer.ref_model)
        # prepare 后模型可能被设置为训练模式，重新设为 eval 并冻结梯度
        trainer.ref_model.eval()
        for param in trainer.ref_model.parameters():
            param.requires_grad = False
    # =================================================================

    print("Starting test training (100 steps)...")
    trainer.train()
    
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if trainer.accelerator.is_main_process:
        print("Test finished. Skipping model save (optional).")
        trainer.state.save_to_json(os.path.join(args.output_dir, "trainer_state.json"))

if __name__ == "__main__":
    main()