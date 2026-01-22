import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from typing import Dict, Any, List, Union, Tuple
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 随机种子设置函数（确保可重复性）
# ---------------------------------------------------------------------------

def seed_everything(seed=42):
    """
    设置所有随机数生成器的种子，确保训练的可重复性。
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# ---------------------------------------------------------------------------
# CustomSymPOTrainer
# ---------------------------------------------------------------------------

class CustomSymPOTrainer(Trainer):
    def __init__(
        self,
        model: Union[torch.nn.Module],
        ref_model: Union[torch.nn.Module], 
        args: TrainingArguments,
        beta_kl: float,
        clip_eps_min: float,
        clip_eps_max: float,
        max_length: int = None,
        **kwargs,
    ):
        super().__init__(model=model, args=args, **kwargs)

        if not hasattr(self, 'processor') or self.processor is None:
            self.processor = self.tokenizer

        
        self.beta_kl = beta_kl
        self.clip_eps_min = clip_eps_min
        self.clip_eps_max = clip_eps_max
        self.max_length = max_length

        if self.is_fsdp_enabled:
            # FSDP 模式：将 Ref Model 包装并分片
            self.ref_model = self.accelerator.prepare_model(
                ref_model, 
                evaluation_mode=True
            )
        else:
            # DDP 或 单卡模式：使用默认的 prepare (会自动放到正确设备)
            self.ref_model = self.accelerator.prepare_model(
                ref_model, 
                evaluation_mode=True
            )
            
        # 再次强制 eval 模式 (双重保险)
        self.ref_model.eval()
    


    def _get_log_probs(
        self, 
        model: AutoModelForCausalLM, 
        prompts: List[str], 
        responses: List[str], 
        max_length: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # 1. 获取 Processor
        if hasattr(self, "processing_class") and self.processing_class is not None:
            processor = self.processing_class
        else:
            processor = self.tokenizer
            
        device = self.args.device

        # 2. Tokenize (关掉特殊 token，避免双重 BOS)
        prompt_encodings = processor(
            prompts, 
            add_special_tokens=False, 
            padding=False, 
            truncation=False
        )
        response_encodings = processor(
            responses, 
            add_special_tokens=False, 
            padding=False, 
            truncation=False
        )

        input_ids_list = []
        loss_masks_list = []

        if max_length is None:
            max_length = 2048

        # 3. 拼接与构建 Mask
        for i in range(len(prompts)):
            p_ids = prompt_encodings['input_ids'][i]
            r_ids = response_encodings['input_ids'][i]
            
            combined_ids = p_ids + r_ids
            # Prompt=0, Response=1
            mask = [0] * len(p_ids) + [1] * len(r_ids)
            
            # 截断处理
            if len(combined_ids) > max_length:
                # 这是一个策略选择：保留尾部
                combined_ids = combined_ids[-max_length:]
                mask = mask[-max_length:]
            
            # 转 Tensor
            input_ids_list.append(torch.tensor(combined_ids, dtype=torch.long))
            loss_masks_list.append(torch.tensor(mask, dtype=torch.float)) # Float 用于后续计算

        # 4. Padding (Left Padding)
        
        # A. 处理 input_ids (使用 processor 以处理 padding_side 和 pad_token)
        # 确保 tokenizer 设置为左填充
        original_padding_side = processor.padding_side
        processor.padding_side = 'left'
        
        padded_inputs = processor.pad(
            {"input_ids": input_ids_list},
            padding='longest',
            max_length=max_length,
            return_tensors='pt'
        )
        input_ids = padded_inputs['input_ids'].to(device)
        attention_mask = padded_inputs['attention_mask'].to(device)
        
        max_batch_len = input_ids.shape[1]
        padded_loss_masks = torch.zeros((len(loss_masks_list), max_batch_len), dtype=torch.float, device=device)
        
        for i, mask_tensor in enumerate(loss_masks_list):
            seq_len = len(mask_tensor)
            # Left Padding: 数据放在最后
            if seq_len > 0:
                padded_loss_masks[i, -seq_len:] = mask_tensor.to(device)
            
        processor.padding_side = original_padding_side # 恢复设置

        # 5. 前向传播
        outputs = model(input_ids, attention_mask=attention_mask, return_dict=True, use_cache=False)
        logits = outputs.logits

        # 6. Shift
        shifted_logits = logits[..., :-1, :]
        shifted_labels = input_ids[..., 1:]
        
        # Mask 也要移位
        shifted_mask = padded_loss_masks[..., 1:] 
        shifted_attention_mask = attention_mask[..., 1:]

        # 结合 Attention Mask (双重保险)
        final_mask = shifted_mask * shifted_attention_mask.float()

        # 7. 计算 Loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        nll_per_token = loss_fct(
            shifted_logits.reshape(-1, shifted_logits.size(-1)), 
            shifted_labels.reshape(-1)
        )
        nll_per_token = nll_per_token.view(shifted_labels.size())
        
        log_probs_per_token = -nll_per_token
        
        # 应用 Mask
        masked_log_probs = log_probs_per_token * final_mask
        
        total_log_probs = masked_log_probs.sum(dim=1)
        valid_response_lens = final_mask.sum(dim=1)
        
        return total_log_probs, valid_response_lens

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        prompts = inputs.pop("prompt")
        chosen_responses = inputs.pop("chosen")
        rejected_responses = inputs.pop("rejected")
        
        f_chosen = inputs.pop("f_Y_chosen").to(self.args.device)
        f_rejected = inputs.pop("f_Y_rejected").to(self.args.device)
    
        batch_size = len(prompts)
        Z = torch.randint(0, 2, (batch_size,), device=self.args.device).float()
        combined_prompts = prompts + prompts
        combined_responses = chosen_responses + rejected_responses
        
        all_log_probs_pi, _ = self._get_log_probs(model, combined_prompts, combined_responses, max_length=self.max_length)        
        with torch.no_grad():
            all_log_probs_ref, _ = self._get_log_probs(self.ref_model, combined_prompts, combined_responses, max_length=self.max_length)
    
        # 拆分
        pi_chosen = all_log_probs_pi[:batch_size]
        pi_rejected = all_log_probs_pi[batch_size:]
        ref_chosen = all_log_probs_ref[:batch_size]
        ref_rejected = all_log_probs_ref[batch_size:]
    
        log_ratio_chosen = pi_chosen - ref_chosen
        log_ratio_rejected = pi_rejected - ref_rejected
        
        log_ratio_y1 = Z * log_ratio_chosen + (1 - Z) * log_ratio_rejected
        f_y2_selected = Z * f_rejected + (1 - Z) * f_chosen

        log_ratio_y1_safe = torch.clamp(log_ratio_y1, -20, 20) 
        ratio_y1 = torch.exp(log_ratio_y1_safe)
    
        
        with torch.no_grad():
            weight = Z - f_y2_selected
    
        # 3. 计算两个 Surrogate Objective (代理目标)
        # 第一项：未截断的原始目标
        surr1 = ratio_y1 * weight
        
        # 第二项：截断后的目标
        # 将 ratio 限制在 [1-eps, 1+eps] 范围内
        ratio_clipped = torch.clamp(ratio_y1, 1.0 - self.clip_eps_min, 1.0 + self.clip_eps_max)
        surr2 = ratio_clipped * weight

        objective = torch.min(surr1, surr2)
        loss_main = - objective.mean()
        total_loss = loss_main
        
        # Logs
        logs = {
            "loss": total_loss.item(),
            # "loss_main": loss_main.item(),
            # "loss_kl": loss_kl.item(),
            "avg_Z": Z.mean().item(), 
            "avg_weight": weight.abs().mean().item(),
            "ratio_mean": ratio_y1.mean().item(),
        }
        self._current_logs = logs
        
        if return_outputs:
            return (total_loss, None)
        return total_loss


    def log(self, logs: dict, *args, **kwargs) -> None:
        if hasattr(self, "_current_logs") and self._current_logs is not None:
            logs.update(self._current_logs)
            self._current_logs = None
        super().log(logs, *args, **kwargs)

@dataclass
class DataCollatorForCustomSymPO:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}
        for key in features[0].keys():
            if key.startswith("ref_logp"):
                continue
            if isinstance(features[0][key], str):
                batch[key] = [feature[key] for feature in features]
            else:
                batch[key] = torch.tensor([feature[key] for feature in features])
        return batch

class PrintingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and logs:
            pass # 避免过度打印，Trainer 自带 tqdm 和 log 打印

def parse_args():
    parser = argparse.ArgumentParser(description="Train a policy model with SymPO algorithm.")
    
    # --- 路径参数 ---
    parser.add_argument("--sft_model_path", type=str, default="/train/Llama-3-8B-Instruct", help="Path to the SFT base model.")
    parser.add_argument("--preprocessed_dataset_path", type=str, default="/train/ds_with_metrics", help="Path to the precomputed dataset.")
    parser.add_argument("--output_dir", type=str, default="/train/output_model/llama3-8b-sympo-5e-6_no_kl_2_4096_seed_42", help="Directory to save checkpoints and final model.")
    
    # --- 训练超参数 ---
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per GPU.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps.")
    parser.add_argument("--gradient_checkpointing", action='store_true', help="Enable gradient checkpointing.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log metrics every N steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save a checkpoint every N steps.")
    parser.add_argument("--save_total_limit", type=int, default=20, help="Limit number of saved checkpoints.")

    # --- SymPO 特定参数 ---
    parser.add_argument("--beta_kl", type=float, default=0.01, help="KL divergence penalty coefficient.")
    parser.add_argument("--clip_eps_min", type=float, default=0.9, help="Min clip value.")
    parser.add_argument("--clip_eps_max", type=float, default=9, help="Max clip value.")
    parser.add_argument("--max_length", type=int, default=4096, help="Max length for log prob calculation.")
    
    # --- W&B ---
    parser.add_argument("--report_to", type=str, default="wandb", help="Integration to report results to.")
    parser.add_argument("--run_name", type=str, default=f"policy-llama3-8b-sympo-default", help="W&B run name.")
    
    # --- 随机种子 ---
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"设置随机种子为: {args.seed}")
    seed_everything(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left' 

    print("从磁盘加载数据集...")
    precomputed_dataset_train = load_from_disk(args.preprocessed_dataset_path)

        # =============== 新增修复代码开始 ===============
    print("正在检查并过滤数据集中的空值 (None)...")
    
    # 1. 定义检查函数，确保关键字段不是 None
    def filter_none(example):
        # 检查关键的数值字段，根据你的数据集字段名调整
        for key in ['f_Y_chosen', 'f_Y_rejected']:
            if example.get(key) is None:
                return False
        return True

    original_len = len(precomputed_dataset_train)
    precomputed_dataset_train = precomputed_dataset_train.filter(filter_none)
    filtered_len = len(precomputed_dataset_train)
    
    if original_len != filtered_len:
        print(f"⚠️ 警告: 过滤掉了 {original_len - filtered_len} 条包含 None 的脏数据！")
    else:
        print("✅ 数据集检查通过，无 None 值。")
    # =============== 新增修复代码结束 ===============

    print("加载用于训练的策略模型 (Policy Model)...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path, 
        dtype=torch.bfloat16
    )

    print("加载作为参考的SFT模型 (Reference Model)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path, 
        dtype=torch.bfloat16
    )
    
    # [修正] 必须启用 eval 模式和禁用梯度，否则内存爆炸且计算错误
    print("冻结参考模型并设为 Eval 模式...")
    ref_model.eval()
    ref_model.requires_grad_(False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        optim="adamw_torch",
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        bf16=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit, 
        eval_strategy="no",
        remove_unused_columns=False,
        report_to=args.report_to,
        run_name=args.run_name,
    )
    
    data_collator = DataCollatorForCustomSymPO()

    print("初始化 CustomSymPOTrainer...")
    trainer = CustomSymPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=precomputed_dataset_train,
        data_collator=data_collator,
        callbacks=[PrintingCallback()],
        beta_kl=args.beta_kl,
        clip_eps_min=args.clip_eps_min,
        clip_eps_max=args.clip_eps_max,
        max_length=args.max_length,
    )
    
    print("开始分布式训练...")
    train_result = trainer.train()
    print("所有任务已完成！")

    ##################################
    # Save final model
    ##################################
    print("\n" + "=" * 60)
    print("*** Save final model ***")
    
    if trainer.is_fsdp_enabled and trainer.accelerator.state.fsdp_plugin is not None:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    
    trainer.save_model(args.output_dir)
    print(f"✅ 模型已保存到: {args.output_dir}")
    
    try:
        tokenizer.save_pretrained(args.output_dir)
    except Exception as e:
        print(f"⚠️  警告: 无法自动保存 tokenizer: {e}")
    
    if trainer.accelerator.is_main_process:
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)

if __name__ == "__main__":
    main()
