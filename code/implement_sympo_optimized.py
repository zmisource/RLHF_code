import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from datasets import load_from_disk, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    PreTrainedTokenizerBase,
    DataCollatorWithPadding
)
from typing import Dict, Any, List, Union, Tuple, Optional
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 随机种子设置函数
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
# Helper Functions & Data Preprocessing (DRPO Style)
# ---------------------------------------------------------------------------

def build_sympo_dataset(
    dataset: Dataset, 
    tokenizer: PreTrainedTokenizerBase, 
    max_length: int = 4096,
    num_proc: int = 4
) -> Dataset:
    """
    预处理数据集：提前进行 Tokenization，避免在训练 Loop 中重复计算。
    """
    
    def tokenize_function(examples):
        new_examples = {
            "chosen_input_ids": [],
            "chosen_attention_mask": [],
            "chosen_labels": [],
            "rejected_input_ids": [],
            "rejected_attention_mask": [],
            "rejected_labels": [],
            "f_Y_chosen": [],
            "f_Y_rejected": []
        }
        
        prompts = examples["prompt"]
        chosens = examples["chosen"]
        rejecteds = examples["rejected"]
        f_y1s = examples["f_Y_chosen"]
        f_y2s = examples["f_Y_rejected"]
        
        for prompt, chosen, rejected, fy1, fy2 in zip(prompts, chosens, rejecteds, f_y1s, f_y2s):
            # 这里的逻辑是：分别构建 (Prompt + Chosen) 和 (Prompt + Rejected)
            # 并且设置 Labels，使得 Prompt 部分为 -100 (被忽略计算 Loss)，Response 部分保留 Token ID
            
            def process_pair(p, r):
                # 1. Full text
                full_text = p + r
                
                # 2. Tokenize full text
                full_tokens = tokenizer(
                    full_text,
                    add_special_tokens=False, # 假设 prompt 已经包含必要的 special tokens 或者由外部控制
                    truncation=True,
                    max_length=max_length
                )
                input_ids = full_tokens["input_ids"]
                attention_mask = full_tokens["attention_mask"]
                
                # 3. Tokenize prompt only (to determine masking length)
                # 注意：为了准确匹配，这里应该使用相同的配置
                prompt_tokens = tokenizer(
                    p,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_length
                )
                prompt_len = len(prompt_tokens["input_ids"])
                
                # 4. Create labels
                # 将 Prompt 部分设为 -100 (Ignore Index)
                labels = [-100] * len(input_ids)
                
                # 如果 prompt 被截断导致比 full_text 还长（理论上不应发生），做个保护
                if prompt_len < len(input_ids):
                    # 将 Response 部分填回 ID
                    for i in range(prompt_len, len(input_ids)):
                        labels[i] = input_ids[i]
                
                return input_ids, attention_mask, labels

            c_ids, c_mask, c_lbls = process_pair(prompt, chosen)
            r_ids, r_mask, r_lbls = process_pair(prompt, rejected)
            
            new_examples["chosen_input_ids"].append(c_ids)
            new_examples["chosen_attention_mask"].append(c_mask)
            new_examples["chosen_labels"].append(c_lbls)
            new_examples["rejected_input_ids"].append(r_ids)
            new_examples["rejected_attention_mask"].append(r_mask)
            new_examples["rejected_labels"].append(r_lbls)
            
            # Pass through scalar values
            new_examples["f_Y_chosen"].append(fy1)
            new_examples["f_Y_rejected"].append(fy2)
            
        return new_examples

    # 移除原始文本列，只保留处理后的 Tensor
    remove_columns = dataset.column_names
    processed_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=remove_columns,
        desc="Tokenizing and Creating Labels"
    )
    return processed_dataset

@dataclass
class SymPODataCollator:
    """
    自定义 Collator：
    1. 分别对 Chosen 和 Rejected 序列进行 Padding。
    2. 将它们组合成一个 Batch 返回。
    """
    tokenizer: PreTrainedTokenizerBase
    model_ignore_index: int = -100
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}
        
        # 提取标量特征
        batch["f_Y_chosen"] = torch.tensor([f["f_Y_chosen"] for f in features], dtype=torch.float32)
        batch["f_Y_rejected"] = torch.tensor([f["f_Y_rejected"] for f in features], dtype=torch.float32)
        
        # Helper to pad a specific key
        def pad_sequence(key_prefix):
            # 收集 input_ids, labels, attention_mask
            input_ids = [f[f"{key_prefix}_input_ids"] for f in features]
            labels = [f[f"{key_prefix}_labels"] for f in features]
            attention_mask = [f[f"{key_prefix}_attention_mask"] for f in features]
            
            # Left padding for generation? No, usually training uses Right padding unless specifically needed.
            # 但是原代码中提到了 left padding。
            # 对于 Causal LM Training (Teacher Forcing)，通常 Right Padding 比较简单，
            # 只要 Attention Mask 设置正确即可。
            # 为了兼容性，这里我们使用 tokenizer.pad_token_id 进行 right padding。
            
            # 使用 torch.nn.utils.rnn.pad_sequence (默认 right pad)
            input_ids_padded = self._pad(input_ids, self.tokenizer.pad_token_id)
            labels_padded = self._pad(labels, self.model_ignore_index)
            attention_mask_padded = self._pad(attention_mask, 0)
            
            return input_ids_padded, attention_mask_padded, labels_padded

        c_ids, c_mask, c_lbls = pad_sequence("chosen")
        r_ids, r_mask, r_lbls = pad_sequence("rejected")
        
        batch["chosen_input_ids"] = c_ids
        batch["chosen_attention_mask"] = c_mask
        batch["chosen_labels"] = c_lbls
        
        batch["rejected_input_ids"] = r_ids
        batch["rejected_attention_mask"] = r_mask
        batch["rejected_labels"] = r_lbls
        
        return batch

    def _pad(self, sequence_list, pad_value):
        # Convert to tensor and pad
        tensors = [torch.tensor(s, dtype=torch.long) for s in sequence_list]
        return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=pad_value)

# ---------------------------------------------------------------------------
# Optimized CustomSymPOTrainer
# ---------------------------------------------------------------------------

class CustomSymPOTrainer(Trainer):
    def __init__(
        self,
        model: Union[torch.nn.Module],
        ref_model: Union[torch.nn.Module], 
        args: TrainingArguments,
        beta_kl: float,
        log_ratio_clip_min: float,
        log_ratio_clip_max: float,
        **kwargs,
    ):
        super().__init__(model=model, args=args, **kwargs)
        
        # 确保 ref_model 在正确的设备上
        self.ref_model = ref_model.to(self.args.device)
        self.beta_kl = beta_kl
        self.log_ratio_clip_min = log_ratio_clip_min
        self.log_ratio_clip_max = log_ratio_clip_max
        self.use_smooth_clip = False 
        
        # Loss function with ignore_index handling
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)

    def _smooth_clamp(self, x: torch.Tensor, min_val: float, max_val: float, temperature: float = 0.1) -> torch.Tensor:
        normalized = 2.0 * (x - min_val) / (max_val - min_val) - 1.0
        clipped_normalized = torch.tanh(normalized / temperature)
        return (clipped_normalized + 1.0) / 2.0 * (max_val - min_val) + min_val
    
    def _compute_per_token_logps(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算逐 Token 的 Log Probabilities。
        返回: [Batch, SeqLen-1] 的 LogProbs (未求和)
        """
        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits
        
        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_labels = labels[..., 1:].contiguous()
        
        # Compute loss (per token) -> NLL
        nll_per_token = self.loss_fct(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_labels.view(-1)
        )
        nll_per_token = nll_per_token.view(shifted_labels.shape)
        
        # Log Prob = - NLL
        # 注意: ignore_index 位置是 0，即 LogProb 为 0，不影响 Sum
        return -nll_per_token

    def _compute_kl_divergence(
        self,
        logps_pi: torch.Tensor,
        logps_ref: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        DRPO 风格的 KL 散度计算 (Estimator with Control Variate)
        From DRPO: ((exp(ref - pi) - (ref - pi) - 1) * mask).sum()
        该估计器是非负且方差较小的 KL(Pi || Ref) 估计。
        """
        # 这里的输入 logps 是 per_token 的 (且已经在 ignore_index 处为 0)
        # 但我们需要 mask 只要在有效位置计算，否则 0 - 0 - 1 = -1 会导致错误积累
        
        # 还原 Mask (False at ignored positions)
        # shifted_labels 对应 per_token 的形状
        shifted_labels = labels[..., 1:].contiguous()
        mask = (shifted_labels != -100).float()
        
        # Diff = Log(Ref) - Log(Pi)
        # 注意：这里传入的 logps 已经在 ignore 处为0了，直接相减也是0
        # 但为了严谨，我们先计算 diff，再乘 mask
        diff = logps_ref - logps_pi
        
        # Formula: e^x - x - 1
        # kl_per_token = (torch.exp(diff) - diff - 1)
        # Apply mask
        kl_per_token = (torch.exp(diff) - diff - 1) * mask
        
        # Sum over sequence, Mean over batch
        return kl_per_token.sum(dim=-1).mean()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 1. 解包数据
        f_y1 = inputs["f_Y_chosen"].to(self.args.device)
        f_y2 = inputs["f_Y_rejected"].to(self.args.device)
        
        chosen_input_ids = inputs["chosen_input_ids"].to(self.args.device)
        chosen_attention_mask = inputs["chosen_attention_mask"].to(self.args.device)
        chosen_labels = inputs["chosen_labels"].to(self.args.device)
        
        rejected_input_ids = inputs["rejected_input_ids"].to(self.args.device)
        rejected_attention_mask = inputs["rejected_attention_mask"].to(self.args.device)
        rejected_labels = inputs["rejected_labels"].to(self.args.device)

        # 2. 计算 Log Probs (Per Token)
        per_token_logps_pi_y1 = self._compute_per_token_logps(model, chosen_input_ids, chosen_attention_mask, chosen_labels)
        per_token_logps_pi_y2 = self._compute_per_token_logps(model, rejected_input_ids, rejected_attention_mask, rejected_labels)
        
        # 3. 计算 Log Probs (Ref Model)
        with torch.no_grad():
            per_token_logps_ref_y1 = self._compute_per_token_logps(self.ref_model, chosen_input_ids, chosen_attention_mask, chosen_labels)
            per_token_logps_ref_y2 = self._compute_per_token_logps(self.ref_model, rejected_input_ids, rejected_attention_mask, rejected_labels)
        
        # 4. 聚合为 Sequence Log Probs (用于 SymPO Core Loss)
        log_probs_pi_y1 = per_token_logps_pi_y1.sum(dim=1)
        log_probs_pi_y2 = per_token_logps_pi_y2.sum(dim=1)
        log_probs_ref_y1 = per_token_logps_ref_y1.sum(dim=1)
        log_probs_ref_y2 = per_token_logps_ref_y2.sum(dim=1)
        

        kl_drpo_chosen = self._compute_kl_divergence(per_token_logps_pi_y1, per_token_logps_ref_y1, chosen_labels)

        if self.state.global_step == 0:
            print(f"--- Sanity Check at Step 0 ---")
            print(f"Sample 0 - ref_logp_chosen: {log_probs_ref_y1[0].item()}")
            print(f"Sample 0 - pi_logp_chosen:  {log_probs_pi_y1[0].item()}")
            print(f"KL (DRPO) - chosen:         {kl_drpo_chosen.item()}")
            print(f"---------------------------------")
        
        # 6. 计算 SymPO Loss
        log_ratio_y1 = log_probs_pi_y1 - log_probs_ref_y1
        log_ratio_y2 = log_probs_pi_y2 - log_probs_ref_y2
        
        with torch.no_grad():
            # [Fix Risk] 如果 log_ratio 极端，kl_term1 会导致 weight1 爆炸。
            # 虽然保持核心 Loss 不变，但为了防止 Nan，可以考虑 clip。
            # 这里暂时保持原样，完全遵守原公式。
            
            weight1 = 1 - f_y2
            weight2 = f_y1

        if self.use_smooth_clip:
            clamped_log_ratio_y1 = self._smooth_clamp(log_ratio_y1, self.log_ratio_clip_min, self.log_ratio_clip_max)
            clamped_log_ratio_y2 = self._smooth_clamp(log_ratio_y2, self.log_ratio_clip_min, self.log_ratio_clip_max)
        else:
            clamped_log_ratio_y1 = torch.clamp(log_ratio_y1, min=self.log_ratio_clip_min, max=self.log_ratio_clip_max)
            clamped_log_ratio_y2 = torch.clamp(log_ratio_y2, min=self.log_ratio_clip_min, max=self.log_ratio_clip_max)
        
        ratio_y1 = torch.exp(clamped_log_ratio_y1)
        ratio_y2 = torch.exp(clamped_log_ratio_y2)
        
        J_sym_objective = ratio_y1 * weight1 - ratio_y2 * weight2 - self.beta_kl * kl_drpo_chosen
        total_loss = -J_sym_objective.mean()
        
        # 7. Logging & Cleanup
        logs = {
            "mean_ratio_chosen": ratio_y1.detach().mean().item(),
            "mean_ratio_rejected": ratio_y2.detach().mean().item(),
            "weight_chosen": weight1.detach().mean().item(),
            "weight_rejected": weight2.detach().mean().item(),
            "kl_drpo_exact_chosen": kl_drpo_chosen.item(), # 新增指标
            "loss": total_loss.item()
        }
        self._current_logs = logs
        
        # Explicit delete to match DRPO style memory management
        del chosen_input_ids, chosen_attention_mask, chosen_labels
        del rejected_input_ids, rejected_attention_mask, rejected_labels
        del per_token_logps_pi_y1, per_token_logps_pi_y2
        
        if return_outputs:
            return (total_loss, None)
        return total_loss

    def log(self, logs: dict, *args, **kwargs) -> None:
        if hasattr(self, "_current_logs") and self._current_logs is not None:
            logs.update(self._current_logs)
            self._current_logs = None
        super().log(logs, *args, **kwargs)

class PrintingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        pass

def parse_args():
    parser = argparse.ArgumentParser(description="Train a policy model with SymPO algorithm (Optimized).")
    
    # --- 路径参数 ---
    parser.add_argument("--sft_model_path", type=str, default="/train/Llama-3-8B-Instruct", help="Path to the SFT base model.")
    parser.add_argument("--preprocessed_dataset_path", type=str, default="/train/ds_with_metrics", help="Path to the precomputed dataset.")
    parser.add_argument("--output_dir", type=str, default="/train/output_model/llama3-8b-sympo-optimized", help="Directory to save checkpoints.")
    
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
    parser.add_argument("--num_proc", type=int, default=8, help="Number of processors for dataset mapping.")

    # --- SymPO 特定参数 ---
    parser.add_argument("--beta_kl", type=float, default=0.01, help="KL divergence penalty coefficient.")
    parser.add_argument("--log_ratio_clip_min", type=float, default=-2.3, help="Min clip value.")
    parser.add_argument("--log_ratio_clip_max", type=float, default=2.3, help="Max clip value.")
    parser.add_argument("--use_smooth_clip", type=bool, default=True, help="Use smooth clipping.")
    parser.add_argument("--max_length", type=int, default=4096, help="Max length for inputs.")
    
    # --- W&B ---
    parser.add_argument("--report_to", type=str, default="wandb", help="Integration to report results to.")
    parser.add_argument("--run_name", type=str, default=f"policy-sympo-optimized", help="W&B run name.")
    
    # --- 随机种子 ---
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"设置随机种子为: {args.seed}")
    seed_everything(args.seed)

    print("加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Note: padding_side is less critical now as Collator handles it, 
    # but strictly speaking for training generation usually left, but for pure training right is fine.
    # DataCollator pad_sequence uses right padding by default.
    tokenizer.padding_side = 'right' 

    print("从磁盘加载数据集...")
    dataset = load_from_disk(args.preprocessed_dataset_path)

    # ----------------------------------------------------
    # 数据集过滤 (None Check)
    # ----------------------------------------------------
    print("正在检查并过滤数据集中的空值 (None)...")
    def filter_none(example):
        for key in ['f_Y_chosen', 'f_Y_rejected', 'prompt', 'chosen', 'rejected']:
            if example.get(key) is None:
                return False
        return True

    original_len = len(dataset)
    dataset = dataset.filter(filter_none)
    filtered_len = len(dataset)
    print(f"数据清洗完成: {original_len} -> {filtered_len}")

    # ----------------------------------------------------
    # 新增: 预处理步骤 (Tokenization)
    # ----------------------------------------------------
    print("开始预处理数据集 (Tokenization)...")
    train_dataset = build_sympo_dataset(
        dataset, 
        tokenizer, 
        max_length=args.max_length,
        num_proc=args.num_proc
    )
    print(f"预处理完成，样本示例 keys: {train_dataset[0].keys()}")

    print("加载模型...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path, 
        dtype=torch.bfloat16,
        use_cache=False # Training doesn't need cache
    )

    ref_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path, 
        dtype=torch.bfloat16,
        use_cache=False
    )
    
    print("冻结参考模型...")
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
        report_to=args.report_to,
        run_name=args.run_name,
        remove_unused_columns=False, # 关键：防止 Dataset 列被 Trainer 自动删除
    )
    
    # 使用自定义 Collator
    data_collator = SymPODataCollator(tokenizer=tokenizer)

    print("初始化 Trainer...")
    trainer = CustomSymPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[PrintingCallback()],
        beta_kl=args.beta_kl,
        log_ratio_clip_min=args.log_ratio_clip_min,
        log_ratio_clip_max=args.log_ratio_clip_max,
    )
    
    trainer.use_smooth_clip = args.use_smooth_clip

    print("开始训练...")
    train_result = trainer.train()
    
    print("保存模型...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    if trainer.accelerator.is_main_process:
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)

if __name__ == "__main__":
    main()
