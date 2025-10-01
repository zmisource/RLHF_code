# 文件名: train_sympo_v2_argparse.py

# -*- coding: utf-8 -*-
import os
import torch
import argparse  # 1. 导入 argparse 库
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from typing import Dict, Any, List, Union
from dataclasses import dataclass

# CustomSymPOTrainer, DataCollatorForCustomSymPO, PrintingCallback 类定义...
# (这部分代码与原文件完全相同，为保持简洁在此省略)
# ...
# ---------------------------------------------------------------------------
# CustomSymPOTrainer
# ---------------------------------------------------------------------------

class CustomSymPOTrainer(Trainer):
    def __init__(
        self,
        model: Union[torch.nn.Module],
        args: TrainingArguments,
        sft_model_path: str,
        beta_kl: float,
        log_ratio_clip_min: float,
        log_ratio_clip_max: float,
        **kwargs,
    ):
        super().__init__(model=model, args=args, **kwargs)

        if not hasattr(self, 'processor') or self.processor is None:
            self.processor = self.tokenizer

        self.beta_kl = beta_kl
        self.log_ratio_clip_min = log_ratio_clip_min
        self.log_ratio_clip_max = log_ratio_clip_max
        self.sft_model_path = sft_model_path
        self.inference_engine = None

    def _get_log_probs(self, model: AutoModelForCausalLM, prompts: List[str], responses: List[str]) -> torch.Tensor:
        full_texts = [p + r for p, r in zip(prompts, responses)]
        prompt_tokens = self.processor(prompts, padding=False, truncation=False)
        prompt_lengths = [len(p) for p in prompt_tokens['input_ids']]
        unwrapped_model = model.module if hasattr(model, "module") else model
        max_len = unwrapped_model.config.max_position_embeddings
        full_tokens = self.processor(
            full_texts, padding='longest', truncation=True, # 已采纳优化建议
            max_length=max_len,
            return_tensors="pt"
        )
        input_ids = full_tokens['input_ids'].to(self.args.device)
        attention_mask = full_tokens['attention_mask'].to(self.args.device)
        # outputs = model(input_ids, attention_mask=attention_mask, return_dict=True)
        outputs = model(input_ids, attention_mask=attention_mask, return_dict=True, use_cache=False)
        logits = outputs.logits
        shifted_logits = logits[..., :-1, :]
        shifted_labels = input_ids[..., 1:]
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        nll_per_token = loss_fct(shifted_logits.reshape(-1, shifted_logits.size(-1)), shifted_labels.reshape(-1))
        nll_per_token = nll_per_token.view(input_ids.size(0), -1)
        log_probs_per_token = -nll_per_token
        seq_len = shifted_labels.size(1)
        position_ids = torch.arange(seq_len, device=self.args.device).expand_as(shifted_labels)
        prompt_lengths_tensor = torch.tensor(prompt_lengths, device=self.args.device).unsqueeze(1)
        response_start_index = prompt_lengths_tensor - 1
        mask = position_ids >= response_start_index
        attention_mask_shifted = attention_mask[:, 1:].to(torch.bool)
        response_mask = mask & attention_mask_shifted
        masked_log_probs = log_probs_per_token * response_mask
        total_log_probs = masked_log_probs.sum(dim=1)
        return total_log_probs

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        prompts = inputs.pop("prompt")
        chosen_responses = inputs.pop("chosen")
        rejected_responses = inputs.pop("rejected")
        log_probs_ref_y1 = inputs.pop("ref_log_probs_chosen").to(self.args.device)
        log_probs_ref_y2 = inputs.pop("ref_log_probs_rejected").to(self.args.device)
        f_y1 = inputs.pop("rewards_chosen").to(self.args.device)
        f_y2 = inputs.pop("rewards_rejected").to(self.args.device)
        # log_probs_pi_y1 = self._get_log_probs(model, prompts, chosen_responses )
        # log_probs_pi_y2 = self._get_log_probs(model, prompts, rejected_responses)
        batch_size = len(prompts)
        combined_prompts = prompts + prompts
        combined_responses = chosen_responses + rejected_responses
        all_log_probs = self._get_log_probs(model, combined_prompts, combined_responses)
        log_probs_pi_y1 = all_log_probs[:batch_size]
        log_probs_pi_y2 = all_log_probs[batch_size:]
        log_ratio_y1 = log_probs_pi_y1 - log_probs_ref_y1
        log_ratio_y2 = log_probs_pi_y2 - log_probs_ref_y2
        with torch.no_grad():
            kl_term1 = self.beta_kl * log_ratio_y1.detach()
            kl_term2 = self.beta_kl * log_ratio_y2.detach()
            weight1 = 1 - f_y2 - kl_term1
            weight2 = f_y1 + kl_term2
        clamped_log_ratio_y1 = torch.clamp(log_ratio_y1, min=self.log_ratio_clip_min, max=self.log_ratio_clip_max)
        ratio_y1 = torch.exp(clamped_log_ratio_y1)
        clamped_log_ratio_y2 = torch.clamp(log_ratio_y2, min=self.log_ratio_clip_min, max=self.log_ratio_clip_max)
        ratio_y2 = torch.exp(clamped_log_ratio_y2)
        J_sym_objective = ratio_y1 * weight1 - ratio_y2 * weight2
        total_loss = -J_sym_objective.mean()
        
        logs = {
            "mean_ratio_chosen": ratio_y1.detach().mean().item(),
            "mean_ratio_rejected": ratio_y2.detach().mean().item(),
            "weight_chosen": weight1.detach().mean().item(),
            "weight_rejected": weight2.detach().mean().item(),
        }
        self._current_logs = logs
        
        return total_loss

    def log(self, logs: dict, *args, **kwargs) -> None:
        if hasattr(self, "_current_logs") and self._current_logs is not None:
            logs.update(self._current_logs)
            self._current_logs = None
        super().log(logs, *args, **kwargs)

# ---------------------------------------------------------------------------
# Data Collator 和 Callback
# ---------------------------------------------------------------------------

@dataclass
class DataCollatorForCustomSymPO:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}
        for key in features[0].keys():
            if isinstance(features[0][key], str):
                batch[key] = [feature[key] for feature in features]
            else:
                batch[key] = torch.tensor([feature[key] for feature in features])
        return batch

class PrintingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and logs:
            print(logs)

# 2. 新建一个函数来管理所有参数
def parse_args():
    parser = argparse.ArgumentParser(description="Train a policy model with SymPO algorithm.")
    
    # --- 路径参数 ---
    parser.add_argument("--sft_model_path", type=str, default="/train/Llama-3-8B-Instruct", help="Path to the SFT base model.")
    parser.add_argument("--preprocessed_dataset_path", type=str, default="/train/preprocessed_traindataset", help="Path to the precomputed dataset.")
    parser.add_argument("--output_dir", type=str, default="/train/output_model/llama3-8b-sympo-default", help="Directory to save checkpoints and final model.")
    
    # --- 训练超参数 ---
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per GPU.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps.")
    parser.add_argument("--gradient_checkpointing", action='store_true', help="Enable gradient checkpointing to save memory.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for learning rate scheduler.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log metrics every N steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save a checkpoint every N steps.")
    parser.add_argument("--save_total_limit", type=int, default=20, help="Limit the total number of saved checkpoints.")

    # --- SymPO 特定参数 ---
    parser.add_argument("--beta_kl", type=float, default=0.1, help="KL divergence penalty coefficient.")
    parser.add_argument("--log_ratio_clip_min", type=float, default=-2.3, help="Minimum clip value for log probability ratio.")
    parser.add_argument("--log_ratio_clip_max", type=float, default=2.3, help="Maximum clip value for log probability ratio.")
    
    # --- W&B (Weights & Biases) 日志参数 ---
    parser.add_argument("--report_to", type=str, default="wandb", help="The integration to report results to (e.g., 'wandb').")
    parser.add_argument("--run_name", type=str, default=f"policy-llama3-8b-sympo-default", help="A name for the W&B run.")

    return parser.parse_args()

# ---------------------------------------------------------------------------
# 主训练流程
# ---------------------------------------------------------------------------
def main():
    # 3. 在 main 函数开头调用 parse_args
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left' 

    print("从磁盘加载预处理好的数据集...")
    precomputed_dataset_train = load_from_disk(args.preprocessed_dataset_path)

    print("加载用于训练的策略模型...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
    )

    # 4. 使用 args 对象来填充 TrainingArguments
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
        args=training_args,
        # 5. 使用 args 对象来填充 SymPO 参数
        sft_model_path=args.sft_model_path, 
        tokenizer=tokenizer,
        train_dataset=precomputed_dataset_train,
        data_collator=data_collator,
        callbacks=[PrintingCallback()],
        beta_kl=args.beta_kl,
        log_ratio_clip_min=args.log_ratio_clip_min,
        log_ratio_clip_max=args.log_ratio_clip_max,
    )

    print("开始分布式训练...")
    trainer.train()
    print("所有任务已完成！")

if __name__ == "__main__":
    main()