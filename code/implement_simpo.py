import os
import torch
import torch.nn.functional as F # 确保 F 已导入
import argparse
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


class CustomSimPOTrainer(Trainer):
    def __init__(
        self,
        model: Union[torch.nn.Module],
        args: TrainingArguments,
        beta: float,
        gamma: float,
        **kwargs,
    ):
        super().__init__(model=model, args=args, **kwargs)

        if not hasattr(self, 'processor') or self.processor is None:
            self.processor = self.tokenizer

        self.beta = beta 
        self.gamma = gamma 

    def _get_log_probs(self, model: AutoModelForCausalLM, prompts: List[str], responses: List[str]) -> torch.Tensor:
        original_padding_side = self.processor.padding_side
        self.processor.padding_side = 'left'
        full_texts = [p + r for p, r in zip(prompts, responses)]
        prompt_tokens = self.processor(prompts, padding=False, truncation=False)
        prompt_lengths = [len(p) for p in prompt_tokens['input_ids']]
        unwrapped_model = model.module if hasattr(model, "module") else model
        max_len = unwrapped_model.config.max_position_embeddings
        full_tokens = self.processor(
            full_texts, padding='longest', truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
        input_ids = full_tokens['input_ids'].to(self.args.device)
        attention_mask = full_tokens['attention_mask'].to(self.args.device)
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
        self.processor.padding_side = original_padding_side
        return total_log_probs


    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        prompts = inputs.pop("prompt")
        chosen_responses = inputs.pop("chosen")
        rejected_responses = inputs.pop("rejected")

        batch_size = len(prompts)
        combined_prompts = prompts + prompts
        combined_responses = chosen_responses + rejected_responses
        all_log_probs_pi = self._get_log_probs(model, combined_prompts, combined_responses)
        log_probs_pi_chosen = all_log_probs_pi[:batch_size]
        log_probs_pi_rejected = all_log_probs_pi[batch_size:]
        
        # log_probs_pi_chosen = self._get_log_probs(model, prompts, chosen_responses)
        # log_probs_pi_rejected = self._get_log_probs(model, prompts, rejected_responses)
        
        chosen_tokens = self.processor(chosen_responses, add_special_tokens=False)
        chosen_lengths = torch.tensor([len(t) for t in chosen_tokens['input_ids']], device=model.device)
        rejected_tokens = self.processor(rejected_responses, add_special_tokens=False)
        rejected_lengths = torch.tensor([len(t) for t in rejected_tokens['input_ids']], device=model.device)
        
        chosen_lengths = torch.clamp(chosen_lengths, min=1)
        rejected_lengths = torch.clamp(rejected_lengths, min=1)

        chosen_scores = self.beta * (log_probs_pi_chosen / chosen_lengths)
        rejected_scores = self.beta * (log_probs_pi_rejected / rejected_lengths)
        
        logits = chosen_scores - rejected_scores - self.gamma
        
        simpo_loss = -F.logsigmoid(logits).mean()
        total_loss = simpo_loss
        
        logs = {"simpo_loss": simpo_loss.detach().item()}
        self._current_logs = logs

        return total_loss


    # log 方法无需修改
    def log(self, logs: dict, *args, **kwargs) -> None:
        if hasattr(self, "_current_logs") and self._current_logs is not None:
            logs.update(self._current_logs)
            self._current_logs = None
        super().log(logs, *args, **kwargs)

@dataclass
class DataCollatorForCustomSimPO: 
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}
        required_keys = ["prompt", "chosen", "rejected"]
        for key in required_keys:
            if key in features[0]:
                batch[key] = [feature[key] for feature in features]
            else:
                raise ValueError(f"DataCollator 缺少必需的键: {key}")
        return batch

class PrintingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and logs:
            print(logs)

# <--- 修改：更新 argparse ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train a policy model with length-normalized SimPO algorithm.")
    
    # --- 路径参数 ---
    parser.add_argument("--sft_model_path", type=str, default="/train/Llama-3-8B-Instruct", help="Path to the SFT base model.")
    parser.add_argument("--preprocessed_dataset_path", type=str, default="/train/precomputed_traindataset", help="Path to the precomputed dataset.")
    parser.add_argument("--output_dir", type=str, default="/train/output_model/llama3-8b-simpo-1e-6", help="Directory to save checkpoints and final model.")
    
    # --- 训练超参数 (不变) ---
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per GPU.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps.")
    parser.add_argument("--gradient_checkpointing", action='store_true', help="Enable gradient checkpointing to save memory.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for learning rate scheduler.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log metrics every N steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save a checkpoint every N steps.")
    parser.add_argument("--save_total_limit", type=int, default=20, help="Limit the total number of saved checkpoints.")

    # --- SimPO 特定参数 (按新 loss 修改) ---
    parser.add_argument("--beta", type=float, default=2.5, help="Beta coefficient for the length-normalized SimPO loss.")
    parser.add_argument("--gamma", type=float, default=1.4, help="Margin parameter gamma for the SimPO loss (default: 0.0).")
    
    # --- W&B (Weights & Biases) 日志参数 ---
    parser.add_argument("--report_to", type=str, default="wandb", help="The integration to report results to (e.g., 'wandb').")
    parser.add_argument("--run_name", type=str, default=f"policy-llama3-8b-simpo-len-norm", help="A name for the W&B run.")

    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left' 

    print("从磁盘加载数据集 (注意: 仅需要 prompt, chosen, rejected)...")
    precomputed_dataset_train = load_from_disk(args.preprocessed_dataset_path)

    print("加载用于训练的策略模型 (Policy Model)...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
    )


    # TrainingArguments 定义无需修改
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        # gradient_checkpointing=False,
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
    
    data_collator = DataCollatorForCustomSimPO()

    print("初始化 CustomSimPOTrainer (Length-Normalized)...")
    trainer = CustomSimPOTrainer(
        model=policy_model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=precomputed_dataset_train,
        data_collator=data_collator,
        callbacks=[PrintingCallback()],
        beta=args.beta, 
        gamma=args.gamma, 
    )

    print("开始分布式训练 (Length-Normalized SimPO 损失)...")
    trainer.train()
    print("所有任务已完成！")

if __name__ == "__main__":
    main()