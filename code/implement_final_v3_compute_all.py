import os
import torch
import argparse
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    AutoModelForSequenceClassification # 1. 新增导入
)
from typing import Dict, Any, List, Union
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# CustomSymPOTrainer
# ---------------------------------------------------------------------------

class CustomSymPOTrainer(Trainer):
    def __init__(
        self,
        model: Union[torch.nn.Module],
        ref_model: Union[torch.nn.Module],
        reward_model: Union[torch.nn.Module], # 2. 新增 reward_model 参数
        args: TrainingArguments,
        beta_kl: float,
        log_ratio_clip_min: float,
        log_ratio_clip_max: float,
        **kwargs,
    ):
        super().__init__(model=model, args=args, **kwargs)

        if not hasattr(self, 'processor') or self.processor is None:
            self.processor = self.tokenizer

        self.ref_model = ref_model.to(self.args.device)
        self.reward_model = reward_model.to(self.args.device) # 3. 保存 reward_model 并移动到设备
        self.beta_kl = beta_kl
        self.log_ratio_clip_min = log_ratio_clip_min
        self.log_ratio_clip_max = log_ratio_clip_max

    def _get_log_probs(self, model: AutoModelForCausalLM, prompts: List[str], responses: List[str]) -> torch.Tensor:
        # ... 此函数无需修改
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
        
    # 4. 新增一个辅助函数用于计算 rewards
    def _get_rewards(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        texts = [p + r for p, r in zip(prompts, responses)]
        # 注意：奖励模型的 tokenizer 可能有不同的 max_length，这里使用其自身的配置
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, 
                                max_length=self.reward_model.config.max_position_embeddings)
        inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.reward_model(**inputs)
            # 直接返回 tensor，不再移回 cpu
            return torch.sigmoid(outputs.logits.squeeze(-1))

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        prompts = inputs.pop("prompt")
        chosen_responses = inputs.pop("chosen")
        rejected_responses = inputs.pop("rejected")
        
        batch_size = len(prompts)
        combined_prompts = prompts + prompts
        combined_responses = chosen_responses + rejected_responses

        # --- 动态计算 Rewards ---
        with torch.no_grad():
            all_rewards = self._get_rewards(combined_prompts, combined_responses)
            f_y1 = all_rewards[:batch_size]
            f_y2 = all_rewards[batch_size:]

        # --- 动态计算参考模型的 log_probs ---
        with torch.no_grad():
            all_log_probs_ref = self._get_log_probs(self.ref_model, combined_prompts, combined_responses)
            log_probs_ref_y1 = all_log_probs_ref[:batch_size]
            log_probs_ref_y2 = all_log_probs_ref[batch_size:]

        # --- 计算策略模型的 log_probs ---
        all_log_probs_pi = self._get_log_probs(model, combined_prompts, combined_responses)
        log_probs_pi_y1 = all_log_probs_pi[:batch_size]
        log_probs_pi_y2 = all_log_probs_pi[batch_size:]
        if self.state.global_step == 0: # 只在第一步打印
            print(f"--- Sanity Check at Step 0 ---")
            print(f"Sample 0 - ref_logp_chosen: {log_probs_ref_y1[0].item()}")
            print(f"Sample 0 - pi_logp_chosen:  {log_probs_pi_y1[0].item()}")
            print(f"Sample 0 - Difference (pi - ref): {log_probs_pi_y1[0].item() - log_probs_ref_y1[0].item()}")
            print(f"---------------------------------")
        # --- 后续损失计算逻辑不变 ---
        log_ratio_y1 = log_probs_pi_y1 - log_probs_ref_y1
        log_ratio_y2 = log_probs_pi_y2 - log_probs_ref_y2
        
        with torch.no_grad():
            kl_term1 = self.beta_kl * log_ratio_y1.detach()
            kl_term2 = self.beta_kl * log_ratio_y2.detach()
            weight1 = 1 - f_y2 - kl_term1
            weight2 = f_y1 + kl_term2
            
        # clamped_log_ratio_y1 = torch.clamp(log_ratio_y1, min=self.log_ratio_clip_min, max=self.log_ratio_clip_max)
        # ratio_y1 = torch.exp(clamped_log_ratio_y1)
        # clamped_log_ratio_y2 = torch.clamp(log_ratio_y2, min=self.log_ratio_clip_min, max=self.log_ratio_clip_max)
        # ratio_y2 = torch.exp(clamped_log_ratio_y2)
        ratio_y1 = torch.exp(log_ratio_y1)
        ratio_y2 = torch.exp(log_ratio_y2)
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

# ... DataCollator 和 Callback 无需修改，但 DataCollator 现在只处理文本字段
@dataclass
class DataCollatorForCustomSymPO:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 现在只剩下 prompt, chosen, rejected 三个文本字段
        return {key: [feature[key] for feature in features] for key in features[0].keys()}
    
class PrintingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and logs:
            print(logs)
# ... argparse 函数需要增加 reward_model_path
def parse_args():
    parser = argparse.ArgumentParser(description="Train a policy model with SymPO algorithm.")
    # ... 其他参数
    parser.add_argument("--sft_model_path", type=str, default="/train/Llama-3-8B-Instruct")
    parser.add_argument("--reward_model_path", type=str, default="/train/f_model") # 新增
    parser.add_argument("--preprocessed_dataset_path", type=str, default="/train/traindataset_1000_v5")
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

def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left' 

    print("从磁盘加载纯文本数据集...")
    text_dataset_train = load_from_disk(args.preprocessed_dataset_path)

    print("加载用于训练的策略模型 (Policy Model)...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
    )

    print("加载作为参考的SFT模型 (Reference Model)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
    )
    ref_model.eval()

    # 5. 新增：加载奖励模型 (Reward Model)
    print("加载奖励模型 (Reward Model)...")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_path, num_labels=1, attn_implementation="flash_attention_2", dtype=torch.bfloat16
    )
    if reward_model.config.pad_token_id is None:
        reward_model.config.pad_token_id = tokenizer.pad_token_id
    reward_model.eval()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing=False,
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
        reward_model=reward_model, # 6. 将 reward_model 传递给 Trainer
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=text_dataset_train,
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