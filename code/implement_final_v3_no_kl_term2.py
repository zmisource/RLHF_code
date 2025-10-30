import os
import torch
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

# ---------------------------------------------------------------------------
# CustomSymPOTrainer
# ---------------------------------------------------------------------------

class CustomSymPOTrainer(Trainer):
    def __init__(
        self,
        model: Union[torch.nn.Module],
        ref_model: Union[torch.nn.Module], # 1. 新增 ref_model 参数
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
        self.beta_kl = beta_kl
        self.log_ratio_clip_min = log_ratio_clip_min
        self.log_ratio_clip_max = log_ratio_clip_max

    def _get_log_probs(self, model: AutoModelForCausalLM, prompts: List[str], responses: List[str]) -> torch.Tensor:
        # 这个辅助函数无需修改，它可以通用地计算任何模型、prompt 和 response 的 log_probs
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
        # 3. 移除从 inputs 中获取 ref_logp 的代码
        # log_probs_ref_y1 = inputs.pop("ref_logp_chosen").to(self.args.device)
        # log_probs_ref_y2 = inputs.pop("ref_logp_rejected").to(self.args.device)
        f_y1 = inputs.pop("reward_chosen").to(self.args.device)
        f_y2 = inputs.pop("reward_rejected").to(self.args.device)

        batch_size = len(prompts)
        combined_prompts = prompts + prompts
        combined_responses = chosen_responses + rejected_responses

        # --- 计算策略模型的 log_probs ---
        all_log_probs_pi = self._get_log_probs(model, combined_prompts, combined_responses)
        log_probs_pi_y1 = all_log_probs_pi[:batch_size]
        log_probs_pi_y2 = all_log_probs_pi[batch_size:]

        # --- 4. 动态计算参考模型的 log_probs ---
        with torch.no_grad():
            all_log_probs_ref = self._get_log_probs(self.ref_model, combined_prompts, combined_responses)
            log_probs_ref_y1 = all_log_probs_ref[:batch_size]
            log_probs_ref_y2 = all_log_probs_ref[batch_size:]
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
            # kl_term2 = self.beta_kl * log_ratio_y2.detach()
            weight1 = 1 - f_y2 - kl_term1
            # weight2 = f_y1 + kl_term2
            weight2 = f_y1

            
        clamped_log_ratio_y1 = torch.clamp(log_ratio_y1, min=self.log_ratio_clip_min, max=self.log_ratio_clip_max)
        ratio_y1 = torch.exp(clamped_log_ratio_y1)
        clamped_log_ratio_y2 = torch.clamp(log_ratio_y2, min=self.log_ratio_clip_min, max=self.log_ratio_clip_max)
        ratio_y2 = torch.exp(clamped_log_ratio_y2)
        
        # ratio_y1 = torch.exp(log_ratio_y1)
        # ratio_y2 = torch.exp(log_ratio_y2)
        J_sym_objective = ratio_y1 * weight1 - ratio_y2 * weight2
        total_loss = -J_sym_objective.mean()
        
        logs = {
            # "mean_log_probs_pi_chosen": log_probs_pi_y1.detach().mean().item(),
            # "mean_log_probs_pi_rejected": log_probs_pi_y2.detach().mean().item(),
            # "mean_log_probs_ref_chosen": log_probs_ref_y1.detach().mean().item(),
            # "mean_log_probs_ref_rejected": log_probs_ref_y2.detach().mean().item(),
            "mean_ratio_chosen": ratio_y1.detach().mean().item(),
            "mean_ratio_rejected": ratio_y2.detach().mean().item(),
            "weight_chosen": weight1.detach().mean().item(),
            "weight_rejected": weight2.detach().mean().item(),
            "kl_term_chosen": kl_term1.detach().mean().item(),
        }
        self._current_logs = logs
        
        return total_loss

    # log 方法无需修改
    def log(self, logs: dict, *args, **kwargs) -> None:
        if hasattr(self, "_current_logs") and self._current_logs is not None:
            logs.update(self._current_logs)
            self._current_logs = None
        super().log(logs, *args, **kwargs)

# Data Collator 和 Callback 无需修改
@dataclass
class DataCollatorForCustomSymPO:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}
        for key in features[0].keys():
            # 移除 ref_logp 相关字段的处理，因为它们已不存在
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
            print(logs)

# argparse 函数无需修改
def parse_args():
    parser = argparse.ArgumentParser(description="Train a policy model with SymPO algorithm.")
    
    # --- 路径参数 ---
    parser.add_argument("--sft_model_path", type=str, default="/train/Llama-3-8B-Instruct", help="Path to the SFT base model.")
    parser.add_argument("--preprocessed_dataset_path", type=str, default="/train/precomputed_traindataset", help="Path to the precomputed dataset.")
    parser.add_argument("--output_dir", type=str, default="/train/output_model/llama3-8b-sympo-1e-6_0.1", help="Directory to save checkpoints and final model.")
    
    # --- 训练超参数 ---
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per GPU.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps.")
    parser.add_argument("--gradient_checkpointing", action='store_true', help="Enable gradient checkpointing to save memory.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for learning rate scheduler.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log metrics every N steps.")
    parser.add_argument("--save_steps", type=int, default=200, help="Save a checkpoint every N steps.")
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

    print("从磁盘加载数据集 (注意: 数据集中不再需要 ref_logp)...")
    precomputed_dataset_train = load_from_disk(args.preprocessed_dataset_path)

    print("加载用于训练的策略模型 (Policy Model)...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
    )

    # 5. 新增：加载作为参考的SFT模型 (Reference Model)
    print("加载作为参考的SFT模型 (Reference Model)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
    )
    # # 将参考模型设置为评估模式，并且不需要计算它的梯度
    # ref_model.eval()

    # TrainingArguments 定义无需修改
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
        ref_model=ref_model, # 6. 将 ref_model 传递给 Trainer
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=precomputed_dataset_train,
        data_collator=data_collator,
        callbacks=[PrintingCallback()],
        beta_kl=args.beta_kl,
        log_ratio_clip_min=args.log_ratio_clip_min,
        log_ratio_clip_max=args.log_ratio_clip_max,
    )
    # 移除 sft_model_path，因为它已经通过 ref_model 对象传入了
    # sft_model_path=args.sft_model_path, 

    print("开始分布式训练...")
    trainer.train()
    print("所有任务已完成！")

if __name__ == "__main__":
    main()