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
from typing import Dict, Any, List, Union
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 随机种子设置函数（确保可重复性）
# ---------------------------------------------------------------------------

def seed_everything(seed=42):
    """
    设置所有随机数生成器的种子，确保训练的可重复性。
    
    Args:
        seed: 随机种子值（默认: 2003）
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 为了可重复性，禁用benchmark
    # 设置环境变量以确保完全确定性（如果使用CUDA）
    os.environ['PYTHONHASHSEED'] = str(seed)

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
        max_length: int = None,
        max_prompt_length: int = None,
        truncation_mode: str = "keep_end",
        **kwargs,
    ):
        super().__init__(model=model, args=args, **kwargs)

        if not hasattr(self, 'processor') or self.processor is None:
            self.processor = self.tokenizer

        self.ref_model = ref_model.to(self.args.device)
        self.beta_kl = beta_kl
        self.log_ratio_clip_min = log_ratio_clip_min
        self.log_ratio_clip_max = log_ratio_clip_max
        self.use_smooth_clip = False  # 默认使用硬裁剪，可通过参数切换
        
        # 截断相关参数
        unwrapped_model = model.module if hasattr(model, "module") else model
        self.max_length = max_length if max_length is not None else unwrapped_model.config.max_position_embeddings
        self.max_prompt_length = max_prompt_length if max_prompt_length is not None else self.max_length // 2
        self.truncation_mode = truncation_mode  # "keep_start" 或 "keep_end"

    def _smooth_clamp(self, x: torch.Tensor, min_val: float, max_val: float, temperature: float = 0.1) -> torch.Tensor:
        """
        平滑裁剪函数，在边界处保持小梯度，避免梯度突然截断。
        
        使用 tanh-based 平滑函数：
        - 在边界处梯度连续，不会突然变为0
        - temperature 控制平滑程度（越小越接近硬裁剪）
        """
        # 将输入归一化到 [-1, 1] 范围
        normalized = 2.0 * (x - min_val) / (max_val - min_val) - 1.0
        # 使用 tanh 进行平滑裁剪
        clipped_normalized = torch.tanh(normalized / temperature)
        # 映射回原始范围
        return (clipped_normalized + 1.0) / 2.0 * (max_val - min_val) + min_val
    
    def _get_log_probs(self, model: AutoModelForCausalLM, prompts: List[str], responses: List[str]) -> tuple[torch.Tensor, torch.Tensor]:
        original_padding_side = self.processor.padding_side        
        self.processor.padding_side = 'right' 
        
        # 1. 先 tokenize prompt 和 response 获取长度（不截断，不添加特殊token）
        prompt_tokens_list = self.processor(prompts, padding=False, truncation=False, add_special_tokens=False)
        response_tokens_list = self.processor(responses, padding=False, truncation=False, add_special_tokens=False)
        
        # 2. 对每个样本进行截断处理（仿照 SimPO 的逻辑）
        all_input_ids = []
        all_attention_masks = []
        prompt_lengths = []
        
        for i, (prompt_tokens, response_tokens) in enumerate(zip(prompt_tokens_list['input_ids'], response_tokens_list['input_ids'])):
            prompt_len = len(prompt_tokens)
            response_len = len(response_tokens)
            longer_response_length = response_len
            
            # 如果 combined sequence 太长，先截断 prompt
            if prompt_len + longer_response_length > self.max_length:
                if self.truncation_mode == "keep_start":
                    # 保留开头部分
                    prompt_tokens = prompt_tokens[:self.max_prompt_length]
                elif self.truncation_mode == "keep_end":
                    # 保留结尾部分
                    prompt_tokens = prompt_tokens[-self.max_prompt_length:] if len(prompt_tokens) > self.max_prompt_length else prompt_tokens
                else:
                    raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")
                prompt_len = len(prompt_tokens)
            
            # 如果截断 prompt 后还是太长，截断 response
            # 注意：这里使用 max_prompt_length 而不是实际的 prompt_len，与 SimPO 保持一致
            if prompt_len + longer_response_length > self.max_length:
                max_response_len = self.max_length - self.max_prompt_length
                response_tokens = response_tokens[:max_response_len] if max_response_len > 0 else []
                response_len = len(response_tokens)
            
            # 直接拼接 prompt 和 response tokens（不添加特殊token，因为已经在原始文本中）
            full_input_ids = prompt_tokens + response_tokens
            full_attention_mask = [1] * len(full_input_ids)
            
            all_input_ids.append(full_input_ids)
            all_attention_masks.append(full_attention_mask)
            prompt_lengths.append(prompt_len)
        
        # 3. Padding（right padding）
        max_seq_len = max(len(ids) for ids in all_input_ids)
        pad_token_id = self.processor.pad_token_id if self.processor.pad_token_id is not None else 0
        
        padded_input_ids = []
        padded_attention_masks = []
        for input_ids, attention_mask in zip(all_input_ids, all_attention_masks):
            pad_length = max_seq_len - len(input_ids)
            padded_input_ids.append(input_ids + [pad_token_id] * pad_length)
            padded_attention_masks.append(attention_mask + [0] * pad_length)
        
        # 4. 转换为 tensor
        input_ids = torch.tensor(padded_input_ids, dtype=torch.long, device=self.args.device)
        attention_mask = torch.tensor(padded_attention_masks, dtype=torch.long, device=self.args.device)
        
        # 5. Forward
        outputs = model(input_ids, attention_mask=attention_mask, return_dict=True, use_cache=False)
        logits = outputs.logits
        
        # 6. Shift logits/labels
        shifted_logits = logits[..., :-1, :]
        shifted_labels = input_ids[..., 1:]
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        nll_per_token = loss_fct(shifted_logits.reshape(-1, shifted_logits.size(-1)), shifted_labels.reshape(-1))
        nll_per_token = nll_per_token.view(input_ids.size(0), -1)
        log_probs_per_token = -nll_per_token
        
        # 7. Create Mask
        seq_len = shifted_labels.size(1)
        position_ids = torch.arange(seq_len, device=self.args.device).expand_as(shifted_labels)
        prompt_lengths_tensor = torch.tensor(prompt_lengths, device=self.args.device).unsqueeze(1)
        
        # Mask 逻辑：保留 index >= (prompt_len - 1) 的部分
        # 因为 padding_side='right'，所以 index 0 确实是文本开头，逻辑成立。
        response_start_index = prompt_lengths_tensor - 1
        mask = position_ids >= response_start_index
        
        attention_mask_shifted = attention_mask[:, 1:].to(torch.bool)
        response_mask = mask & attention_mask_shifted
        
        masked_log_probs = log_probs_per_token * response_mask
        total_log_probs = masked_log_probs.sum(dim=1)
        
        response_lengths = response_mask.float().sum(dim=1)
        
        # 恢复原来的 padding_side (通常是 left)
        self.processor.padding_side = original_padding_side
        
        return total_log_probs, response_lengths
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        prompts = inputs.pop("prompt")
        chosen_responses = inputs.pop("chosen")
        rejected_responses = inputs.pop("rejected")
        f_y1 = inputs.pop("reward_chosen").to(self.args.device)
        f_y2 = inputs.pop("reward_rejected").to(self.args.device)

        batch_size = len(prompts)
        combined_prompts = prompts + prompts
        combined_responses = chosen_responses + rejected_responses

        # --- 计算策略模型的 log_probs 和 长度 ---
        all_log_probs_pi, all_lengths = self._get_log_probs(model, combined_prompts, combined_responses)
        log_probs_pi_y1 = all_log_probs_pi[:batch_size]
        log_probs_pi_y2 = all_log_probs_pi[batch_size:]
        len_y1 = all_lengths[:batch_size] 
        len_y2 = all_lengths[batch_size:]

        # --- 动态计算参考模型的 log_probs ---
        with torch.no_grad():
            # ref模型不需要长度，用 _ 占位
            all_log_probs_ref, _ = self._get_log_probs(self.ref_model, combined_prompts, combined_responses)
            log_probs_ref_y1 = all_log_probs_ref[:batch_size]
            log_probs_ref_y2 = all_log_probs_ref[batch_size:]

        if self.state.global_step == 0:
            print(f"--- Sanity Check at Step 0 ---")
            print(f"Sample 0 - Length Chosen:   {len_y1[0].item()}")
            print(f"Sample 0 - Length Rejected: {len_y2[0].item()}")
            print(f"---------------------------------")

        log_ratio_y1 = log_probs_pi_y1 - log_probs_ref_y1
        log_ratio_y2 = log_probs_pi_y2 - log_probs_ref_y2
        
        with torch.no_grad():
            kl_term1 = self.beta_kl * log_ratio_y1.detach()
            kl_term2 = self.beta_kl * log_ratio_y2.detach()
            
            len_y1 = torch.clamp(len_y1, min=1.0)
            len_y2 = torch.clamp(len_y2, min=1.0)
            
            weight1 = (1 - f_y2 - kl_term1) / len_y1
            weight2 = (f_y1 + kl_term2) / len_y2
            
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
            "kl_term_chosen": kl_term1.detach().mean().item(),
            "mean_len_chosen": len_y1.detach().mean().item(),
            "mean_len_rejected": len_y2.detach().mean().item()
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
    parser.add_argument("--output_dir", type=str, default="/train/output_model/llama3-8b-sympo-1e-6_0.5_length_seed_42_2048", help="Directory to save checkpoints and final model.")
    
    # --- 训练超参数 ---
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per GPU.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps.")
    parser.add_argument("--gradient_checkpointing", action='store_true', help="Enable gradient checkpointing to save memory.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for learning rate scheduler.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log metrics every N steps.")
    parser.add_argument("--save_steps", type=int, default=10000, help="Save a checkpoint every N steps.")
    parser.add_argument("--save_total_limit", type=int, default=20, help="Limit the total number of saved checkpoints.")

    # --- SymPO 特定参数 ---
    parser.add_argument("--beta_kl", type=float, default=0.5, help="KL divergence penalty coefficient.")
    parser.add_argument("--log_ratio_clip_min", type=float, default=-2.3, help="Minimum clip value for log probability ratio.")
    parser.add_argument("--log_ratio_clip_max", type=float, default=2.3, help="Maximum clip value for log probability ratio.")
    parser.add_argument("--use_smooth_clip", action='store_true', help="Use smooth clipping for better gradient stability (recommended).")
    
    # --- 截断参数（仿照 SimPO） ---
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum length of the combined prompt+response sequence. If None, uses model's max_position_embeddings.")
    parser.add_argument("--max_prompt_length", type=int, default=1800, help="Maximum length of the prompt. If None, uses max_length // 2.")
    parser.add_argument("--truncation_mode", type=str, default="keep_end", choices=["keep_start", "keep_end"], help="Truncation mode: 'keep_start' keeps the beginning of prompt, 'keep_end' keeps the end of prompt.")
    
    # --- W&B (Weights & Biases) 日志参数 ---
    parser.add_argument("--report_to", type=str, default="wandb", help="The integration to report results to (e.g., 'wandb').")
    parser.add_argument("--run_name", type=str, default=f"policy-llama3-8b-sympo-default", help="A name for the W&B run.")
    
    # --- 随机种子参数 ---
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility .")

    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置随机种子以确保可重复性
    print(f"设置随机种子为: {args.seed}")
    seed_everything(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left' 

    print("从磁盘加载数据集 (注意: 数据集中不再需要 ref_logp)...")
    precomputed_dataset_train = load_from_disk(args.preprocessed_dataset_path)

    print("加载用于训练的策略模型 (Policy Model)...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path, dtype=torch.bfloat16
    )

    # 5. 新增：加载作为参考的SFT模型 (Reference Model)
    print("加载作为参考的SFT模型 (Reference Model)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path, dtype=torch.bfloat16
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
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        truncation_mode=args.truncation_mode,
    )
    # 设置是否使用平滑裁剪
    trainer.use_smooth_clip = args.use_smooth_clip
    # 移除 sft_model_path，因为它已经通过 ref_model 对象传入了
    # sft_model_path=args.sft_model_path, 

        # 设置是否使用平滑裁剪
    trainer.use_smooth_clip = args.use_smooth_clip
    # 移除 sft_model_path，因为它已经通过 ref_model 对象传入了
    # sft_model_path=args.sft_model_path, 

    print("开始分布式训练...")
    train_result = trainer.train()
    print("所有任务已完成！")

    ##################################
    # Save final model (HuggingFace format)
    ##################################
    print("\n" + "=" * 60)
    print("*** Save final model ***")
    print("=" * 60)
    print(f"⚠️  注意: 如果使用 FSDP，此步骤会自动合并分片权重并保存为 HuggingFace 格式")
    print(f"保存路径: {args.output_dir}")
    
    # 确保 FSDP 使用 FULL_STATE_DICT 模式保存最终模型
    if trainer.is_fsdp_enabled and trainer.accelerator.state.fsdp_plugin is not None:
        print("检测到 FSDP，设置状态字典类型为 FULL_STATE_DICT...")
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    
    # 保存最终模型（自动处理 FSDP 合并）
    trainer.save_model(args.output_dir)
    print(f"✅ 模型已保存到: {args.output_dir}")
    
    # 保存 tokenizer
    try:
        tokenizer.save_pretrained(args.output_dir)
        print("✅ Tokenizer 已保存")
    except Exception as e:
        print(f"⚠️  警告: 无法自动保存 tokenizer: {e}")
    
    # 保存训练指标
    if trainer.accelerator.is_main_process:
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        print("✅ 训练指标已保存")
    
    print("\n" + "=" * 60)
    print("🎉 训练和模型保存完成！")
    print("=" * 60)
    print(f"\n最终 HuggingFace 模型已保存到: {args.output_dir}")
    print("可以直接使用 HuggingFace 的 from_pretrained() 加载模型")
    print(f"\n注意: checkpoint-* 目录是训练过程中的中间检查点（FSDP 分片格式）")
    print(f"最终模型保存在 {args.output_dir} 根目录（HuggingFace 格式）")
if __name__ == "__main__":
    main()