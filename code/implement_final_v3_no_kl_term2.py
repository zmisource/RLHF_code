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
        log_ratio_clip_min: float,
        log_ratio_clip_max: float,
        max_length: int = None,
        **kwargs,
    ):
        super().__init__(model=model, args=args, **kwargs)

        if not hasattr(self, 'processor') or self.processor is None:
            self.processor = self.tokenizer

        # 确保 ref_model 在正确的设备上
        self.ref_model = ref_model.to(self.args.device)
        self.beta_kl = beta_kl
        self.log_ratio_clip_min = log_ratio_clip_min
        self.log_ratio_clip_max = log_ratio_clip_max
        self.max_length = max_length

    def _smooth_clamp(self, x: torch.Tensor, min_val: float, max_val: float, temperature: float = 0.1) -> torch.Tensor:
        normalized = 2.0 * (x - min_val) / (max_val - min_val) - 1.0
        clipped_normalized = torch.tanh(normalized / temperature)
        return (clipped_normalized + 1.0) / 2.0 * (max_val - min_val) + min_val
    
    def _get_log_probs(
        self, 
        model: AutoModelForCausalLM, 
        prompts: List[str], 
        responses: List[str], 
        max_length: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self, "processing_class") and self.processing_class is not None:
            processor = self.processing_class
        else:
            processor = self.tokenizer
        device = self.args.device

        # 1. 保存现场
        original_padding_side = processor.padding_side
        original_truncation_side = processor.truncation_side
        
        try:
            # 2. 强制配置
            processor.padding_side = 'left'
            processor.truncation_side = 'left' 

            # 3. 文本拼接
            full_texts = [p + r for p, r in zip(prompts, responses)]

            # 4. 确定 Max Length
            if max_length is None:
                if hasattr(model.config, "max_position_embeddings"):
                    max_length = model.config.max_position_embeddings
                else:
                    max_length = 2048

            # 5. 获取 Response 物理长度
            response_inputs = processor(
                responses, 
                padding=False, 
                truncation=False, 
                add_special_tokens=False, 
                return_attention_mask=False
            )
            response_lengths = [len(ids) for ids in response_inputs['input_ids']]
            response_lengths_tensor = torch.tensor(response_lengths, device=device, dtype=torch.long)

            # 6. Full Text 编码
            full_tokens = processor(
                full_texts,
                padding='longest',
                truncation=True,
                max_length=max_length,
                return_tensors='pt',
            )
            
            input_ids = full_tokens['input_ids'].to(device)
            attention_mask = full_tokens['attention_mask'].to(device)

            # 7. 模型前向
            # 注意：Policy Model 需要梯度，Ref Model 不需要，由外部调用者控制 context
            outputs = model(input_ids, attention_mask=attention_mask, return_dict=True, use_cache=False)
            
            logits = outputs.logits
            shifted_logits = logits[..., :-1, :]
            shifted_labels = input_ids[..., 1:]

            # 8. 计算逐 Token Loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            nll_per_token = loss_fct(
                shifted_logits.reshape(-1, shifted_logits.size(-1)), 
                shifted_labels.reshape(-1)
            )
            nll_per_token = nll_per_token.view(shifted_labels.size())
            log_probs_per_token = -nll_per_token

            # 9. Mask 计算
            shifted_attention_mask = attention_mask[:, 1:]
            seq_len = shifted_labels.size(1)
            position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
            
            # Left Padding 下，Response 在序列末尾
            # seq_len 是 L-1, response_len 是 R
            # 有效 Label 的起始索引是 L-1 - R (对应 input_ids 的 L-R)
            response_start_indices = seq_len - response_lengths_tensor
            response_start_indices = torch.clamp(response_start_indices, min=0)
            
            response_mask = (position_ids >= response_start_indices.unsqueeze(1)) & shifted_attention_mask.bool()

            # 10. 汇总结果
            masked_log_probs = log_probs_per_token * response_mask
            total_log_probs = masked_log_probs.sum(dim=1)
            valid_response_lengths = response_mask.sum(dim=1)
            
            return total_log_probs, valid_response_lengths
            
        finally:
            # 11. 恢复配置 (确保即使出错也恢复)
            processor.padding_side = original_padding_side
            processor.truncation_side = original_truncation_side

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        prompts = inputs.pop("prompt")
        chosen_responses = inputs.pop("chosen")
        rejected_responses = inputs.pop("rejected")
        
        f_y1 = inputs.pop("f_Y_chosen").to(self.args.device)
        f_y2 = inputs.pop("f_Y_rejected").to(self.args.device)

        batch_size = len(prompts)
        combined_prompts = prompts + prompts
        combined_responses = chosen_responses + rejected_responses

        # --- 计算策略模型的 log_probs ---
        all_log_probs_pi, _ = self._get_log_probs(
            model, 
            combined_prompts, 
            combined_responses, 
            max_length=self.max_length
        )        
        log_probs_pi_y1 = all_log_probs_pi[:batch_size]
        log_probs_pi_y2 = all_log_probs_pi[batch_size:]

        # --- 计算参考模型的 log_probs ---
        with torch.no_grad():
            all_log_probs_ref, _ = self._get_log_probs(
                self.ref_model, 
                combined_prompts, 
                combined_responses, 
                max_length=self.max_length
            )
            log_probs_ref_y1 = all_log_probs_ref[:batch_size]
            log_probs_ref_y2 = all_log_probs_ref[batch_size:]
            
        if self.state.global_step == 0:
            print(f"--- Sanity Check at Step 0 ---")
            print(f"Sample 0 - ref_logp_chosen: {log_probs_ref_y1[0].item()}")
            print(f"Sample 0 - pi_logp_chosen:  {log_probs_pi_y1[0].item()}")
            print(f"Difference: {log_probs_pi_y1[0].item() - log_probs_ref_y1[0].item()}")
            print(f"---------------------------------")
        
        log_ratio_y1 = log_probs_pi_y1 - log_probs_ref_y1
        log_ratio_y2 = log_probs_pi_y2 - log_probs_ref_y2
        
        # SymPO 核心计算逻辑
        with torch.no_grad():
            # kl_term1 = self.beta_kl * log_ratio_y1.detach()
            # kl_term2 = self.beta_kl * log_ratio_y2.detach()
            weight1 = 1 - f_y2
            # weight2 = f_y1
            # weight1 = torch.clamp(1.0 - f_y2, min=0.01) # 保证至少有 0.01 的权重
            # weight2 = torch.clamp(f_y1, min=0.01)


        clamped_log_ratio_y1 = torch.clamp(log_ratio_y1, min=self.log_ratio_clip_min, max=self.log_ratio_clip_max)
        # clamped_log_ratio_y2 = torch.clamp(log_ratio_y2, min=self.log_ratio_clip_min, max=self.log_ratio_clip_max)
        
        ratio_y1 = torch.exp(clamped_log_ratio_y1)
        # ratio_y2 = torch.exp(clamped_log_ratio_y2)
        
        J_sym_objective = ratio_y1 * weight1
        total_loss = -J_sym_objective.mean()
        
        logs = {
            "mean_ratio_chosen": ratio_y1.detach().mean().item(),
            # "mean_ratio_rejected": ratio_y2.detach().mean().item(),
            "weight_chosen": weight1.detach().mean().item(),
            # "weight_rejected": weight2.detach().mean().item(),
            # "kl_term_chosen": kl_term1.detach().mean().item(),
            # "kl_term_rejected": kl_term2.detach().mean().item(),
            "loss": total_loss.item()
        }
        self._current_logs = logs
        
        # 遵循 Trainer 接口，如果需要返回 outputs (虽然这里通常不需要)
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
    parser.add_argument("--log_ratio_clip_min", type=float, default=-2.3, help="Min clip value.")
    parser.add_argument("--log_ratio_clip_max", type=float, default=2.3, help="Max clip value.")
    parser.add_argument("--use_smooth_clip", type=bool, default=True, help="Use smooth clipping.")
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
        log_ratio_clip_min=args.log_ratio_clip_min,
        log_ratio_clip_max=args.log_ratio_clip_max,
        max_length=args.max_length,
    )
    
    # 设置是否使用平滑裁剪
    trainer.use_smooth_clip = args.use_smooth_clip
    
    # [修正] 移除了此处重复的 trainer 配置代码块

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