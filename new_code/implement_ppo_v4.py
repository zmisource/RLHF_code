import json
import os
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from torch.nn.utils.rnn import pad_sequence
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
        gamma: float,
        clip_eps_min: float,    
        clip_eps_max: float,
        max_length: int = None,
        debug_fixed_batch: bool = False,   # 新增调试标志
        **kwargs,
    ):
        super().__init__(model=model, args=args, **kwargs)
        self.debug_fixed_batch = debug_fixed_batch

        # 用于累积日志
        self._step_logs_accumulator = {}
        self._step_counter = 0

        if not hasattr(self, 'processor') or self.processor is None:
            self.processor = self.tokenizer

        
        self.beta_kl = beta_kl
        self.gamma = gamma
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

    def get_train_dataloader(self):
        """重写 DataLoader 创建方法，调试模式下使用 SequentialSampler 固定顺序"""
        if self.debug_fixed_batch:
            from torch.utils.data import DataLoader, SequentialSampler
            train_dataset = self.train_dataset
            data_collator = self.data_collator
            dataloader_params = {
                "batch_size": self.args.per_device_train_batch_size,
                "collate_fn": data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
            }
            # 使用顺序采样器
            dataloader_params["sampler"] = SequentialSampler(train_dataset)
            return DataLoader(train_dataset, **dataloader_params)
        else:
            # 正常模式：调用父类方法（使用默认采样器，可能随机）
            return super().get_train_dataloader()

    def _get_log_probs(
            self, 
            model: AutoModelForCausalLM, 
            prompts: List[str], 
            responses: List[str], 
            max_length: int = 2048
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            tokenizer = self.processing_class
            device = self.args.device
            full_texts = [p + r for p, r in zip(prompts, responses)]
            
            inputs = tokenizer(
                full_texts,
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
                add_special_tokens=False 
            ).to(device)

            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            prompt_inputs = tokenizer(
                prompts, 
                add_special_tokens=False, 
                return_attention_mask=False,
                truncation=False 
            )
            
            for i, p_ids in enumerate(prompt_inputs["input_ids"]):
                p_len = len(p_ids)
                
                valid_indices = (attention_mask[i] == 1).nonzero(as_tuple=True)[0]
                
                if len(valid_indices) > 0:
                    start_index = valid_indices[0].item()
                else:
                    start_index = 0 
                prompt_end_index = start_index + p_len
                
                if prompt_end_index > input_ids.shape[1]:
                    prompt_end_index = input_ids.shape[1]
                
                labels[i, start_index:prompt_end_index] = -100
            outputs = model(input_ids, attention_mask=attention_mask, return_dict=True, use_cache=False)
            logits = outputs.logits

            shifted_logits = logits[..., :-1, :].contiguous()
            shifted_labels = labels[..., 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
            
            nll_per_token = loss_fct(
                shifted_logits.view(-1, shifted_logits.size(-1)), 
                shifted_labels.view(-1)
            )
            nll_per_token = nll_per_token.view(shifted_labels.size())
            
            valid_mask = (shifted_labels != -100).float()
            masked_log_probs = (-nll_per_token) * valid_mask
            
            total_log_probs = masked_log_probs.sum(dim=1)
            
            valid_response_lens = valid_mask.sum(dim=1)
            valid_response_lens = torch.clamp(valid_response_lens, min=1e-8)
            
            return total_log_probs, valid_response_lens

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 从输入中获取数据
        prompts = inputs.pop("prompt")
        chosen_responses = inputs.pop("chosen")
        rejected_responses = inputs.pop("rejected")
        
        # 获取f值
        f_X_Y2 = inputs.pop("f_Y_rejected").to(self.args.device)  # f*(X, rejected)
        f_X_Y1 = inputs.pop("f_Y_chosen").to(self.args.device)    # f*(X, chosen)
        
        batch_size = len(prompts)
        
        # 计算策略模型的对数概率
        # chosen part
        chosen_log_probs_pi, chosen_lens = self._get_log_probs(
            model, prompts, chosen_responses, max_length=self.max_length
        )
        
        # rejected part
        rejected_log_probs_pi, rejected_lens = self._get_log_probs(
            model, prompts, rejected_responses, max_length=self.max_length
        )

        # 计算参考模型的对数概率
        with torch.no_grad():
            chosen_log_probs_ref, _ = self._get_log_probs(
                self.ref_model, prompts, chosen_responses, max_length=self.max_length
            )
            rejected_log_probs_ref, _ = self._get_log_probs(
                self.ref_model, prompts, rejected_responses, max_length=self.max_length
            )
        
        # 计算重要性权重比
        log_ratio_chosen = chosen_log_probs_pi - chosen_log_probs_ref
        log_ratio_rejected = rejected_log_probs_pi - rejected_log_probs_ref
        
        # 随机选择 Z 为 0 或 1
        Z = torch.randint(0, 2, (batch_size,), device=self.args.device).float()

        log_ratio = Z * log_ratio_chosen + (1 - Z) * log_ratio_rejected
        f_selected = Z * f_X_Y2 + (1 - Z) * f_X_Y1
        # log_probs_pi = Z * chosen_log_probs_pi + (1 - Z) * rejected_log_probs_pi
        
        # include kl term in the weight, but do not backprop through it
        with torch.no_grad():
            kl_selected = self.beta_kl * log_ratio.detach()
            weight = Z - f_selected - kl_selected

        log_ratio_safe = torch.clamp(log_ratio, -20, 20) 
        ratio_selected = torch.exp(log_ratio_safe)
        
        clip_min = Z * 1.0 + (1 - Z) * (1 - self.clip_eps_min)
        clip_max = Z * (1 + self.clip_eps_max) + (1 - Z) * 1.0
        
        term_raw = ratio_selected * weight

        ratio_clipped = torch.clamp(ratio_selected, min=clip_min, max=clip_max)

        term_clipped = ratio_clipped * weight
        objective = term_clipped

        
        # total loss
        total_loss = -objective.mean()
        
        # 记录日志
        micro_logs = {
            "loss": total_loss.item(),
            "objective": objective.mean().item(),
            # "kl_penalty": kl_div.item(),
            "term_raw": term_raw.mean().item(),
            "term_clipped": term_clipped.mean().item(),
            "ratio_selected_mean": ratio_selected.mean().item(),
            "log_ratio_selected_mean": log_ratio_safe.mean().item(),
            "f_X_Y2_mean": f_X_Y2.mean().item(),
            "f_X_Y1_mean": f_X_Y1.mean().item(),
        }
        
        # 累积日志
        if not hasattr(self, '_step_logs_accumulator'):
            self._step_logs_accumulator = {}
            self._step_counter = 0
        
        for key, value in micro_logs.items():
            if key not in self._step_logs_accumulator:
                self._step_logs_accumulator[key] = value
            else:
                self._step_logs_accumulator[key] += value
        
        self._step_counter += 1
        
        if return_outputs:
            return (total_loss, None)
        return total_loss

    def log(self, logs: dict, *args, **kwargs) -> None:
        """重写log方法，记录step级别的平均值"""
        if hasattr(self, '_step_counter') and self._step_counter > 0:
            # 计算当前step所有micro-batch的平均值
            avg_logs = {
                k: v / self._step_counter 
                for k, v in self._step_logs_accumulator.items()
            }
            
            # 合并到logs中
            logs.update(avg_logs)
            
            # 重置累加器
            self._step_logs_accumulator = {}
            self._step_counter = 0
        
        super().log(logs, *args, **kwargs)

class LogSavingCallback(TrainerCallback):
    """自动保存训练日志的回调"""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and logs and args.output_dir:
            try:
                # 创建日志文件路径
                log_file = os.path.join(args.output_dir, "training_logs.jsonl")
                
                # 准备日志条目
                log_entry = {
                    "timestamp": time.time(),
                    "step": state.global_step,
                    "epoch": state.epoch,
                    **logs
                }
                
                # 追加写入
                with open(log_file, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
                    
            except Exception as e:
                print(f"⚠️ 保存日志时出错: {e}")

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
    parser.add_argument("--sft_model_path", type=str, default="/root/autodl-fs/train/Llama-3-8B-Instruct", help="Path to the SFT base model.")
    parser.add_argument("--preprocessed_dataset_path", type=str, default="/root/autodl-fs/train/ds_with_metrics", help="Path to the precomputed dataset.")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-fs/train/output_model/Llama-3-8B-Instruct", help="Directory to save checkpoints and final model.")
    
    # --- 训练超参数 ---
    parser.add_argument("--learning_rate", type=float, default=1e-7, help="Learning rate.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per GPU.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of gradient accumulation steps.")
    parser.add_argument("--gradient_checkpointing", action='store_true', help="Enable gradient checkpointing.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log metrics every N steps.")
    parser.add_argument("--save_steps", type=int, default=2000, help="Save a checkpoint every N steps.")
    parser.add_argument("--save_total_limit", type=int, default=20, help="Limit number of saved checkpoints.")

    # --- SymPO 特定参数 ---
    parser.add_argument("--beta_kl", type=float, default=2.5, help="KL divergence penalty coefficient.") # 0.05
    parser.add_argument("--gamma", type=float, default=1.4, help="Refer to gamma in SimPO, usually 0.5-1.5.")
    parser.add_argument("--clip_eps_min", type=float, default=0.9, help="Min clip value.")
    parser.add_argument("--clip_eps_max", type=float, default=9, help="Max clip value.")
    parser.add_argument("--max_length", type=int, default=2048, help="Max length for log prob calculation.")
    
    # --- W&B ---
    parser.add_argument("--report_to", type=str, default="wandb", help="Integration to report results to.")
    parser.add_argument("--run_name", type=str, default=f"policy-Llama-3-8B-Instruct-sympo-default", help="W&B run name.")
    
    # --- 随机种子 ---
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # --- debug ---
    parser.add_argument("--debug_fixed_batch", action="store_true",
                    help="Debug mode: use a fixed small batch and train repeatedly.")
    parser.add_argument("--debug_batch_size", type=int, default=8,
                    help="Size of the fixed batch for debugging (only used if --debug_fixed_batch is set).")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"设置随机种子为: {args.seed}")
    seed_everything(args.seed)

    # =============== 新增代码：获取本地 GPU ID ===============
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    print(f"Process rank: {local_rank}, device set to cuda:{local_rank}")
    # =======================================================

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

    # =============== 新增调试代码：固定小批量模式 ===============
    if args.debug_fixed_batch:
        print(f"\n🔧 调试模式：使用固定 batch，大小 = {args.debug_batch_size}")
        # 取前 debug_batch_size 个样本（如果数据集不足则取全部）
        subset_size = min(args.debug_batch_size, len(precomputed_dataset_train))
        precomputed_dataset_train = precomputed_dataset_train.select(range(subset_size))
        print(f"🔧 训练数据集已缩小为 {subset_size} 个样本")

        # 强制覆盖训练参数
        args.per_device_train_batch_size = subset_size
        args.gradient_accumulation_steps = 1     # 关闭梯度累积，简化调试
        # 提示用户建议使用单卡运行
        if int(os.environ.get("WORLD_SIZE", 1)) > 1:
            print("⚠️  警告：调试模式下建议使用单卡（设置 CUDA_VISIBLE_DEVICES=0 或使用 --local_rank=-1）")
    # =========================================================

    # =============== 修改代码：加载模型时指定 device_map ===============
    print("加载用于训练的策略模型 (Policy Model)...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path, 
        dtype=torch.bfloat16,
        device_map={"": local_rank}  # <--- 强制加载到当前 GPU
    )

    print("加载作为参考的SFT模型 (Reference Model)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path, 
        dtype=torch.bfloat16,
        device_map={"": local_rank}  # <--- 强制加载到当前 GPU
    )
    # =================================================================
    
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
        # save_total_limit=args.save_total_limit, 
        save_total_limit=1,
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
        processing_class=tokenizer,
        train_dataset=precomputed_dataset_train,
        data_collator=data_collator,
        callbacks=[PrintingCallback(), LogSavingCallback()],
        beta_kl=args.beta_kl,
        gamma=args.gamma,
        clip_eps_min=args.clip_eps_min,
        clip_eps_max=args.clip_eps_max,
        max_length=args.max_length,
        debug_fixed_batch=args.debug_fixed_batch,   # 新增
    )
    
    print("开始分布式训练...")
    train_result = trainer.train()
    print("所有任务已完成！")

    ##################################
    # Save final model and logs
    ##################################
    print("\n" + "=" * 60)
    print("*** Saving final model and logs ***")
    
    if trainer.is_fsdp_enabled and trainer.accelerator.state.fsdp_plugin is not None:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    
    trainer.save_model(args.output_dir)
    print(f"✅ 模型已保存到: {args.output_dir}")
    
    try:
        tokenizer.save_pretrained(args.output_dir)
    except Exception as e:
        print(f"⚠️  警告: 无法自动保存 tokenizer: {e}")
    
    if trainer.accelerator.is_main_process:
        # 保存训练指标
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        
        # 确保保存完整的训练状态
        try:
            trainer_state_file = os.path.join(args.output_dir, "trainer_state.json")
            trainer.state.save_to_json(trainer_state_file)
            print(f"✅ 训练状态已保存到: {trainer_state_file}")
        except Exception as e:
            print(f"⚠️  警告: 无法保存训练状态: {e}")
        
        # 如果有日志历史，也单独保存
        if hasattr(trainer.state, 'log_history') and trainer.state.log_history:
            log_history_file = os.path.join(args.output_dir, "log_history.json")
            with open(log_history_file, "w") as f:
                json.dump(trainer.state.log_history, f, indent=2)
            print(f"✅ 训练日志历史已保存到: {log_history_file}")


if __name__ == "__main__":
    main()
