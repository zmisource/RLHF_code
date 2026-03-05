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
# 随机种子设置
# ---------------------------------------------------------------------------
def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# ---------------------------------------------------------------------------
# CustomSymPOTrainer (已针对 apply_chat_template 优化)
# ---------------------------------------------------------------------------
class CustomSymPOTrainer(Trainer):
    def __init__(
        self,
        model: Union[torch.nn.Module],
        args: TrainingArguments,
        beta: float,
        gamma: float,
        max_length: int = None,
        **kwargs,
    ):
        super().__init__(model=model, args=args, **kwargs)
        if not hasattr(self, 'processor') or self.processor is None:
            self.processor = self.tokenizer
        self.beta = beta
        self.gamma = gamma
        self.max_length = max_length

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

        # 1. Tokenize
        # 注意：这里假设 prompts 已经是经过 apply_chat_template 的字符串
        # 包含了 <|begin_of_text|> 和 header
        prompt_encodings = processor(
            prompts, 
            add_special_tokens=False, # 模板里通常已经有了，所以 False
            padding=False, 
            truncation=False
        )
        response_encodings = processor(
            responses, 
            add_special_tokens=False, # 响应里已经手动加了 <|eot_id|>，所以 False
            padding=False, 
            truncation=False
        )

        input_ids_list = []
        loss_masks_list = []

        if max_length is None:
            max_length = 2048

        # 获取 BOS ID 用于检测
        bos_token_id = processor.bos_token_id
        
        for i in range(len(prompts)):
            p_ids = prompt_encodings['input_ids'][i]
            r_ids = response_encodings['input_ids'][i]
            
            # --- [关键逻辑修改] 检测并处理 BOS ---
            # 如果 prompt 已经是通过模板生成的，通常 p_ids[0] 就是 BOS (128000)
            # 如果不是，我们需要手动加上，确保注意力机制正常工作
            if len(p_ids) == 0: # 异常保护
                continue
                
            if bos_token_id is not None and p_ids[0] != bos_token_id:
                # 只有当开头缺 BOS 时才补
                combined_ids = [bos_token_id] + p_ids + r_ids
                mask = [0] * (len(p_ids) + 1) + [1] * len(r_ids)
            else:
                # 已经有 BOS 了，直接拼
                combined_ids = p_ids + r_ids
                mask = [0] * len(p_ids) + [1] * len(r_ids)
            
            # 截断处理 (保留尾部，这对对话模型很重要)
            if len(combined_ids) > max_length:
                combined_ids = combined_ids[-max_length:]
                mask = mask[-max_length:]
            
            input_ids_list.append(torch.tensor(combined_ids, dtype=torch.long))
            loss_masks_list.append(torch.tensor(mask, dtype=torch.float))

        # 2. Padding (Left Padding)
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
        
        # 手动处理 Mask 的 Padding
        max_batch_len = input_ids.shape[1]
        padded_loss_masks = torch.zeros((len(loss_masks_list), max_batch_len), dtype=torch.float, device=device)
        
        for i, mask_tensor in enumerate(loss_masks_list):
            seq_len = len(mask_tensor)
            if seq_len > 0:
                padded_loss_masks[i, -seq_len:] = mask_tensor.to(device)
            
        processor.padding_side = original_padding_side

        # 3. 前向传播
        outputs = model(input_ids, attention_mask=attention_mask, return_dict=True, use_cache=False)
        logits = outputs.logits

        # 4. Shift & Loss Calculation
        shifted_logits = logits[..., :-1, :]
        shifted_labels = input_ids[..., 1:]
        shifted_mask = padded_loss_masks[..., 1:] 
        shifted_attention_mask = attention_mask[..., 1:]

        # 这里的 Mask 逻辑：既要是 Response 部分，又要是 非Padding 部分
        final_mask = shifted_mask * shifted_attention_mask.float()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        nll_per_token = loss_fct(
            shifted_logits.reshape(-1, shifted_logits.size(-1)), 
            shifted_labels.reshape(-1)
        )
        nll_per_token = nll_per_token.view(shifted_labels.size())
        
        masked_log_probs = (-nll_per_token) * final_mask
        
        total_log_probs = masked_log_probs.sum(dim=1)
        valid_response_lens = final_mask.sum(dim=1)
        
        return total_log_probs, valid_response_lens

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        prompts = inputs.pop("prompt")
        chosen_responses = inputs.pop("chosen")
        rejected_responses = inputs.pop("rejected")
    
        batch_size = len(prompts)
        combined_prompts = prompts + prompts
        combined_responses = chosen_responses + rejected_responses
        
        all_log_probs_pi, valid_response_lens = self._get_log_probs(
            model, combined_prompts, combined_responses, max_length=self.max_length
        )        

        pi_chosen = all_log_probs_pi[:batch_size]
        pi_rejected = all_log_probs_pi[batch_size:]
        len_chosen = valid_response_lens[:batch_size]
        len_rejected = valid_response_lens[batch_size:]

        # SimPO 公式: Average Log Prob
        reward_chosen = (pi_chosen / len_chosen) * self.beta
        reward_rejected = (pi_rejected / len_rejected) * self.beta
        
        margin = reward_chosen - reward_rejected - self.gamma
        loss = -F.logsigmoid(margin).mean()
        
        # Logging
        accuracy = (margin > 0).float().mean().item()
        self._current_logs = {
            "loss": loss.item(),
            "rewards/chosen": reward_chosen.detach().mean().item(),
            "rewards/rejected": reward_rejected.detach().mean().item(),
            "rewards/accuracy": accuracy,
            "rewards/margin": margin.detach().mean().item(),
        }
        
        if return_outputs:
            return (loss, None)
        return loss

    def log(self, logs: dict, *args, **kwargs) -> None:
        if hasattr(self, "_current_logs") and self._current_logs is not None:
            logs.update(self._current_logs)
            self._current_logs = None
        super().log(logs, *args, **kwargs)

@dataclass
class DataCollatorForSimPO:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}
        # 确保只取这三个字段，避免取到 dataset 里可能存在的 tensor 字段导致报错
        for key in ["prompt", "chosen", "rejected"]:
            batch[key] = [feature[key] for feature in features]
        return batch

def parse_args():
    parser = argparse.ArgumentParser()
    # 路径参数
    parser.add_argument("--sft_model_path", type=str, default="/train/Llama-3-8B-Instruct")
    parser.add_argument("--preprocessed_dataset_path", type=str, default="/train/ds_with_metrics")
    parser.add_argument("--output_dir", type=str, default="/train/output_model/llama3-8b-simpo")
    
    # 训练参数
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--gradient_checkpointing", action='store_true')
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=5)

    # SimPO 参数
    parser.add_argument("--beta", type=float, default=2.5)
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--max_length", type=int, default=2048)
    
    # 杂项
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--run_name", type=str, default="simpo-llama3")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    seed_everything(args.seed)

    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)
    # Llama-3 Pad Token 设置：
    # 如果 <|eot_id|> (128009) 用作结束符，pad 最好设为 <|reserved_special_token_0|> 或 eos_token
    # 只要不和 eot_id 冲突即可
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left' 

    print("从磁盘加载数据集...")
    dataset = load_from_disk(args.preprocessed_dataset_path)

    # 2. 数据过滤 (适配 preprocess_ultrafeedback 的返回值)
    print("正在过滤无效数据...")
    def filter_fn(example):
        # 你的预处理函数失败时返回空字符串，这里必须剔除
        if not example.get('prompt') or len(example['prompt']) == 0:
            return False
        if not example.get('chosen') or len(example['chosen']) == 0:
            return False
        if not example.get('rejected') or len(example['rejected']) == 0:
            return False
        return True

    original_len = len(dataset)
    dataset = dataset.filter(filter_fn)
    print(f"过滤前: {original_len}, 过滤后: {len(dataset)}")
    
    # 3. 加载模型 (注意 torch_dtype)
    print("加载 Policy Model...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path, 
        dtype=torch.bfloat16
    )

    if args.gradient_checkpointing:
        policy_model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={'use_reentrant': False} if args.gradient_checkpointing else None,
        optim="adamw_torch",
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        bf16=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit, 
        report_to=args.report_to,
        run_name=args.run_name,
        remove_unused_columns=False # 防止 Dataset 中的 string column 被移除
    )
    
    trainer = CustomSymPOTrainer(
        model=policy_model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset,
        data_collator=DataCollatorForSimPO(),
        beta=args.beta,
        gamma=args.gamma,
        max_length=args.max_length,
    )
    
    print("开始训练...")
    trainer.train()
    
    print("保存模型...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()