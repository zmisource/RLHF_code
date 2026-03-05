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
)
from typing import Dict, Any, List, Union, Tuple
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence

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
# CustomSymPOTrainer (最终修复版)
# ---------------------------------------------------------------------------
class CustomSymPOTrainer(Trainer):
    def __init__(
        self,
        model: Union[torch.nn.Module],
        args,
        beta: float,
        gamma: float,
        max_length: int = None,
        **kwargs,
    ):
        super().__init__(model=model, args=args, **kwargs)

        self.beta = beta
        self.gamma = gamma
        self.max_length = max_length
        
        self._metrics_buffer = {} 
        self._batch_cnt = 0

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
            seq_len = input_ids.shape[1]
            
            valid_indices = (attention_mask[i] == 1).nonzero(as_tuple=True)[0]
            
            if len(valid_indices) > 0:
                start_index = valid_indices[0].item()
            else:
                start_index = 0 
            # 限制 prompt mask 不超出截断后序列长度
            prompt_end_index = min(start_index + p_len, seq_len)
            
            # 如果 prompt 占满了整个截断后的序列，response 被完全截掉
            if prompt_end_index >= seq_len:
                import warnings
                warnings.warn(
                    f"Sample {i}: prompt ({p_len} tokens) fills entire "
                    f"truncated sequence ({seq_len} tokens). "
                    f"Response has 0 valid tokens."
                )
            
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

        reward_chosen = (pi_chosen / len_chosen) * self.beta
        reward_rejected = (pi_rejected / len_rejected) * self.beta
        
        margin = reward_chosen - reward_rejected - self.gamma
        loss = -F.logsigmoid(margin).mean()
        
        acc_strict = (margin > 0).float().mean().item()
        acc_simple = (reward_chosen > reward_rejected).float().mean().item()
        
        current_metrics = {
            "loss": loss.item(),
            "rewards/chosen": reward_chosen.detach().mean().item(),
            "rewards/rejected": reward_rejected.detach().mean().item(),
            "rewards/accuracy": acc_strict,   
            "rewards/win_rate": acc_simple,
            "rewards/margins": margin.detach().mean().item(),
        }

        for k, v in current_metrics.items():
            if k not in self._metrics_buffer:
                self._metrics_buffer[k] = 0.0
            self._metrics_buffer[k] += v
        
        self._batch_cnt += 1
        
        if return_outputs:
            return (loss, None)
        return loss

    def log(self, logs: dict, *args, **kwargs) -> None:
        if self._batch_cnt > 0:
            for k, v in self._metrics_buffer.items():
                logs[k] = v / self._batch_cnt
            self._metrics_buffer = {}
            self._batch_cnt = 0
        super().log(logs, *args, **kwargs)

@dataclass
class DataCollatorForSimPO:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}
        for key in ["prompt", "chosen", "rejected"]:
            batch[key] = [feature[key] for feature in features]
        return batch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_model_path", type=str, default="/root/autodl-tmp/.autodl/Llama-3-8B-Instruct")
    parser.add_argument("--preprocessed_dataset_path", type=str, default="/root/autodl-tmp/.autodl/ds_with_metrics")
    parser.add_argument("--output_dir", type=str, default="/train/output_model/llama3-8b-simpo")
    
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--gradient_checkpointing", action='store_true')
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    
    parser.add_argument("--beta", type=float, default=2.5)
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--max_length", type=int, default=2048)
    
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--run_name", type=str, default="simpo-llama3")
    parser.add_argument("--seed", type=int, default=42)

    # --- 单 batch 调试模式 ---
    parser.add_argument("--single_batch_debug", action='store_true',
                        help="只取前 N 条样本反复训练（用于 overfitting 调试）")
    parser.add_argument("--single_batch_size", type=int, default=8,
                        help="single_batch_debug 模式下保留的样本数")
    parser.add_argument("--max_steps", type=int, default=100,
                        help="single_batch_debug 模式下的最大训练步数")
    return parser.parse_args()

def main():
    args = parse_args()
    seed_everything(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)
    
    if tokenizer.pad_token is None:
        if '<|reserved_special_token_0|>' in tokenizer.get_vocab():
            tokenizer.pad_token = '<|reserved_special_token_0|>'
        else:
            tokenizer.pad_token = tokenizer.eos_token
            
    tokenizer.padding_side = 'left' 

    print("从磁盘加载数据集...")
    dataset = load_from_disk(args.preprocessed_dataset_path)

    def filter_fn(example):
        if not example.get('prompt') or len(example['prompt']) == 0: return False
        if not example.get('chosen') or len(example['chosen']) == 0: return False
        if not example.get('rejected') or len(example['rejected']) == 0: return False
        if len(example['prompt']) + len(example['chosen']) > args.max_length * 5: return False
        if len(example['prompt']) + len(example['rejected']) > args.max_length * 5: return False
        return True

    original_len = len(dataset)
    dataset = dataset.filter(filter_fn)
    print(f"过滤前: {original_len}, 过滤后: {len(dataset)}")

    # --- 单 batch 调试：只保留前 N 条样本 ---
    if args.single_batch_debug:
        n = min(args.single_batch_size, len(dataset))
        dataset = dataset.select(range(n))
        print(f"[single_batch_debug] 只保留前 {n} 条样本，将反复训练")
    
    print("加载 Policy Model...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path, 
        torch_dtype=torch.bfloat16
    )

    if args.gradient_checkpointing:
        policy_model.gradient_checkpointing_enable()

    # --- 根据是否为 single_batch_debug 构造 TrainingArguments ---
    ta_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
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
        remove_unused_columns=False,
    )

    if args.single_batch_debug:
        # 用 max_steps 控制训练步数，不设 epoch
        ta_kwargs["max_steps"] = args.max_steps
        # 保证每个 step 用完整 batch（= 数据集大小），gradient_accumulation=1
        ta_kwargs["per_device_train_batch_size"] = args.single_batch_size
        ta_kwargs["gradient_accumulation_steps"] = 1
        ta_kwargs["dataloader_drop_last"] = False
        print(f"[single_batch_debug] max_steps={args.max_steps}, "
              f"batch_size={args.single_batch_size}, grad_accum=1")
    else:
        ta_kwargs["num_train_epochs"] = args.num_train_epochs

    training_args = TrainingArguments(**ta_kwargs)
    
    trainer = CustomSymPOTrainer(
        model=policy_model,
        args=training_args,
        processing_class=tokenizer, 
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