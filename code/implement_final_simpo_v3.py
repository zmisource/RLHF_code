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

# --- [新增] 引入 PEFT 库 ---
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)

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
# CustomSymPOTrainer (保持不变)
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

        if not hasattr(self, 'processor') or self.processor is None:
            self.processor = self.tokenizer

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
        max_length: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        tokenizer = self.tokenizer
        device = self.args.device
        if max_length is None:
            max_length = 2048

        prompt_ids_list = tokenizer(prompts, add_special_tokens=False)["input_ids"]
        response_ids_list = tokenizer(responses, add_special_tokens=False)["input_ids"]
        
        input_ids_list = []
        labels_list = []
        
        for p_ids, r_ids in zip(prompt_ids_list, response_ids_list):
            full_ids = p_ids + r_ids
            curr_labels = [-100] * len(p_ids) + r_ids
            
            if len(full_ids) > max_length:
                full_ids = full_ids[:max_length]
                curr_labels = curr_labels[:max_length]
            
            input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
            labels_list.append(torch.tensor(curr_labels, dtype=torch.long))

        input_ids_flipped = [t.flip(0) for t in input_ids_list]
        labels_flipped = [t.flip(0) for t in labels_list]
        
        input_ids_padded = pad_sequence(input_ids_flipped, batch_first=True, padding_value=tokenizer.pad_token_id).flip(1)
        labels_padded = pad_sequence(labels_flipped, batch_first=True, padding_value=-100).flip(1)
        
        input_ids = input_ids_padded.to(device)
        labels = labels_padded.to(device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

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
    parser.add_argument("--sft_model_path", type=str, default="/train/Llama-3-8B-Instruct")
    parser.add_argument("--preprocessed_dataset_path", type=str, default="/train/ds_with_metrics")
    parser.add_argument("--output_dir", type=str, default="/train/output_model/llama3-8b-simpo")
    
    # --- [修改] LoRA 学习率 ---
    # LoRA 通常需要比全量微调更大的 LR。推荐 5e-5 到 1e-4。
    parser.add_argument("--learning_rate", type=float, default=1e-5) 
    
    parser.add_argument("--num_train_epochs", type=int, default=1)
    
    # --- [修改] Batch Size ---
    # 有了 LoRA，显存压力骤减，你可以尝试把 Batch Size 改回 2 或 4 (具体看显存)
    # SimPO 内部会翻倍，建议先设为 2 试试
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    
    parser.add_argument("--gradient_checkpointing", action='store_true', default=True)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=3)
    
    parser.add_argument("--beta", type=float, default=2.5)
    parser.add_argument("--gamma", type=float, default=1.4)
    # 有了 LoRA，显存允许的话，Max Length 可以适当调大回 2048，或者保持 1024
    parser.add_argument("--max_length", type=int, default=1024)
    
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--run_name", type=str, default="simpo-llama3-lora")
    parser.add_argument("--seed", type=int, default=42)
    
    # --- [新增] LoRA 参数 ---
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    return parser.parse_args()

def main():
    args = parse_args()
    seed_everything(args.seed)

    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)
    
    if tokenizer.pad_token is None:
        if '<|reserved_special_token_0|>' in tokenizer.get_vocab():
            tokenizer.pad_token = '<|reserved_special_token_0|>'
        else:
            tokenizer.pad_token = tokenizer.eos_token
            
    tokenizer.padding_side = 'left' 

    print("从磁盘加载数据集...")
    dataset = load_from_disk(args.preprocessed_dataset_path)

    # 2. 数据过滤
    def filter_fn(example):
        if not example.get('prompt') or len(example['prompt']) == 0: return False
        if not example.get('chosen') or len(example['chosen']) == 0: return False
        if not example.get('rejected') or len(example['rejected']) == 0: return False
        if len(example['prompt']) + len(example['chosen']) > args.max_length * 5: return False 
        return True

    original_len = len(dataset)
    dataset = dataset.filter(filter_fn)
    print(f"过滤前: {original_len}, 过滤后: {len(dataset)}")
    
    # 3. 加载模型
    print("加载 Policy Model...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path, 
        torch_dtype=torch.bfloat16)

    # --- [新增] LoRA 配置与应用 ---
    print(f"应用 LoRA 配置 (Rank: {args.lora_rank})...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        # Llama-3 的所有线性层
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    policy_model = get_peft_model(policy_model, peft_config)
    
    # 打印可训练参数量，确认 LoRA 生效
    policy_model.print_trainable_parameters()
    # -----------------------------

    # LoRA 兼容的 Gradient Checkpointing 设置
    if args.gradient_checkpointing:
        # 确保输入层可以计算梯度
        if hasattr(policy_model, "enable_input_require_grads"):
            policy_model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            policy_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            
        policy_model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        # LoRA 训练中，adamw_torch 就可以
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
        remove_unused_columns=False
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
    # Trainer 会自动处理 PeftModel 的保存，只保存 adapter
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()