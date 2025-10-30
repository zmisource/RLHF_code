import os
import torch
import random
import argparse
import bitsandbytes
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    AutoModelForSequenceClassification,
    TrainerCallback,
)
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Dict, Any, List, Union
from tqdm import tqdm
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# CustomSymPOTrainer (from implement_KL.ipynb)
# ---------------------------------------------------------------------------

class CustomSymPOTrainer(Trainer):
    def __init__(
        self,
        model: Union[PeftModel, torch.nn.Module],
        ref_model: Union[PeftModel, torch.nn.Module],
        reward_model: Union[PeftModel, torch.nn.Module],
        args: TrainingArguments,
        beta_kl: float,
        log_ratio_clip_min: float,
        log_ratio_clip_max: float,
        **kwargs,
    ):
        """
        用于 SymPO 算法的自定义 Trainer。

        Args:
            model (PeftModel): 策略模型 (policy_model)。
            ref_model (PeftModel): 参考模型 (ref_model)。
            reward_model (torch.nn.Module): 奖励模型 (reward_model)。
            args (TrainingArguments): 训练参数。
            beta_kl (float): KL散度惩罚项的系数。
            log_ratio_clip_min (float): 对数概率比的最小裁剪值。
            log_ratio_clip_max (float): 对数概率比的最大裁剪值。
        """
        super().__init__(model=model, args=args, **kwargs)

        if not hasattr(self, 'processor') or self.processor is None:
            self.processor = self.tokenizer

        self.ref_model = ref_model
        self.reward_model = reward_model
        self.beta_kl = beta_kl
        self.log_ratio_clip_min = log_ratio_clip_min
        self.log_ratio_clip_max = log_ratio_clip_max

    def _get_log_probs(self, model: AutoModelForCausalLM, prompts: List[str], responses: List[str]) -> torch.Tensor:
        full_texts = [p + r for p, r in zip(prompts, responses)]

        original_padding_side = self.processor.padding_side
        self.processor.padding_side = 'left'

        prompt_tokens = self.processor(prompts, padding=False, truncation=False)
        prompt_lengths = [len(p) for p in prompt_tokens['input_ids']]

        # 检查模型是否被`torch.nn.DataParallel`或`DistributedDataParallel`包裹
        unwrapped_model = model.module if hasattr(model, "module") else model
        max_len = unwrapped_model.config.max_position_embeddings

        full_tokens = self.processor(
            full_texts, padding=True, truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
        self.processor.padding_side = original_padding_side

        input_ids = full_tokens['input_ids'].to(self.args.device)
        attention_mask = full_tokens['attention_mask'].to(self.args.device)

        outputs = model(input_ids, attention_mask=attention_mask, return_dict=True)
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
        del logits, shifted_logits, shifted_labels, nll_per_token, log_probs_per_token, response_mask, masked_log_probs

        return total_log_probs

    def _get_rewards(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        texts = [p + r for p, r in zip(prompts, responses)]
        original_padding_side = self.processor.padding_side
        self.processor.padding_side = 'right'
        
        # 检查模型是否被`torch.nn.DataParallel`或`DistributedDataParallel`包裹
        unwrapped_model = self.reward_model.module if hasattr(self.reward_model, "module") else self.reward_model
        reward_max_len = unwrapped_model.config.max_position_embeddings
        
        inputs = self.processor(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=reward_max_len
        ).to(self.args.device)
        self.processor.padding_side = original_padding_side
        with torch.no_grad():
            outputs = self.reward_model(**inputs)
            return torch.sigmoid(outputs.logits.squeeze(-1))

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        prompts = inputs.pop("prompt")
        chosen_responses = inputs.pop("chosen")
        rejected_responses = inputs.pop("rejected")
        y1s = chosen_responses
        y2s = rejected_responses

        log_probs_pi_y1 = self._get_log_probs(model, prompts, y1s)
        log_probs_pi_y2 = self._get_log_probs(model, prompts, y2s)

        with torch.no_grad():
            with self.model.disable_adapter():
                log_probs_ref_y1 = self._get_log_probs(self.model, prompts, y1s)
                log_probs_ref_y2 = self._get_log_probs(self.model, prompts, y2s)

        log_ratio_y1 = log_probs_pi_y1 - log_probs_ref_y1
        log_ratio_y2 = log_probs_pi_y2 - log_probs_ref_y2

        with torch.no_grad():
            f_y1 = self._get_rewards(prompts, y1s) 
            f_y2 = self._get_rewards(prompts, y2s)

            log_ratio_weight_y1 = log_ratio_y1.detach()
            log_ratio_weight_y2 = log_ratio_y2.detach()

            kl_term1 = self.beta_kl * log_ratio_weight_y1
            kl_term2 = self.beta_kl * log_ratio_weight_y2

            weight1 = 1 - f_y2 - kl_term1 # ipynb 中已注释
            weight2 = f_y1 + kl_term2 # ipynb 中已注释

        clamped_log_ratio_y1 = torch.clamp(log_ratio_y1, min=self.log_ratio_clip_min, max=self.log_ratio_clip_max)
        ratio_y1 = torch.exp(clamped_log_ratio_y1)

        clamped_log_ratio_y2 = torch.clamp(log_ratio_y2, min=self.log_ratio_clip_min, max=self.log_ratio_clip_max)
        ratio_y2 = torch.exp(clamped_log_ratio_y2)

        J_sym_objective = ratio_y1 * weight1 - ratio_y2 * weight2
        total_loss = -J_sym_objective.mean()

        logs = {
            "mean_ratio_chosen": ratio_y1.detach().mean().item(),
            "mean_ratio_rejected": ratio_y2.detach().mean().item(),
            "mean_weight_chosen": weight1.detach().mean().item(),
            "mean_weight_rejected": weight2.detach().mean().item(),
            # "mean_logprob_pi_chosen": log_probs_pi_y1.detach().mean().item(),
            # "mean_logprob_ref_chosen": log_probs_ref_y1.detach().mean().item(),
            "f_chosen": f_y1.detach().mean().item(),
            "f_rejected": f_y2.detach().mean().item(),
            # "debug_ratio_chosen_max": ratio_y1.detach().max().item(),
            # "debug_ratio_rejected_max": ratio_y2.detach().max().item(),
            # "debug_weight_chosen_max": weight1.detach().max().item(),
            # "debug_weight_rejected_max": weight2.detach().max().item(),
            # "debug_logprob_pi_chosen_max": log_probs_pi_y1.detach().max().item(),
            # "debug_logprob_ref_chosen_max": log_probs_ref_y1.detach().max().item(),
            # "debug_objective_mean": J_sym_objective.detach().mean().item()
        }
        self._current_logs = logs

        if self.args.device == "cuda:0" or self.args.device == "cuda":
            torch.cuda.empty_cache()

        return total_loss

    def log(self, logs: dict, *args, **kwargs) -> None:
        if hasattr(self, "_current_logs") and self._current_logs is not None:
            logs.update(self._current_logs)
            self._current_logs = None

        super().log(logs, *args, **kwargs)

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Evaluation requires an eval_dataset.")

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.eval()

        total_samples = 0
        total_accuracy_count = 0
        total_policy_margins = 0
        total_rewards_chosen = 0
        total_rewards_rejected = 0

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                prompts = batch["prompt"]
                chosen_responses = batch["chosen"]
                rejected_responses = batch["rejected"]

                batch_size = len(prompts)
                total_samples += batch_size

                policy_chosen_logps = self._get_log_probs(self.model, prompts, chosen_responses)
                policy_rejected_logps = self._get_log_probs(self.model, prompts, rejected_responses)
                rewards_chosen = self._get_rewards(prompts, chosen_responses)
                rewards_rejected = self._get_rewards(prompts, rejected_responses)

                correct_predictions = (policy_chosen_logps > policy_rejected_logps).sum().item()
                total_accuracy_count += correct_predictions

                total_policy_margins += (policy_chosen_logps - policy_rejected_logps).sum().item()
                total_rewards_chosen += rewards_chosen.sum().item()
                total_rewards_rejected += rewards_rejected.sum().item()

        accuracy = total_accuracy_count / total_samples if total_samples > 0 else 0
        policy_margins = total_policy_margins / total_samples if total_samples > 0 else 0
        rewards_chosen_mean = total_rewards_chosen / total_samples if total_samples > 0 else 0
        rewards_rejected_mean = total_rewards_rejected / total_samples if total_samples > 0 else 0
        reward_margins = rewards_chosen_mean - rewards_rejected_mean

        metrics = {
            f"{metric_key_prefix}_accuracy": accuracy,
            f"{metric_key_prefix}_policy_margins": policy_margins,
            f"{metric_key_prefix}_rewards_chosen_mean": rewards_chosen_mean,
            f"{metric_key_prefix}_rewards_rejected_mean": rewards_rejected_mean,
            f"{metric_key_prefix}_rewards_margins": reward_margins,
        }

        self.model.train()
        self.log(metrics)
        return metrics

# ---------------------------------------------------------------------------
# Data Collator (from implement_KL.ipynb)
# ---------------------------------------------------------------------------

@dataclass
class DataCollatorForCustomSymPO:
    """
    一个简单的 Data Collator，它将一批字典（每个包含字符串）
    转换为一个包含字符串列表的字典。
    """
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}
        for key in features[0].keys():
            batch[key] = [feature[key] for feature in features]
        return batch

# ---------------------------------------------------------------------------
# Callback (from implement_KL.ipynb)
# ---------------------------------------------------------------------------

class PrintingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and logs:
            print(logs)

# ---------------------------------------------------------------------------
# Argument Parsing (inspired by implement_final_v3_no_kl_term2.py)
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train a policy model with SymPO algorithm using 4-bit quantization.")
    
    # --- 路径参数 (来自 ipynb Cell 7) ---
    parser.add_argument("--sft_model_path", type=str, default="/train/Llama-3-8B-Instruct", help="Path to the SFT base model.")
    parser.add_argument("--reward_model_path", type=str, default="/train/f_model", help="Path to the reward model.")
    parser.add_argument("--dataset_train_name", type=str, default="/train/traindataset", help="Path to the training dataset.")
    parser.add_argument("--output_dir", type=str, default="/train/output_model/llama3-8b-sympo-5e-5_0.1", help="Directory to save checkpoints and final model.")
    
    # --- 训练超参数 (来自 ipynb Cell 7 & 9) ---
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per GPU.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Number of gradient accumulation steps.")
    parser.add_argument("--gradient_checkpointing", action='store_true', default=False, help="Enable gradient checkpointing (default is False in ipynb).")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for learning rate scheduler.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log metrics every N steps.")
    parser.add_argument("--save_steps", type=int, default=300, help="Save a checkpoint every N steps.")
    parser.add_argument("--save_total_limit", type=int, default=20, help="Limit the total number of saved checkpoints.")

    # --- SymPO 特定参数 (来自 ipynb Cell 9) ---
    parser.add_argument("--beta_kl", type=float, default=0.1, help="KL divergence penalty coefficient.")
    parser.add_argument("--log_ratio_clip_min", type=float, default=-2.3, help="Minimum clip value for log probability ratio.")
    parser.add_argument("--log_ratio_clip_max", type=float, default=2.3, help="Maximum clip value for log probability ratio.")
    
    # --- W&B (Weights & Biases) 日志参数 ---
    parser.add_argument("--report_to", type=str, default=None, help="The integration to report results to (e.g., 'wandb', 'none'). Default is None from ipynb.")
    parser.add_argument("--run_name", type=str, default="sympo-llama3-8b-run", help="A name for the W&B run.")

    # --- QLoRA/Device 参数 (来自 ipynb Cell 8) ---
    parser.add_argument("--device_map", type=str, default="auto", help="Device map for model loading.")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2", help="Attention implementation (e.g., 'flash_attention_2').")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main Function (combining ipynb Cells 8, 9, 10, 11)
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # 1. 4-bit 量化配置 (from ipynb Cell 8)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # 2. LoRA 配置 (from ipynb Cell 8)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 3. 加载 Tokenizer (from ipynb Cell 8)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 注意：Trainer 内部的 _get_log_probs 会自行处理 padding_side
    # 但我们可以在这里设置一个默认值
    tokenizer.padding_side = 'left' 

    # 4. 加载模型 (from ipynb Cell 8)
    print("Loading policy model (for training)...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path,
        quantization_config=quantization_config,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        torch_dtype=torch.bfloat16,
    )
    policy_model = prepare_model_for_kbit_training(policy_model)
    policy_model = get_peft_model(policy_model, lora_config)

    print("Loading reward model (frozen)...")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_path,
        quantization_config=quantization_config,
        device_map=args.device_map,
        num_labels=1,
        attn_implementation=args.attn_implementation,
        torch_dtype=torch.bfloat16,
    )
    if reward_model.config.pad_token_id is None:
        reward_model.config.pad_token_id = tokenizer.pad_token_id

    for param in reward_model.parameters():
        param.requires_grad = False
    
    # 5. 加载和预处理数据集 (from ipynb Cell 10)
    
    # 将 ipynb Cell 10 中的预处理函数定义在 main 内部，使其可以访问 'tokenizer'
    def preprocess_ultrafeedback(example: dict) -> dict:
        """
        Args:
            example: 数据集中的一个样本字典。
        Returns:
            一个包含 'prompt', 'chosen', 'rejected' 键的新字典。
        """
        chosen_response = example['chosen'][-1]['content']+ "<|eot_id|>\n"
        rejected_response = example['rejected'][-1]['content']+ "<|eot_id|>\n"
        conversation_history = example['chosen'][:-1]

        full_prompt = tokenizer.apply_chat_template(
            conversation_history,
            tokenize=False,
            add_generation_prompt=True
        )
        if not full_prompt or not chosen_response or not rejected_response:
            return {"prompt": "", "chosen": "", "rejected": ""}

        return {
            "prompt": full_prompt,
            "chosen": chosen_response,
            "rejected": rejected_response
        }

    dataset_train = load_from_disk(args.dataset_train_name)

    print("Preprocessing dataset...")
    processed_dataset_train = dataset_train.map(
        preprocess_ultrafeedback,
        num_proc=4, # 可以考虑也加入 argparse
        remove_columns=dataset_train.column_names
    )

    processed_dataset_train = processed_dataset_train.filter(
        lambda x: x['prompt'] and x['chosen'] and x['rejected']
    )
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        optim="adamw_8bit", # ipynb 使用 8-bit adam
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        bf16=True, # ipynb 启用 bf16
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_strategy="no", # ipynb 设置为 "no"
        remove_unused_columns=False,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
        run_name=args.run_name,
    )
    
    data_collator = DataCollatorForCustomSymPO()

    print("Initializing CustomSymPOTrainer...")
    trainer = CustomSymPOTrainer(
        model=policy_model,
        ref_model=policy_model, 
        reward_model=reward_model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=processed_dataset_train,
        data_collator=data_collator,
        callbacks=[PrintingCallback()],
        beta_kl=args.beta_kl,
        log_ratio_clip_min=args.log_ratio_clip_min,
        log_ratio_clip_max=args.log_ratio_clip_max,
    )

    # 8. 开始训练 (from ipynb Cell 11)
    print("Starting training...")
    trainer.train()
    
    print("Training finished.")



if __name__ == "__main__":
    main()