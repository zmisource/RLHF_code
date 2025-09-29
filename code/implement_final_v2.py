# 文件名: train_sympo.py

# -*- coding: utf-8 -*-
import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    TrainerCallback
)
from typing import Dict, Any, List, Union
import tempfile
import pandas as pd
from dataclasses import dataclass

# 假设这些是您的自定义或第三方库
# from oumi.datasets.evaluation import AlpacaEvalDataset
# from alpaca_eval import evaluate as alpaca_evaluate
# from oumi.inference import VLLMInferenceEngine
# from oumi.core.configs import ModelParams, InferenceConfig, GenerationParams

# from huggingface_hub import login


# ---------------------------------------------------------------------------
# CustomSymPOTrainer
# ---------------------------------------------------------------------------

class CustomSymPOTrainer(Trainer):
    def __init__(
        self,
        model: Union[torch.nn.Module],
        args: TrainingArguments,
        sft_model_path: str,
        beta_kl: float,
        log_ratio_clip_min: float,
        log_ratio_clip_max: float,
        **kwargs,
    ):
        super().__init__(model=model, args=args, **kwargs)

        if not hasattr(self, 'processor') or self.processor is None:
            self.processor = self.tokenizer

        self.beta_kl = beta_kl
        self.log_ratio_clip_min = log_ratio_clip_min
        self.log_ratio_clip_max = log_ratio_clip_max
        self.sft_model_path = sft_model_path

        # # [专家建议] 推理引擎只在主进程中初始化，以节省其他进程的内存
        # if self.args.local_rank == 0:
        #     model_params = ModelParams(
        #         model_name=self.sft_model_path,
        #         model_max_length=8192,
        #         tokenizer_kwargs={"pad_token": "<|end_of_text|>"}
        #     )
        #     # 注意：VLLM可能需要指定GPU。请根据VLLM文档确认多卡环境下如何最佳初始化
        #     self.inference_engine = VLLMInferenceEngine(model_params)
        # else:
        #     self.inference_engine = None
        self.inference_engine = None
    def _get_log_probs(self, model: AutoModelForCausalLM, prompts: List[str], responses: List[str]) -> torch.Tensor:
        full_texts = [p + r for p, r in zip(prompts, responses)]
        # original_padding_side = self.processor.padding_side
        # self.processor.padding_side = 'left'
        prompt_tokens = self.processor(prompts, padding=False, truncation=False)
        prompt_lengths = [len(p) for p in prompt_tokens['input_ids']]
        unwrapped_model = model.module if hasattr(model, "module") else model
        max_len = unwrapped_model.config.max_position_embeddings
        full_tokens = self.processor(
            full_texts, padding='max_length', truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
        # self.processor.padding_side = original_padding_side
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
        return total_log_probs

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        prompts = inputs.pop("prompt")
        chosen_responses = inputs.pop("chosen")
        rejected_responses = inputs.pop("rejected")
        log_probs_ref_y1 = inputs.pop("ref_logp_chosen").to(self.args.device)
        log_probs_ref_y2 = inputs.pop("ref_logp_rejected").to(self.args.device)
        f_y1 = inputs.pop("reward_chosen").to(self.args.device)
        f_y2 = inputs.pop("reward_rejected").to(self.args.device)
        log_probs_pi_y1 = self._get_log_probs(model, prompts, chosen_responses )
        log_probs_pi_y2 = self._get_log_probs(model, prompts, rejected_responses)
        log_ratio_y1 = log_probs_pi_y1 - log_probs_ref_y1
        log_ratio_y2 = log_probs_pi_y2 - log_probs_ref_y2
        with torch.no_grad():
            kl_term1 = self.beta_kl * log_ratio_y1.detach()
            kl_term2 = self.beta_kl * log_ratio_y2.detach()
            weight1 = 1 - f_y2 - kl_term1
            weight2 = f_y1 + kl_term2
        clamped_log_ratio_y1 = torch.clamp(log_ratio_y1, min=self.log_ratio_clip_min, max=self.log_ratio_clip_max)
        ratio_y1 = torch.exp(clamped_log_ratio_y1)
        clamped_log_ratio_y2 = torch.clamp(log_ratio_y2, min=self.log_ratio_clip_min, max=self.log_ratio_clip_max)
        ratio_y2 = torch.exp(clamped_log_ratio_y2)
        J_sym_objective = ratio_y1 * weight1 - ratio_y2 * weight2
        total_loss = -J_sym_objective.mean()
        
        # 日志记录
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

    # def evaluate(
    #     self,
    #     eval_dataset=None,
    #     ignore_keys=None,
    #     metric_key_prefix: str = "eval",
    # ):
    #     """
    #     [修改] 使用 VLLM 引擎和 AlpacaEval 重写评估方法，并增加多GPU保护。
    #     """
    #     if not self.accelerator.is_main_process:
    #         return {}

    #     with tempfile.TemporaryDirectory() as temp_dir:
    #         print(f"主进程 (rank 0) 创建临时模型目录: {temp_dir}")
    #         model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
    #         model_to_save.save_pretrained(temp_dir)
    #         self.tokenizer.save_pretrained(temp_dir)
    #         torch.cuda.empty_cache()

    #         alpaca_dataset = AlpacaEvalDataset(dataset_name="tatsu-lab/alpaca_eval").conversations()
    #         generation_params = GenerationParams(max_new_tokens=8192, temperature=0.9)
    #         model_params = ModelParams(
    #             model_name=temp_dir,
    #             model_max_length=8192,
    #             tokenizer_kwargs={"pad_token": "<|end_of_text|>"}
    #         )
    #         inference_config = InferenceConfig(model=model_params, generation=generation_params)
    #         responses = self.inference_engine.infer(alpaca_dataset, inference_config)
    #         model_outputs = pd.DataFrame({
    #             'instruction': [conv.messages[0].content for conv in alpaca_dataset],
    #             'output': [conv.messages[1].content for conv in responses],
    #             'generator': 'my_policy_model'
    #         })
    #         df_leaderboard, _ = alpaca_evaluate(
    #             model_outputs=model_outputs,
    #             annotators_config="weighted_alpaca_eval_gpt4_turbo",
    #             is_return_instead_of_print=True,
    #         )
    #         metrics_series = df_leaderboard.loc['my_policy_model']
    #         metrics = {f"{metric_key_prefix}_{k}": v for k, v in metrics_series.items()}
    #         lc_win_rate = metrics_series.get('length_controlled_winrate', 0.0)
    #         metrics[f"{metric_key_prefix}_lc_win_rate"] = lc_win_rate
    #         print(f"\nAlpacaEval 评估结果 for `my_policy_model` (在主进程上执行):")
    #         print(metrics_series)
    #         self.log(metrics)
    #     print("主进程 (rank 0) 清理临时模型目录。")
    #     return metrics

# ---------------------------------------------------------------------------
# Data Collator 和 Callback
# ---------------------------------------------------------------------------

@dataclass
class DataCollatorForCustomSymPO:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}
        for key in features[0].keys():
            if isinstance(features[0][key], str):
                batch[key] = [feature[key] for feature in features]
            else:
                batch[key] = torch.tensor([feature[key] for feature in features])
        return batch

class PrintingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and logs:
            print(logs)


# ---------------------------------------------------------------------------
# 主训练流程
# ---------------------------------------------------------------------------
def main():
    # --- 模型和路径配置 ---
    sft_model_path = "/train/Llama-3-8B-Instruct"
    preprocessed_dataset_path = "/train/precomputed_traindataset"
    output_dir  = "/train/output_model/llama3-8b-advan_5e-7_beta_0.2"
    
    # --- 超参数 ---
    lr = 5e-7

    tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left' 

    print("从磁盘加载预处理好的数据集...")
    precomputed_dataset_train = load_from_disk(preprocessed_dataset_path)
    # dummy_eval_dataset = precomputed_dataset_train.select(range(10)) 
    # def preprocess_for_length_sampler(examples):
    #     # 使用 prompt + chosen 的长度作为分组依据
    #     texts = [p + c for p, c in zip(examples['prompt'], examples['chosen'])]
    #     model_inputs = tokenizer(texts, truncation=False)
    #     # 返回所有原始列，并添加新的 input_ids 列
    #     examples['input_ids'] = model_inputs['input_ids']
    #     return examples

    # precomputed_dataset_train = precomputed_dataset_train.map(
    #     preprocess_for_length_sampler,
    #     batched=True,
    # )   
    # --- 策略模型训练 ---
    print("加载用于训练的策略模型...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        sft_model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        # gradient_checkpointing=True,
        # gradient_checkpointing_kwargs={'use_reentrant': False},
        optim="adamw_torch",
        learning_rate=lr,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_strategy="steps",
        logging_steps=10,
        logging_first_step=True,
        # group_by_length=True,
        bf16=True,
        save_strategy="steps",
        save_steps=300,
        eval_strategy="no",
        # eval_steps=10,
        save_total_limit=40, 
        # metric_for_best_model="eval_lc_win_rate",
        # greater_is_better=True,
        remove_unused_columns=False,
        report_to="wandb",
        run_name=f"policy-llama3-8b-sympo-lr{lr}",
        load_best_model_at_end=False,
    )
    
    data_collator = DataCollatorForCustomSymPO()

    print("初始化 CustomSymPOTrainer...")
    trainer = CustomSymPOTrainer(
        model=policy_model,
        args=training_args,
        sft_model_path=sft_model_path, 
        tokenizer=tokenizer,
        train_dataset=precomputed_dataset_train,
        # eval_dataset=dummy_eval_dataset,
        data_collator=data_collator,
        callbacks=[PrintingCallback()],
        beta_kl=0.2,
        log_ratio_clip_min=-3,
        log_ratio_clip_max=3,
    )

    print("开始分布式训练...")
    trainer.train()
    print("所有任务已完成！")

if __name__ == "__main__":
    main()