import argparse
import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_scheduler,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

# ---------------------------------------------------------------------------
# 日志记录器
# ---------------------------------------------------------------------------
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 数据处理和 Collator
# ---------------------------------------------------------------------------
def preprocess_ultrafeedback(example: dict, tokenizer: AutoTokenizer) -> dict:
    """
    Args:
        example: 数据集中的一个样本字典。
        tokenizer: 用于编码的分词器。

    Returns:
        一个包含 'prompt', 'chosen', 'rejected' 键的新字典。
    """
    chosen_response = example['chosen'][-1]['content'] + "<|eot_id|>\n"
    rejected_response = example['rejected'][-1]['content'] + "<|eot_id|>\n"
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

def get_collate_fn():
    def collate_fn(features):
        batch = {}
        for key in features[0].keys():
            batch[key] = [feature[key] for feature in features]
        return batch
    return collate_fn


# ---------------------------------------------------------------------------
# 核心计算逻辑 (从 CustomTrainer 中提取)
# ---------------------------------------------------------------------------

def _get_log_probs(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompts: list, responses: list, device: str) -> torch.Tensor:
    full_texts = [p + r for p, r in zip(prompts, responses)]

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'

    prompt_tokens = tokenizer(prompts, padding=False, truncation=False)
    prompt_lengths = [len(p) for p in prompt_tokens['input_ids']]

    max_len = model.config.max_position_embeddings
    full_tokens = tokenizer(
        full_texts, padding=True, truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    tokenizer.padding_side = original_padding_side
    
    # Accelerator 会自动处理张量到设备的移动，但这里我们是动态创建的，所以需要手动移动
    input_ids = full_tokens['input_ids'].to(device)
    attention_mask = full_tokens['attention_mask'].to(device)

    outputs = model(input_ids, attention_mask=attention_mask, return_dict=True)
    logits = outputs.logits

    shifted_logits = logits[..., :-1, :]
    shifted_labels = input_ids[..., 1:]
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    nll_per_token = loss_fct(shifted_logits.reshape(-1, shifted_logits.size(-1)), shifted_labels.reshape(-1))
    nll_per_token = nll_per_token.view(input_ids.size(0), -1)
    log_probs_per_token = -nll_per_token

    seq_len = shifted_labels.size(1)
    position_ids = torch.arange(seq_len, device=device).expand_as(shifted_labels)
    prompt_lengths_tensor = torch.tensor(prompt_lengths, device=device).unsqueeze(1)
    response_start_index = prompt_lengths_tensor - 1
    mask = position_ids >= response_start_index
    attention_mask_shifted = attention_mask[:, 1:].to(torch.bool)
    response_mask = mask & attention_mask_shifted

    masked_log_probs = log_probs_per_token * response_mask
    total_log_probs = masked_log_probs.sum(dim=1)
    
    return total_log_probs

def _get_rewards(reward_model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, prompts: list, responses: list, device: str) -> torch.Tensor:
    texts = [p + r for p, r in zip(prompts, responses)]
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'right'
    reward_max_len = reward_model.config.max_position_embeddings
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=reward_max_len
    )
    # Accelerator 会自动处理张量到设备的移动，但这里我们是动态创建的，所以需要手动移动
    inputs = {k: v.to(device) for k, v in inputs.items()}
    tokenizer.padding_side = original_padding_side
    
    with torch.no_grad():
        outputs = reward_model(**inputs)
        return torch.sigmoid(outputs.logits.squeeze(-1))

def compute_sympo_loss(
    policy_model, ref_model, reward_model, tokenizer, batch, beta_kl, log_ratio_clip_min, log_ratio_clip_max, device
):
    prompts = batch["prompt"]
    y1s = batch["chosen"]
    y2s = batch["rejected"]
    
    # 使用 policy_model 计算 log probs
    log_probs_pi_y1 = _get_log_probs(policy_model, tokenizer, prompts, y1s, device)
    log_probs_pi_y2 = _get_log_probs(policy_model, tokenizer, prompts, y2s, device)

    # 使用 ref_model 计算参考模型的 log probs (在 no_grad 上下文中)
    with torch.no_grad():
        log_probs_ref_y1 = _get_log_probs(ref_model, tokenizer, prompts, y1s, device)
        log_probs_ref_y2 = _get_log_probs(ref_model, tokenizer, prompts, y2s, device)

    log_ratio_y1 = log_probs_pi_y1 - log_probs_ref_y1
    log_ratio_y2 = log_probs_pi_y2 - log_probs_ref_y2

    with torch.no_grad():
        f_y1 = _get_rewards(reward_model, tokenizer, prompts, y1s, device)
        f_y2 = _get_rewards(reward_model, tokenizer, prompts, y2s, device)

        log_ratio_weight_y1 = log_ratio_y1.detach()
        log_ratio_weight_y2 = log_ratio_y2.detach()

        kl_term1 = beta_kl * log_ratio_weight_y1
        kl_term2 = beta_kl * log_ratio_weight_y2

        weight1 = 1 - f_y2 - kl_term1
        weight2 = f_y1 + kl_term2

    clamped_log_ratio_y1 = torch.clamp(log_ratio_y1, min=log_ratio_clip_min, max=log_ratio_clip_max)
    ratio_y1 = torch.exp(clamped_log_ratio_y1)

    clamped_log_ratio_y2 = torch.clamp(log_ratio_y2, min=log_ratio_clip_min, max=log_ratio_clip_max)
    ratio_y2 = torch.exp(clamped_log_ratio_y2)

    J_sym_objective = ratio_y1 * weight1 - ratio_y2 * weight2
    total_loss = -J_sym_objective.mean()

    # 日志信息
    logs = {
        "loss": total_loss.item(),
        "mean_ratio_chosen": ratio_y1.detach().mean().item(),
        "mean_ratio_rejected": ratio_y2.detach().mean().item(),
        "f_chosen": f_y1.detach().mean().item(),
        "f_rejected": f_y2.detach().mean().item(),
    }
    return total_loss, logs


# ---------------------------------------------------------------------------
# 主训练函数
# ---------------------------------------------------------------------------
def main(args):
    # 1. 初始化 Accelerator
    accelerator = Accelerator(log_with="tensorboard", project_dir=args.output_dir)
    set_seed(args.seed)

    # 在主进程上打印信息
    logger.info(f"Accelerator state: {accelerator.state}", main_process_only=True)
    logger.info(f"Training args: {args}", main_process_only=True)

    # 2. 加载 Tokenizer 和模型
    # 在所有进程上加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 使用 accelerator.main_process_first() 确保主进程先下载模型
    with accelerator.main_process_first():
        policy_model = AutoModelForCausalLM.from_pretrained(
            args.sft_model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.sft_model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            args.reward_model_path,
            num_labels=1,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )

    # 冻结不需要训练的模型参数
    for param in ref_model.parameters():
        param.requires_grad = False
    for param in reward_model.parameters():
        param.requires_grad = False
        
    if reward_model.config.pad_token_id is None:
        reward_model.config.pad_token_id = tokenizer.pad_token_id

    # 3. 加载和预处理数据集
    with accelerator.main_process_first():
        raw_dataset = load_from_disk(args.dataset_train_name)
        processed_dataset = raw_dataset.map(
            preprocess_ultrafeedback,
            fn_kwargs={"tokenizer": tokenizer},
            num_proc=args.num_workers,
            remove_columns=raw_dataset.column_names
        )
        processed_dataset = processed_dataset.filter(
            lambda x: x['prompt'] and x['chosen'] and x['rejected']
        )

    train_dataloader = DataLoader(
        processed_dataset,
        batch_size=args.per_device_train_batch_size,
        collate_fn=get_collate_fn(),
        shuffle=True,
        num_workers=args.num_workers,
    )
    
    # 4. 准备优化器和学习率调度器
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.learning_rate)
    
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(max_train_steps * args.warmup_ratio),
        num_training_steps=max_train_steps,
    )

    # 5. 使用 accelerator.prepare() 准备所有组件
    policy_model, ref_model, reward_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        policy_model, ref_model, reward_model, optimizer, train_dataloader, lr_scheduler
    )

    # 6. 开始训练
    global_step = 0
    logger.info("***** Starting training *****", main_process_only=True)
    for epoch in range(args.num_train_epochs):
        policy_model.train()
        
        progress_bar = tqdm(
            range(max_train_steps // args.num_train_epochs),
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch+1}/{args.num_train_epochs}"
        )
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(policy_model):
                loss, logs = compute_sympo_loss(
                    policy_model=policy_model,
                    ref_model=ref_model,
                    reward_model=reward_model,
                    tokenizer=tokenizer,
                    batch=batch,
                    beta_kl=args.beta_kl,
                    log_ratio_clip_min=args.log_ratio_clip_min,
                    log_ratio_clip_max=args.log_ratio_clip_max,
                    device=accelerator.device,
                )
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(policy_model.parameters(), args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # 日志记录
                if global_step % args.logging_steps == 0:
                    accelerator.log(logs, step=global_step)
                    progress_bar.set_postfix(logs)
                    
                # 保存模型
                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved checkpoint to {save_path}", main_process_only=True)
    
    # 训练结束后保存最终模型
    logger.info("***** Finished training *****", main_process_only=True)
    final_save_path = os.path.join(args.output_dir, "final_model")
    accelerator.save_state(final_save_path)
    logger.info(f"Saved final model to {final_save_path}", main_process_only=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with SymPO using Accelerate")
    
    # 路径参数
    parser.add_argument("--sft_model_path", type=str, default="/train/Llama-3-8B-Instruct", help="Path to the SFT base model.")
    parser.add_argument("--reward_model_path", type=str, default="/train/f_model", help="Path to the reward model.")
    parser.add_argument("--dataset_train_name", type=str, default="/train/preprocessed_traindataset", help="Path to the training dataset.")
    parser.add_argument("--output_dir", type=str, default="/train/output_model_direct/llama3-8b-advan_2e-5_beta_0.2_accelerate", help="Directory to save checkpoints and final model.")
    
    # 训练超参数
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per GPU.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for learning rate scheduler.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    # SymPO 特定参数
    parser.add_argument("--beta_kl", type=float, default=0.2, help="KL divergence penalty coefficient.")
    parser.add_argument("--log_ratio_clip_min", type=float, default=-3.0, help="Minimum clip value for log probability ratio.")
    parser.add_argument("--log_ratio_clip_max", type=float, default=3.0, help="Maximum clip value for log probability ratio.")
    
    # 其他参数
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloading.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log metrics every N steps.")
    parser.add_argument("--save_steps", type=int, default=150, help="Save a checkpoint every N steps.")

    args = parser.parse_args()
    main(args)