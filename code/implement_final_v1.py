import argparse
import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helper function to get log probabilities for the policy model
# ---------------------------------------------------------------------------
def _get_log_probs(model, tokenizer, prompts, responses, device):
    """
    Computes the log probabilities of responses given prompts for the policy model.
    """
    full_texts = [p + r for p, r in zip(prompts, responses)]
    
    tokenizer.padding_side = 'left'
    prompt_tokens = tokenizer(prompts, padding=False, truncation=False)
    prompt_lengths = [len(p) for p in prompt_tokens['input_ids']]

    max_len = model.config.max_position_embeddings
    full_tokens = tokenizer(
        full_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
    )
    input_ids = full_tokens['input_ids'].to(device)
    attention_mask = full_tokens['attention_mask'].to(device)
    outputs = model(input_ids, attention_mask=attention_mask, return_dict=True)
    logits = outputs.logits

    shifted_logits = logits[..., :-1, :]
    shifted_labels = input_ids[..., 1:]
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    nll_per_token = loss_fct(shifted_logits.reshape(-1, shifted_logits.size(-1)), shifted_labels.reshape(-1))
    log_probs_per_token = -nll_per_token.view(input_ids.size(0), -1)

    seq_len = shifted_labels.size(1)
    position_ids = torch.arange(seq_len, device=device).expand_as(shifted_labels)
    prompt_lengths_tensor = torch.tensor(prompt_lengths, device=device).unsqueeze(1)
    response_start_index = prompt_lengths_tensor - 1
    mask = position_ids >= response_start_index
    attention_mask_shifted = attention_mask[:, 1:].to(torch.bool)
    response_mask = mask & attention_mask_shifted

    masked_log_probs = log_probs_per_token * response_mask
    total_log_probs = masked_log_probs.sum(dim=1)
    
    tokenizer.padding_side = 'right' # Reset to default
    return total_log_probs

# ---------------------------------------------------------------------------
# Custom Data Collator
# ---------------------------------------------------------------------------
@dataclass
class DataCollatorForPrecomputedSymPO:
    """
    Collates batches of data, separating text from pre-computed numerical values.
    """
    def __call__(self, features):
        batch = {}
        text_keys = ['prompt', 'chosen', 'rejected']
        numerical_keys = ['ref_logp_chosen', 'ref_logp_rejected', 'reward_chosen', 'reward_rejected']
        
        for key in text_keys:
            batch[key] = [feature[key] for feature in features]
        
        for key in numerical_keys:
            batch[key] = torch.tensor([feature[key] for feature in features], dtype=torch.float32)
            
        return batch

# ---------------------------------------------------------------------------
# Loss Computation
# ---------------------------------------------------------------------------
def compute_sympo_loss_precomputed(
    policy_model, tokenizer, batch, beta_kl, log_ratio_clip_min, log_ratio_clip_max, device
):
    """
    Computes the SymPO loss using pre-computed values for ref and reward models.
    """
    prompts = batch["prompt"]
    y1s, y2s = batch["chosen"], batch["rejected"]
    
    # Load precomputed values from the batch and move to device
    log_probs_ref_y1 = batch["ref_logp_chosen"].to(device)
    log_probs_ref_y2 = batch["ref_logp_rejected"].to(device)
    f_y1 = batch["reward_chosen"].to(device)
    f_y2 = batch["reward_rejected"].to(device)

    # Only policy model log probabilities are computed on the fly
    log_probs_pi_y1 = _get_log_probs(policy_model, tokenizer, prompts, y1s, device)
    log_probs_pi_y2 = _get_log_probs(policy_model, tokenizer, prompts, y2s, device)

    log_ratio_y1 = log_probs_pi_y1 - log_probs_ref_y1
    log_ratio_y2 = log_probs_pi_y2 - log_probs_ref_y2
    
    with torch.no_grad():
        kl_term1 = beta_kl * log_ratio_y1.detach()
        kl_term2 = beta_kl * log_ratio_y2.detach()
        weight1 = 1 - f_y2 - kl_term1
        weight2 = f_y1 + kl_term2
    
    clamped_log_ratio_y1 = torch.clamp(log_ratio_y1, min=log_ratio_clip_min, max=log_ratio_clip_max)
    ratio_y1 = torch.exp(clamped_log_ratio_y1)
    
    clamped_log_ratio_y2 = torch.clamp(log_ratio_y2, min=log_ratio_clip_min, max=log_ratio_clip_max)
    ratio_y2 = torch.exp(clamped_log_ratio_y2)
    
    J_sym_objective = ratio_y1 * weight1 - ratio_y2 * weight2
    total_loss = -J_sym_objective.mean()

    # Log metrics
    logs = {
        "loss": total_loss.item(),
        "mean_ratio_chosen": ratio_y1.detach().mean().item(),
        "mean_ratio_rejected": ratio_y2.detach().mean().item(),
        "weight_chosen": weight1.detach().mean().item(),
        "weight_rejected": weight2.detach().mean().item(),
    }
    return total_loss, logs

# ---------------------------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------------------------
def main(args):
    # 1. Initialize Accelerator
    accelerator = Accelerator(log_with="tensorboard", project_dir=args.output_dir)
    set_seed(args.seed)

    logger.info(f"Accelerator state: {accelerator.state}", main_process_only=True)
    logger.info(f"Training args: {args}", main_process_only=True)

    # 2. Load Tokenizer and the Policy Model ONLY
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with accelerator.main_process_first():
        policy_model = AutoModelForCausalLM.from_pretrained(
            args.sft_model_path,
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16,
        )

    # 3. Load the NEW preprocessed dataset
    with accelerator.main_process_first():
        processed_dataset = load_from_disk(args.dataset_train_name)
    
    data_collator = DataCollatorForPrecomputedSymPO()
    train_dataloader = DataLoader(
        processed_dataset,
        batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator,
        shuffle=True,
    )
    
    # 4. Prepare optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.learning_rate)
    
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(max_train_steps * args.warmup_ratio),
        num_training_steps=max_train_steps,
    )

    # 5. Use accelerator.prepare() on the simplified set of objects
    policy_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        policy_model, optimizer, train_dataloader, lr_scheduler
    )

    # 6. Start training
    global_step = 0
    logger.info("***** Starting training with precomputed values *****", main_process_only=True)
    for epoch in range(args.num_train_epochs):
        policy_model.train()
        
        progress_bar = tqdm(
            range(max_train_steps // args.num_train_epochs),
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch+1}/{args.num_train_epochs}"
        )
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(policy_model):
                loss, logs = compute_sympo_loss_precomputed(
                    policy_model=policy_model,
                    tokenizer=tokenizer,
                    batch=batch,
                    beta_kl=args.beta_kl,
                    log_ratio_clip_min=args.log_ratio_clip_min,
                    log_ratio_clip_max=args.log_ratio_clip_max,
                    device=accelerator.device,
                )
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    total_norm = 0
                    for p in policy_model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_grad_norm = total_norm ** 0.5
                    logs["grad_norm"] = total_grad_norm
                    accelerator.clip_grad_norm_(policy_model.parameters(), args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # Log metrics
                if global_step % args.logging_steps == 0:
                    accelerator.log(logs, step=global_step)
                    progress_bar.set_postfix(logs)
                    
                # Save model checkpoint
                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved checkpoint to {save_path}", main_process_only=True)
    
    # Save final model
    logger.info("***** Finished training *****", main_process_only=True)
    final_save_path = os.path.join(args.output_dir, "final_model")
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(policy_model)
    unwrapped_model.save_pretrained(
        final_save_path,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(policy_model),
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(final_save_path)
    logger.info(f"Saved final model to {final_save_path}", main_process_only=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with SymPO using Accelerate and precomputed values")
    
    # Path arguments
    parser.add_argument("--sft_model_path", type=str, required=True, help="Path to the SFT base model (policy model).")
    parser.add_argument("--dataset_train_name", type=str, required=True, help="Path to the NEW precomputed training dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints and final model.")
    
    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per GPU.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for learning rate scheduler.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    # SymPO-specific arguments
    parser.add_argument("--beta_kl", type=float, default=0.1, help="KL divergence penalty coefficient.")
    parser.add_argument("--log_ratio_clip_min", type=float, default=-3.0, help="Minimum clip value for log probability ratio.")
    parser.add_argument("--log_ratio_clip_max", type=float, default=3.0, help="Maximum clip value for log probability ratio.")
    
    # Other arguments
    parser.add_argument("--logging_steps", type=int, default=10, help="Log metrics every N steps.")
    parser.add_argument("--save_steps", type=int, default=150, help="Save a checkpoint every N steps.")

    args = parser.parse_args()
    main(args)