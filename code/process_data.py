import argparse
import random
import numpy as np
import torch
from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
import os

def seed_everything(seed=2003):
    """Set random seed for reproducibility across all libraries."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = True,
    label_pad_token_id: int = -100,
    is_encoder_decoder: bool = False,
) -> torch.FloatTensor:
    """
    Compute the log probabilities of the given labels under the given logits.
    This function is adapted from SimPOTrainer.get_batch_logps.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. 
                Label tokens with a value of label_pad_token_id are ignored. 
                Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. 
                         Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
        label_pad_token_id: The label pad token id.
        is_encoder_decoder: Whether the model is an encoder-decoder model.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities 
        of the given labels under the given logits.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

    if not is_encoder_decoder:
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
    
    loss_mask = labels != label_pad_token_id

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == label_pad_token_id] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)

def _get_log_probs(model, tokenizer, prompts, responses, device, average_log_prob=True, label_pad_token_id=-100):
    """
    Compute log probabilities for prompts + responses using the get_batch_logps format.
    Adapted to match SimPOTrainer's get_batch_logps implementation.

    Args:
        model: The model to compute log probabilities from.
        tokenizer: Tokenizer for text processing.
        prompts: List of prompt strings.
        responses: List of response strings.
        device: Device to run computation on.
        average_log_prob: If True, return average log prob per token. Otherwise, return sum.
        label_pad_token_id: Token ID to use for padding labels (default -100 to ignore prompt tokens).

    Returns:
        List of log probabilities (one per prompt-response pair).
    """
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    
    # Tokenize prompts separately to get prompt lengths
    prompt_tokens = tokenizer(prompts, padding=False, truncation=False)
    prompt_lengths = [len(p) for p in prompt_tokens['input_ids']]
    
    # Get model max length (handle wrapped models)
    unwrapped_model = model.module if hasattr(model, "module") else model
    max_len = unwrapped_model.config.max_position_embeddings
    
    # Tokenize full texts (prompt + response)
    full_texts = [p + r for p, r in zip(prompts, responses)]
    full_tokens = tokenizer(
        full_texts, 
        padding='longest', 
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    
    input_ids = full_tokens['input_ids'].to(device)
    attention_mask = full_tokens['attention_mask'].to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, use_cache=False)
        logits = outputs.logits

    # Create labels: set prompt tokens to label_pad_token_id to exclude them from log prob calculation
    labels = input_ids.clone()
    for i, prompt_len in enumerate(prompt_lengths):
        labels[i, :prompt_len] = label_pad_token_id

    # Compute log probabilities using get_batch_logps format
    log_probs = get_batch_logps(
        logits=logits,
        labels=labels,
        average_log_prob=average_log_prob,
        label_pad_token_id=label_pad_token_id,
        is_encoder_decoder=False,
    )
    
    tokenizer.padding_side = original_padding_side
    return log_probs.cpu().tolist()

# CHANGED: Added 'device' argument and explicit .to(device) call
def _get_rewards(reward_model, tokenizer, prompts, responses, device):
    texts = [p + r for p, r in zip(prompts, responses)]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=reward_model.config.max_position_embeddings)
    # Move tensors to the correct device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = reward_model(**inputs)
        return torch.sigmoid(outputs.logits.squeeze(-1)).cpu().tolist()

def preprocess_ultrafeedback(example: dict, tokenizer: AutoTokenizer) -> dict:
    chosen_response = example['chosen'][-1]['content'] + "<|eot_id|>\n"
    rejected_response = example['rejected'][-1]['content'] + "<|eot_id|>\n"
    conversation_history = example['chosen'][:-1]
    full_prompt = tokenizer.apply_chat_template(
        conversation_history, tokenize=False, add_generation_prompt=True
    )
    if not full_prompt or not chosen_response or not rejected_response:
        return {"prompt": "", "chosen": "", "rejected": ""}
    return {"prompt": full_prompt, "chosen": chosen_response, "rejected": rejected_response}

def main(args):
    # Set random seed for reproducibility
    if args.seed is not None:
        seed_everything(args.seed)
        print(f"Random seed set to: {args.seed}")
    
    accelerator = Accelerator()

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ref_model = AutoModelForCausalLM.from_pretrained(args.sft_model_path, dtype=torch.bfloat16)
    reward_model = AutoModelForSequenceClassification.from_pretrained(args.reward_model_path, num_labels=1, dtype=torch.bfloat16)
    if reward_model.config.pad_token_id is None:
        reward_model.config.pad_token_id = tokenizer.pad_token_id

    with accelerator.main_process_first():
        original_dataset = load_from_disk(args.dataset_path)
        print("Preprocessing raw dataset to extract prompts and responses...")
        original_dataset = original_dataset.map(
            preprocess_ultrafeedback,
            fn_kwargs={"tokenizer": tokenizer},
            num_proc=4,
            remove_columns=original_dataset.column_names
        )
        original_dataset = original_dataset.filter(
            lambda x: x['prompt'] and x['chosen'] and x['rejected']
        )

    dataloader = DataLoader(original_dataset, batch_size=args.batch_size)

    ref_model, reward_model, dataloader = accelerator.prepare(ref_model, reward_model, dataloader)

    all_results = []
    for batch in tqdm(dataloader, disable=not accelerator.is_main_process):
        prompts = batch['prompt']
        chosen_responses = batch['chosen']
        rejected_responses = batch['rejected']

        # CHANGED: Pass accelerator.device to the helper functions
        # Use get_batch_logps format with average_log_prob option
        ref_logp_chosen = _get_log_probs(
            ref_model, tokenizer, prompts, chosen_responses, 
            accelerator.device, 
            average_log_prob=args.average_log_prob
        )
        ref_logp_rejected = _get_log_probs(
            ref_model, tokenizer, prompts, rejected_responses, 
            accelerator.device,
            average_log_prob=args.average_log_prob
        )
        reward_chosen = _get_rewards(reward_model, tokenizer, prompts, chosen_responses, accelerator.device)
        reward_rejected = _get_rewards(reward_model, tokenizer, prompts, rejected_responses, accelerator.device)
        
        batch_results = []
        for i in range(len(prompts)):
            batch_results.append({
                'prompt': prompts[i], 'chosen': chosen_responses[i], 'rejected': rejected_responses[i],
                'ref_logp_chosen': ref_logp_chosen[i], 'ref_logp_rejected': ref_logp_rejected[i],
                'reward_chosen': reward_chosen[i], 'reward_rejected': reward_rejected[i],
            })
        
        gathered_results = accelerator.gather_for_metrics(batch_results)
        if accelerator.is_main_process:
            all_results.extend(gathered_results)

    if accelerator.is_main_process:
        with accelerator.main_process_first():
            final_data_dict = {
                'prompt': [], 'chosen': [], 'rejected': [],
                'ref_logp_chosen': [], 'ref_logp_rejected': [],
                'reward_chosen': [], 'reward_rejected': []
            }
            seen_prompts = set()
            for item in all_results:
                # Using a combination of prompt and chosen to define uniqueness
                unique_id = item['prompt'] + "|||" + item['chosen']
                if unique_id not in seen_prompts:
                    for key in final_data_dict.keys():
                        final_data_dict[key].append(item[key])
                    seen_prompts.add(unique_id)

            preprocessed_dataset = Dataset.from_dict(final_data_dict)
            print(f"Total processed examples: {len(preprocessed_dataset)}")
            print(f"Saving new dataset to {args.output_path}")
            preprocessed_dataset.save_to_disk(args.output_path)
            print("Preprocessing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-GPU preprocessing script for 4xA100 GPUs.")
    parser.add_argument("--sft_model_path", type=str, required=True, help="Path to SFT reference model")
    parser.add_argument("--reward_model_path", type=str, required=True, help="Path to reward model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to input dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save processed dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="Per-device batch size. For 4xA100, recommend 8-16")
    parser.add_argument("--seed", type=int, default=2003, help="Random seed for reproducibility")
    parser.add_argument("--no-average_log_prob", dest="average_log_prob", action="store_false",
                       help="Return sum of log probabilities instead of average (default: average).")
    parser.set_defaults(average_log_prob=True)
    args = parser.parse_args()
    main(args)