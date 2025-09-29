import argparse
import torch
from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
import os

# CHANGED: Added 'device' argument and explicit .to(device) calls
def _get_log_probs(model, tokenizer, prompts, responses, device):
    full_texts = [p + r for p, r in zip(prompts, responses)]
    
    tokenizer.padding_side = 'left'
    prompt_tokens = tokenizer(prompts, padding=False, truncation=False)
    prompt_lengths = [len(p) for p in prompt_tokens['input_ids']]

    max_len = model.config.max_position_embeddings
    full_tokens = tokenizer(
        full_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
    )
    # Move tensors to the correct device
    input_ids = full_tokens['input_ids'].to(device)
    attention_mask = full_tokens['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
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
    
    tokenizer.padding_side = 'right'
    return total_log_probs.cpu().tolist()

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
        ref_logp_chosen = _get_log_probs(ref_model, tokenizer, prompts, chosen_responses, accelerator.device)
        ref_logp_rejected = _get_log_probs(ref_model, tokenizer, prompts, rejected_responses, accelerator.device)
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
    parser = argparse.ArgumentParser(description="Multi-GPU preprocessing script.")
    parser.add_argument("--sft_model_path", type=str, required=True)
    parser.add_argument("--reward_model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=__name__)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8, help="Per-device batch size.")
    args = parser.parse_args()
    main(args)