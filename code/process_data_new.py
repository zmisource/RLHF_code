import argparse
import torch
from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
import os
import json
# CHANGED: Added 'device' argument and explicit .to(device) calls
def _get_log_probs(model, tokenizer, prompts, responses, device):
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'

    # 核心逻辑
    full_texts = [p + r for p, r in zip(prompts, responses)]
    prompt_tokens = tokenizer(prompts, padding=False, truncation=False)
    prompt_lengths = [len(p) for p in prompt_tokens['input_ids']]
    
    # 优点2 (来自方法二): 模型解包
    unwrapped_model = model.module if hasattr(model, "module") else model
    max_len = unwrapped_model.config.max_position_embeddings
    
    full_tokens = tokenizer(
        full_texts, padding='longest', truncation=True,
        max_length=max_len, return_tensors="pt"
    )
    input_ids = full_tokens['input_ids'].to(device)
    attention_mask = full_tokens['attention_mask'].to(device)
    
    # 优点3 (来自方法二): 关闭缓存
    outputs = model(input_ids, attention_mask=attention_mask, return_dict=True, use_cache=False)
    logits = outputs.logits

    # full_texts = [p + r for p, r in zip(prompts, responses)]
    
    # tokenizer.padding_side = 'left'
    # prompt_tokens = tokenizer(prompts, padding=False, truncation=False)
    # prompt_lengths = [len(p) for p in prompt_tokens['input_ids']]

    # max_len = model.config.max_position_embeddings
    # full_tokens = tokenizer(
    #     full_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
    # )
    # # Move tensors to the correct device
    # input_ids = full_tokens['input_ids'].to(device)
    # attention_mask = full_tokens['attention_mask'].to(device)

    # with torch.no_grad():
    #     outputs = model(input_ids, attention_mask=attention_mask)
    #     logits = outputs.logits

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
    
    # tokenizer.padding_side = 'right'
    tokenizer.padding_side = original_padding_side

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

    # --- 优化点 1: 为每个进程创建一个唯一的临时文件名 ---
    # 这样每个GPU都会写入自己的文件，互不干扰
    temp_output_path = f"{args.output_path}_temp_{accelerator.process_index}.jsonl"

    # 如果临时文件已存在，先删除，避免重复写入
    if accelerator.is_main_process:
        # 确保目录存在
        output_dir = os.path.dirname(args.output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # 清理可能存在的旧临时文件
        for i in range(accelerator.num_processes):
            old_temp_file = f"{args.output_path}_temp_{i}.jsonl"
            if os.path.exists(old_temp_file):
                os.remove(old_temp_file)
    
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ref_model = AutoModelForCausalLM.from_pretrained(args.sft_model_path, attn_implementation="flash_attention_2", dtype=torch.bfloat16)
    reward_model = AutoModelForSequenceClassification.from_pretrained(args.reward_model_path, num_labels=1, attn_implementation="flash_attention_2", dtype=torch.bfloat16)
    # with accelerator.main_process_first():
    #     ref_model = AutoModelForCausalLM.from_pretrained(args.sft_model_path, ...)
    #     reward_model = AutoModelForSequenceClassification.from_pretrained(args.reward_model_path, ...)

    # # prepare 会处理好模型的设备放置
    # ref_model, reward_model, dataloader = accelerator.prepare(ref_model, reward_model, dataloader)
    if reward_model.config.pad_token_id is None:
        reward_model.config.pad_token_id = tokenizer.pad_token_id

    with accelerator.main_process_first():
        original_dataset = load_from_disk(args.dataset_path).select(range(1000)) # 您可以按需添加 .select(range(300))
        print("Preprocessing raw dataset to extract prompts and responses...")
        original_dataset = original_dataset.map(
            preprocess_ultrafeedback,
            fn_kwargs={"tokenizer": tokenizer},
            num_proc=os.cpu_count(), # 使用所有可用的CPU核心进行map操作
            remove_columns=original_dataset.column_names,
            load_from_cache_file=False
        )
        original_dataset = original_dataset.filter(
            lambda x: x['prompt'] and x['chosen'] and x['rejected']
        )

    dataloader = DataLoader(original_dataset, batch_size=args.batch_size)

    ref_model, reward_model, dataloader = accelerator.prepare(ref_model, reward_model, dataloader)

    # --- 优化点 2: 在循环中，将结果直接写入文件，而不是收集到内存列表 ---
    with open(temp_output_path, "w", encoding="utf-8") as f:
        for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
            prompts = batch['prompt']
            chosen_responses = batch['chosen']
            rejected_responses = batch['rejected']

            ref_logp_chosen = _get_log_probs(ref_model, tokenizer, prompts, chosen_responses, accelerator.device)
            ref_logp_rejected = _get_log_probs(ref_model, tokenizer, prompts, rejected_responses, accelerator.device)
            reward_chosen = _get_rewards(reward_model, tokenizer, prompts, chosen_responses, accelerator.device)
            reward_rejected = _get_rewards(reward_model, tokenizer, prompts, rejected_responses, accelerator.device)
            
            for i in range(len(prompts)):
                record = {
                    'prompt': prompts[i], 'chosen': chosen_responses[i], 'rejected': rejected_responses[i],
                    'ref_logp_chosen': ref_logp_chosen[i], 'ref_logp_rejected': ref_logp_rejected[i],
                    'reward_chosen': reward_chosen[i], 'reward_rejected': reward_rejected[i],
                }
                f.write(json.dumps(record) + "\n")

    # --- 优化点 3: 等待所有进程完成文件写入 ---
    accelerator.wait_for_everyone()

    # --- 优化点 4: 仅在主进程上进行文件合并、去重和保存 ---
    if accelerator.is_main_process:
        print("Main process is now consolidating results...")
        final_data_dict = {
            'prompt': [], 'chosen': [], 'rejected': [],
            'ref_logp_chosen': [], 'ref_logp_rejected': [],
            'reward_chosen': [], 'reward_rejected': []
        }
        seen_prompts = set()

        for i in range(accelerator.num_processes):
            temp_file = f"{args.output_path}_temp_{i}.jsonl"
            with open(temp_file, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    unique_id = item['prompt'] + "|||" + item['chosen']
                    if unique_id not in seen_prompts:
                        for key in final_data_dict.keys():
                            final_data_dict[key].append(item[key])
                        seen_prompts.add(unique_id)
            # 删除已处理的临时文件
            os.remove(temp_file)

        preprocessed_dataset = Dataset.from_dict(final_data_dict)
        print(f"Total processed examples after deduplication: {len(preprocessed_dataset)}")
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