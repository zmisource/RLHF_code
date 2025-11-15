"""
预计算数据集中每个样本的 ref_logp_chosen 和 ref_logp_rejected

这个脚本使用与训练代码完全相同的计算逻辑，确保预计算的值与训练时step 0的pi_logp完全一致。
使用相同的随机种子可以进一步保证可重复性。

使用方法:
    python precompute_ref_logp.py \
        --sft_model_path /root/Llama-3-8B-Instruct \
        --input_dataset_path /path/to/input/dataset \
        --output_dataset_path /path/to/output/dataset \
        --batch_size 16 \
        --seed 42
"""

import os
import random
import numpy as np
import torch
import argparse
from datasets import load_from_disk, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from tqdm import tqdm
from typing import List

def seed_everything(seed=42):
    """设置所有随机数生成器的种子，确保可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def _get_log_probs(model, tokenizer, prompts: List[str], responses: List[str], device: str) -> torch.Tensor:
    """
    计算log_probs，与训练代码中的_get_log_probs函数完全一致
    
    Args:
        model: 参考模型（应在eval模式）
        tokenizer: 分词器
        prompts: prompt列表
        responses: response列表
        device: 计算设备
        
    Returns:
        torch.Tensor: 每个样本的log_probs总和（形状: [batch_size]）
    """
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    
    full_texts = [p + r for p, r in zip(prompts, responses)]
    prompt_tokens = tokenizer(prompts, padding=False, truncation=False)
    prompt_lengths = [len(p) for p in prompt_tokens['input_ids']]
    
    unwrapped_model = model.module if hasattr(model, "module") else model
    max_len = unwrapped_model.config.max_position_embeddings
    
    full_tokens = tokenizer(
        full_texts, padding='longest', truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    input_ids = full_tokens['input_ids'].to(device)
    attention_mask = full_tokens['attention_mask'].to(device)
    
    outputs = model(input_ids, attention_mask=attention_mask, return_dict=True, use_cache=False)
    logits = outputs.logits
    del outputs
    
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = input_ids[..., 1:].contiguous()
    del logits
    
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
    
    tokenizer.padding_side = original_padding_side
    return total_log_probs

def precompute_ref_log_probs_batch(batch: dict, ref_model, tokenizer, device: str) -> dict:
    """
    批量预计算ref_logp_chosen和ref_logp_rejected
    
    Args:
        batch: 包含'prompt', 'chosen', 'rejected'的批次数据
        ref_model: 参考模型（应在eval模式）
        tokenizer: 分词器
        device: 计算设备
        
    Returns:
        添加了'ref_logp_chosen'和'ref_logp_rejected'的批次数据
    """
    prompts = batch["prompt"]
    chosen_responses = batch["chosen"]
    rejected_responses = batch["rejected"]
    
    # 确保ref_model在eval模式
    ref_model.eval()
    
    # 计算chosen和rejected的log_probs
    with torch.no_grad():
        log_probs_chosen = _get_log_probs(ref_model, tokenizer, prompts, chosen_responses, device)
        log_probs_rejected = _get_log_probs(ref_model, tokenizer, prompts, rejected_responses, device)
    
    # 转换为float32并转为列表，确保精度
    batch["ref_logp_chosen"] = log_probs_chosen.cpu().float().numpy().tolist()
    batch["ref_logp_rejected"] = log_probs_rejected.cpu().float().numpy().tolist()
    
    return batch

def parse_args():
    parser = argparse.ArgumentParser(description="预计算数据集中的ref_logp_chosen和ref_logp_rejected")
    
    parser.add_argument("--sft_model_path", type=str, required=True, 
                       help="SFT模型路径（用于作为参考模型）")
    parser.add_argument("--input_dataset_path", type=str, required=True,
                       help="输入数据集路径（应包含prompt, chosen, rejected字段）")
    parser.add_argument("--output_dataset_path", type=str, required=True,
                       help="输出数据集路径（将保存包含ref_logp的数据集）")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="批量处理大小")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子（与训练时保持一致）")
    parser.add_argument("--num_proc", type=int, default=1,
                       help="并行处理的进程数")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 设置随机种子
    print(f"设置随机种子为: {args.seed}")
    seed_everything(args.seed)
    
    # 初始化Accelerator（用于分布式处理）
    accelerator = Accelerator()
    
    # 加载tokenizer
    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    # 加载参考模型
    print("加载参考模型...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # 将模型移动到设备
    device = accelerator.device
    ref_model = ref_model.to(device)
    
    # 加载数据集
    print(f"从 {args.input_dataset_path} 加载数据集...")
    dataset = load_from_disk(args.input_dataset_path)
    
    # 检查必需字段
    required_fields = ["prompt", "chosen", "rejected"]
    missing_fields = [f for f in required_fields if f not in dataset.column_names]
    if missing_fields:
        raise ValueError(f"数据集缺少必需字段: {missing_fields}")
    
    # 检查是否已经包含ref_logp字段
    if "ref_logp_chosen" in dataset.column_names or "ref_logp_rejected" in dataset.column_names:
        print("⚠️  警告: 数据集中已包含ref_logp字段，将被覆盖")
        # 可以选择移除旧字段
        dataset = dataset.remove_columns([col for col in dataset.column_names 
                                         if col.startswith("ref_logp")])
    
    print(f"数据集大小: {len(dataset)}")
    print(f"数据集字段: {dataset.column_names}")
    
    # 批量处理数据集
    print("开始预计算ref_logp...")
    print(f"批量大小: {args.batch_size}, 设备: {device}")
    
    # 使用DataLoader进行批量处理
    from torch.utils.data import DataLoader
    from datasets import Dataset as HFDataset
    
    # 创建DataLoader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # 使用accelerator.prepare包装模型和dataloader
    ref_model, dataloader = accelerator.prepare(ref_model, dataloader)
    
    # 收集所有结果
    all_results = []
    
    print("开始预计算ref_logp...")
    for batch in tqdm(dataloader, desc="预计算ref_logp", disable=not accelerator.is_main_process):
        prompts = batch['prompt']
        chosen_responses = batch['chosen']
        rejected_responses = batch['rejected']
        
        # 计算ref_logp
        log_probs_chosen = _get_log_probs(ref_model, tokenizer, prompts, chosen_responses, device)
        log_probs_rejected = _get_log_probs(ref_model, tokenizer, prompts, rejected_responses, device)
        
        # 转换为列表
        batch_results = []
        for i in range(len(prompts)):
            batch_results.append({
                'ref_logp_chosen': log_probs_chosen[i].cpu().item(),
                'ref_logp_rejected': log_probs_rejected[i].cpu().item(),
            })
        
        # 收集所有进程的结果
        gathered_results = accelerator.gather_for_metrics(batch_results)
        if accelerator.is_main_process:
            all_results.extend(gathered_results)
        
        # 定期清理缓存
        if len(all_results) % (args.batch_size * 10) == 0:
            torch.cuda.empty_cache()
    
    # 在主进程上创建新数据集并保存
    if accelerator.is_main_process:
        print("创建新的数据集...")
        
        # 提取原始数据
        original_data = {
            'prompt': [dataset[i]['prompt'] for i in range(len(dataset))],
            'chosen': [dataset[i]['chosen'] for i in range(len(dataset))],
            'rejected': [dataset[i]['rejected'] for i in range(len(dataset))],
        }
        
        # 添加预计算的ref_logp
        original_data['ref_logp_chosen'] = [result['ref_logp_chosen'] for result in all_results]
        original_data['ref_logp_rejected'] = [result['ref_logp_rejected'] for result in all_results]
        
        # 如果数据集中有其他字段，也保留
        other_fields = {k: v for k, v in dataset[0].items() 
                       if k not in ['prompt', 'chosen', 'rejected']}
        for key in other_fields:
            if key not in ['ref_logp_chosen', 'ref_logp_rejected']:
                original_data[key] = [dataset[i][key] for i in range(len(dataset))]
        
        # 创建新数据集
        processed_dataset = HFDataset.from_dict(original_data)
        
        # 保存处理后的数据集
        print(f"保存处理后的数据集到 {args.output_dataset_path}...")
        processed_dataset.save_to_disk(args.output_dataset_path)
        print("✅ 预计算完成！")
        
        # 打印一些示例值用于验证
        print(f"\n数据集大小: {len(processed_dataset)}")
        print("示例值（前3个样本）:")
        for i in range(min(3, len(processed_dataset))):
            sample = processed_dataset[i]
            print(f"样本 {i}:")
            print(f"  ref_logp_chosen: {sample['ref_logp_chosen']:.6f}")
            print(f"  ref_logp_rejected: {sample['ref_logp_rejected']:.6f}")
    
    accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()

