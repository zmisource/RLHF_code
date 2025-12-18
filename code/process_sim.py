import argparse
from datasets import load_from_disk
from transformers import AutoTokenizer
import os

def preprocess_ultrafeedback(example: dict, tokenizer: AutoTokenizer) -> dict:
    """
    提取文本并保留 f_chosen 和 f_rejected 字段。
    """
    try:
        # 1. 处理文本逻辑（保持不变）
        chosen_response = example['chosen'][-1]['content'] + "<|eot_id|>\n"
        rejected_response = example['rejected'][-1]['content'] + "<|eot_id|>\n"
        conversation_history = example['chosen'][:-1]
        
        full_prompt = tokenizer.apply_chat_template(
            conversation_history, tokenize=False, add_generation_prompt=True
        )
        
        # 2. 提取需要保留的原始字段
        # 使用 .get 以防万一字段不存在，或者直接用 example['f_chosen']
        f_chosen = example.get('f_chosen', None)
        f_rejected = example.get('f_rejected', None)

        # 3. 数据有效性检查 (只检查文本是否为空)
        if not full_prompt or not chosen_response or not rejected_response:
            # 标记为空，稍后 filter 会过滤掉
            return {
                "prompt": "", "chosen": "", "rejected": "", 
                "f_chosen": f_chosen, "f_rejected": f_rejected
            }
            
        # 4. 返回包含所有目标列的字典
        return {
            "prompt": full_prompt,
            "chosen": chosen_response,
            "rejected": rejected_response,
            "f_chosen": f_chosen,   # <--- 新增
            "f_rejected": f_rejected # <--- 新增
        }
    except Exception as e:
        return {"prompt": "", "chosen": "", "rejected": "", "f_chosen": None, "f_rejected": None}

def main(args):
    print(f"Loading tokenizer from: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading dataset from: {args.dataset_path}")
    original_dataset = load_from_disk(args.dataset_path)
    
    print(f"Preprocessing dataset with {args.num_proc} processes...")
    
    # map 会根据返回的字典 keys 自动建立新的 schema
    processed_dataset = original_dataset.map(
        preprocess_ultrafeedback,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=args.num_proc,
        remove_columns=original_dataset.column_names, # 依然移除旧列，确保输出干净，只包含上面 return 的 5 列
        desc="Formatting Data & Keeping Scores"
    )

    print("Filtering invalid entries...")
    # 过滤掉 prompt/chosen/rejected 为空的行
    final_dataset = processed_dataset.filter(
        lambda x: x['prompt'] and x['chosen'] and x['rejected']
    )

    print(f"Original size: {len(original_dataset)} -> Processed size: {len(final_dataset)}")
    
    # (可选) 打印一条数据看看结构
    if len(final_dataset) > 0:
        print("Sample example:", final_dataset[0].keys()) 
        # 预期输出: dict_keys(['prompt', 'chosen', 'rejected', 'f_chosen', 'f_rejected'])

    print(f"Saving to {args.output_path}...")
    final_dataset.save_to_disk(args.output_path)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing Script (Keeping f_chosen/f_rejected)")
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_proc", type=int, default=8)
    
    args = parser.parse_args()
    main(args)