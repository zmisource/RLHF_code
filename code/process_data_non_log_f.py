import argparse
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer
import os

# 不再需要 torch, DataLoader, Accelerator, reward model 等

def preprocess_ultrafeedback(example: dict, tokenizer: AutoTokenizer) -> dict:
    # 此函数保持不变
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
    # 只需要 tokenizer 来应用模板
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)

    # 加载、映射、过滤、保存
    print("加载原始数据集...")
    original_dataset = load_from_disk(args.dataset_path).select(range(1000))

    print("应用模板，转换为 prompt/chosen/rejected 格式...")
    text_dataset = original_dataset.map(
        preprocess_ultrafeedback,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=os.cpu_count(),
        remove_columns=original_dataset.column_names
    )
    
    print("过滤空数据...")
    text_dataset = text_dataset.filter(
        lambda x: x['prompt'] and x['chosen'] and x['rejected']
    )
    
    # 注意：可以增加一个去重步骤，但如果源数据是干净的，则非必须
    
    print(f"Total processed examples: {len(text_dataset)}")
    print(f"Saving new text-only dataset to {args.output_path}")
    text_dataset.save_to_disk(args.output_path)
    print("Text-only preprocessing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text-only preprocessing script.")
    # 只需要 sft_model_path 来加载 tokenizer
    parser.add_argument("--sft_model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args)

