import os
import json
import argparse
import pandas as pd
from pathlib import Path
from importlib.metadata import version

import torch
# 移除了 alpaca_eval 和 login，因为推理阶段不需要
from oumi.datasets.evaluation import AlpacaEvalDataset, utils
from oumi.inference import VLLMInferenceEngine
from oumi.core.configs import GenerationParams, InferenceConfig, ModelParams

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="在[无网络]服务器上对模型进行AlpacaEval推理")
    parser.add_argument("--model_path", type=str, default="/train/output_model/llama3-8b-sympo-1e-6_no_term2_merged/checkpoint-300", help="要评估的合并后模型的路径")
    parser.add_argument("--model_display_name", type=str, default="advan-1e-6_no_term2-checkpoint-3753", help="在排行榜中显示的模型名称")
    # 修改：输出目录现在只用于存放临时的JSON文件
    parser.add_argument("--output_dir", type=str, default="/train/output_model/alpaca_outputs", help="保存模型输出(JSON)的目录")
    parser.add_argument("--num_examples", type=int, default=805, help="用于评估的样本数量 (默认: 805, 即完整数据集)")
    parser.add_argument("--max_new_tokens", type=int, default=8192, help="模型生成时允许的最大token数")
    parser.add_argument("--temperature", type=float, default=0.9, help="生成时的温度参数")
    # 移除了 --annotators_config，推理阶段不需要
    return parser.parse_args()

def main():
    args = parse_args()

    # --- 1. 准备工作：创建目录 ---
    print("--- 步骤 1: 准备工作 ---")
    # 移除了 OPENAI_API_KEY 检查
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"模型输出JSON将保存在: {output_dir}")

    # --- 2. 加载 AlpacaEval 数据集 ---
    print("\n--- 步骤 2: 加载 AlpacaEval 数据集 ---")
    # 注意：这里假设 oumi 或 huggingface datasets 已经缓存了数据集。
    # 如果没有，您需要先在有网的机器上下载 "tatsu-lab/alpaca_eval" 数据集 (eval.json)，
    # 然后上传到服务器，并修改 AlpacaEvalDataset 让它从本地文件加载。
    try:
        alpaca_dataset = AlpacaEvalDataset(dataset_name="tatsu-lab/alpaca_eval").conversations()
    except Exception as e:
        print(f"加载数据集失败: {e}")
        print("请确保 'tatsu-lab/alpaca_eval' 数据集已在本地缓存，")
        print("或者修改代码以从本地JSON文件加载 (e.g., /path/to/eval.json)。")
        return

    if args.num_examples < len(alpaca_dataset):
        alpaca_dataset = alpaca_dataset[:args.num_examples]
    print(f"已加载 {len(alpaca_dataset)} 条评估样本。")

    # --- 3. 配置并运行 VLLM 推理 ---
    print("\n--- 步骤 3: 配置并运行 VLLM 推理 ---")
    generation_params = GenerationParams(max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    model_params = ModelParams(
        model_name=args.model_path,
        model_max_length=args.max_new_tokens,
        tokenizer_kwargs={"pad_token": "<|end_of_text|>"}
    )
    inference_config = InferenceConfig(model=model_params, generation=generation_params)
    
    inference_engine = VLLMInferenceEngine(model_params)
    
    print("开始生成模型回复...")
    responses = inference_engine.infer(alpaca_dataset, inference_config)
    print("模型回复生成完毕。")

    # --- 4. 格式化输出并保存到文件 ---
    print("\n--- 步骤 4: 格式化输出并保存为JSON ---")
    responses_json = utils.conversations_to_alpaca_format(responses)
    responses_df = pd.DataFrame(responses_json)
    responses_df["generator"] = args.model_display_name

    # 定义输出文件路径
    output_json_path = output_dir / f"model_outputs_{args.model_display_name}.json"
    
    # 保存为JSON文件，供评估脚本使用
    responses_df.to_json(output_json_path, orient='records', indent=2)
    print(f"模型输出已保存到: {output_json_path}")

    # --- 5. 保存推理配置 ---
    print("\n--- 步骤 5: 保存推理配置 ---")
    inference_config_dict = {
        "packages": {
            "oumi": version("oumi"),
        },
        "configs": {
            "inference_config": str(inference_config),
        },
        "model_display_name": args.model_display_name,
    }
    
    config_filename = f"inference_config_{args.model_display_name}.json"
    config_filepath = output_dir / config_filename

    print(f"正在将推理配置文件保存到: {config_filepath}")
    with open(config_filepath, "w") as output_file:
        output_file.write(json.dumps(inference_config_dict, indent=2))
    
    print("\n推理流程完成！请将JSON文件拷贝到有网络的服务器进行评估。")

if __name__ == "__main__":
    main()