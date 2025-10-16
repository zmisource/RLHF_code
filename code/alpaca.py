import os
import json
import argparse
import pandas as pd
from pathlib import Path
from importlib.metadata import version

import torch
from huggingface_hub import login
from oumi.datasets.evaluation import AlpacaEvalDataset, utils
from oumi.inference import VLLMInferenceEngine
from oumi.core.configs import GenerationParams, InferenceConfig, ModelParams
from alpaca_eval import evaluate

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="在服务器上对模型进行AlpacaEval推理和评估")
    parser.add_argument("--model_path", type=str, default="/train/output_model/consolidate_model/checkpoint-500", help="要评估的合并后模型的路径")
    parser.add_argument("--model_display_name", type=str, default="checkpoint-500", help="在排行榜中显示的模型名称")
    parser.add_argument("--output_dir", type=str, default="/train/output_model/alpaca", help="保存评估结果和配置文件的目录")
    parser.add_argument("--num_examples", type=int, default=805, help="用于评估的样本数量 (默认: 805, 即完整数据集)")
    parser.add_argument("--max_new_tokens", type=int, default=8192, help="模型生成时允许的最大token数")
    parser.add_argument("--temperature", type=float, default=0.9, help="生成时的温度参数")
    parser.add_argument("--annotators_config", type=str, default="weighted_alpaca_eval_gpt4_turbo", help="AlpacaEval使用的裁判模型配置")

    return parser.parse_args()

def main():
    args = parse_args()

    # --- 1. 准备工作：登录和创建目录 ---
    print("--- 步骤 1: 准备工作 ---")
    # 检查环境变量是否存在
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("请设置 OPENAI_API_KEY 环境变量！")

    # 创建输出目录
    eval_output_dir = Path(args.output_dir) / args.model_display_name
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"结果将保存在: {eval_output_dir}")

    # --- 2. 加载 AlpacaEval 数据集 ---
    print("\n--- 步骤 2: 加载 AlpacaEval 数据集 ---")
    alpaca_dataset = AlpacaEvalDataset(dataset_name="tatsu-lab/alpaca_eval").conversations()
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
    
    # 初始化推理引擎
    inference_engine = VLLMInferenceEngine(model_params)
    
    # 开始推理
    print("开始生成模型回复...")
    responses = inference_engine.infer(alpaca_dataset, inference_config)
    print("模型回复生成完毕。")

    # --- 4. 格式化输出并进行评估 ---
    print("\n--- 步骤 4: 格式化输出并调用 AlpacaEval 进行评估 ---")
    responses_json = utils.conversations_to_alpaca_format(responses)
    responses_df = pd.DataFrame(responses_json)
    responses_df["generator"] = args.model_display_name

    print("正在使用裁判模型进行评估，这可能需要一些时间并消耗OpenAI API额度...")
    df_leaderboard, _ = evaluate(
        model_outputs=responses_df,
        annotators_config=args.annotators_config,
        is_return_instead_of_print=True,
        output_path=str(eval_output_dir),
    )

    # --- 5. 显示并保存结果 ---
    print("\n--- 步骤 5: 显示并保存评估结果 ---")
    metrics = df_leaderboard.loc[args.model_display_name]
    print(f"模型 `{args.model_display_name}` 的评估指标:")
    for metric, value in metrics.items():
        print(f" - {metric} = {value}")

    # 保存完整的配置文件
    evaluation_config_dict = {
        "packages": {
            "alpaca_eval": version("alpaca_eval"),
            "oumi": version("oumi"),
        },
        "configs": {
            "inference_config": str(inference_config),
            "annotators_config": args.annotators_config,
        },
        "eval_metrics": metrics.to_dict(),
    }
    
    config_filename = f"evaluation_config_{args.model_display_name}.json"
    config_filepath = Path(args.output_dir) / config_filename

    print(f"\n正在将配置文件保存到: {config_filepath}")
    with open(config_filepath, "w") as output_file:
        output_file.write(json.dumps(evaluation_config_dict, indent=2))
    
    print("\n评估流程全部完成！")


if __name__ == "__main__":
    main()