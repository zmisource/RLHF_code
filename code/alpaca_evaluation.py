import os
import json
import argparse
import pandas as pd
from pathlib import Path
from importlib.metadata import version
from alpaca_eval import evaluate

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="在[有网络]服务器上运行AlpacaEval评估")
    parser.add_argument("--input_json_path", type=str, required=True, help="由 run_inference.py 生成的模型输出JSON文件路径")
    parser.add_argument("--model_display_name", type=str, required=True, help="在排行榜中显示的模型名称 (必须与JSON中的 'generator' 字段匹配)")
    parser.add_argument("--output_dir", type=str, default="./alpaca_eval_results", help="保存最终评估结果和配置文件的目录")
    parser.add_argument("--annotators_config", type=str, default="weighted_alpaca_eval_gpt4_turbo", help="AlpacaEval使用的裁判模型配置")
    
    return parser.parse_args()

def main():
    args = parse_args()

    # --- 1. 准备工作：检查API Key和创建目录 ---
    print("--- 步骤 1: 准备工作 ---")
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("请设置 OPENAI_API_KEY 环境变量！")

    eval_output_dir = Path(args.output_dir) / args.model_display_name
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"评估结果将保存在: {eval_output_dir}")

    # --- 2. 加载模型输出文件 ---
    print(f"\n--- 步骤 2: 加载模型输出 {args.input_json_path} ---")
    try:
        responses_df = pd.read_json(args.input_json_path, orient='records')
    except Exception as e:
        print(f"加载JSON文件失败: {e}")
        print("请确保文件路径正确，且格式为 'records' (JSON对象列表)。")
        return
    
    # 确保 'generator' 字段与 display_name 匹配
    if "generator" not in responses_df.columns or responses_df["generator"].iloc[0] != args.model_display_name:
        print(f"警告: JSON中的 'generator' 字段 ('{responses_df.get('generator', ['N/A'])[0]}') ")
        print(f"与 --model_display_name ('{args.model_display_name}') 不匹配。")
        print(f"将强制设置为: {args.model_display_name}")
        responses_df["generator"] = args.model_display_name
        
    print(f"已加载 {len(responses_df)} 条模型回复。")

    # --- 3. 调用 AlpacaEval 进行评估 ---
    print("\n--- 步骤 3: 调用 AlpacaEval 进行评估 ---")
    print("正在使用裁判模型进行评估，这可能需要一些时间并消耗OpenAI API额度...")
    
    df_leaderboard, _ = evaluate(
        model_outputs=responses_df,
        annotators_config=args.annotators_config,
        is_return_instead_of_print=True,
        output_path=str(eval_output_dir),
    )

    # --- 4. 显示并保存结果 ---
    print("\n--- 步骤 4: 显示并保存评估结果 ---")
    
    # 检查模型是否在排行榜中
    if args.model_display_name not in df_leaderboard.index:
        print(f"错误：模型 '{args.model_display_name}' 未出现在评估结果中。")
        print("请检查评估过程中是否有错误。排行榜内容：")
        print(df_leaderboard)
        return

    metrics = df_leaderboard.loc[args.model_display_name]
    print(f"模型 `{args.model_display_name}` 的评估指标:")
    for metric, value in metrics.items():
        print(f" - {metric} = {value}")

    # 保存评估配置文件
    evaluation_config_dict = {
        "packages": {
            "alpaca_eval": version("alpaca_eval"),
        },
        "configs": {
            "annotators_config": args.annotators_config,
        },
        "eval_metrics": metrics.to_dict(),
    }
    
    config_filename = f"evaluation_config_{args.model_display_name}.json"
    config_filepath = Path(args.output_dir) / config_filename # 注意：保存在顶层output_dir

    print(f"\n正在将评估配置文件保存到: {config_filepath}")
    with open(config_filepath, "w") as output_file:
        output_file.write(json.dumps(evaluation_config_dict, indent=2))
    
    print("\n评估流程全部完成！")

if __name__ == "__main__":
    main()