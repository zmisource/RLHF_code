#!/usr/bin/env python3
"""
将 FSDP SHARDED_STATE_DICT 格式的 checkpoint 转换为 HuggingFace 格式

使用方法:
    python convert_fsdp_to_hf.py \
        --base_model_path /path/to/base/model \
        --checkpoint_path /path/to/checkpoint-500 \
        --output_path /path/to/output/model

或者使用 accelerate 运行（如果 checkpoint 需要多 GPU 环境）:
    accelerate launch convert_fsdp_to_hf.py \
        --base_model_path /path/to/base/model \
        --checkpoint_path /path/to/checkpoint-500 \
        --output_path /path/to/output/model
"""

import argparse
import os
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator


def convert_fsdp_to_hf(
    base_model_path: str,
    checkpoint_path: str,
    output_path: str,
    dtype: str = "bfloat16",
    model_name_in_checkpoint: str = None,
):
    """
    将 FSDP 分片 checkpoint 转换为 HuggingFace 格式
    
    Args:
        base_model_path: 基础模型路径（用于获取 config 和 tokenizer）
        checkpoint_path: FSDP checkpoint 目录路径（包含 pytorch_model_fsdp_0 等子目录）
        output_path: 输出 HuggingFace 格式模型的保存路径
        dtype: 模型数据类型，可选 "bfloat16", "float16", "float32"
        model_name_in_checkpoint: checkpoint 中模型权重子目录名（如 "pytorch_model_fsdp_0"），
                                   如果为 None 则自动检测
    """
    print("=" * 60)
    print("FSDP Checkpoint → HuggingFace 格式转换工具")
    print("=" * 60)
    
    # 1. 确定数据类型
    torch_dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype not in torch_dtype_map:
        raise ValueError(f"不支持的 dtype: {dtype}，请选择 {list(torch_dtype_map.keys())}")
    torch_dtype = torch_dtype_map[dtype]
    
    # 2. 从基础模型加载配置
    print(f"\n步骤 1/5: 从基础模型加载配置...")
    print(f"  基础模型路径: {base_model_path}")
    config = AutoConfig.from_pretrained(base_model_path)
    print("  ✅ 配置加载完成")
    
    # 3. 初始化 Accelerator（单进程模式，用于加载 FSDP checkpoint）
    print(f"\n步骤 2/5: 初始化 Accelerator...")
    accelerator = Accelerator()
    print("  ✅ Accelerator 初始化完成")
    
    # 4. 从基础模型加载完整模型结构（在CPU上，避免显存占用）
    # 注意：使用 from_pretrained 而不是 from_config，因为 FSDP 需要完整的模型结构
    print(f"\n步骤 3/5: 从基础模型加载模型结构...")
    print(f"  ⚠️  注意: 此步骤会从基础模型加载权重，但后续会被 FSDP 权重覆盖")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map="cpu",  # 在 CPU 上加载，节省显存
    )
    print(f"  ✅ 模型结构加载完成，参数量: {model.num_parameters() / 1e9:.2f}B")
    
    # 5. 使用 Accelerator 包装模型（这是 FSDP 加载所必需的）
    print(f"\n步骤 3.5/5: 使用 Accelerator 包装模型...")
    model = accelerator.prepare(model)
    print("  ✅ 模型包装完成")
    
    # 6. 自动检测或使用指定的模型权重子目录
    if model_name_in_checkpoint is None:
        print(f"\n步骤 4/5: 检测 checkpoint 中的模型权重目录...")
        print(f"  Checkpoint 路径: {checkpoint_path}")
        
        # 查找 pytorch_model_fsdp_* 目录
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint 路径不存在: {checkpoint_path}")
        
        model_name_in_checkpoint = None
        for item in os.listdir(checkpoint_path):
            if item.startswith("pytorch_model_fsdp_"):
                model_name_in_checkpoint = item
                break
        
        if model_name_in_checkpoint is None:
            raise FileNotFoundError(
                f"在 '{checkpoint_path}' 中找不到 'pytorch_model_fsdp_*' 目录。\n"
                f"请检查 checkpoint 路径是否正确，或使用 --model_name_in_checkpoint 参数手动指定。"
            )
        print(f"  ✅ 检测到模型权重目录: {model_name_in_checkpoint}")
    else:
        print(f"\n步骤 4/5: 使用指定的模型权重目录: {model_name_in_checkpoint}")
    
    # 7. 加载 FSDP 分片权重
    print(f"\n步骤 4.5/5: 加载 FSDP 分片权重...")
    print(f"  ⚠️  注意: 此步骤可能需要几分钟时间，请耐心等待...")
    
    # 使用 accelerator.load_state 加载 FSDP 检查点
    # 这是最可靠的方法，兼容不同版本的 accelerate
    try:
        # 尝试使用 model_name 参数（较新版本）
        accelerator.load_state(
            checkpoint_path,
            model_name=model_name_in_checkpoint,
            strict=False,  # 只加载模型权重，忽略优化器等
        )
    except TypeError:
        # 如果 model_name 参数不存在，尝试不使用该参数
        try:
            accelerator.load_state(
                checkpoint_path,
                strict=False,
            )
            print("  ⚠️  注意: 使用了不带 model_name 的加载方式，如果失败请检查 checkpoint 结构")
        except Exception as e:
            raise RuntimeError(
                f"加载 checkpoint 失败: {e}\n"
                f"请检查 accelerate 版本，或尝试使用 Accelerator 的 load_state 方法。\n"
                f"错误详情: {type(e).__name__}: {str(e)}"
            )
    
    print("  ✅ FSDP 权重加载完成")
    
    # 8. 解包模型，获取底层的 HuggingFace 模型
    print(f"\n步骤 4.6/5: 解包模型...")
    model = accelerator.unwrap_model(model)
    print("  ✅ 模型解包完成")
    
    # 9. 绑定权重（对于 LLaMA 等模型很重要）
    if hasattr(model, "tie_weights"):
        print("\n步骤 4.7/5: 绑定权重...")
        model.tie_weights()
        print("  ✅ 权重绑定完成")
    
    # 10. 保存为 HuggingFace 格式
    print(f"\n步骤 5/5: 保存为 HuggingFace 格式...")
    print(f"  输出路径: {output_path}")
    print(f"  ⚠️  注意: 此步骤可能需要几分钟时间，取决于模型大小...")
    
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)
    print("  ✅ 模型权重保存完成")
    
    # 11. 保存 tokenizer
    print(f"\n额外步骤: 保存 tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        tokenizer.save_pretrained(output_path)
        print("  ✅ Tokenizer 保存完成")
    except Exception as e:
        print(f"  ⚠️  警告: 无法自动保存 tokenizer: {e}")
        print(f"  请手动从 {base_model_path} 复制 tokenizer 文件到 {output_path}")
    
    # 完成
    print("\n" + "=" * 60)
    print("🎉 转换完成！")
    print("=" * 60)
    print(f"\n合并后的 HuggingFace 模型已保存到: {output_path}")
    print("\n使用方法:")
    print(f"  from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{output_path}')")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{output_path}')")
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="将 FSDP SHARDED_STATE_DICT 格式的 checkpoint 转换为 HuggingFace 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法
  python convert_fsdp_to_hf.py \\
      --base_model_path /path/to/Llama-3-8B-Instruct \\
      --checkpoint_path /path/to/checkpoint-500 \\
      --output_path /path/to/merged-model

  # 指定模型权重子目录名
  python convert_fsdp_to_hf.py \\
      --base_model_path /path/to/Llama-3-8B-Instruct \\
      --checkpoint_path /path/to/checkpoint-500 \\
      --output_path /path/to/merged-model \\
      --model_name_in_checkpoint pytorch_model_fsdp_0

  # 使用 accelerate 运行（如果需要多 GPU 环境）
  accelerate launch convert_fsdp_to_hf.py \\
      --base_model_path /path/to/Llama-3-8B-Instruct \\
      --checkpoint_path /path/to/checkpoint-500 \\
      --output_path /path/to/merged-model
        """
    )
    
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="基础模型路径（用于获取 config.json 和 tokenizer，例如: /path/to/Llama-3-8B-Instruct）"
    )
    
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="FSDP checkpoint 目录路径（包含 pytorch_model_fsdp_0 等子目录，例如: /path/to/checkpoint-500）"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="输出 HuggingFace 格式模型的保存路径"
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="模型数据类型（默认: bfloat16）"
    )
    
    parser.add_argument(
        "--model_name_in_checkpoint",
        type=str,
        default=None,
        help="checkpoint 中模型权重子目录名（如 'pytorch_model_fsdp_0'），如果为 None 则自动检测"
    )
    
    args = parser.parse_args()
    
    convert_fsdp_to_hf(
        base_model_path=args.base_model_path,
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        dtype=args.dtype,
        model_name_in_checkpoint=args.model_name_in_checkpoint,
    )


if __name__ == "__main__":
    main()

