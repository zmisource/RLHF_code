# # # merge_fsdp_checkpoint.py
# # from accelerate import Accelerator
# # from transformers import AutoModelForCausalLM

# # accelerator = Accelerator()
# # model_name = "/train/output_model/llama3-8b-sympo-default"
# # checkpoint = f"{model_name}/checkpoint-500"
# # merge_model_name = "/train/output_model/consolidate_model"
# # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")

# # # 从 checkpoint 恢复分片参数
# # accelerator.load_state(checkpoint)

# # # 解除 FSDP 包装，得到完整模型
# # model = accelerator.unwrap_model(model)

# # # 保存合并后的模型
# # model.save_pretrained(f"{model_name}/checkpoint-500", safe_serialization=True)

# import os
# from accelerate import Accelerator
# from transformers import AutoConfig, AutoModelForCausalLM
# import torch
# import argparse
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# parser = argparse.ArgumentParser()
# parser.add_argument("--model_dir", type=str, default="/train/output_model/llama3-8b-sympo-default",
#                     help="Directory containing model config and tokenizer (not necessarily weights)")
# parser.add_argument("--checkpoint", type=str, default="/train/output_model/llama3-8b-sympo-default/checkpoint-500",
#                     help="Path to the accelerate/FSDP checkpoint folder (e.g. .../checkpoint-500)")
# parser.add_argument("--out_dir", type=str, default="/train/output_model/consolidate_model/checkpoint-500",
#                     help="Where to save merged model (if omitted, saves under model_dir/merged)")
# args = parser.parse_args()

# model_dir = args.model_dir
# checkpoint_dir = args.checkpoint
# out_dir = args.out_dir 

# logger.info(f"model_dir={model_dir}, checkpoint={checkpoint_dir}, out_dir={out_dir}")

# accelerator = Accelerator()
# # 1) 加载配置并基于 config 构建模型结构（不依赖单个权重文件）
# config = AutoConfig.from_pretrained("/train/Llama-3-8B-Instruct")
# logger.info("Config loaded.")

# # 2) 使用 from_config 构建模型（随机初始化），之后 accelerator.load_state 会用 FSDP 的分片参数覆盖
# model = AutoModelForCausalLM.from_config(config)
# logger.info("Model created from config (random init).")

# # 3) 在多卡下，accelerate 会包装模型，之后 load_state 加载 FSDP checkpoint 到包装后的模型
# model = accelerator.prepare(model)
# logger.info("Model prepared by accelerator.")

# logger.info(f"Loading checkpoint state from {checkpoint_dir} ...")
# accelerator.load_state(checkpoint_dir)
# logger.info("Checkpoint loaded into model.")

# # 4) unwrap 并保存合并后的完整模型（仅主进程保存）
# unwrapped = accelerator.unwrap_model(model)
# if accelerator.is_main_process:
#     logger.info("Saving merged model ...")
#     # safe_serialization=True 使用 safetensors 或安全方式保存，减小出错几率
#     unwrapped.save_pretrained(out_dir, safe_serialization=True)
#     logger.info(f"Merged model saved to {out_dir}")

# logger.info("Done.")
# import socket
# import httpcore

# # --- 开始：强制使用 IPv4 ---
# # 这个代码块强制网络后端首选 IPv4，以解决在某些系统上的 "Address family not supported by protocol" 错误。
# # 它通过“猴子补丁”的方式修改了 httpcore 库的默认连接行为。

# _orig_connect_tcp = httpcore._backends.sync.SyncStream.connect_tcp

# def _force_ipv4_connect_tcp(self, *args, **kwargs):
#     """强制TCP连接使用IPv4"""
#     kwargs["local_address"] = "0.0.0.0"
#     return _orig_connect_tcp(self, *args, **kwargs)

# httpcore._backends.sync.SyncStream.connect_tcp = _force_ipv4_connect_tcp

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
import os
import shutil

# --- 1. 配置参数 ---
base_model_path = "/train/Llama-3-8B-Instruct" # <--- ‼️ 请务必修改为你的基础模型
fsdp_checkpoint_path = "/train/output_model/llama3-8b-sympo-5e-5/checkpoint-40" # <--- ‼️ 请修改为你的 checkpoint 路径
consolidated_model_path = "/train/output_model/llama3-8b-sympo-5e-5_merged/checkpoint-40" # <--- ‼️ 你可以自定义保存路径

# --- 2. 初始化Accelerator和模型 ---
# 初始化Accelerator，它会自动处理设备放置
accelerator = Accelerator()

print("从基础模型路径加载模型结构...")
# 首先，我们需要一个模型的“骨架”，所以从预训练路径加载配置
# device_map="auto"在这里有助于在CPU上加载，避免占用GPU
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16, # 根据您的训练设置调整
    device_map="cpu", # 在CPU上执行合并以节省显存
)

print(f"模型结构加载完毕。总参数量: {model.num_parameters() / 1e9:.2f}B")

# --- 3. 加载FSDP分片权重 ---
# print(f"正在从FSDP检查点加载分片权重: {fsdp_checkpoint_path}")

# # Accelerator的load_state会自动处理FSDP的分片加载逻辑
# # 注意：这里我们只关心模型权重，所以先将模型传递给prepare
# # 即使我们在CPU上，也需要这个步骤来匹配FSDP的状态字典结构
# model = accelerator.prepare(model)
# accelerator.load_state(fsdp_checkpoint_path)
# print("FSDP权重加载成功。")
# --- 3. 加载FSDP分片权重 ---
print(f"正在从FSDP检查点加载分片权重: {fsdp_checkpoint_path}")

# 找到包含权重的真实子目录名 (例如 'pytorch_model_fsdp_0')
model_name = None
for f in os.listdir(fsdp_checkpoint_path):
    if f.startswith("pytorch_model_fsdp_"):
        model_name = f
        break

if model_name is None:
    raise FileNotFoundError(f"在 '{fsdp_checkpoint_path}' 中找不到 'pytorch_model_fsdp_*' 目录")

print(f"检测到模型权重目录为: {model_name}")

# 在 load_state 中明确指定模型权重的子目录
# 使用 `strict=False` 是因为我们只关心加载模型权重，而不关心优化器等状态
accelerator.load_state(fsdp_checkpoint_path, model_name=model_name, strict=False)

print("FSDP权重加载成功。")
# --- 4. 提取并保存完整模型 ---
print("正在提取完整的模型...")
# 使用unwrap_model来获取底层的Hugging Face模型
unwrapped_model = accelerator.unwrap_model(model)

print(f"正在将合并后的完整模型保存到: {consolidated_model_path}")
# 使用save_pretrained保存完整的模型权重、配置文件等
# safe_serialization=True 会保存为更安全、更快的.safetensors格式
unwrapped_model.save_pretrained(
    consolidated_model_path,
    safe_serialization=True
)
print("完整模型保存成功！")

# --- 5. 保存Tokenizer ---
print("正在保存对应的Tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(consolidated_model_path)
    print(f"Tokenizer已成功保存到: {consolidated_model_path}")
except Exception as e:
    print(f"警告：无法自动保存Tokenizer。请从 '{base_model_path}' 手动复制。错误: {e}")

print("\n合并流程完成！")


from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from accelerate import load_checkpoint_in_model
import torch, os

base_model = "meta-llama/Llama-3-8b"           # your base arch (must match training)
ckpt_dir   = "path/to/checkpoint-500"          # folder with pytorch_model_fsdp_0, optimizer_0, etc.
out_dir    = "path/to/consolidated-model"

# 1) Build an equivalent, UNWRAPPED model (CPU is fine)
cfg   = AutoConfig.from_pretrained(base_model)
model = AutoModelForCausalLM.from_config(cfg)
model.to("cpu")  # stays on CPU while loading

# 2) Load the FSDP sharded weights into this model
#    If your checkpoint has a subfolder for the model shards, specify it.
#    Depending on Accelerate version, the arg can be `subfolder` or `model_name`.
load_checkpoint_in_model(
    model,
    checkpoint=ckpt_dir,
    # one of these depending on your layout/version:
    # subfolder="pytorch_model_fsdp_0",
    # model_name="pytorch_model_fsdp_0",
    strict=False,                # helpful if heads/tied weights differ slightly
)

# (Optional) Tie weights if your arch expects it
if hasattr(model, "tie_weights"):
    model.tie_weights()

# 3) Save a single-file consolidated HF checkpoint
os.makedirs(out_dir, exist_ok=True)
model.save_pretrained(out_dir, safe_serialization=True)   # writes model.safetensors + config.json

# 4) Also save tokenizer so from_pretrained works out-of-the-box later
try:
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    tok.save_pretrained(out_dir)
except Exception:
    pass