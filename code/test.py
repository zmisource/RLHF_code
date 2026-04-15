import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.is_bf16_supported())

# import os
# from huggingface_hub import snapshot_download

# # 1. 强制设置镜像环境变量
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# # 2. 设置 Token 和 模型ID
# token = "REMOVEDsbovewyBnAMeJNMBKNpkjClCQomtyWvqHO"  # 替换你的 Token
# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# # 3. 下载模型
# snapshot_download(
#     repo_id=model_id,
#     local_dir="/train/Meta-Llama-3-8B-Instruct",
#     token=token,
#     resume_download=True  # 支持断点续传
# )

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 指定你刚才下载的本地路径
model_path = "/train/Meta-Llama-3-8B-Instruct"

print("正在加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("正在加载 Model (这可能需要一两分钟)...")
# 如果显存够 (>=16GB)，可以用 float16 加载；如果显存只有 12G-16G，建议加 load_in_8bit=True
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

print("加载成功！")

# 简单测试
input_text = "Who are you?"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))