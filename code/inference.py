import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# --- 1. 配置 ---
# 这是上一步合并后模型的保存路径
model_path = "/train/output_model/consolidate_model/checkpoint-3753"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"正在从路径加载模型和Tokenizer: {model_path}")
print(f"使用的设备: {device}")

# --- 2. 加载模型和Tokenizer ---
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 加载模型
# torch_dtype="auto" 会自动选择最佳精度
# device_map="auto" 会自动将模型分配到可用的GPU上
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)

# 如果没有设置device_map="auto"，则需要手动移动模型到设备
# model.to(device)

print("模型和Tokenizer加载成功！")

# --- 3. 使用Pipeline进行简单推理 (推荐) ---
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

prompt = "I'm planning a trip to Southeast Asia. What are the top 3 must-see destinations?"

# Llama 3需要特定的聊天模板格式
messages = [
    {"role": "system", "content": "You are a helpful travel assistant."},
    {"role": "user", "content": prompt},
]

# apply_chat_template 会自动格式化输入
prompt_formatted = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


print("\n--- 推理开始 ---")
outputs = pipe(
    prompt_formatted,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)

print(f"提示 (格式化后): \n{prompt_formatted}")
print("\n模型输出: \n")
print(outputs[0]["generated_text"])
print("\n--- 推理结束 ---")


# --- 4. 手动进行推理 (更灵活) ---
# input_ids = tokenizer(prompt_formatted, return_tensors="pt").to(device)
#
# with torch.no_grad():
#     outputs = model.generate(
#         **input_ids,
#         max_new_tokens=256,
#         do_sample=True,
#         temperature=0.7,
#         top_p=0.9
#     )
#
# response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print("\n--- 手动推理输出 ---")
# print(response_text)