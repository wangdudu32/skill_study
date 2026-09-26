import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 当前目录就是模型目录
model_path = "../Qwen3-8B-AWQ"

print("正在加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("正在加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
)

print("模型加载完成")
print("模型设备:", model.device)
print("GPU:", torch.cuda.get_device_name(0))


# =========================
# 输入问题
# =========================

messages = [
    {
        "role": "user",
        "content": "请简单介绍一下你自己。"
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)

inputs = tokenizer(
    text,
    return_tensors="pt"
).to(model.device)


# =========================
# 模型推理
# =========================

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256
    )


# 只获取模型新生成的部分
generated_ids = outputs[0][inputs["input_ids"].shape[1]:]

response = tokenizer.decode(
    generated_ids,
    skip_special_tokens=True
)

print("\n========== 模型回答 ==========")
print(response)
