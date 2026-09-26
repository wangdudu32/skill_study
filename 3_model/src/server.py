import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "../Qwen3-8B-AWQ"

app = FastAPI()

print("正在加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("正在加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
)

model.eval()

print("模型加载完成")
print("GPU:", torch.cuda.get_device_name(0))


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def chat(request: ChatRequest):

    messages = [
        {
            "role": "user",
            "content": request.message
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )

    return {
        "response": response
    }
