# serve.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# ==== 模型加载 ====
model_name = "Qwen3-4B"
model_path = f"../../models/{model_name}"
lora_path = f"../../loraResult/{model_name}"

print("加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
tokenizer.model_max_length = 1024

print("加载 base model + LoRA...")
config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=True,
)
base_model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", trust_remote_code=True, quantization_config=config
)
model = PeftModel.from_pretrained(base_model, lora_path, device_map="auto").eval()

# ==== FastAPI ====
app = FastAPI()
# 允许所有源，所有方法，所有头（不推荐生产环境使用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    messages = [m.dict() for m in request.messages]
    messages = compress_context(messages, tokenizer, model, max_tokens=12000)
    prompt = format_messages_to_prompt(messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    result = extract_response(response_text)
    return {"response": result}

# ==== 上下文摘要压缩 ====
def compress_context(messages, tokenizer, model, max_tokens=12000):
    if len(messages) <= 6:
        return messages
    head = [m for m in messages if m['role'] == 'system'][:1]
    body = [m for m in messages if m['role'] != 'system']
    recent = body[-4:]
    history = body[:-4]

    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])
    summary_prompt = f"<|im_start|>user\n请总结以下对话的核心信息：\n{history_text}\n<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(summary_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.5,
            top_p=0.8,
            do_sample=False,
        )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    summary = summary.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
    return head + [{"role": "system", "content": f"对话摘要：{summary}"}] + recent

def format_messages_to_prompt(messages):
    return "\n".join([
        f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages
    ]) + "\n<|im_start|>assistant\n"

def extract_response(text):
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant")[-1]
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>")[0]
    return text.strip()

# ==== 启动服务 ====
if __name__ == "__main__":
    print("启动服务：http://localhost:8000/chat")
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=False)
