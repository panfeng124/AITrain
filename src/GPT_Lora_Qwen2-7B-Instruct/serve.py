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
# model_name = "Qwen2.5-Coder-7B-Instruct"
model_path = f"../../models/{model_name}"
lora_path = f"../../loraResult/{model_name}"

print("加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
tokenizer.model_max_length = 32768

print("加载 base model + LoRA...")
config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=False,
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
    messages = compress_context(messages)

    prompt = format_messages_to_prompt(messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 根据 prompt token 剩余空间设置 max_new_tokens
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]
    MAX_TOTAL_TOKENS = tokenizer.model_max_length  # 32768 for Qwen3-4B
    MAX_NEW = min(2048, MAX_TOTAL_TOKENS - prompt_len)


    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    new_tokens = outputs[0][input_len:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return {"response": response_text.strip()}

# ==== 上下文摘要压缩 ====
def compress_context(messages, max_rounds: int = 5):
    """
    保留最近 max_rounds 轮问答（user + assistant 为一轮），其余历史丢弃。
    system message 保留。
    """
    system_msgs = [m for m in messages if m["role"] == "system"]
    dialog_msgs = [m for m in messages if m["role"] != "system"]

    # 分割成轮次对话（每轮包括 user 和可能的 assistant）
    rounds = []
    temp = []
    for m in dialog_msgs:
        temp.append(m)
        if m["role"] == "assistant":
            rounds.append(temp)
            temp = []
    if temp:  # 如果最后一轮还没 assistant
        rounds.append(temp)

    # 仅保留最近 max_rounds 轮
    retained = rounds[-max_rounds:]
    flat_retained = [m for round_ in retained for m in round_]

    # 拼接最终上下文
    context = system_msgs + flat_retained

    print(f"【DEBUG】保留 system 消息数: {len(system_msgs)}")
    print(f"【DEBUG】保留最近轮数: {len(retained)}")
    for i, m in enumerate(context):
        print(f"\n[{i+1}] {m['role']}:\n{m['content']}\n{'-'*50}")

    return context


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
    print("启动服务：http://localhost:7101/chat")
    uvicorn.run("serve:app", host="0.0.0.0", port=7101, reload=False)
