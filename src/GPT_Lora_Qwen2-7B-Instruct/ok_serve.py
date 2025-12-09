# serve.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# ==== 配置 ====
model_name = "Qwen2.5-Coder-7B-Instruct"
model_path = f"../../models/{model_name}"
lora_path = f"../../loraResult/{model_name}"

# ==== 全局變量 ====
tokenizer: Optional[AutoTokenizer] = None
model: Optional[PeftModel] = None
device = "cuda"

# ==== 模型加載函數 ====
def load_model():
    global tokenizer, model
    if model is not None:
        return

    print("正在加載 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("正在加載模型 (4-bit + LoRA)...")
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        quantization_config=config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, lora_path, device_map="auto").eval()
    print("✅ 模型加載完成")

# ==== FastAPI App ====
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

@app.on_event("startup")
def startup():
    load_model()

def extract_response(text: str) -> str:
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant")[-1]
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>")[0]
    return text.strip()

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # 確保模型已加載
    if model is None or tokenizer is None:
        load_model()

    # 轉為 dict 列表
    messages = [msg.dict() for msg in request.messages]

    # ✅ 關鍵：使用官方 chat template（不要手動拼接！）
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  # 自動加上 <|im_start|>assistant
        )
    except Exception as e:
        return {"error": f"Prompt 構造失敗: {str(e)}"}
    print("【SERVICE PROMPT】")
    print(repr(prompt))  # 查看真实字符串，含转义
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

    input_len = inputs.input_ids.shape[1]
    # 限制最大生成長度，避免 OOM（可根據顯存調整）
    max_new_tokens = min(1024, 2048 - input_len)  # 如果要 2048，請確保輸入很短

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    response = extract_response(full_output)
    return {"response": response}

# ==== 啟動 ====
if __name__ == "__main__":
    print("啟動服務：http://localhost:7101/chat")
    uvicorn.run(
        "serve:app",
        host="0.0.0.0",
        port=7101,
        workers=1,
        reload=False,
        loop="asyncio",
        http="h11",
    )