import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ==== 配置 ====
model_name = "Qwen2-7B-Instruct"
model_path = f"../../models/{model_name}"
lora_path = f"../../loraResult/{model_name}"

# ==== 工具函数 ====
def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    return tokenizer

def load_lora_model():
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
    return PeftModel.from_pretrained(base_model, lora_path, device_map="auto").eval()

def format_history(messages):
    formatted = ""
    for msg in messages:
        role, content = msg["role"], msg["content"]
        formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return formatted + "<|im_start|>assistant\n"

def extract_response(text):
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant")[-1]
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>")[0]
    return text.strip()

def generate_response(history, model, tokenizer):
    prompt = format_history(history)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return extract_response(decoded)

# ==== 多轮对话 ====
def chat():
    tokenizer = load_tokenizer()
    model = load_lora_model()
    history = []

    print("🤖 开始对话（输入 'exit' 退出）")
    while True:
        user_input = input("你：").strip()
        if user_input.lower() in ["exit", "quit", "q"]:
            print("👋 再见")
            break

        history.append({"role": "user", "content": user_input})
        assistant_reply = generate_response(history, model, tokenizer)
        print("助手：", assistant_reply)
        history.append({"role": "assistant", "content": assistant_reply})

# ==== 启动 ====
if __name__ == "__main__":
    chat()
