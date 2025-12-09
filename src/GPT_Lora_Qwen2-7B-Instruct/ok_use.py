from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

model_name = "Qwen2.5-Coder-7B-Instruct"
model_path = f"../../models/{model_name}"
lora_path = f"../../loraResult/{model_name}"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    model_path, quantization_config=config, device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(model, lora_path, device_map="auto").eval()

# === 構造和服務完全相同的 messages ===
messages = [
    {"role": "system", "content": "代碼生成助手"},
    {"role": "user", "content": "使用LAB語言，幫我寫一個信息展示組件"}
]

# 使用 apply_chat_template（和服務一樣）
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("【CORRECT SCRIPT PROMPT】")
print(repr(prompt))

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)

def extract_response(text):
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant")[-1]
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>")[0]
    return text.strip()

response = extract_response(full_output)
print("\n輸出：", response)