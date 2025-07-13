import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ==== 配置 ====
model_name = "Qwen2-7B-Instruct"
model_path = f"../../models/{model_name}"
lora_path = f"../../loraResult/{model_name}"

# ==== 加载函数 ====
def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.model_max_length = 2048  # 避免超长文本 OOM

    return tokenizer

def load_lora_model():
    config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", trust_remote_code=True, quantization_config=config
    )
    return PeftModel.from_pretrained(base_model, lora_path, device_map="auto").eval()

def generate_response(prompt, model, tokenizer):
    input_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return extract_response(text)

def extract_response(text):
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant")[-1]
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>")[0]
    return text.strip()

# ==== 运行推理 ====
tokenizer = load_tokenizer()
model = load_lora_model()

prompt = "你好，请介绍一下你自己。"
print("输入：", prompt)
print("输出：", generate_response(prompt, model, tokenizer))
