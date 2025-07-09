# test_lora.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ==== 配置 ====
modelName="Qwen2-7B-Instruct"
model_id = f"../../models/{modelName}" 
lora_model_path =f"../../loraResult/{modelName}"

# ==== 加载 tokenizer ====
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ==== 加载基础模型 ====
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
)


# ==== 加载 LoRA adapter ====

model = PeftModel.from_pretrained(base_model, lora_model_path, device_map="auto")
model.eval()
# print(model)


# ==== 推理函数 ====
def generate_response(prompt):
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
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "<|im_start|>assistant" in output_text:
        output_text = output_text.split("<|im_start|>assistant")[-1]
    if "<|im_end|>" in output_text:
        output_text = output_text.split("<|im_end|>")[0]
    return output_text.strip()

# ==== 示例推理 ====
prompt = "潘峰是谁"
print(f"输入: {prompt}")
print(f"输出: {generate_response(prompt)}")
