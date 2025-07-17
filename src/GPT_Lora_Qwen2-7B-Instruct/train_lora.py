import torch, json
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments,
    DataCollatorForLanguageModeling, Trainer
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# ==== 配置 ====
# model_name = "Qwen2-7B-Instruct"
# model_name = "Qwen2.5-Coder-7B-Instruct"
# model_name = "Qwen2.5-Coder-3B-Instruct"
model_name = "Qwen3-4B"
model_path = f"../../models/{model_name}"
output_dir = f"../../loraResult/{model_name}"
data_path = "../../trainData/merged_2025_07_15_19_11.jsonl"

# ==== 工具函数 ====
def load_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.model_max_length = 32768  # 避免超长文本 OOM
    return tokenizer

def format_chatml(example):
    text = []
    for msg in example["messages"]:
        text.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
    return {"text": "\n".join(text)}

def load_dataset(path, tokenizer):
    with open(path, "r", encoding="utf-8") as f:
        raw_data = [json.loads(line) for line in f if line.strip()]
    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(format_chatml)
    dataset = dataset.map(lambda ex: tokenizer(ex["text"], truncation=True, padding="max_length", max_length=512), batched=True)
    dataset = dataset.remove_columns(["messages", "text"])
    dataset.set_format("torch")
    return dataset

def prepare_model(path):
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=False,  # 尽量关闭，除非你显存不够
    )
    model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", trust_remote_code=True, quantization_config=config)
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,                        # LoRA 的秩，一般 8 或 16
        lora_alpha=16,              # 通常为 r 的 2 倍
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # 适配 Qwen 架构
    )
    return get_peft_model(model, lora_cfg)

def print_trainable_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔧 Trainable params: {trainable} / {total} ({trainable / total:.2%})")

# ==== 主流程 ====
tokenizer = load_tokenizer(model_path)
dataset = load_dataset(data_path, tokenizer)
model = prepare_model(model_path)
print_trainable_parameters(model)

args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,                        # 建议训练多轮，提升学习效果
    per_device_train_batch_size=1,             # 4070S 显存可支撑 batch size 2~6，建议从4起试验
    gradient_accumulation_steps=4,             # 累积梯度扩大有效 batch size（如总 batch = 4x4 = 16）
    learning_rate=1e-4,                        # 3e-4 对大模型偏高，建议尝试 2e-4 更稳
    lr_scheduler_type="cosine",                # 学习率调度：cosine 收敛更平滑
    warmup_ratio=0.03,                         # 用 warmup_ratio 替代 warmup_steps，适配不同步数
    logging_steps=10,
    save_strategy="epoch",
    max_grad_norm=1.0,
    bf16=False,                                # 4070 不支持 BF16
    fp16=True,                                 # 开启 FP16 更省显存（推荐）
    report_to="none"
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

trainer.train()
model.save_pretrained(output_dir)
print("LoRA 微调完成，模型已保存到:", output_dir)
