import torch, json
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments,
    DataCollatorForLanguageModeling, Trainer
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# ==== 配置 ====
model_name = "Qwen2-7B-Instruct"
model_path = f"../../models/{model_name}"
output_dir = f"../../loraResult/{model_name}"
data_path = "../../trainData/data.jsonl"

# ==== 工具函数 ====
def load_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
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
        llm_int8_enable_fp32_cpu_offload=True,
    )
    model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", trust_remote_code=True, quantization_config=config)
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=8, lora_alpha=32, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
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
    num_train_epochs=1,
    per_device_train_batch_size=1,
    learning_rate=3e-4,
    logging_steps=1,
    save_strategy="epoch",
    warmup_steps=0,
    max_grad_norm=1.0,
    fp16=False,
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
print("✅ LoRA 微调完成，模型已保存到:", output_dir)
