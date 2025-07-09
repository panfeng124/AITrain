# train_lora.py

import torch
import json
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments,
    DataCollatorForLanguageModeling, Trainer
)
from datasets import Dataset
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig, get_peft_model
)

# ==== 配置 ====
modelName="Qwen2-7B-Instruct"
model_id = f"../../models/{modelName}" 
output_dir =f"../../loraResult/{modelName}"
data_path = "../../trainData/data.jsonl"

# ==== 加载 tokenizer ====
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ==== 数据预处理 ====
def format_chatml(example):
    formatted = ""
    for msg in example["messages"]:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    return {"text": formatted}

def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    dataset = Dataset.from_list(data)
    dataset = dataset.map(format_chatml)
    dataset = dataset.map(lambda ex: tokenizer(ex["text"], truncation=True, padding="max_length", max_length=512), batched=True)
    dataset = dataset.remove_columns(["messages", "text"])
    dataset.set_format("torch")
    return dataset

train_dataset = load_dataset(data_path)

# ==== 加载量化模型 ====
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
)

# ==== LoRA 配置 ====
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    #target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    target_modules=["q_proj", "v_proj"]  # 先只微调 q/v，节省显存
)

model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.4f}%"
    )

def mark_lora_as_trainable(model):
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True


            # 强制标记 LoRA 参数为可训练
mark_lora_as_trainable(model)

# 打印验证
print_trainable_parameters(model)

# ==== 训练参数 ====
training_args = TrainingArguments(
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

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
)

trainer.train()
model.save_pretrained(output_dir)
print("✅ LoRA 微调完成，模型保存在:", output_dir)
