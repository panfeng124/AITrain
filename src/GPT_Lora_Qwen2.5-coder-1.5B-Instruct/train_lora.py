import torch, json
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments,
    DataCollatorForLanguageModeling, Trainer
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel,   # ✅ 补上这一行
)

# ==== 配置 ====
# model_name = "Qwen2-7B-Instruct"
# model_name = "Qwen2.5-Coder-7B-Instruct"
# model_name = "Qwen2.5-Coder-3B-Instruct"
model_name = "Qwen2.5-Coder-1.5B-Instruct"
model_path = f"../../models/{model_name}"
output_dir = f"../../loraResult/{model_name}"
data_path = "../../trainData/data.jsonl"
#data_path = "../../trainData/data3.jsonl"

# ==== 工具函数 ====
def load_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.model_max_length = 32768  # 避免超长文本 OOM
    return tokenizer

def format_with_chat_template(example, tokenizer):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=True  # ⭐ 关键
    )
    return {"text": text}

def load_dataset(path, tokenizer):
    with open(path, "r", encoding="utf-8") as f:
        raw_data = [json.loads(line) for line in f if line.strip()]

    dataset = Dataset.from_list(raw_data)

    dataset = dataset.map(
        lambda ex: format_with_chat_template(ex, tokenizer),
        remove_columns=["messages"],
        num_proc=4
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=3072
        )

    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        num_proc=4
    )

    dataset.set_format("torch")
    return dataset

def prepare_model(path, resume_lora=False, lora_path=None):
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=False,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=config
    )

    base_model = prepare_model_for_kbit_training(base_model)
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    if resume_lora:
        assert lora_path is not None, "resume_lora=True 时必须提供 lora_path"
        print(f"🔁 从已有 LoRA 加载继续训练: {lora_path}")
        model = PeftModel.from_pretrained(
            base_model,
            lora_path,
            is_trainable=True,   # ⚠️ 关键：否则不会继续更新
        )
    else:
        print("🆕 创建新的 LoRA")
        lora_cfg = LoraConfig(
            task_type="CAUSAL_LM",
            r=24,
            lora_alpha=48,
            lora_dropout=0.05,
            bias="none",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        )
        model = get_peft_model(base_model, lora_cfg)

    return model

def print_trainable_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔧 Trainable params: {trainable} / {total} ({trainable / total:.2%})")

# ==== 主流程 ====
tokenizer = load_tokenizer(model_path)
dataset = load_dataset(data_path, tokenizer)
resume_lora = False   # ← 你现在要的模式
model = prepare_model(
    model_path,
    resume_lora=resume_lora,
    lora_path=output_dir
)

if resume_lora:
    # ===== 继续训练（精修 / 打磨）=====
    num_train_epochs = 3        # 或 5
    learning_rate = 5e-5        # 或 8e-5
    warmup_ratio = 0.02
    phase_name = "resume"
else:
    # ===== 从头 LoRA（语言塑形期）=====
    num_train_epochs = 12
    learning_rate = 1.5e-4
    warmup_ratio = 0.02
    phase_name = "from_scratch"

print_trainable_parameters(model)

args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,                        # 建议训练多轮，提升学习效果
    per_device_train_batch_size=2,             # 4070S 显存可支撑 batch size 2~6，建议从4起试验
    gradient_accumulation_steps=6,             # 累积梯度扩大有效 batch size（如总 batch = 4x4 = 16）
    learning_rate=learning_rate,                        # 3e-4 对大模型偏高，建议尝试 2e-4 更稳
    lr_scheduler_type="cosine",                # 学习率调度：cosine 收敛更平滑
    warmup_ratio=warmup_ratio,                         # 用 warmup_ratio 替代 warmup_steps，适配不同步数
    logging_steps=10,
    save_strategy="epoch",
    max_grad_norm=1.0,
    bf16=False,                                # 4070 不支持 BF16
    fp16=True,                                 # 开启 FP16 更省显存（推荐）
    report_to="none",
    run_name=f"LAB-lora-{phase_name}",  # ⭐ 非常推荐
    optim="paged_adamw_8bit",   # ⭐ 关键：显存更省
    gradient_checkpointing=True,
    group_by_length=True,
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

print(f"""
==============================
 Training Phase: {phase_name}
 resume_lora   : {resume_lora}
 epochs        : {num_train_epochs}
 lr            : {learning_rate}
 warmup_ratio  : {warmup_ratio}
==============================
""")

trainer.train()
model.save_pretrained(output_dir)
print("LoRA 微调完成，模型已保存到:", output_dir)
