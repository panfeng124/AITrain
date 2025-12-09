import json

# 替换为你的实际路径
data_path = "../../trainData/merged_2025_07_15_19_11.jsonl"
model_path = "../../models/Qwen2.5-Coder-7B-Instruct"

# 手动加载 tokenizer（最小化依赖）
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

lengths = []
total = 0

with open(data_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        ex = json.loads(line)
        # 构造 ChatML 格式文本（与你训练时一致）
        text = "\n".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in ex["messages"])
        tokens = tokenizer(text, add_special_tokens=True)["input_ids"]
        lengths.append(len(tokens))
        total += 1

# 排序
lengths.sort()

def percentile(arr, p):
    idx = int(p * (len(arr) - 1))
    return arr[idx]

print(f"总样本数: {total}")
print(f"最短长度: {lengths[0]}")
print(f"中位数 (50%): {percentile(lengths, 0.5)}")
print(f"90% 分位数: {percentile(lengths, 0.9)}")
print(f"95% 分位数: {percentile(lengths, 0.95)}")
print(f"最长长度: {lengths[-1]}")