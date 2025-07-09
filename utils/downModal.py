# 设置代理（根据你的代理服务器配置修改）
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 或使用 ModelScope 镜像
# os.environ["HF_ENDPOINT"] = "https://modelscope.cn/api/v1/models"


from huggingface_hub import snapshot_download

# 模型配置
# model_id = "Qwen/Qwen3-4B"  
model_id = "Qwen/Qwen2.5-Coder-7B-Instruct" 
local_dir = f"./models/{model_id.split('/')[-1]}"  # 本地保存路径

try:
    # 下载模型到指定本地目录
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,  # 不使用符号链接，便于模型迁移
        resume_download=True,  # 支持断点续传
    )
    
    print(f"模型 {model_id} 已成功下载到: {local_dir}")
    
except Exception as e:
    print(f"下载失败: {e}")
    print("请检查网络连接或模型ID是否正确")
