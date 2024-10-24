import torch
from diffusers import FluxPipeline
import os

# 从环境变量获取令牌
token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not token:
    raise ValueError("请设置 HUGGING_FACE_HUB_TOKEN 环境变量")

# 设置模型 ID
model_id = "black-forest-labs/FLUX.1-dev"

# 设置保存路径（按照 ComfyUI 的要求）
save_path = os.path.join("models", "checkpoints", "FLUX.1-dev")

# 确保目录存在
os.makedirs(save_path, exist_ok=True)

# 下载模型
pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_auth_token=token)

# 保存模型
pipe.save_pretrained(save_path)

print(f"模型已保存到: {save_path}")
