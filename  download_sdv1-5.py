import torch
from diffusers import StableDiffusionPipeline
import os

# 设置模型 ID
model_id = "runwayml/stable-diffusion-v1-5"

# 设置保存路径
save_path = os.path.join("models", "checkpoints", "stable-diffusion-v1-5")

# 确保目录存在
os.makedirs(save_path, exist_ok=True)

# 下载模型
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

# 保存模型
pipe.save_pretrained(save_path)

print(f"模型已保存到: {save_path}")
