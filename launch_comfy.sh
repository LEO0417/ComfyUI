#!/bin/bash

echo "开始执行脚本"

# ComfyUI 的目录路径
COMFY_DIR="/Users/leowang/github/ComfyUI"

# PyTorch 虚拟环境的名称
VENV_NAME="pytorch"

echo "切换到 ComfyUI 目录: $COMFY_DIR"
cd "$COMFY_DIR"

echo "激活 PyTorch 虚拟环境"
source ~/anaconda3/bin/activate PyTorch
# 如果您使用的是 venv 而不是 Anaconda，请使用下面的命令替代上面的行：
# source ~/path/to/pytorch/venv/bin/activate

echo "启动 ComfyUI"
python main.py --auto-launch &

echo "等待 15 秒"
sleep 15

echo "尝试打开浏览器"
open http://localhost:8188

echo "脚本执行完毕"
