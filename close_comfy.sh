#!/bin/bash

echo "开始关闭 ComfyUI"

# 查找运行 ComfyUI 的进程
COMFY_PID=$(pgrep -f "python.*main.py")

if [ -z "$COMFY_PID" ]; then
    echo "没有找到运行中的 ComfyUI 进程"
else
    echo "找到 ComfyUI 进程，PID: $COMFY_PID"
    echo "正在关闭 ComfyUI..."
    kill $COMFY_PID
    sleep 2
    
    # 检查进程是否还在运行
    if ps -p $COMFY_PID > /dev/null; then
        echo "ComfyUI 没有立即响应，强制终止..."
        kill -9 $COMFY_PID
    fi
    
    echo "ComfyUI 已关闭"
fi

# 检查 8188 端口是否仍被占用
if lsof -i :8188 > /dev/null; then
    echo "警告：端口 8188 仍被占用，可能需要手动检查"
else
    echo "端口 8188 已释放"
fi

echo "关闭脚本执行完毕"
