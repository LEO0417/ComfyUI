import torch
import torchvision
import torchaudio
import numpy as np

print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"Torchaudio version: {torchaudio.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# 测试 MPS
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(f"Tensor on MPS: {x}")
    print(f"Tensor device: {x.device}")
else:
    print("MPS not available. Using CPU.")
    x = torch.ones(1)
    print(f"Tensor on CPU: {x}")
    print(f"Tensor device: {x.device}")
