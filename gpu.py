import torch
import os

# Check PyTorch version
print("PyTorch version:", torch.__version__)
# Check CUDA availability
cuda_available = torch.cuda.is_available()

if cuda_available:
    print("CUDA version:", torch.version.cuda)
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA device name:", torch.cuda.get_device_name(0))
print(f"Number of CPU cores: {os.cpu_count()}")