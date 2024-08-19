import torch
print(f"Torch Version: {torch.__version__}")
print(f"CUDA is Available: {torch.cuda.is_available()}")
print(f"CUDA Device Count: {torch.cuda.device_count()}")
print(f"CUDA Current Device: {torch.cuda.current_device()}")

