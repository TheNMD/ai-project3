import sys
import platform
import torch

print("Python version: ", sys.version)
print("Ubuntu version: ", platform.release())
print("Torch GPU is available: ", torch.cuda.is_available())