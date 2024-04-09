import os
import sys
import platform
import time
import warnings
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(filename='errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

import torch
import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from memory_profiler import profile

# Set ENV to be 'local', 'server' or 'colab'
ENV = "server".lower()

if ENV == "local":
  image_path = "image"
  result_path = "result"
elif ENV == "server":
  image_path = "image"
  result_path = "result"
elif ENV == "colab":
  from google.colab import drive
  drive.mount('/content/drive')
  image_path = "drive/MyDrive/Coding/image"
  result_path = "drive/MyDrive/Coding/result"
    
if __name__ == '__main__':
  print("Python version: ", sys.version)
  print("Ubuntu version: ", platform.release())
  print("Torch GPU is available: ", torch.cuda.is_available())
        
  if not os.path.exists("result"):
    os.makedirs("result")
    
  # Define the pre-trained SwinV2 Transformer
  model_name = 'swinv2_tiny_window16_256.ms_in1k'
  model = timm.create_model(model_name, pretrained=True)
  
  # Save pre-trained model and its architecture
  with open('model/pretrained/swinv2_pretrained_architecture.txt', 'w') as f:
    f.write(str(model))
  torch.save(model.state_dict(), 'model/pretrained/swinv2_pretrained.pth')