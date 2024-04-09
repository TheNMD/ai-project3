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

def load_model(name, type="pretrained"):
  if type == "custom":
    pass
  elif type == "pretrained":
    if name == "vit":
      pass
    elif name == "swin":
      if os.path.exists("model/pretrained/swinv2_pretrained.pth"):
        model = torch.load("model/pretrained/swinv2_pretrained.pth")
      else:
        model_name = 'swinv2_tiny_window16_256.ms_in1k'
        model = timm.create_model(model_name, pretrained=True)
        torch.save(model, 'model/pretrained/swinv2_pretrained.pth')
        with open('model/pretrained/swinv2_pretrained_architecture.txt', 'w') as f:
          f.write(str(model))
  return model

if __name__ == '__main__':
  print("Python version: ", sys.version)
  print("Ubuntu version: ", platform.release())
  print("Torch GPU is available: ", torch.cuda.is_available())
        
  if not os.path.exists("result"):
    os.makedirs("result")
  
  model = load_model("swinv2", "pretrained")