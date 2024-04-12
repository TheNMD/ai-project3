import os
import shutil
import zipfile
import sys
import time
import platform
import warnings
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(filename='errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from memory_profiler import profile


ENV = "server".lower()

if ENV == "local":
  image_path = "image"
  model_path = "model"
  result_path = "result"
elif ENV == "server":
  image_path = "image"
  model_path = "model"
  result_path = "result"
elif ENV == "colab":
  from google.colab import drive
  drive.mount('/content/drive')
  if not os.path.exists("image.zip"):
    shutil.copy('drive/MyDrive/Coding/image.zip', 'image.zip')
    with zipfile.ZipFile('image.zip', 'r') as zip_ref:
      zip_ref.extractall()
  image_path = "image"
  model_path = "drive/MyDrive/Coding/model"
  result_path = "drive/MyDrive/Coding/result"
  
def load_model(model_name, model_option):
  if os.path.exists(f"{model_path}/{model_name}-{model_option}.pth"):
    model = torch.load(f"{model_path}/{model_name}-{model_option}.pth")
  else:
    if model_name == "vit":
      model = timm.create_model('vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=True)
      # clear, light_rain, moderate_rain, heavy_rain, very_heavy_rain
      num_feature = model.head.in_features
      model.head = nn.Linear(in_features=num_feature, out_features=5)
    elif model_name == "swinv2":
      model = timm.create_model('swinv2_base_window16_256.ms_in1k', pretrained=True)
      # clear, light_rain, moderate_rain, heavy_rain, very_heavy_rain
      num_feature = model.head.fc.in_features
      model.head.fc = nn.Linear(in_features=num_feature, out_features=5)
    elif model_name == "effnetv2":
      model = timm.create_model('tf_efficientnetv2_m.in21k_ft_in1k', pretrained=True)
      # clear, light_rain, moderate_rain, heavy_rain, very_heavy_rain
      num_feature = model.classifier.in_features
      model.classifier = nn.Linear(in_features=num_feature, out_features=5)
    elif model_name == "convnext":
      model = timm.create_model('convnext_small.fb_in22k', pretrained=True)
      # clear, light_rain, moderate_rain, heavy_rain, very_heavy_rain
      num_feature = model.head.fc.in_features
      model.head.fc = nn.Linear(in_features=num_feature, out_features=5)

    torch.save(model, f'{model_path}/{model_name}-{model_option}.pth')
    with open(f'{model_path}/{model_name}-{model_option}_architecture.txt', 'w') as f:
      f.write(str(model))

  return model

load_model("vit", "pretrained")
load_model("swinv2", "pretrained")
load_model("effnetv2", "pretrained")
load_model("convnext", "pretrained")