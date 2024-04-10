import os
import sys
import platform
import time
import warnings
from collections import Counter, defaultdict
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(filename='errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from memory_profiler import profile
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import timm

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
    elif name == "swinv2":
      if os.path.exists("model/pretrained/swinv2-pretrained.pth"):
        model = torch.load("model/pretrained/swinv2-pretrained.pth")
      else:
        model_name = 'swinv2_tiny_window16_256.ms_in1k'
        model = timm.create_model(model_name, pretrained=True)
        torch.save(model, 'model/pretrained/swinv2-pretrained.pth')
        with open('model/pretrained/swinv2-pretrained_architecture.txt', 'w') as f:
          f.write(str(model))
  return model

def load_data(image_size=256, 
              batch_size=32, 
              num_workers=16):
  
  # Preprocessing data
  # 1/ Resize images to fit the image size used when training
  # 2/ Convert to Tensor
  # 3/ Normalize based on ImageNet statistics
  data_transforms = transforms.Compose([transforms.Resize(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                             std=[0.229, 0.224, 0.225])])
  
  image_dataset = datasets.ImageFolder(root="image/labeled",
                                       transform=data_transforms)
  
  # Split dataset into train, val, and test sets
  train_size = int(0.8 * len(image_dataset))
  val_size   = int(0.1 * len(image_dataset))
  test_size  = len(image_dataset) - train_size - val_size
  
  train_set, val_set, test_set = random_split(image_dataset, 
                                              [train_size, val_size, test_size], 
                                              generator=torch.Generator().manual_seed(42))
  
  train_loader = DataLoader(train_set,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers)
  
  val_loader   = DataLoader(val_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)
  
  test_loader  = DataLoader(test_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)
  
  # with open("image/train_val_test_summary.txt", 'w') as file:
  #   file.write('### Label to Index ###\n')
  #   file.write(f'{image_dataset.class_to_idx}\n')
  #   file.write('### Train set ###\n')
  #   file.write(f'{count_instances(train_loader)}\n')
  #   file.write('### Val set ###\n')
  #   file.write(f'{count_instances(val_loader)}\n')
  #   file.write('### Test set ###\n')
  #   file.write(f'{count_instances(test_loader)}\n')
  
  return train_loader, val_loader, test_loader

def count_instances(data_loader):
  label_counter = defaultdict(int)
  for _, labels in data_loader:
      for label in labels:
          label_counter[label.item()] += 1
  
  label_counter = dict(label_counter)
  label_counter = dict(sorted(label_counter.items()))
  
  return label_counter

if __name__ == '__main__':
  print("Python version: ", sys.version)
  print("Ubuntu version: ", platform.release())
  print("Torch GPU is available: ", torch.cuda.is_available())
  
  if not os.path.exists("model"):
    os.makedirs("model")
    os.makedirs("model/pretrained")
    os.makedirs("model/custom")
     
  if not os.path.exists("result"):
    os.makedirs("result")
    os.makedirs("result/checkpoint")
  
  # # Load and split data
  # train_loader, val_loader, test_loader = load_data()
  
  # Load model
  model = load_model("swinv2", "pretrained")