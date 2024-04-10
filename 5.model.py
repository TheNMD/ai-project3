import os
import sys
import platform
import time
from collections import Counter
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(filename='errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
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

class FinetuneModule(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-4):
      super().__init__()
      self.model = model
      self.lr = learning_rate

    def forward(self, x):
      return self.model(x)

    def training_step(self, batch, batch_idx):
      x, y = batch
      logits = self(x)
      loss =  F.cross_entropy(logits, y)
      preds = torch.argmax(logits, dim=1)
      acc = torch.sum(preds == y).item() / len(x)
      self.log('train_loss', loss, on_step=False, on_epoch=True)
      self.log('train_acc', acc, on_step=False, on_epoch=True)  
      return loss

    def validation_step(self, batch, batch_idx):
      x, y = batch
      logits = self(x)
      loss = F.cross_entropy(logits, y)
      preds = torch.argmax(logits, dim=1)
      acc = torch.sum(preds == y).item() / len(x)
      self.log('val_loss', loss, on_step=False, on_epoch=True)
      self.log('val_acc', acc, on_step=False, on_epoch=True)
      return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

def load_model(name, option, checkpoint=False):
  if checkpoint:
    pass
  else:
    if os.path.exists(f"model/{name}-{option}.pth"):
      model = torch.load(f"model/{name}-{option}.pth")
    else:
      if name == "swinv2":
        model = timm.create_model('swinv2_base_window16_256.ms_in1k', pretrained=True)
        # Replace the final classification layer to match the dataset
        # clear, light_rain, moderate_rain, heavy_rain, very_heavy_rain    
        num_feature = model.head.fc.in_features
        model.head.fc = nn.Linear(in_features=num_feature, out_features=5) 
      elif name == "vit":
        model = timm.create_model('vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=True)
        # Replace the final classification layer to match the dataset
        # clear, light_rain, moderate_rain, heavy_rain, very_heavy_rain
        num_feature = model.head.in_features
        model.head = nn.Linear(in_features=num_feature, out_features=5)  
        
      torch.save(model, f'model/{name}-{option}.pth')
      with open(f'model/{name}-{option}_architecture.txt', 'w') as f:
        f.write(str(model))
          
  return model

def load_data(image_size, 
              batch_size, 
              num_workers=16):
  
  # Preprocessing data
  # 1/ Resize images to fit the image size used when training
  # 2/ Convert to Tensor
  # 3/ Normalize based on ImageNet statistics
  data_transforms = transforms.Compose([transforms.Resize((image_size, image_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
  
  train_dataset = datasets.ImageFolder(root="image/sets/train",
                                       transform=data_transforms)
  
  val_dataset = datasets.ImageFolder(root="image/sets/val",
                                     transform=data_transforms)
  
  test_dataset = datasets.ImageFolder(root="image/sets/test",
                                      transform=data_transforms)
  
  
  train_loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers)
  
  val_loader   = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)
  
  test_loader  = DataLoader(test_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)
  
  with open("image/train_val_test_summary.txt", 'w') as file:
    file.write('### Label to Index ###\n')
    file.write(f'{train_dataset.class_to_idx}\n')
    file.write('### Train set ###\n')
    file.write(f'{dict(Counter(train_dataset.targets))}\n')
    file.write('### Val set ###\n')
    file.write(f'{dict(Counter(val_dataset.targets))}\n')
    file.write('### Test set ###\n')
    file.write(f'{dict(Counter(test_dataset.targets))}\n')
  
  return train_loader, val_loader, test_loader

if __name__ == '__main__':
  print("Python version: ", sys.version)
  print("Ubuntu version: ", platform.release())
  if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Torch GPU is available")
    for i in range(torch.cuda.device_count()):
      print(torch.cuda.get_device_name(i))
  else:
    device = torch.device("cpu")
    print("Only Torch CPU is available")
  
  if not os.path.exists("model"):
    os.makedirs("model")
     
  if not os.path.exists("result"):
    os.makedirs("result")
    os.makedirs("result/checkpoint")
    
  # Load model
  model_name = "vit"
  option = "pretrained"
  checkpoint = False
  
  model = load_model(model_name, option, checkpoint)
  
  model = FinetuneModule(model, 0.0001)
  
  # Load and split data
  if model_name == "swinv2":
    image_size = 256
  elif model_name == "vit":
    image_size = 224
  batch_size = 32
  
  train_loader, val_loader, test_loader = load_data(image_size=image_size,
                                                    batch_size=batch_size)
  
  trainer = pl.Trainer(devices=1, 
                       accelerator="gpu", 
                       strategy="ddp",
                       max_epochs=10,)

  trainer.fit(model, train_loader, val_loader)

  