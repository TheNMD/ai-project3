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

# Set ENV to be 'local', 'server' or 'colab'
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

class FinetuneModule(pl.LightningModule):
  def __init__(self, model, loader, optimizer, learning_rate):
    super().__init__()
    self.model = model
    self.train_loader = loader[0]
    self.val_loader = loader[1]
    self.test_loader = loader[2]
    self.optimizer = optimizer
    self.learning_rate = learning_rate

  def forward(self, x):
    return self.model(x)

  def common_step(self, batch, batch_idx):
      x, y = batch
      logits = self(x)

      criterion = nn.CrossEntropyLoss()
      loss = criterion(logits, y)
      predictions = logits.argmax(-1)
      correct = (predictions == y).sum().item()
      accuracy = correct / x.shape[0]

      return loss, accuracy

  def training_step(self, batch, batch_idx):
      train_loss, train_acc = self.common_step(batch, batch_idx)

      self.log("train_loss", train_loss)
      self.log("train_acc", train_acc)

      return train_loss
  
  def validation_step(self, batch, batch_idx):
      val_loss, val_acc = self.common_step(batch, batch_idx)   

      self.log("val_loss", val_loss, on_epoch=True)
      self.log("val_acc", val_acc, on_epoch=True)

      return val_loss

  def test_step(self, batch, batch_idx):
      loss, accuracy = self.common_step(batch, batch_idx)     

      return loss
    
  def configure_optimizers(self):
    if self.optimizer == "adam":
      optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    elif self.optimizer == "sgd":
      optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=0)
    # TODO Add scheduler to addjust learning_rate
    return optimizer
    
  def train_dataloader(self):
      return self.train_loader

  def val_dataloader(self):
      return self.val_loader

  def test_dataloader(self):
      return self.test_loader

def load_model(name, option):
  if os.path.exists(f"{model_path}/{name}-{option}.pth"):
    model = torch.load(f"{model_path}/{name}-{option}.pth")
  else:
    if name == "vit":
      model = timm.create_model('vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=True)
      # clear, light_rain, moderate_rain, heavy_rain, very_heavy_rain
      num_feature = model.head.in_features
      model.head = nn.Linear(in_features=num_feature, out_features=5)  
    elif name == "swinv2":
      model = timm.create_model('swinv2_base_window16_256.ms_in1k', pretrained=True)
      # clear, light_rain, moderate_rain, heavy_rain, very_heavy_rain    
      num_feature = model.head.fc.in_features
      model.head.fc = nn.Linear(in_features=num_feature, out_features=5)
    elif name == "effnetv2":
      model = timm.create_model('tf_efficientnetv2_m.in21k_ft_in1k', pretrained=True)
      # clear, light_rain, moderate_rain, heavy_rain, very_heavy_rain    
      num_feature = model.classifier.in_features
      model.classifier = nn.Linear(in_features=num_feature, out_features=5)
    elif name == "convnext":
      model = timm.create_model('convnext_small.fb_in22k', pretrained=True)
      # clear, light_rain, moderate_rain, heavy_rain, very_heavy_rain    
      num_feature = model.head.fc.in_features
      model.head.fc = nn.Linear(in_features=num_feature, out_features=5)  
      
    torch.save(model, f'{model_path}/{name}-{option}.pth')
    with open(f'{model_path}/{name}-{option}_architecture.txt', 'w') as f:
      f.write(str(model))
          
  return model

def load_data(option, image_size, batch_size, shuffle, num_workers=20):
  # Preprocessing data
  # 1/ Resize images to fit the image size used when training
  # 2/ Convert to Tensor
  # 3/ Normalize based on ImageNet statistics
  data_transforms = transforms.Compose([transforms.Resize((image_size, image_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
  
  dataset = datasets.ImageFolder(root=f"{image_path}/sets/{option}", transform=data_transforms)
  
  data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
  
  return data_loader

if __name__ == '__main__':
  print("Python version: ", sys.version)
  print("Ubuntu version: ", platform.release())
  if torch.cuda.is_available():
    device = torch.device("cuda")
    num_gpus = torch.cuda.device_count()
    print(f"Torch GPU is available: {num_gpus}")
    for i in range(num_gpus):
      print(torch.cuda.get_device_name(i))
  else:
    device = torch.device("cpu")
    print("Only Torch CPU is available")
  
  if not os.path.exists("model"):
    os.makedirs("model")
     
  if not os.path.exists("result"):
    os.makedirs("result")
    os.makedirs("result/checkpoint")
    os.makedirs("result/final")
  
  # Hyperparameters
  ## For model
  model_name = "vit"
  option = "pretrained"
  checkpoint = False

  ## For optimizer
  learning_rate = 1e-4
  optimizer = "adam"

  ## For callbacks
  patience = 5
  min_delta = 1e-3

  ## For training loop
  batch_size = 16
  # batch_size = 8
  # num_epochs = 10
  num_epochs = 30

  ## For training loop
  # batch_size = 16
  batch_size = 8
  num_epochs = 10
  epoch_ratio = 0.5 # check val every percent of an epoch
  
  # Load model
  model = load_model(model_name, option)

  # Load data
  if model_name == "vit" or model_name == "convnext":
    train_loader = load_data(option="train", image_size=224, batch_size=batch_size, shuffle=True)
    val_loader   = load_data(option="val", image_size=224, batch_size=batch_size, shuffle=True)
    test_loader  = load_data(option="test", image_size=224, batch_size=batch_size, shuffle=True)
  elif model_name == "swinv2":
    train_loader = load_data(option="train", image_size=256, batch_size=batch_size, shuffle=True)
    val_loader   = load_data(option="val", image_size=256, batch_size=batch_size, shuffle=True)
    test_loader  = load_data(option="test", image_size=256, batch_size=batch_size, shuffle=True)
  elif model_name == "effnetv2":
    train_loader = load_data(option="train", image_size=384, batch_size=batch_size, shuffle=True)
    val_loader   = load_data(option="val", image_size=480, batch_size=batch_size, shuffle=True)
    test_loader  = load_data(option="test", image_size=480, batch_size=batch_size, shuffle=True)

  # Make Lightning module
  module = FinetuneModule(model, [train_loader, val_loader, test_loader], optimizer, learning_rate)
  if checkpoint:
    module = module.load_from_checkpoint(f"{result_path}/checkpoint/{model_name}-{option}.ckpt")
  
  # Logger
  logger = CSVLogger(save_dir=f'{result_path}/checkpoint', name=f'{model_name}-{option}_results')

  # Callbacks
  early_stop_callback = EarlyStopping(monitor='val_acc',
                                      mode='max',
                                      patience=patience,
                                      min_delta=min_delta,
                                      verbose=True)

  checkpoint_callback = ModelCheckpoint(monitor='val_acc',
                                        mode='max',
                                        save_top_k=1,
                                        filename=f'{model_name}-{option}',
                                        dirpath=f'{result_path}/checkpoint')
  
  # Combine all elements
  if num_gpus > 1:
    accelerator = 'gpu'
    devices = num_gpus
    strategy = 'ddp'
  elif num_gpus == 1:
    accelerator = 'gpu'
    devices = 1
    strategy = 'auto'
  else:
    accelerator = 'cpu'
    devices = 'auto'
    strategy = 'auto'

  trainer = pl.Trainer(accelerator=accelerator, 
                      devices=devices, 
                      strategy=strategy,
                      max_epochs=num_epochs,
                      logger=logger,
                      callbacks=[early_stop_callback, checkpoint_callback],
                      val_check_interval=epoch_ratio,
                      log_every_n_steps=50,    # log train_loss and train_acc every 50 batches
                      precision=16)            # use mixed precision to speed up training

  # Training loop
  start_time = time.time()
  trainer.fit(module)
  end_time = time.time()
  print(f"Training time: {end_time - start_time} seconds")
  
  # Inference
  start_time = time.time()
  trainer.test()
  end_time = time.time()
  print(f"Inference time: {end_time - start_time} seconds")

  