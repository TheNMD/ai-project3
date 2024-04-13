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
  result_path = "result"
elif ENV == "server":
  image_path = "image"
  result_path = "result"
elif ENV == "colab":
  from google.colab import drive
  drive.mount('/content/drive')
  if not os.path.exists("image.zip"):
    shutil.copy('drive/MyDrive/Coding/image.zip', 'image.zip')
    with zipfile.ZipFile('image.zip', 'r') as zip_ref:
      zip_ref.extractall()
  image_path = "image"
  result_path = "drive/MyDrive/Coding/result"

class FinetuneModule(pl.LightningModule):
  def __init__(self, model_settings, optimizer_settings, loop_settings):
    super().__init__()
    # self.save_hyperparameters()

    self.model_name = model_settings[0]
    self.model_option = model_settings[1]
    self.model, train_size, test_size = load_model(self.model_name, self.model_option)

    self.optimizer_name = optimizer_settings[0]
    self.learning_rate = optimizer_settings[1]

    self.batch_size = loop_settings[0]

    self.train_loader = load_data(option="train", image_size=train_size, batch_size=self.batch_size, shuffle=True)
    self.val_loader   = load_data(option="val", image_size=test_size, batch_size=self.batch_size, shuffle=False)
    self.test_loader  = load_data(option="test", image_size=test_size, batch_size=self.batch_size, shuffle=False)

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
    test_loss, test_acc = self.common_step(batch, batch_idx)
    self.log("test_loss", test_loss)
    self.log("test_acc", test_acc)
    return test_loss

  def configure_optimizers(self):
    if self.optimizer_name == "adam":
      optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    elif self.optimizer_name == "sgd":
      optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=0)
    # TODO Add scheduler to adjust learning_rate
    return optimizer

  def train_dataloader(self):
    return self.train_loader

  def val_dataloader(self):
    return self.val_loader

  def test_dataloader(self):
    return self.test_loader

def load_model(model_name, model_option):
  if model_name == "vit-b":
    model = timm.create_model('vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=True)
    num_feature = model.head.in_features
    model.head = nn.Linear(in_features=num_feature, out_features=5)
    train_size, test_size = 224, 224
  elif model_name == "vit-l":
    model = timm.create_model('vit_large_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
    num_feature = model.head.in_features
    model.head = nn.Linear(in_features=num_feature, out_features=5)
    train_size, test_size = 224, 224
  elif model_name == "swinv2-t":
    model = timm.create_model('swinv2_tiny_window16_256.ms_in1k', pretrained=True)
    num_feature = model.head.fc.in_features
    model.head.fc = nn.Linear(in_features=num_feature, out_features=5)
    train_size, test_size = 256, 256
  elif model_name == "effnetv2-s":
    model = timm.create_model('tf_efficientnetv2_s.in21k_ft_in1k', pretrained=True)
    num_feature = model.classifier.in_features
    model.classifier = nn.Linear(in_features=num_feature, out_features=5)
    train_size, test_size = 300, 384
  elif model_name == "effnetv2-m":
    model = timm.create_model('tf_efficientnetv2_m.in21k_ft_in1k', pretrained=True)
    num_feature = model.classifier.in_features
    model.classifier = nn.Linear(in_features=num_feature, out_features=5)
    train_size, test_size = 384, 480
  elif model_name == "convnext-s":
    model = timm.create_model('convnext_small.fb_in22k', pretrained=True)
    num_feature = model.head.fc.in_features
    model.head.fc = nn.Linear(in_features=num_feature, out_features=5)
    train_size, test_size = 224, 224
  elif model_name == "convnext-b":
    model = timm.create_model('convnext_base.fb_in22k', pretrained=True)
    num_feature = model.head.fc.in_features
    model.head.fc = nn.Linear(in_features=num_feature, out_features=5)
    train_size, test_size = 224, 224

  if not os.path.exists(f"{result_path}/checkpoint/{model_name}-{model_option}"):
    os.makedirs(f"{result_path}/checkpoint/{model_name}-{model_option}")
  with open(f'{result_path}/checkpoint/{model_name}-{model_option}/architecture.txt', 'w') as f:
    f.write(str(model))

  return model, train_size, test_size

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

def plot_results(model_name, model_option, latest_version):
  log_results = pd.read_csv(f"{result_path}/checkpoint/{model_name}-{model_option}/{latest_version}/metrics.csv")
  train_results = log_results[['epoch', 'step', 'train_loss', 'train_acc']].dropna()
  val_results = log_results[['epoch', 'step', 'val_loss', 'val_acc']].dropna()

  # Plotting loss
  plt.plot(train_results['step'], train_results['train_loss'], label='train_loss')
  plt.plot(val_results['step'], val_results['val_loss'], label='val_loss')
  plt.legend()
  plt.xlabel('step')
  plt.ylabel('value')
  plt.title(f'Loss of {model_name}-{model_option}')
  plt.legend()
  plt.savefig(f'{result_path}/checkpoint/{model_name}-{model_option}/{latest_version}/graph_loss.png')

  plt.clf()

  # Plotting acc
  plt.plot(train_results['step'], train_results['train_acc'], label='train_acc')
  plt.plot(val_results['step'], val_results['val_acc'], label='val_acc')
  plt.legend()
  plt.xlabel('step')
  plt.ylabel('value')
  plt.title(f'Accuracy of {model_name}-{model_option}')
  plt.legend()
  plt.savefig(f'{result_path}/checkpoint/{model_name}-{model_option}/{latest_version}/graph_acc.png')

  test_results = log_results[['epoch', 'step', 'test_loss', 'test_acc']].dropna()
  print(f"Testing loss: {test_results['test_loss'].tolist()[0]}")
  print(f"Testing acc: {test_results['test_acc'].tolist()[0]}")

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
     
  if not os.path.exists("result"):
    os.makedirs("result")
    os.makedirs("result/checkpoint")
    os.makedirs("result/final")
  
  # Hyperparameters
  ## For model
  model_name = "vit-b" # vit-b | vit-l | swinv2-t | effnetv2-s | effnetv2-m | convnext-s | convnext-b
  model_option = "pretrained"
  checkpoint = False
  print(f"Model: {model_name}-{model_option}")

  ## For optimizer
  optimizer_name = "adam"
  learning_rate = 1e-4

  ## For callbacks
  patience = 5
  min_delta = 1e-3

  ## For training loop
  batch_size = 32
  num_epochs = 20
  epoch_ratio = 0.5 # check val every percent of an epoch

  # Make Lightning module
  model_settings = [model_name, model_option]
  optimizer_settings = [optimizer_name, learning_rate]
  loop_settings = [batch_size]

  if checkpoint:
    version = "version_0"
    
    module = FinetuneModule.load_from_checkpoint(f"{result_path}/checkpoint/{model_name}-{model_option}/best_model.ckpt", 
                                                 model_settings=model_settings,
                                                 optimizer_settings=optimizer_settings, 
                                                 loop_settings=loop_settings)
    
    trainer = pl.Trainer()
    
    # Evaluation
    start_time = time.time()
    trainer.test(module)
    end_time = time.time()
    print(f"Evaluation time: {end_time - start_time} seconds")
  else:
    versions = sorted([folder for folder in os.listdir(f'{result_path}/checkpoint/{model_name}-{model_option}') 
                       if os.path.isdir(f'{result_path}/checkpoint/{model_name}-{model_option}/{folder}')])
    latest_version = f"version_{len(versions) + 1}"
    
    module = FinetuneModule(model_settings, optimizer_settings, loop_settings)
  
    # Logger
    logger = CSVLogger(save_dir=f'{result_path}/checkpoint', name=f'{model_name}-{model_option}')

    # Callbacks
    early_stop_callback = EarlyStopping(monitor='val_acc',
                                        mode='max',
                                        patience=patience,
                                        min_delta=min_delta,
                                        verbose=True)

    checkpoint_callback = ModelCheckpoint(monitor='val_acc',
                                          mode='max',
                                          save_top_k=1,
                                          filename='best_model',
                                          dirpath=f'{result_path}/checkpoint/{model_name}-{model_option}/{latest_version}')
    
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
                        log_every_n_steps=200,    # log train_loss and train_acc every 200 batches
                        precision=16)             # use mixed precision to speed up training

    # Training loop
    start_time = time.time()
    trainer.fit(module)
    end_time = time.time()
    print(f"Training time: {end_time - start_time} seconds")
    
    # Evaluation
    start_time = time.time()
    trainer.test(module)
    end_time = time.time()
    print(f"Evaluation time: {end_time - start_time} seconds")
    
    # Plot loss and accuracy
    plot_results(model_name, model_option, latest_version)

  