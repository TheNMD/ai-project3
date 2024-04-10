import os
import sys
import platform
import time
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(filename='errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import torch.optim as optim
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
              batch_size=32, 
              num_workers=16):
  
  # Preprocessing data
  # 1/ Resize images to fit the image size used when training
  # 2/ Convert to Tensor
  # 3/ Normalize based on ImageNet statistics
  data_transforms = transforms.Compose([transforms.Resize((image_size, image_size)),
                                        transforms.ToTensor()])
  
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
  
  return train_loader, train_size, val_loader, val_size, test_loader, test_size

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
    
  with open('result/currently_training.txt', 'w') as f: pass
  
  # Load model
  name = "vit"
  option = "pretrained"
  checkpoint = False
  
  model = load_model(name, option, checkpoint)
  
  # Load and split data
  if name == "swinv2":
    image_size = 256
  elif name == "vit":
    image_size = 224
  batch_size = 32
  
  train_loader, train_size, val_loader, val_size, test_loader, test_size = load_data(image_size=image_size,
                                                                                     batch_size=batch_size)
  
  print(f"Train size: {train_size}")
  print(f"Train size: {val_size}")
  print(f"Train size: {test_size}")
  
  # Loss function and optimizer
  learning_rate = 0.001
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  
  # Training loop
  best_accuracy = 0.0
  epochs = 10
  
  model = nn.DataParallel(model)
  model.to(device)
  for epoch in range(epochs):
    epoch_start_time = time.time()
    try:
      # Training phase
      model.train()
      batch = 0
      for images, labels in train_loader:
        batch_start_time = time.time()
        
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        batch += 1
        
        batch_end_time = time.time() - batch_start_time
        print(f"Train: Epoch {epoch} | Batch {batch} | {batch_end_time}")
        with open('result/currently_training.txt', 'a') as f:
          f.write(f"Train: Epoch {epoch} | Batch {batch} | {batch_end_time}\n")
      
      # Validation phase
      model.eval()
      val_loss = 0.0
      correct = 0
      total = 0
      batch = 0
      with torch.no_grad():
        for images, labels in val_loader:
            batch_start_time = time.time()
            
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            batch += 1
            
            batch_end_time = time.time() - batch_start_time
            print(f"Val: Epoch {epoch} | Batch {batch} | {batch_end_time}")
            with open('result/currently_training.txt', 'a') as f:
              f.write(f"Val: Epoch {epoch} | Batch {batch} | {batch_end_time}\n")
      
      # Calculate validation accuracy
      val_acc = correct / total
      val_loss /= val_size
      epoch_end_time = time.time() - epoch_start_time
      
      # Print epoch statistics
      print(f"Epoch {epoch + 1}/{epochs} | val_loss: {val_loss} | val_acc: {val_acc} | time: {epoch_end_time}")
      with open('result/currently_training.txt', 'a') as f:
        f.write(f"Epoch {epoch + 1}/{epochs} | val_loss: {val_loss} | val_acc: {val_acc} | time: {epoch_end_time}\n")
      
      # Save the best model
      if val_acc > best_accuracy:
          best_accuracy = val_acc
          torch.save(model, f'result/checkpoint/{name}-{option}.pth')
          
      # Save training results
      training_record = {'epoch' : [epoch], 'val_loss' : [val_loss], 'val_acc': [val_acc], 'time': [epoch_end_time]}
      if not os.path.exists(f"result/{name}-{option}_training.csv") or checkpoint == False:
        training_result = pd.DataFrame(training_record)
      else:
        training_result = pd.read_csv(f"result/{name}-{option}_training.csv")
        training_result = training_result.append(training_record, ignore_index=True)
      training_result.to_csv(f"result/{name}-{option}_training.csv", index=False)
      
    except Exception as e:
        print(e)
        logging.error(e, exc_info=True)
        break

  