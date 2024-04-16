import os, sys, platform, shutil, time
import zipfile
import warnings
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(filename='errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torchvision import datasets
from torchvision.transforms import v2
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import timm
from torchsummary import summary

import pandas as pd
import matplotlib.pyplot as plt

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
    self.freeze = model_settings[2]
    self.model, train_size, test_size = load_model(self.model_name, self.model_option, self.freeze)

    self.optimizer_name = optimizer_settings[0]
    self.learning_rate = optimizer_settings[1]
    self.scheduler_name = optimizer_settings[2]

    self.batch_size = loop_settings[0]

    self.train_loader = load_data(set_name="train", image_size=train_size, batch_size=self.batch_size, shuffle=True)
    self.val_loader   = load_data(set_name="val", image_size=test_size, batch_size=self.batch_size, shuffle=False)
    self.test_loader  = load_data(set_name="test", image_size=test_size, batch_size=self.batch_size, shuffle=False)

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
      optimizer = Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), weight_decay=0)
    elif self.optimizer_name == "sgd":
      optimizer = SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=0)
      
    if self.scheduler_name == "none":
      return {"optimizer": optimizer}
    elif self.scheduler_name == "cawr":
      scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=int(len(self.train_loader) * 0.4), T_mult=1) 

    return {"optimizer": optimizer, "lr_scheduler": scheduler}

  def train_dataloader(self):
    return self.train_loader

  def val_dataloader(self):
    return self.val_loader

  def test_dataloader(self):
    return self.test_loader

def load_model(model_name, model_option, freeze=False):
  if model_option == "custom":
    pass
  elif model_option == "pretrained":
    if model_name == "vit-b":
      model = timm.create_model('vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=True)
      if freeze:
        for param in model.parameters():
          param.requires_grad = False
      num_feature = model.head.in_features
      model.head = nn.Linear(in_features=num_feature, out_features=5)
      train_size, test_size = 224, 224
    elif model_name == "vit-l":
      model = timm.create_model('vit_large_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
      if freeze:
        for param in model.parameters():
          param.requires_grad = False
      num_feature = model.head.in_features
      model.head = nn.Linear(in_features=num_feature, out_features=5)
      train_size, test_size = 224, 224
    elif model_name == "swinv2-t":
      model = timm.create_model('swinv2_tiny_window16_256.ms_in1k', pretrained=True)
      if freeze:
        for param in model.parameters():
          param.requires_grad = False
      num_feature = model.head.fc.in_features
      model.head.fc = nn.Linear(in_features=num_feature, out_features=5)
      train_size, test_size = 256, 256
    elif model_name == "swinv2-b":
      model = timm.create_model('swinv2_base_window8_256.ms_in1k', pretrained=True)
      if freeze:
        for param in model.parameters():
          param.requires_grad = False
      num_feature = model.head.fc.in_features
      model.head.fc = nn.Linear(in_features=num_feature, out_features=5)
      train_size, test_size = 256, 256
    elif model_name == "convnext-s":
      model = timm.create_model('convnext_small.fb_in22k', pretrained=True)
      if freeze:
        for param in model.parameters():
          param.requires_grad = False
      num_feature = model.head.fc.in_features
      model.head.fc = nn.Linear(in_features=num_feature, out_features=5)
      train_size, test_size = 224, 224
    elif model_name == "convnext-b":
      model = timm.create_model('convnext_base.fb_in22k', pretrained=True)
      if freeze:
        for param in model.parameters():
          param.requires_grad = False
      num_feature = model.head.fc.in_features
      model.head.fc = nn.Linear(in_features=num_feature, out_features=5)
      train_size, test_size = 224, 224
    elif model_name == "convnext-l":
      model = timm.create_model('convnext_large.fb_in22k', pretrained=True)
      if freeze:
        for param in model.parameters():
          param.requires_grad = False
      num_feature = model.head.fc.in_features
      model.head.fc = nn.Linear(in_features=num_feature, out_features=5)
      train_size, test_size = 224, 224

  with open(f'{result_path}/checkpoint/{model_name}-{model_option}/architecture.txt', 'w') as f:
    f.write("### Summary ###\n")
    f.write(f"{summary(model, (3, train_size, train_size))}\n\n")
    f.write("### Full ###\n")
    f.write(str(model))

  return model, train_size, test_size

def load_data(set_name, image_size, batch_size, shuffle, num_workers=4):
  # Preprocessing data
  # TODO Add more preprocessing methods
  # TODO Different methods for train and val and test
  transforms = v2.Compose([v2.ToImage(), 
                           v2.Resize((image_size, image_size)),
                           # v2.RandomVerticalFlip(p=0.1),
                           # v2.RandomHorizontalFlip(p=0.1),
                           v2.Grayscale(num_output_channels=1),
                           v2.ToDtype(torch.float32, scale=True),
                           # v2.GaussianBlur(kernel_size=3, sigma=0.2),
                           v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                          ])

  dataset = datasets.ImageFolder(root=f"{image_path}/sets/{set_name}", transform=transforms )
  
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
  
  return dataloader

def plot_results(model_name, model_option, latest_version):
  log_results = pd.read_csv(f"{result_path}/checkpoint/{model_name}-{model_option}/{latest_version}/metrics.csv")
  train_results = log_results[['epoch', 'train_loss', 'train_acc']].dropna()
  train_results = train_results.groupby(['epoch'], as_index=False).mean()
  val_results = log_results[['epoch', 'val_loss', 'val_acc']].dropna()
  val_results = val_results.groupby(['epoch'], as_index=False).mean()

  # Plotting loss
  plt.plot(train_results['epoch'], train_results['train_loss'], label='train_loss')
  plt.plot(val_results['epoch'], val_results['val_loss'], label='val_loss')
  plt.legend()
  plt.xlabel('epoch')
  plt.ylabel('value')
  plt.title(f'Loss of {model_name}-{model_option}')
  plt.legend()
  plt.savefig(f'{result_path}/checkpoint/{model_name}-{model_option}/{latest_version}/graph_loss.png')

  plt.clf()

  # Plotting acc
  plt.plot(train_results['epoch'], train_results['train_acc'], label='train_acc')
  plt.plot(val_results['epoch'], val_results['val_acc'], label='val_acc')
  plt.legend()
  plt.xlabel('epoch')
  plt.ylabel('value')
  plt.title(f'Accuracy of {model_name}-{model_option}')
  plt.legend()
  plt.savefig(f'{result_path}/checkpoint/{model_name}-{model_option}/{latest_version}/graph_acc.png')

  test_results = log_results[['test_loss', 'test_acc']].dropna()
  test_loss = test_results['test_loss'].tolist()[0]
  test_acc = test_results['test_acc'].tolist()[0]
  
  return test_loss, test_acc

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
    num_gpus = 0
    print("Only Torch CPU is available")
     
  if not os.path.exists("result"):
    os.makedirs("result")
    os.makedirs("result/checkpoint")
    os.makedirs("result/final")
  
  # Hyperparameters
  ## For model
  # vit-b | vit-l | swinv2-t | swinv2-b
  # convnext-s | convnext-b | convnext-l
  model_name = "convnext-b" 
  model_option = "pretrained"
  freeze = False
  checkpoint = False
  print(f"Model: {model_name}-{model_option}")
  if not os.path.exists(f"{result_path}/checkpoint/{model_name}-{model_option}"):
    os.makedirs(f"{result_path}/checkpoint/{model_name}-{model_option}")

  ## For optimizer & scheduler
  optimizer_name = "adam" # adam | sgdr
  learning_rate = 1e-4
  scheduler_name = "none" # none | cawr
  print(f"Optimizer: {optimizer_name}")
  print(f"Learning rate: {learning_rate}")
  print(f"Scheduler: {scheduler_name}")

  ## For callbacks
  patience = 5
  min_delta = 1e-3

  ## For training loop
  batch_size = 32
  num_epochs = 30
  epoch_ratio = 0.5 # check val every percent of an epoch
  print(f"Batch size: {batch_size}")

  # Make Lightning module
  model_settings = [model_name, model_option, freeze]
  optimizer_settings = [optimizer_name, learning_rate, scheduler_name]
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
    latest_version = f"version_{len(versions)}"
    
    try:
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
        devices = 2
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
      train_start_time = time.time()
      trainer.fit(module)
      train_end_time = time.time() - train_start_time
      print(f"Training time: {train_end_time} seconds")
      
      # Evaluation
      test_start_time = time.time()
      trainer.test(module)
      test_end_time = time.time() - test_start_time
      print(f"Evaluation time: {test_end_time} seconds")
      
      # Plot loss and accuracy
      test_loss, tess_acc = plot_results(model_name, model_option, latest_version)
      
      # Write down hyperparameters and results
      with open(f'{result_path}/checkpoint/{model_name}-{model_option}/{latest_version}/notes.txt', 'w') as file:
        file.write('### For models ###\n')
        file.write(f'Model name: {model_name}\n')
        file.write(f'Model Option: {model_option}\n')
        file.write(f'Freeze: {model_option}\n\n')
        
        file.write('### For optimizer & scheduler ###\n')
        file.write(f'Optimizer: {optimizer_name}\n')
        file.write(f'Learning rate: {learning_rate}\n')
        file.write(f'Scheduler: {scheduler_name}\n\n')
        
        file.write('### For callbacks ###\n')
        file.write(f'Patience: {patience}\n')
        file.write(f'Min delta: {min_delta}\n\n')
        
        file.write('### For training loop ###\n')
        file.write(f'Batch size: {batch_size}\n')
        file.write(f'Epochs: {num_epochs}\n')
        file.write(f'Epoch ratio: {epoch_ratio}\n')
        file.write(f'Num GPUs: {num_gpus}\n\n')
        
        file.write('### Results ###\n')
        file.write(f"Test loss: {test_loss}\n")
        file.write(f"Test accuracy: {tess_acc}\n")
        file.write(f"Training time: {train_end_time} seconds\n")
        file.write(f"Evaluation time: {test_end_time} seconds\n")
    except Exception as e:
      print(e)
      if os.path.exists(f'{result_path}/checkpoint/{model_name}-{model_option}/{latest_version}'):
        shutil.rmtree(f'{result_path}/checkpoint/{model_name}-{model_option}/{latest_version}')
  