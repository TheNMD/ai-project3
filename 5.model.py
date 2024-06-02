import os, sys, platform, shutil, time, random
import zipfile
import warnings
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(filename='errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

import torch, torchvision, timm, torchsummary, pickle
from torchvision.transforms import v2
import pytorch_lightning as pl
import numpy as np
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

class CustomRandAugment(v2.RandAugment):
  def __init__(self, num_ops, magnitude, fill):
      super().__init__(num_ops=num_ops, magnitude=magnitude, fill=fill)

      try:
        # del self._AUGMENTATION_SPACE['Brightness']
        # del self._AUGMENTATION_SPACE['Color']
        # del self._AUGMENTATION_SPACE['Contrast']
        # del self._AUGMENTATION_SPACE['Sharpness']
        # del self._AUGMENTATION_SPACE['Posterize']
        # del self._AUGMENTATION_SPACE['Solarize']
        # del self._AUGMENTATION_SPACE['Equalize']
        del self._AUGMENTATION_SPACE['AutoContrast']
      except Exception as e:
        pass


class FinetuneModule(pl.LightningModule):
  def __init__(self, model_settings, optimizer_settings, loop_settings):
    super().__init__()

    self.interval = model_settings['interval']
    self.model_name = model_settings['model_name']
    self.model_option = model_settings['model_option']
    self.num_classes = model_settings['num_classes']
    self.stochastic_depth = model_settings['stochastic_depth']
    self.freeze = model_settings['freeze']
    self.model, train_size, test_size = self.load_model()

    self.optimizer_name = optimizer_settings['optimizer_name']
    self.learning_rate = optimizer_settings['learning_rate']
    self.lr_decay = optimizer_settings['lr_decay']
    self.weight_decay = optimizer_settings['weight_decay']
    self.scheduler_name = optimizer_settings['scheduler_name']

    self.batch_size = loop_settings['batch_size']
    self.epochs = loop_settings['epochs']
    self.label_smoothing = loop_settings['label_smoothing']

    self.train_loader = self.load_data("train", train_size, True)
    self.val_loader   = self.load_data("val", test_size, False)
    self.test_loader  = self.load_data("test", test_size, False)
    
    self.label_list = []
    self.prediction_list = []

  def load_model(self):
    def add_stochastic_depth(model_name, model, drop_prob):
      if drop_prob == 0: return model
      if model_name == "convnext":
          for layer in model.modules():
              if isinstance(layer, timm.models.convnext.ConvNeXtBlock):
                  layer.drop_path = timm.layers.DropPath(drop_prob)
      return model
    
    name_and_size = self.model_name.split('-')
    name, size = name_and_size[0], name_and_size[1]
    
    if self.model_option == "custom":
      is_pretrained = False
    elif self.model_option == "pretrained":
      is_pretrained = True
      
    if name == "vit":
      if size == "s":
        model = timm.create_model('vit_small_patch16_224.augreg_in21k_ft_in1k', pretrained=is_pretrained)
        train_size, test_size = 224, 224
      elif size == "b":
        model = timm.create_model('vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=is_pretrained)
        train_size, test_size = 224, 224
      elif size == "l":
        model = timm.create_model('vit_large_patch16_224.augreg_in21k_ft_in1k', pretrained=is_pretrained)
        train_size, test_size = 224, 224
      if self.freeze:
        for param in model.parameters(): param.requires_grad = False
      num_feature = model.head.in_features
      model.head = torch.nn.Linear(in_features=num_feature, out_features=self.num_classes)
      model.head.weight.data.mul_(0.001)
      # model = add_stochastic_depth(name, model, self.stochastic_depth)
    
    elif name == "swin":
      if size == "s":
        model = timm.create_model('swin_small_patch4_window7_224.ms_in22k_ft_in1k', pretrained=is_pretrained)
        train_size, test_size = 224, 224
      elif size == "b":
        model = timm.create_model('swin_base_patch4_window7_224.ms_in22k_ft_in1k', pretrained=is_pretrained)
        train_size, test_size = 224, 224
      if self.freeze:
        for param in model.parameters(): param.requires_grad = False
      num_feature = model.head.fc.in_features
      model.head.fc = torch.nn.Linear(in_features=num_feature, out_features=5)
      model.head.fc.weight.data.mul_(0.001)
      # model = add_stochastic_depth(name, model, self.stochastic_depth)
    
    elif name == "effnetv2":
      if size == "s":
        model = timm.create_model('tf_efficientnetv2_s.in21k', pretrained=is_pretrained)
        train_size, test_size = 300, 384
      elif size == "m":
        model = timm.create_model('tf_efficientnetv2_m.in21k', pretrained=is_pretrained)
        train_size, test_size = 384, 480
      if self.freeze:
        for param in model.parameters(): param.requires_grad = False
      num_feature = model.classifier.in_features
      model.classifier = torch.nn.Linear(in_features=num_feature, out_features=5)
      model.classifier.weight.data.mul_(0.001)
      # model = add_stochastic_depth(name, model, self.stochastic_depth)
    
    elif name == "convnext":
      if size == "s":
        model = timm.create_model('convnext_small.fb_in22k', pretrained=is_pretrained)
        train_size, test_size = 224, 224
      elif size == "b":
        model = timm.create_model('convnext_base.fb_in22k', pretrained=is_pretrained)
        train_size, test_size = 224, 224
      elif size == "l":
        # model = timm.create_model('convnext_large.fb_in22k', pretrained=is_pretrained)
        with open('result/convnext-l.pickle', 'rb') as f:
          model = pickle.load(f)
        train_size, test_size = 224, 224
      if self.freeze:
        for param in model.parameters(): param.requires_grad = False
      num_feature = model.head.fc.in_features
      model.head.fc = torch.nn.Linear(in_features=num_feature, out_features=self.num_classes)
      model.head.fc.weight.data.mul_(0.001)
      model = add_stochastic_depth(name, model, self.stochastic_depth)
      
    with open(f'{result_path}/checkpoint/{self.interval}/{self.model_name}-{self.model_option}/architecture.txt', 'w') as f:
      f.write("### Summary ###\n")
      f.write(f"{torchsummary.summary(model, (3, train_size, train_size))}\n\n")
      f.write("### Full ###\n")
      f.write(str(model))

    return model, train_size, test_size

  def load_data(self, set_name, image_size, shuffle, num_workers=4):
    def median_blur(image, kernel_size=5):
        pil_image = v2.functional.to_pil_image(image)
        blurred_img = cv.medianBlur(np.array(pil_image), kernel_size)
        return v2.functional.to_image(blurred_img)
    
    # Preprocessing data
    # TODO Add more preprocessing methods
    if set_name == "train":
      transforms = v2.Compose([
                               v2.ToImage(), 
                               v2.Resize((image_size, image_size)),
                              #  v2.RandomErasing(p=0.25, value=255),
                              #  v2.RandAugment(num_ops=2, magnitude=round(random.gauss(9, 0.5)), fill=255),
                               CustomRandAugment(num_ops=2, magnitude=round(random.gauss(9, 0.5)), fill=255),
                               v2.Lambda(lambda image: median_blur(image, kernel_size=5)),
                               v2.Lambda(lambda image: v2.functional.autocontrast(image)),
                               v2.ToDtype(torch.float32, scale=True),
                               v2.Normalize(mean=[0.9844, 0.9930, 0.9632], 
                                            std=[0.0641, 0.0342, 0.1163]), # mean and std of Nha Be dataset
                              ])
      
    elif set_name == "val" or set_name == "test":
      transforms = v2.Compose([
                               v2.ToImage(), 
                               v2.Resize((image_size, image_size)),
                               v2.Lambda(lambda image: median_blur(image, kernel_size=5)),
                               v2.Lambda(lambda image: v2.functional.autocontrast(image)),
                               v2.ToDtype(torch.float32, scale=True),
                               v2.Normalize(mean=[0.9844, 0.9930, 0.9632], 
                                            std=[0.0641, 0.0342, 0.1163]), # mean and std of Nha Be dataset
                              ]) 

    dataset = torchvision.datasets.ImageFolder(root=f"{image_path}/labeled/{self.interval}/{set_name}",
                                               transform=transforms)
    
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=self.batch_size, 
                                             shuffle=shuffle, 
                                             num_workers=num_workers)
    
    return dataloader

  def forward(self, x):
    return self.model(x)

  def training_step(self, batch, batch_idx):
    x, y = batch
    logits = self(x)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
    train_loss = criterion(logits, y)
    predictions = logits.argmax(-1)
    correct = (predictions == y).sum().item()
    train_acc = correct / x.shape[0]
    
    self.log("train_loss", train_loss)
    self.log("train_acc", train_acc)
    return train_loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    logits = self(x)
    criterion = torch.nn.CrossEntropyLoss()
    val_loss = criterion(logits, y)
    predictions = logits.argmax(-1)
    correct = (predictions == y).sum().item()
    val_acc = correct / x.shape[0]
    
    self.log("val_loss", val_loss, on_epoch=True)
    self.log("val_acc", val_acc, on_epoch=True)
    return val_loss

  def test_step(self, batch, batch_idx):
    x, y = batch
    logits = self(x)
    criterion = torch.nn.CrossEntropyLoss()
    test_loss = criterion(logits, y)
    predictions = logits.argmax(-1)
    correct = (predictions == y).sum().item()
    test_acc = correct / x.shape[0]

    self.label_list += y.tolist()
    self.prediction_list += predictions.tolist()
    
    self.log("test_loss", test_loss, on_epoch=True)
    self.log("test_acc", test_acc, on_epoch=True)
    
    return test_loss
  
  def configure_optimizers(self):
    def get_optimizer_settings():
      if self.optimizer_name == "adam" or self.optimizer_name == "adamw":
        if self.lr_decay == 0:
          optimizer_settings = [{'params': self.model.parameters(), 
                                 'lr': self.learning_rate, 
                                 'betas' : (0.9, 0.999), 
                                 'weight_decay' : self.weight_decay}]
        else:
          # layer-wise lr decay
          optimizer_settings = []
          learning_rate = self.learning_rate
          
          layer_names = [n for n, _ in self.model.named_parameters()]
          layer_names.reverse()
          
          previous_group_name = layer_names[0].split('.')[0]

          for name in layer_names:
            current_group_name = name.split('.')[0]
            
            if current_group_name == "stages":
                current_block_num = int(name.split('.')[3])
                if current_block_num % 3 == 1: current_block_num += 1 
                elif current_block_num % 3 == 0: current_block_num += 2
                current_group_name = f"{name.split('.')[0]}{name.split('.')[1]}{name.split('.')[2]}{current_block_num}"
            
            if current_group_name != previous_group_name:
                learning_rate *= self.lr_decay
                
            previous_group_name = current_group_name

            optimizer_settings += [{'params': [p for n, p in self.model.named_parameters() 
                                               if n == name and p.requires_grad], 
                                    'lr': learning_rate, 
                                    'betas' : (0.9, 0.999), 
                                    'weight_decay' : self.weight_decay}]
          
      elif self.optimizer_name == "sgd":
        if self.lr_decay == 0:
          optimizer_settings = [{'params': self.model.parameters(), 
                                'lr': self.learning_rate, 
                                'momentum' : 0.9, 
                                'weight_decay' : 0}]
        else:
          # layer-wise lr decay
          optimizer_settings = []
          learning_rate = self.learning_rate
          
          layer_names = [name for name, _ in self.model.named_parameters()]
          layer_names.reverse()
          
          previous_group_name = layer_names[0].split('.')[0]

          for name in layer_names:
            current_group_name = name.split('.')[0]
            
            if current_group_name != previous_group_name:
                learning_rate *= self.lr_decay
                
            previous_group_name = current_group_name
          
            optimizer_settings += [{'params': [p for n, p in self.model.named_parameters() 
                                               if n == name and p.requires_grad], 
                                    'lr': learning_rate, 
                                    'momentum' : 0.9, 
                                    'weight_decay' : 0}]
    
      return optimizer_settings
      
    
    if self.optimizer_name == "adam":
      optimizer = torch.optim.Adam(get_optimizer_settings())
      
    elif self.optimizer_name == "adamw":
      optimizer = torch.optim.AdamW(get_optimizer_settings())
      
    elif self.optimizer_name == "sgd":
      optimizer = torch.optim.SGD(get_optimizer_settings())
        
    if self.scheduler_name == "none":
      return {"optimizer": optimizer}
    
    elif self.scheduler_name == "cd":
      # Cosine decay
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                             T_max=5)
      
    elif self.scheduler_name == "cdwr":
      # Cosine decay warm restart
      scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                                       T_0=int(len(self.train_loader) * 0.4), 
                                                                       T_mult=1) 

    return {"optimizer": optimizer, "lr_scheduler": scheduler}

  def train_dataloader(self):
    return self.train_loader

  def val_dataloader(self):
    return self.val_loader

  def test_dataloader(self):
    return self.test_loader

def plot_loss_acc(monitor_value, min_delta, save_path, draw=True):
  if not os.path.exists(f"{save_path}/metrics.csv"):
    return None, None, None
  
  log_results = pd.read_csv(f"{save_path}/metrics.csv")
  train_results = log_results[['epoch', 'train_loss', 'train_acc']].dropna()
  train_results = train_results.groupby(['epoch'], as_index=False).mean()
  val_results = log_results[['epoch', 'val_loss', 'val_acc']].dropna()
  
  monitor_result_list = val_results[monitor_value].tolist()
  if monitor_value == "val_loss":
      min_loss = monitor_result_list[0]
      for value in monitor_result_list[1:]:
          if value - min_loss < min_delta: min_loss = value
      best_idx = val_results[val_results['val_acc'] == min_loss].index.tolist()[0]
  elif monitor_value == "val_acc":
      max_acc = monitor_result_list[0]
      for value in monitor_result_list[1:]:
          if value - max_acc > min_delta: max_acc = value
      best_idx = val_results[val_results['val_acc'] == max_acc].index.tolist()[0]
  best_epoch = val_results.loc[best_idx, 'epoch']
  
  val_results = val_results.groupby(['epoch'], as_index=False).mean()
  
  if draw:
    # Plotting loss
    plt.plot(train_results['epoch'], train_results['train_loss'], label='train_loss')
    plt.plot(val_results['epoch'], val_results['val_loss'], label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.title('Loss Graph')
    plt.legend()
    plt.savefig(f'{save_path}/graph_loss.png')

    plt.clf()

    # Plotting acc
    plt.plot(train_results['epoch'], train_results['train_acc'], label='train_acc')
    plt.plot(val_results['epoch'], val_results['val_acc'], label='val_acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.title('Accuracy Graph')
    plt.legend()
    plt.savefig(f'{save_path}/graph_acc.png')

  if "test_loss" in log_results.columns:
    test_results = log_results[['test_loss', 'test_acc']].dropna()
    test_loss = test_results['test_loss'].tolist()
    test_acc = test_results['test_acc'].tolist()
  else:
    test_loss = None
    test_acc = None
    
  return test_loss, test_acc, best_epoch

def plot_confusion_matrix(labels, predictions, save_path, draw=True):
  def calculate_metrics(confusion_matrix):
      num_classes = confusion_matrix.shape[0]
      precision = np.zeros(num_classes)
      recall = np.zeros(num_classes)
      f1 = np.zeros(num_classes)
      
      for i in range(num_classes):
          TP = confusion_matrix[i, i]
          FP = np.sum(confusion_matrix[:, i]) - TP
          FN = np.sum(confusion_matrix[i, :]) - TP
          TN = np.sum(confusion_matrix) - TP - FP - FN
          precision[i] = TP / (TP + FP) if TP + FP != 0 else 0
          recall[i] = TP / (TP + FN) if TP + FN != 0 else 0
          f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if precision[i] + recall[i] != 0 else 0
      return precision, recall, f1

  for i in range(len(labels)):
    if str(labels[i]) == "0":
        labels[i] = "clear"
    elif str(labels[i]) == "1":
        labels[i] = "heavy_rain"
    elif str(labels[i]) == "2":
        labels[i] = "light_rain"
    elif str(labels[i]) == "3":
        labels[i] = "moderate_rain"
    elif str(labels[i]) == "4":
        labels[i] = "very_heavy_rain"
        
  for i in range(len(predictions)):
    if str(predictions[i]) == "0":
        predictions[i] = "clear"
    elif str(predictions[i]) == "1":
        predictions[i] = "heavy_rain"
    elif str(predictions[i]) == "2":
        predictions[i] = "light_rain"
    elif str(predictions[i]) == "3":
        predictions[i] = "moderate_rain"
    elif str(predictions[i]) == "4":
        predictions[i] = "very_heavy_rain"
  
  if draw:
    _, ax = plt.subplots(figsize=(10.5, 8))
    display_labels = ['clear', 'heavy_rain', 'light_rain', 'moderate_rain', 'very_heavy_rain']
    ConfusionMatrixDisplay.from_predictions(labels,
                                            predictions,
                                            display_labels=display_labels,
                                            normalize='true', 
                                            ax=ax)
    plt.title('Confusion Matrix')
    plt.savefig(f'{save_path}/confusion_matrix.png')

  cm = confusion_matrix(labels, predictions)
  return calculate_metrics(cm)
    
if __name__ == '__main__':
  print("Python version: ", sys.version)
  print("Ubuntu version: ", platform.release())
  if torch.cuda.is_available():
    device = torch.device("cuda")
    num_gpus = torch.cuda.device_count()
    print(f"Torch GPU is available: {num_gpus}")
    for i in range(num_gpus):
      print(torch.cuda.get_device_name(i), "\n")
  else:
    device = torch.device("cpu")
    num_gpus = 0
    print("Only Torch CPU is available\n")
     
  if not os.path.exists("result"):
    os.makedirs("result")
    os.makedirs("result/checkpoint")
    os.makedirs("result/final")
  
  # Hyperparameters
  ## For model
  # 0 | 3600 | 7200 | 10800 | 14400 | 18000 | 21600 | 43200
  interval = 0
  # convnext-s | convnext-b | convnext-l 
  # vit-s      | vit-b      | vit-l 
  # swin-s     | swin-b 
  # effnetv2-s | effnetv2-m
  model_name = "convnext-b"
  model_option = "pretrained" # pretrained | custom
  num_classes = 5
  stochastic_depth = 0.3 # 0.0 | 0.1 | 0.2 | 0.3 
  freeze = False
  checkpoint = False
  ckpt_version = "version_3"
  train_from_checkpoint = False
  continue_training = False
  
  print(f"Interval: {interval}")
  print(f"Model: {model_name}-{model_option}")
  print(f"Stochastic depth: {stochastic_depth}")
  print(f"Freeze: {freeze}")
  if not checkpoint:  print(f"Load from checkpoint: {checkpoint}")
  else: print(f"Load from checkpoint: {checkpoint} [{ckpt_version}]")
  print(f"Train from checkpoint: {train_from_checkpoint}")
  print(f"Continue training: {continue_training}\n")
  
  if not os.path.exists(f"{result_path}/checkpoint/{interval}"):
    os.makedirs(f"{result_path}/checkpoint/{interval}")
  if not os.path.exists(f"{result_path}/checkpoint/{interval}/{model_name}-{model_option}"):
    os.makedirs(f"{result_path}/checkpoint/{interval}/{model_name}-{model_option}")
  model_path = f"{result_path}/checkpoint/{interval}/{model_name}-{model_option}"

  ## For optimizer & scheduler
  optimizer_name = "adamw"  # adam | adamw | sgd
  learning_rate = 5e-5      # 1e-3 | 1e-4  | 5e-5
  lr_decay = 0.0            # 0.0  | 0.8 
  weight_decay = 1e-8       # 0    | 1e-8 
  scheduler_name = "cd"     # none | cd    | cdwr  
  
  print(f"Optimizer: {optimizer_name}")
  print(f"Learning rate: {learning_rate}")
  print(f"Layer-wise learning rate decay: {lr_decay}")
  print(f"Weight decay: {weight_decay}")
  print(f"Scheduler: {scheduler_name}\n")

  ## For callbacks
  monitor_value = "val_acc" # val_acc | val_loss
  patience = 22
  min_delta = 1e-4 # 1e-4 | 5e-4
  min_epochs = 21 # 21 | 41 | 61

  ## For training loop
  batch_size = 128 # 32 | 64 | 128 | 256
  epochs = 200
  epoch_ratio = 0.5 # Check val every percentage of an epoch
  label_smoothing = 0.1
  
  print(f"Batch size: {batch_size}")
  print(f"Epoch: {epochs}")
  print(f"Label smoothing: {label_smoothing}\n")

  # Combine all settings
  model_settings = {'interval': interval,
                    'model_name': model_name, 
                    'model_option': model_option,
                    'num_classes': num_classes,
                    'stochastic_depth': stochastic_depth, 
                    'freeze': freeze}
  
  optimizer_settings = {'optimizer_name': optimizer_name, 
                        'learning_rate': learning_rate,
                        'lr_decay': lr_decay, 
                        'weight_decay': weight_decay, 
                        'scheduler_name': scheduler_name}
  
  loop_settings = {'batch_size': batch_size, 
                   'epochs': epochs,
                   'label_smoothing': label_smoothing}
  
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
  
  versions = [folder for folder in 
              os.listdir(f'{result_path}/checkpoint/{interval}/{model_name}-{model_option}') 
              if os.path.isdir(f'{result_path}/checkpoint/{interval}/{model_name}-{model_option}/{folder}')]
  if len(versions) == 0:
    new_version = "version_0"
  else:
    latest_version = sorted([int(version.split('_')[1]) for version in versions])[-1]
    new_version = f"version_{latest_version + 1}"
  
  # Logger
  logger = pl.loggers.CSVLogger(save_dir=f'{result_path}/checkpoint/{interval}', 
                                name=f'{model_name}-{model_option}')

  # Callbacks
  if monitor_value == "val_acc": monitor_mode = "max"
  elif monitor_value == "val_loss": monitor_mode = "min" 
  
  early_stopping_callback = pl.callbacks.EarlyStopping(monitor=monitor_value,
                                                       mode='max',
                                                       patience=patience,
                                                       min_delta=min_delta,
                                                       verbose=True,)

  checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor=monitor_value,
                                                     mode=monitor_mode,
                                                     save_top_k=1,
                                                     filename='best_model',
                                                     dirpath=f'{model_path}/{new_version}',
                                                     verbose=True,)
  
  if checkpoint:
    module = FinetuneModule.load_from_checkpoint(f"{model_path}/{ckpt_version}/best_model.ckpt", 
                                                 model_settings=model_settings,
                                                 optimizer_settings=optimizer_settings, 
                                                 loop_settings=loop_settings)
    
    if train_from_checkpoint:
      trainer = pl.Trainer(accelerator=accelerator, 
                           devices=devices, 
                           strategy=strategy,
                           max_epochs=epochs,
                           min_epochs=min_epochs,
                           logger=logger,
                           callbacks=[early_stopping_callback, checkpoint_callback],
                           val_check_interval=epoch_ratio,
                           log_every_n_steps=50,    # log train_loss and train_acc every n batches
                           precision=16)            # use mixed precision to speed up training
      
      try:
        # Training loop
        train_start_time = time.time()
        if continue_training:
          trainer.fit(module, ckpt_path=f"{model_path}/{ckpt_version}/best_model.ckpt")
        else:
          trainer.fit(module)
        train_end_time = time.time() - train_start_time
        print(f"Training time: {train_end_time} seconds")
        
        # Evaluation
        module_test = FinetuneModule.load_from_checkpoint(f"{model_path}/{new_version}/best_model.ckpt", 
                                                          model_settings=model_settings,
                                                          optimizer_settings=optimizer_settings, 
                                                          loop_settings=loop_settings)
        test_start_time = time.time()
        trainer.test(module_test)
        test_end_time = time.time() - test_start_time
        print(f"Evaluation time: {test_end_time} seconds")

        # Plot loss and accuracy
        test_loss, tess_acc, best_epoch = plot_loss_acc(monitor_value, 
                                                        min_delta, 
                                                        f"{model_path}/{new_version}",
                                                        draw=True)
        print(f"Best epoch [{monitor_value}]: {best_epoch}")
        
        # Plot testing accuracy by class
        precision, recall, f1 = plot_confusion_matrix(module.label_list,
                                                        module.prediction_list,
                                                        f"{model_path}/{ckpt_version}",
                                                        draw=True)
        print(f"Precision:{precision}\nRecall: {recall}\nF1: {f1}")
        
        # Write down hyperparameters and results
        with open(f"{model_path}/{new_version}/notes.txt", 'w') as file:
          file.write('### Hyperparameters ###\n')
          file.write(f'model_settings = {model_settings}\n')
          file.write(f'optimizer_settings = {optimizer_settings}\n')
          file.write(f'loop_settings {loop_settings}\n\n')

          file.write('### Results ###\n')
          file.write(f"Test loss: {test_loss}\n")
          file.write(f"Test accuracy: {tess_acc}\n")
          file.write(f"Precision:{precision}\nRecall: {recall}\nF1: {f1}\n")
          file.write(f"Best epoch ({monitor_value}): {best_epoch}\n")
          file.write(f"Training time: {train_end_time} seconds\n")
          file.write(f"Load from: {interval}-{ckpt_version}\n")
          file.write(f"Continue training: {continue_training}\n")

        # Move architecture file to the corresponding version
        if os.path.exists(f"{model_path}/architecture.txt"):
          shutil.move(f"{model_path}/architecture.txt", f"{model_path}/{new_version}/architecture.txt")
      except Exception as e:
        print(e)
        logging.error(e, exc_info=True)
        if os.path.exists(f'{result_path}/checkpoint/{interval}/{model_name}-{model_option}/{new_version}'):
          shutil.rmtree(f'{result_path}/checkpoint/{interval}/{model_name}-{model_option}/{new_version}')
          
    else:
      trainer = pl.Trainer(accelerator=accelerator, 
                           devices=devices, 
                           strategy=strategy,
                           logger=False,
                           enable_checkpointing=False)
      
      try:
        # Evaluation
        start_time = time.time()
        trainer.test(module)
        end_time = time.time()
        print(f"Evaluation time: {end_time - start_time} seconds")
        
        test_loss, tess_acc, best_epoch = plot_loss_acc(monitor_value, 
                                                        min_delta, 
                                                        f"{model_path}/{ckpt_version}",
                                                        draw=True)
        print(f"Best epoch [{monitor_value}]: {best_epoch}")
        
        precision, recall, f1 = plot_confusion_matrix(module.label_list,
                                                      module.prediction_list,
                                                      f"{model_path}/{ckpt_version}",
                                                      draw=True)
        print(f"Precision:{precision}\nRecall: {recall}\nF1: {f1}")
        
        # Move architecture file to the corresponding version
        if os.path.exists(f"{model_path}/architecture.txt"):
          shutil.move(f"{model_path}/architecture.txt", f"{model_path}/{ckpt_version}/architecture.txt")
      except Exception as e:
        print(e)
        logging.error(e, exc_info=True)
    
  else:    
    module = FinetuneModule(model_settings=model_settings, 
                            optimizer_settings=optimizer_settings, 
                            loop_settings=loop_settings)

    trainer = pl.Trainer(accelerator=accelerator, 
                         devices=devices, 
                         strategy=strategy,
                         max_epochs=epochs,
                         min_epochs=min_epochs,
                         logger=logger,
                         callbacks=[early_stopping_callback, checkpoint_callback],
                         val_check_interval=epoch_ratio,
                         log_every_n_steps=50,    # log train_loss and train_acc every n batches
                         precision=16)             # use mixed precision to speed up training
    
    try:
      # Training loop
      train_start_time = time.time()
      trainer.fit(module)
      train_end_time = time.time() - train_start_time
      print(f"Training time: {train_end_time} seconds")
      
      # Evaluation
      module_test = FinetuneModule.load_from_checkpoint(f"{model_path}/{new_version}/best_model.ckpt", 
                                                        model_settings=model_settings,
                                                        optimizer_settings=optimizer_settings, 
                                                        loop_settings=loop_settings)
      test_start_time = time.time()
      trainer.test(module_test)
      test_end_time = time.time() - test_start_time
      print(f"Evaluation time: {test_end_time} seconds")

      # Plot loss and accuracy
      test_loss, tess_acc, best_epoch = plot_loss_acc(monitor_value, 
                                                     min_delta, 
                                                     f"{model_path}/{new_version}",
                                                     draw=True)
      print(f"Best epoch [{monitor_value}]: {best_epoch}")
      
      # Plot testing accuracy by class
      precision, recall, f1 = plot_confusion_matrix(module.label_list,
                                                    module.prediction_list,
                                                    f"{model_path}/{ckpt_version}",
                                                    draw=True)
      print(f"Precision:{precision}\nRecall: {recall}\nF1: {f1}")
      
      # Write down hyperparameters and results
      with open(f"{model_path}/{new_version}/notes.txt", 'w') as file:
        file.write('### Hyperparameters ###\n')
        file.write(f'model_settings = {model_settings}\n')
        file.write(f'optimizer_settings = {optimizer_settings}\n')
        file.write(f'loop_settings {loop_settings}\n\n')

        file.write('### Results ###\n')
        file.write(f"Test loss: {test_loss}\n")
        file.write(f"Test accuracy: {tess_acc}\n")
        file.write(f"Precision:{precision}\nRecall: {recall}\nF1: {f1}\n")
        file.write(f"Best epoch ({monitor_value}): {best_epoch}\n")
        file.write(f"Training time: {train_end_time} seconds\n")
      
      # Move architecture file to the corresponding version
      if os.path.exists(f"{model_path}/architecture.txt"):
        shutil.move(f"{model_path}/architecture.txt", f"{model_path}/{new_version}/architecture.txt")
    except Exception as e:
      print(e)
      logging.error(e, exc_info=True)
      # if os.path.exists(f'{result_path}/checkpoint/{interval}/{model_name}-{model_option}/{new_version}'):
      #   shutil.rmtree(f'{result_path}/checkpoint/{interval}/{model_name}-{model_option}/{new_version}')

  