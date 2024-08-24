import os, shutil, random, zipfile
import warnings, logging
warnings.filterwarnings('ignore')
logging.basicConfig(filename='errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

import torch, torchvision, timm, torchsummary, pickle
from torchvision.transforms import v2
from torch.utils.data import Dataset
from torchvision.io import read_image
import pytorch_lightning as pl
import numpy as np
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Set ENV to be 'local', 'server' or 'colab'
ENV = "server".lower()

if ENV == "local" or ENV == "server":
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

def load_model(model_name, model_opt, classes, sdepth, save_path):
  def add_sdepth(model_name, model, drop_prob):
    if drop_prob == 0: return model
    if model_name == "convnext":
        for layer in model.modules():
            if isinstance(layer, timm.models.convnext.ConvNeXtBlock):
                layer.drop_path = timm.layers.DropPath(drop_prob)
    return model
  
  name_and_size = model_name.split('-')
  name, size = name_and_size[0], name_and_size[1]
  
  if model_opt == "custom":
    is_pretrained = False
  elif model_opt == "pretrained":
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
    num_feature = model.head.in_features
    model.head = torch.nn.Linear(in_features=num_feature, out_features=classes)
    model.head.weight.data.mul_(0.001)
    # model = add_sdepth(name, model, sdepth)
  
  elif name == "swin":
    if size == "s":
      model = timm.create_model('swin_small_patch4_window7_224.ms_in22k_ft_in1k', pretrained=is_pretrained)
      train_size, test_size = 224, 224
    elif size == "b":
      model = timm.create_model('swin_base_patch4_window7_224.ms_in22k_ft_in1k', pretrained=is_pretrained)
      train_size, test_size = 224, 224
    num_feature = model.head.fc.in_features
    model.head.fc = torch.nn.Linear(in_features=num_feature, out_features=5)
    model.head.fc.weight.data.mul_(0.001)
    # model = add_sdepth(name, model, sdepth)
  
  elif name == "effnetv2":
    if size == "s":
      model = timm.create_model('tf_efficientnetv2_s.in21k', pretrained=is_pretrained)
      train_size, test_size = 300, 384
    elif size == "m":
      model = timm.create_model('tf_efficientnetv2_m.in21k', pretrained=is_pretrained)
      train_size, test_size = 384, 480
    num_feature = model.classifier.in_features
    model.classifier = torch.nn.Linear(in_features=num_feature, out_features=5)
    model.classifier.weight.data.mul_(0.001)
    # model = add_sdepth(name, model, sdepth)
  
  elif name == "convnext":
    if size == "s":
      # model = timm.create_model('convnext_small.fb_in22k', pretrained=is_pretrained)
      # with open('result/pretrained/convnext_small.pkl', 'wb') as f: pickle.dump(model, f)
      with open('result/pretrained/convnext_small.pkl', 'rb') as f: model = pickle.load(f)
      train_size, test_size = 224, 224
    elif size == "b":
      # model = timm.create_model('convnext_base.fb_in22k', pretrained=is_pretrained)
      # with open('result/pretrained/convnext_base.pkl', 'wb') as f: pickle.dump(model, f)
      with open('result/pretrained/convnext_base.pkl', 'rb') as f: model = pickle.load(f)
      train_size, test_size = 224, 224
    elif size == "l":
      # model = timm.create_model('convnext_large.fb_in22k', pretrained=is_pretrained)
      # with open('result/pretrained/convnext_large.pkl', 'wb') as f: pickle.dump(model, f)
      with open('result/pretrained/convnext_large.pkl', 'rb') as f: model = pickle.load(f)
      train_size, test_size = 224, 224
    num_feature = model.head.fc.in_features
    model.head.fc = torch.nn.Linear(in_features=num_feature, out_features=classes)
    model.head.fc.weight.data.mul_(0.001)
    model = add_sdepth(name, model, sdepth)
  
  if save_path:
    with open(f'{save_path}/architecture.txt', 'w') as f:
      f.write("### Summary ###\n")
      f.write(f"{torchsummary.summary(model, (3, train_size, train_size))}\n\n")
      f.write("### Full ###\n")
      f.write(str(model))

  return model, train_size, test_size

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

class CustomImageDataset(Dataset):
    def __init__(self, img_labels, img_dir, transform=None, past_image_num=6, full_image_list=None):
        self.img_labels = img_labels
        self.img_dir = img_dir
        self.transform = transform
        self.past_image_num = past_image_num
        self.full_image_list = full_image_list

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        label = self.img_labels.iloc[idx, 2]
        
        img_name = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = read_image(img_path)
        transformed_image = self.transform(image)
        
        past_img_names = self.load_past_images(img_name)
        past_img_paths = [os.path.join(self.img_dir, past_image) for past_image in past_img_names]
        past_images = [read_image(path) for path in past_img_paths]
        transformed_past_images = [self.transform(img) for img in past_images]
        
        transformed_image += np.sum(transformed_past_images)
        mean = torch.mean(transformed_image)
        std = torch.std(transformed_image)
        transformed_image = (transformed_image - mean) / std
            
        return transformed_image, label
    
    def load_past_images(self, img_name):
        idx = self.full_image_list.index[self.full_image_list == img_name][0]
        if idx >= self.past_image_num:
            past_images = self.img_labels['image_name'].iloc[idx - self.past_image_num : idx].tolist()
        else:
            past_images = self.img_labels['image_name'].iloc[: idx].tolist()
        return past_images

def load_data(radar_range, interval, set_name, image_size, batch_size, shuffle, num_workers=8):
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
  
  label_file = pd.read_csv(f"image/sets/{radar_range}_{interval}_{set_name}.csv")
  full_image_list = pd.read_csv("image/labels.csv")
  full_image_list = full_image_list[full_image_list['range'] == radar_range]['image_name']
  dataset = CustomImageDataset(img_labels=label_file, 
                               img_dir="image/combined", 
                               transform=transforms,
                               past_image_num=6,
                               full_image_list=full_image_list)
  
  dataloader = torch.utils.data.DataLoader(dataset, 
                                           batch_size=batch_size, 
                                           shuffle=shuffle, 
                                           num_workers=num_workers)
  
  return dataloader

class FinetuneModule(pl.LightningModule):
  def __init__(self, model_settings, optimizer_settings, loop_settings, save_path=None):
    super().__init__()

    self.radar_range = model_settings['radar_range']
    self.interval = model_settings['interval']
    self.model_name = model_settings['model_name']
    self.model_opt = model_settings['model_opt']
    self.classes = model_settings['classes']
    self.sdepth = model_settings['sdepth']
    self.model, train_size, test_size = load_model(self.model_name, self.model_opt, self.classes, self.sdepth, save_path)

    self.optimizer_name = optimizer_settings['optimizer_name']
    self.learning_rate = optimizer_settings['learning_rate']
    self.weight_decay = optimizer_settings['weight_decay']
    self.scheduler_name = optimizer_settings['scheduler_name']

    self.batch_size = loop_settings['batch_size']
    self.epochs = loop_settings['epochs']
    self.label_smoothing = loop_settings['label_smoothing']

    self.train_loader = load_data(self.radar_range, self.interval, "train", train_size, self.batch_size, True)
    self.val_loader   = load_data(self.radar_range, self.interval, "val", test_size, self.batch_size, False)
    self.test_loader  = load_data(self.radar_range, self.interval, "test", test_size, self.batch_size, False)
    
    self.label_list = []
    self.pred_list = []

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
    self.pred_list += predictions.tolist()
    
    self.log("test_loss", test_loss, on_epoch=True)
    self.log("test_acc", test_acc, on_epoch=True)
    
    return test_loss
  
  def configure_optimizers(self):
    def get_optimizer_settings():
      if self.optimizer_name == "adam" or self.optimizer_name == "adamw":
        optimizer_settings = [{'params': self.model.parameters(), 
                                'lr': self.learning_rate, 
                                'betas' : (0.9, 0.999), 
                                'weight_decay' : self.weight_decay}]
          
      elif self.optimizer_name == "sgd":
        optimizer_settings = [{'params': self.model.parameters(), 
                              'lr': self.learning_rate, 
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

def plot_loss_acc(monitor_value, min_delta, plot_name, save_path, draw=True):
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
    plt.title(f"{plot_name['range']}_{plot_name['interval']}_{plot_name['model']}")
    plt.legend()
    plt.savefig(f'{save_path}/graph_loss.png')

    plt.clf()

    # Plotting acc
    plt.plot(train_results['epoch'], train_results['train_acc'], label='train_acc')
    plt.plot(val_results['epoch'], val_results['val_acc'], label='val_acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.title(f"{plot_name['range']}_{plot_name['interval']}_{plot_name['model']}")
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

def plot_cmatrix(labels, predictions, plot_name, save_path, draw=True):
  def calculate_metrics(confusion_matrix):
    classes = confusion_matrix.shape[0]
    precision = np.zeros(classes)
    recall = np.zeros(classes)
    f1 = np.zeros(classes)
    
    for i in range(classes):
        tp = confusion_matrix[i, i]
        fp = np.sum(confusion_matrix[:, i]) - tp
        fn = np.sum(confusion_matrix[i, :]) - tp
        tn = np.sum(confusion_matrix) - tp - fp - fn
        precision[i] = tp / (tp + fp) if tp + fp != 0 else 0
        recall[i] = tp / (tp + fn) if tp + fn != 0 else 0
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
    if "very_heavy_rain" in labels or "very_heavy_rain" in predictions:
      display_labels = ['clear', 'heavy_rain', 'light_rain', 'moderate_rain', 'very_heavy_rain']
    else:
      display_labels = ['clear', 'heavy_rain', 'light_rain', 'moderate_rain']
    ConfusionMatrixDisplay.from_predictions(labels, predictions, display_labels=display_labels, normalize='true', ax=ax)
    plt.title(f"{plot_name['range']}_{plot_name['interval']}_{plot_name['model']}")
    plt.savefig(f'{save_path}/cmatrix.png')

  cmatrix = confusion_matrix(labels, predictions)
  return calculate_metrics(cmatrix)
    

  