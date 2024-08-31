import os, sys, platform, shutil, time, zipfile
import warnings, logging
warnings.filterwarnings('ignore')
logging.basicConfig(filename='errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

import torch
import pytorch_lightning as pl
import pandas as pd

from _init_model import FinetuneModule, plot_loss_acc, plot_cmatrix

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

def check_version(model_path):
  versions = [folder for folder in os.listdir(model_path) if os.path.isdir(f'{model_path}/{folder}')]
  if len(versions) == 0:
    new_version = "version_0"
  else:
    latest_version = sorted([int(version.split('_')[1]) for version in versions])[-1]
    new_version = f"version_{latest_version + 1}"
    
  save_path = f"{model_path}/{new_version}"
  if not os.path.exists(f"{save_path}"):
    os.makedirs(f"{save_path}")
    
  return new_version, save_path

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
  radar_range = "120km" # 120km | 300km
  interval = "0h" # 0h | 1h | 2h | 3h | 4h | 5h | 6h
  # convnext-s | convnext-b | convnext-l 
  # vit-s      | vit-b      | vit-l 
  # swin-s     | swin-b 
  # effnetv2-s | effnetv2-m
  model_name = "convnext-b"
  model_opt = "pretrained" # pretrained | custom
  classes = 5
  sdepth = 0.2 # 0.0 | 0.1 | 0.2 | 0.3
  past_image_num = 0 # 0 | 6 | 12 | 18
  combined_method = "sum" # sum | concat
  checkpoint = False
  ckpt_version = "version_0"
  
  print(f"Interval: {interval}")
  print(f"Model: {model_name}-{model_opt}")
  print(f"Stochastic depth: {sdepth}")
  print(f"Past image num: {past_image_num}")
  print(f"Combine method: {combined_method}")

  ## For optimizer & scheduler
  optimizer_name = "adamw"  # adam | adamw | sgd
  learning_rate = 5e-5      # 1e-3 | 1e-4  | 5e-5
  weight_decay = 1e-8       # 0    | 1e-8 
  scheduler_name = "cd"     # none | cd    | cdwr  
  
  print(f"Optimizer: {optimizer_name}")
  print(f"Learning rate: {learning_rate}")
  print(f"Weight decay: {weight_decay}")
  print(f"Scheduler: {scheduler_name}\n")

  ## For training loop
  batch_size = 128 # 32 | 64 | 128 | 256
  epochs = 200
  epoch_ratio = 0.5 # Check val every percentage of an epoch
  label_smoothing = 0.1
  
  print(f"Batch size: {batch_size}")
  print(f"Epoch: {epochs}")
  print(f"Epoch ratio: {epoch_ratio}")
  print(f"Label smoothing: {label_smoothing}\n")

  ## For callbacks
  monitor_value = "val_acc" # val_acc | val_loss
  patience = 22
  min_delta = 1e-4 # 1e-4 | 5e-4
  min_epochs = 41 # 0 | 21 | 41 | 61

  print(f"Monitor value: {monitor_value}")
  print(f"Patience: {patience}")
  print(f"Min delta: {min_delta}")
  print(f"Min epoch: {min_epochs}\n")

  # Combine all settings
  model_settings = {'radar_range': radar_range,
                    'interval': interval,
                    'model_name': model_name, 
                    'model_opt': model_opt,
                    'classes': classes,
                    'sdepth': sdepth,
                    'past_image_num': past_image_num,
                    'combined_method': combined_method}
  
  optimizer_settings = {'optimizer_name': optimizer_name, 
                        'learning_rate': learning_rate,
                        'weight_decay': weight_decay, 
                        'scheduler_name': scheduler_name}
  
  loop_settings = {'batch_size': batch_size, 
                   'epochs': epochs,
                   'label_smoothing': label_smoothing}
  
  callback_settings = {'monitor_value': monitor_value, 
                       'patience': patience,
                       'min_delta': min_delta,
                       'min_epochs': min_epochs}
  
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
  
  if not os.path.exists(f"{result_path}/checkpoint/{radar_range}"):
    os.makedirs(f"{result_path}/checkpoint/{radar_range}/")    
  if not os.path.exists(f"{result_path}/checkpoint/{radar_range}/{interval}"):
    os.makedirs(f"{result_path}/checkpoint/{radar_range}/{interval}")
  if not os.path.exists(f"{result_path}/checkpoint/{radar_range}/{interval}/{model_name}-{model_opt}"):
    os.makedirs(f"{result_path}/checkpoint/{radar_range}/{interval}/{model_name}-{model_opt}")
  model_path = f"{result_path}/checkpoint/{radar_range}/{interval}/{model_name}-{model_opt}"
  
  # Load a checkpoint
  if checkpoint:
    save_path = f"{model_path}/{ckpt_version}"
    print(f"Checkpoint version: {ckpt_version}")
    
    # Init module and trainer
    module_test = FinetuneModule.load_from_checkpoint(f"{model_path}/{ckpt_version}/best_model.ckpt", 
                                                      model_settings=model_settings,
                                                      optimizer_settings=optimizer_settings, 
                                                      loop_settings=loop_settings)

    trainer = pl.Trainer(accelerator=accelerator, 
                         devices=devices, 
                         strategy=strategy,
                         logger=False,
                         enable_checkpointing=False)
    
    try:
      # Evaluation
      start_time = time.time()
      trainer.test(module_test)
      end_time = time.time()
      print(f"Evaluation time: {end_time - start_time} seconds")
      
      # image_save_info = {"range": radar_range, "interval": interval, "model": model_name, 'save_path': save_path}
      
      result_log = pd.read_csv(f"{save_path}/metrics.csv")
      test_loss, tess_acc, best_epoch = plot_loss_acc(result_log, monitor_value, min_delta)
      print(f"Best epoch [{monitor_value}]: {best_epoch}")
      
      precision, recall, f1 = plot_cmatrix(module_test.label_list, module_test.pred_list)
      print(f"Precision:{precision}\nRecall: {recall}\nF1: {f1}")
      
    except Exception as e:
      print(e)
      logging.error(e, exc_info=True)
  
  # Train a new model
  else:
    new_version, save_path = check_version(model_path)
    print(f"New version: {new_version}")
    
    # Logger
    logger = pl.loggers.CSVLogger(save_dir=f"{result_path}/checkpoint/{radar_range}/{interval}",
                                  name=f"{model_name}-{model_opt}",
                                  version=new_version,)

    # Callbacks
    if monitor_value == "val_acc": monitor_mode = "max"
    elif monitor_value == "val_loss": monitor_mode = "min" 
    
    early_stopping_callback = pl.callbacks.EarlyStopping(monitor=monitor_value,
                                                         mode=monitor_mode,
                                                         patience=patience,
                                                         min_delta=min_delta,
                                                         verbose=True,)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor=monitor_value,
                                                       mode=monitor_mode,
                                                       save_top_k=1,
                                                       filename='best_model',
                                                       dirpath=save_path,
                                                       verbose=True,)
    
    # Init module and trainer
    module = FinetuneModule(model_settings=model_settings, 
                            optimizer_settings=optimizer_settings, 
                            loop_settings=loop_settings,
                            save_path=save_path)

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
    
    # Training loop
    try:
      train_start_time = time.time()
      trainer.fit(module)
      train_end_time = time.time() - train_start_time
      print(f"Training time: {train_end_time} seconds")
      
      # Evaluation
      module_test = FinetuneModule.load_from_checkpoint(f"{save_path}/best_model.ckpt", 
                                                        model_settings=model_settings,
                                                        optimizer_settings=optimizer_settings, 
                                                        loop_settings=loop_settings)
      test_start_time = time.time()
      trainer.test(module_test)
      test_end_time = time.time() - test_start_time
      print(f"Evaluation time: {test_end_time} seconds")

      image_save_info = {"range": radar_range, "interval": interval, "model": model_name, 'save_path': save_path}
      
      result_log = pd.read_csv(f"{save_path}/metrics.csv")
      test_loss, tess_acc, best_epoch = plot_loss_acc(result_log, monitor_value, min_delta, image_save_info)
      print(f"Best epoch [{monitor_value}]: {best_epoch}")
      
      precision, recall, f1 = plot_cmatrix(module_test.label_list, module_test.pred_list, image_save_info)
      print(f"Precision:{precision}\nRecall: {recall}\nF1: {f1}")
      
      # Write down hyperparameters and results
      with open(f"{save_path}/notes.txt", 'w') as file:
        file.write('### Hyperparameters ###\n')
        file.write(f'Model settings: {model_settings}\n')
        file.write(f'Optimizer settings: {optimizer_settings}\n')
        file.write(f'Loop settings: {loop_settings}\n')
        file.write(f'Callback settings: {callback_settings}\n\n')

        file.write('### Results ###\n')
        file.write(f"Test loss: {test_loss}\n")
        file.write(f"Test accuracy: {tess_acc}\n")
        file.write(f"Precision: {precision}\nRecall: {recall}\nF1: {f1}\n")
        file.write(f"Best epoch [{monitor_value}]: {best_epoch}\n")
        file.write(f"Training time: {train_end_time} seconds\n")
        file.write(f"Version: {new_version}\n")
        
    except Exception as e:
      print(e)
      logging.error(e, exc_info=True)
      # if os.path.exists(f'{save_path}'):
      #   shutil.rmtree(f'{save_path}')

  