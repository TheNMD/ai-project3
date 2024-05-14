import os, sys, platform, shutil, time
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(filename='errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

import numpy as np
import pandas as pd

# Set ENV to be 'local', 'server' or 'colab'
ENV = "server".lower()

if ENV == "local":
  data_path = "data/NhaBe"
elif ENV == "server":
  data_path = "data"
elif ENV == "colab":
    from google.colab import drive
    drive.mount('/content/drive')
    # %cd drive/MyDrive/Coding/
    data_path = "data/NhaBe"

# 0 | 3600 | 7200 | 10800 | 14400 | 18000 | 21600 | 43200
interval = 43200 

def split_df(interval, seed=42):
    metadata = pd.read_csv("metadata.csv")
    
    big_df = metadata[['timestamp_0', f'label_{interval}']]
    big_df = big_df[big_df[f'label_{interval}'] != 'NotAvail']
    big_df = big_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    split_idx1 = int(len(big_df) * 0.8)
    split_idx2 = int(len(big_df) * 0.1)

    # Split the DataFrame into three parts
    train_set = big_df.iloc[:split_idx1]
    val_set = big_df.iloc[split_idx1:(split_idx1 + split_idx2)]
    test_set = big_df.iloc[(split_idx1 + split_idx2):]

    train_set.reset_index(drop=True, inplace=True)
    val_set.reset_index(drop=True, inplace=True)
    test_set.reset_index(drop=True, inplace=True)
    
    with open(f'image/labeled/{interval}-temp/train_val_test_summary.txt', 'w') as file:
        file.write("### Train set ###\n")
        frequency_train = train_set[f'label_{interval}'].value_counts()
        file.write(f"{frequency_train}\n\n")
        
        file.write("### Val set ###\n")
        frequency_val = val_set[f'label_{interval}'].value_counts()
        file.write(f"{frequency_val}\n\n")
        
        file.write("### Test set ###\n")
        frequency_test = test_set[f'label_{interval}'].value_counts()
        file.write(f"{frequency_test}\n\n")
    
    return train_set, val_set, test_set

def move_to_train(metadata_chunk):
    for idx, row in metadata_chunk.iterrows():
        timestamp = row['timestamp_0']
        label = row[f'label_{interval}']
        if os.path.exists(f"image/unlabeled1/{timestamp}.jpg"):
            shutil.copy(f"image/unlabeled1/{timestamp}.jpg", f"image/labeled/{interval}-temp/train/{label}/{timestamp}.jpg")
        elif os.path.exists(f"image/unlabeled2/{timestamp}.jpg"):
            shutil.copy(f"image/unlabeled2/{timestamp}.jpg", f"image/labeled/{interval}-temp/train/{label}/{timestamp}.jpg")
        print(idx)

def move_to_val(metadata_chunk):
    for idx, row in metadata_chunk.iterrows():
        timestamp = row['timestamp_0']
        label = row[f'label_{interval}']
        if os.path.exists(f"image/unlabeled1/{timestamp}.jpg"):
            shutil.copy(f"image/unlabeled1/{timestamp}.jpg", f"image/labeled/{interval}-temp/val/{label}/{timestamp}.jpg")
        elif os.path.exists(f"image/unlabeled2/{timestamp}.jpg"):
            shutil.copy(f"image/unlabeled2/{timestamp}.jpg", f"image/labeled/{interval}-temp/val/{label}/{timestamp}.jpg")
        print(idx)
        
def move_to_test(metadata_chunk):
    for idx, row in metadata_chunk.iterrows():
        timestamp = row['timestamp_0']
        label = row[f'label_{interval}']
        if os.path.exists(f"image/unlabeled1/{timestamp}.jpg"):
            shutil.copy(f"image/unlabeled1/{timestamp}.jpg", f"image/labeled/{interval}-temp/test/{label}/{timestamp}.jpg")
        elif os.path.exists(f"image/unlabeled2/{timestamp}.jpg"):
            shutil.copy(f"image/unlabeled2/{timestamp}.jpg", f"image/labeled/{interval}-temp/test/{label}/{timestamp}.jpg")
        print(idx)

if __name__ == '__main__':
    print("Python version: ", sys.version)
    print("Ubuntu version: ", platform.release())
    
    if not os.path.exists(f"image/labeled/{interval}-temp"):
        os.makedirs(f"image/labeled/{interval}-temp")
        shutil.move(f"image/labeled/{interval}_avg_reflectivity_dist.png", f"image/labeled/{interval}-temp/{interval}_avg_reflectivity_dist.png")
        shutil.move(f"image/labeled/{interval}_label_dist.png", f"image/labeled/{interval}-temp/{interval}_label_dist.png")
        shutil.move(f"image/labeled/{interval}_label_dist.txt", f"image/labeled/{interval}-temp/{interval}_label_dist.txt")
                
        os.makedirs(f"image/labeled/{interval}-temp/train")
        os.makedirs(f"image/labeled/{interval}-temp/train/clear")
        os.makedirs(f"image/labeled/{interval}-temp/train/light_rain")
        os.makedirs(f"image/labeled/{interval}-temp/train/moderate_rain")
        os.makedirs(f"image/labeled/{interval}-temp/train/heavy_rain")
        os.makedirs(f"image/labeled/{interval}-temp/train/very_heavy_rain")
        
        os.makedirs(f"image/labeled/{interval}-temp/val")
        os.makedirs(f"image/labeled/{interval}-temp/val/clear")
        os.makedirs(f"image/labeled/{interval}-temp/val/light_rain")
        os.makedirs(f"image/labeled/{interval}-temp/val/moderate_rain")
        os.makedirs(f"image/labeled/{interval}-temp/val/heavy_rain")
        os.makedirs(f"image/labeled/{interval}-temp/val/very_heavy_rain")
        
        os.makedirs(f"image/labeled/{interval}-temp/test")
        os.makedirs(f"image/labeled/{interval}-temp/test/clear")
        os.makedirs(f"image/labeled/{interval}-temp/test/light_rain")
        os.makedirs(f"image/labeled/{interval}-temp/test/moderate_rain")
        os.makedirs(f"image/labeled/{interval}-temp/test/heavy_rain")
        os.makedirs(f"image/labeled/{interval}-temp/test/very_heavy_rain")
    
    train_set, val_set, test_set = split_df(interval)
    
    num_processes = 16
    chunk_size = 100 * num_processes
    
    # Move to train set
    try:
        # Use multiprocessing to iterate over the metadata 
        with mp.Pool(processes=num_processes) as pool:
            start_time = time.time()
            pool.map(move_to_train, np.array_split(train_set, num_processes))
            end_time = time.time() - start_time
    except Exception as e:
        print(e)
        logging.error(e, exc_info=True)
    
    # Move to val set
    try:
        # Use multiprocessing to iterate over the metadata 
        with mp.Pool(processes=num_processes) as pool:
            start_time = time.time()
            pool.map(move_to_val, np.array_split(val_set, num_processes))
            end_time = time.time() - start_time
    except Exception as e:
        print(e)
        logging.error(e, exc_info=True)
    
    # Move to test set
    try:
        # Use multiprocessing to iterate over the metadata 
        with mp.Pool(processes=num_processes) as pool:
            start_time = time.time()
            pool.map(move_to_test, np.array_split(test_set, num_processes))
            end_time = time.time() - start_time
    except Exception as e:
        print(e)
        logging.error(e, exc_info=True)
        
