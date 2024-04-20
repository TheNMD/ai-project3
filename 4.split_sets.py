import os, sys, platform, shutil, time
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(filename='errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

import numpy as np
import pandas as pd
from memory_profiler import profile

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

interval = 0

def split_df(interval, seed=42):
    metadata = pd.read_csv("metadata_lite.csv")
    
    big_df = metadata[['timestamp_0', f'label_{interval}']]
    big_df = big_df[(big_df[f'label_{interval}'] != 'NotAvail') & (big_df[f'label_{interval}'] != 'Error')]
    big_df = big_df.rename(columns={'timestamp_0': 'timestamp', f'label_{interval}': 'label'})
    big_df = big_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    split_idx1 = int(len(big_df) * 0.8)
    split_idx2 = int(len(big_df) * 0.1)

    # Split the DataFrame into three parts
    train_set = big_df.iloc[:split_idx1]
    val_set = big_df.iloc[split_idx1:(split_idx1 + split_idx2)]
    test_set = big_df.iloc[(split_idx1 + split_idx2):]

    # Reset the indices if necessary
    train_set.reset_index(drop=True, inplace=True)
    val_set.reset_index(drop=True, inplace=True)
    test_set.reset_index(drop=True, inplace=True)
    
    return train_set, val_set, test_set

def move_to_train(metadata_chunk):
    for _, row in metadata_chunk.iterrows():
        timestamp = row['timestamp']
        label = row['label']
        shutil.copy(f"image/labeled/{interval}/{label}/{timestamp}.jpg", f"image/labeled/{interval}/train/{label}/{timestamp}.jpg")

def move_to_val(metadata_chunk):
    for _, row in metadata_chunk.iterrows():
        timestamp = row['timestamp']
        label = row['label']
        shutil.copy(f"image/labeled/{interval}/{label}/{timestamp}.jpg", f"image/labeled/{interval}/val/{label}/{timestamp}.jpg")
        
def move_to_test(metadata_chunk):
    for _, row in metadata_chunk.iterrows():
        timestamp = row['timestamp']
        label = row['label']
        shutil.copy(f"image/labeled/{interval}/{label}/{timestamp}.jpg", f"image/labeled/{interval}/test/{label}/{timestamp}.jpg")

if __name__ == '__main__':
    print("Python version: ", sys.version)
    print("Ubuntu version: ", platform.release())
    
    num_processes = 16
    chunk_size = 100 * num_processes
    
    if not os.path.exists("image/sets"):
        os.makedirs("image/sets")
        os.makedirs(f"image/labeled/{interval}")
        
        os.makedirs(f"image/labeled/{interval}/train")
        os.makedirs(f"image/labeled/{interval}/train/clear")
        os.makedirs(f"image/labeled/{interval}/train/light_rain")
        os.makedirs(f"image/labeled/{interval}/train/moderate_rain")
        os.makedirs(f"image/labeled/{interval}/train/heavy_rain")
        os.makedirs(f"image/labeled/{interval}/train/very_heavy_rain")
        
        os.makedirs(f"image/labeled/{interval}/val")
        os.makedirs(f"image/labeled/{interval}/val/clear")
        os.makedirs(f"image/labeled/{interval}/val/light_rain")
        os.makedirs(f"image/labeled/{interval}val/moderate_rain")
        os.makedirs(f"image/labeled/{interval}/val/heavy_rain")
        os.makedirs(f"image/labeled/{interval}/val/very_heavy_rain")
        
        os.makedirs(f"image/labeled/{interval}/test")
        os.makedirs(f"image/labeled/{interval}/clear")
        os.makedirs(f"image/labeled/{interval}/light_rain")
        os.makedirs(f"image/labeled/{interval}/moderate_rain")
        os.makedirs(f"image/labeled/{interval}/heavy_rain")
        os.makedirs(f"image/labeled/{interval}/very_heavy_rain")
    
    train_set, val_set, test_set = split_df(interval)
    
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
        
