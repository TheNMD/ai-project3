import os
import shutil
import time
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

def split_df(seed=42):
    metadata = pd.read_csv("metadata_lite.csv")
    
    big_df = metadata.sample(frac=1, random_state=seed).reset_index(drop=True)
    
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
        future_label = row['future_label']
        shutil.copy(f"image/labeled/{future_label}/{timestamp}.jpg", f"image/sets/train/{future_label}/{timestamp}.jpg")

def move_to_val(metadata_chunk):
    for _, row in metadata_chunk.iterrows():
        timestamp = row['timestamp']
        future_label = row['future_label']
        shutil.copy(f"image/labeled/{future_label}/{timestamp}.jpg", f"image/sets/val/{future_label}/{timestamp}.jpg")
        
def move_to_test(metadata_chunk):
    for _, row in metadata_chunk.iterrows():
        timestamp = row['timestamp']
        future_label = row['future_label']
        shutil.copy(f"image/labeled/{future_label}/{timestamp}.jpg", f"image/sets/test/{future_label}/{timestamp}.jpg")

if __name__ == '__main__':    
    # view_sample_images()
    
    num_processes = 4
    chunk_size = 100 * num_processes
    
    if not os.path.exists("image/sets"):
        os.makedirs("image/sets")
        
        os.makedirs("image/sets/train")
        os.makedirs("image/sets/train/clear")
        os.makedirs("image/sets/train/light_rain")
        os.makedirs("image/sets/train/moderate_rain")
        os.makedirs("image/sets/train/heavy_rain")
        os.makedirs("image/sets/train/very_heavy_rain")
        
        os.makedirs("image/sets/val")
        os.makedirs("image/sets/val/clear")
        os.makedirs("image/sets/val/light_rain")
        os.makedirs("image/sets/val/moderate_rain")
        os.makedirs("image/sets/val/heavy_rain")
        os.makedirs("image/sets/val/very_heavy_rain")
        
        os.makedirs("image/sets/test")
        os.makedirs("image/sets/test/clear")
        os.makedirs("image/sets/test/light_rain")
        os.makedirs("image/sets/test/moderate_rain")
        os.makedirs("image/sets/test/heavy_rain")
        os.makedirs("image/sets/test/very_heavy_rain")
    
    train_set, val_set, test_set = split_df()
    
    # Move to train set
    try:
        # Use multiprocessing to iterate over the metadata 
        with mp.Pool(processes=num_processes) as pool:
            start_time = time.time()
            pool.map(move_to_train, np.array_split(train_set, num_processes))
            end_time = time.time() - start_time
    except Exception as e:
        # If crash due to lack of memory, restart the process (progress is saved)
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
        # If crash due to lack of memory, restart the process (progress is saved)
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
        # If crash due to lack of memory, restart the process (progress is saved)
        print(e)
        logging.error(e, exc_info=True)
        
