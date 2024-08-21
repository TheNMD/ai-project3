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

if ENV == "local" or ENV == "server":
  data_path = "data"
elif ENV == "colab":
    from google.colab import drive
    drive.mount('/content/drive')
    # %cd drive/MyDrive/Coding/
    data_path = "data/NhaBe"
    
def split_df(interval, seed=42):
    metadata = pd.read_csv("image/labels.csv")
    
    big_df = metadata[['image_name', 'type', f'label_{interval}']]
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
    
    with open(f'image/{interval}_summary.txt', 'w') as file:
        file.write("### Train set ###\n")
        frequency_train = train_set[f'label_{interval}'].value_counts()
        file.write(f"{frequency_train}\n\n")
        
        file.write("### Val set ###\n")
        frequency_val = val_set[f'label_{interval}'].value_counts()
        file.write(f"{frequency_val}\n\n")
        
        file.write("### Test set ###\n")
        frequency_test = test_set[f'label_{interval}'].value_counts()
        file.write(f"{frequency_test}\n\n")
    
    train_set.to_csv(f"image/{interval}_train.csv", index=False)
    val_set.to_csv(f"image/{interval}_val.csv", index=False)
    test_set.to_csv(f"image/{interval}_test.csv", index=False)

if __name__ == '__main__':
    print("Python version: ", sys.version)
    print("Ubuntu version: ", platform.release())
    
    # 0 | 3600 | 7200 | 10800 | 14400 | 18000 | 21600 | 43200
    interval = 10800 
    split_df(interval)
        
