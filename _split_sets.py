import os, sys, platform
import warnings, logging
warnings.filterwarnings('ignore')
logging.basicConfig(filename='errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

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

# clear = 0
# heavy_rain = 1
# light_rain = 2
# moderate_rain = 3
# very_heavy_rain = 4

def split_df(radar_range, interval, seed=42):
    metadata = pd.read_csv("image/labels.csv")
    big_df = metadata[['image_name', 'range', f'label_{interval}']]
    big_df = big_df[big_df['range'] == radar_range]
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
    
    with open(f'image/sets/{radar_range}_{interval}_summary.txt', 'w') as file:
        file.write("### Train set ###\n")
        frequency_train = train_set[f'label_{interval}'].value_counts()
        file.write(f"{frequency_train}\n\n")
        
        file.write("### Val set ###\n")
        frequency_val = val_set[f'label_{interval}'].value_counts()
        file.write(f"{frequency_val}\n\n")
        
        file.write("### Test set ###\n")
        frequency_test = test_set[f'label_{interval}'].value_counts()
        file.write(f"{frequency_test}\n\n")
    
    train_set.to_csv(f"image/sets/{radar_range}_{interval}_train.csv", index=False)
    val_set.to_csv(f"image/sets/{radar_range}_{interval}_val.csv", index=False)
    test_set.to_csv(f"image/sets/{radar_range}_{interval}_test.csv", index=False)

if __name__ == '__main__':
    print("Python version: ", sys.version)
    print("Ubuntu version: ", platform.release())
    
    if not os.path.exists("image/sets"):
        os.makedirs("image/sets")
    
    # 0 | 3600 | 7200 | 10800 | 14400 | 18000 | 21600 | 43200
    interval = 10800 
    # split_df("100km", interval)
    split_df("250km", interval)
        
