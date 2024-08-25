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

def split_df(radar_range, interval, past_image_num, seed=42):
    metadata = pd.read_csv(f"image/labels_{radar_range}.csv")
    full_set = metadata[['image_name', 'range', f'label_{interval}']]
    full_set = full_set[past_image_num:]
    full_set = full_set[full_set[f'label_{interval}'] != 'NotAvail']
    full_set = full_set.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    split_idx1 = int(len(full_set) * 0.8)
    split_idx2 = int(len(full_set) * 0.1)

    # Split the DataFrame into three parts
    train_set = full_set.iloc[:split_idx1]
    val_set = full_set.iloc[split_idx1:(split_idx1 + split_idx2)]
    test_set = full_set.iloc[(split_idx1 + split_idx2):]

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
    # 6 | 12 | 18
    past_image_num = 6
    split_df("100km", interval, past_image_num)
    split_df("250km", interval, past_image_num)
        
