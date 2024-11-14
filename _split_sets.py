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

def split_df(radar_range, interval, past_image_num, seed=42):
    labels = pd.read_csv(f"image/labels_{radar_range}.csv")
    full_set = labels[['image_name', 'range', f'label_{interval}']]
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
    
    # Generate label files
    labels = pd.read_csv("metadata.csv")[[
            'timestamp_0h', 'range', 'label_0h',
            'label_1h', 'label_2h','label_3h',
            'label_4h', 'label_5h','label_6h'
        ]]

    labels = labels.rename(columns={'timestamp_0h': 'image_name'})
    labels = labels.replace('clear', '0')
    labels = labels.replace('heavy_rain', '1')
    labels = labels.replace('light_rain', '2')
    labels = labels.replace('moderate_rain', '3')
    labels = labels.replace('very_heavy_rain', '4')
    labels['image_name'] = labels['image_name'].astype(str) + ".jpg"
    
    labels_120km = (labels[labels['range'] == '120km']).reset_index(drop=True)
    labels_120km.to_csv("image/labels_120km.csv", index=False)
    labels_300km = (labels[labels['range'] == '300km']).reset_index(drop=True)
    labels_300km.to_csv("image/labels_300km.csv", index=False)
    
    intervals = ["0h", "1h", "2h", "3h", "4h", "5h", "6h"]
    past_image_num = 6 # 6 | 12 | 18
    for interval in intervals:
        split_df("120km", interval, past_image_num)
        split_df("300km", interval, past_image_num)
        
