import os, sys, platform, time
import multiprocessing as mp
import warnings, logging
warnings.filterwarnings('ignore')
logging.basicConfig(filename='errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

import pyart
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

def find_future_images(interval):
    interval_num = int(interval[:-1]) * 3600
    sub_metadata = []
    for i in ["120km", "300km"]:
        metadata = pd.read_csv("metadata.csv")
        metadata = (metadata[metadata['range'] == i]).reset_index(drop=True)
        metadata['timestamp_0h'] = pd.to_datetime(metadata['timestamp_0h'], format="%Y-%m-%d %H-%M-%S")
        
        timestamp_col = f"timestamp_{interval}"
        label_col = f"label_{interval}"

        if timestamp_col not in metadata.columns: metadata[timestamp_col] = np.nan
        if label_col not in metadata.columns: metadata[label_col] = np.nan

        for idx, row in metadata.iterrows():
            if type(row[timestamp_col]) is str:
                continue
            
            current_time = row['timestamp_0h']
            time_difference = metadata['timestamp_0h'] - current_time
            future_metadata = metadata[(time_difference >= pd.Timedelta(interval_num - 60, "s")) &
                                       (time_difference <= pd.Timedelta(interval_num + 600, "s"))].head(1)
            
            if future_metadata.empty:
                metadata.loc[idx, [timestamp_col]] = "NotAvail"
                metadata.loc[idx, [label_col]] = "NotAvail"
            else:
                future_timestamp = future_metadata['timestamp_0h'].tolist()[0]
                metadata.loc[idx, [timestamp_col]] = future_timestamp
                future_label =  metadata.loc[metadata['timestamp_0h'] == future_timestamp, 'label_0h'].tolist()[0]
                metadata.loc[idx, [label_col]] = future_label
            
            # print(f"{current_time} - Done")

        metadata['timestamp_0h'] = metadata['timestamp_0h'].astype(str).str.replace(':', '-')
        metadata[timestamp_col] = metadata[timestamp_col].astype(str).str.replace(':', '-')
        sub_metadata += [metadata]
        
    merged_df = pd.concat(sub_metadata, ignore_index=True)
    merged_df = (merged_df.sort_values(by='timestamp_0h')).reset_index(drop=True)
    merged_df.to_csv(f"metadata_{interval}.csv", index=False)

def update_metadata(intervals):
    metadata = pd.read_csv("metadata.csv")

    for interval in intervals:
        small_metadata = pd.read_csv(f"metadata_{interval}.csv")
        metadata[f'timestamp_{interval}'] = small_metadata[f'timestamp_{interval}']
        metadata[f'label_{interval}'] = small_metadata[f'label_{interval}']
    
    metadata.to_csv("metadata.csv", index=False)
    
if __name__ == '__main__':
    print("Python version: ", sys.version)
    print("Ubuntu version: ", platform.release())
    
    # Label future images
    intervals = ["1h", "2h", "3h", "4h", "5h", "6h", "12h", "24h", "48h"]
    num_processes = len(intervals)
    try:
        # Use multiprocessing to iterate over the metadata
        with mp.Pool(processes=num_processes) as pool:
            start_time = time.time()
            pool.map(find_future_images, intervals)
            end_time = time.time() - start_time

            print(f"Time: {end_time}")
    except Exception as e:
        print(e)
        logging.error(e, exc_info=True)
        
    # Update metadata
    update_metadata(intervals)

        
        
        
        