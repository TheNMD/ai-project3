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
    for i in ["120km", "300km"]:
        metadata = pd.read_csv("metadata.csv")
        metadata = metadata[metadata['range'] == i]
        metadata['timestamp_0'] = pd.to_datetime(metadata['timestamp_0'], format="%Y-%m-%d %H-%M-%S")
        
        timestamp_col = f"timestamp_{interval}"
        label_col = f"label_{interval}"

        if timestamp_col not in metadata.columns: metadata[timestamp_col] = np.nan
        if label_col not in metadata.columns: metadata[label_col] = np.nan

        for idx, row in metadata.iterrows():
            if type(row[timestamp_col]) is str:
                continue
                    
            current_time = row['timestamp_0']
            future_metadata = metadata[(metadata['timestamp_0'] - current_time >= pd.Timedelta(interval - 60, "s")) &
                                    (metadata['timestamp_0'] - current_time <= pd.Timedelta(interval + 600, "s"))].head(1)
            
            if future_metadata.empty:
                metadata.loc[idx, [timestamp_col]] = "NotAvail"
                metadata.loc[idx, [label_col]] = "NotAvail"
            else:
                future_timestamp = future_metadata['timestamp_0'].tolist()[0]
                metadata.loc[idx, [timestamp_col]] = future_timestamp
                metadata.loc[idx, [label_col]] = metadata.loc[metadata['timestamp_0'] == future_timestamp, 'label_0'].tolist()[0]
            
            print(current_time)

        metadata['timestamp_0'] = metadata['timestamp_0'].astype(str).str.replace(':', '-')
        metadata[timestamp_col] = metadata[timestamp_col].astype(str).str.replace(':', '-')
        metadata.to_csv(f"metadata_{i}_{interval}.csv", index=False)
        
    df_120km = pd.read_csv(f"metadata_120km_{interval}.csv")
    df_300km = pd.read_csv(f"metadata_300km_{interval}.csv")
    merged_df = pd.concat([df_120km, df_300km], ignore_index=True)
    merged_df = merged_df.sort_values(by='timestamp_0').reset_index(drop=True)
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
    timestamps = [3600, 7200, 10800, 14400, 18000, 21600, 43200, 86400, 172800]
    num_processes = len(timestamps)
    try:
        # Use multiprocessing to iterate over the metadata
        with mp.Pool(processes=num_processes) as pool:
            start_time = time.time()
            pool.map(find_future_images, timestamps)
            end_time = time.time() - start_time

            print(f"Time: {end_time}")
    except Exception as e:
        print(e)
        logging.error(e, exc_info=True)
        
    # Update metadata
    update_metadata(timestamps)


        
        
        
        