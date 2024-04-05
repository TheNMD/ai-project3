import os
import time
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(filename='errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

import numpy as np
import pandas as pd
import pyart

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

# Iterate over all years -> months -> days -> files
# Create a df to contain paths and timestamps for all files
def create_metadata(year):
  for month in sorted(os.listdir(f"{data_path}/{year}")):
    if os.path.exists(f"metadata/metadata_{year}_{month}.csv"): continue
    
    paths = []
    timestamps = []
    for day in sorted(os.listdir(f"{data_path}/{year}/{month}")):
      for file in sorted(os.listdir(f"{data_path}/{year}/{month}/{day}")):
        path =  f"{year}/{month}/{day}/{file}"
        paths += [path]
        
        data = pyart.io.read_sigmet(f"{data_path}/{path}")
        timestamp = str(pyart.util.datetime_from_radar(data)).replace(':', '-')
        timestamps += [timestamp]
        
        print(f"{timestamp}")

    metadata = pd.DataFrame(list(zip(paths, timestamps)), columns=['path', 'timestamp'])
    metadata['generated'] = False
    metadata['future_path'] = np.nan
    metadata['future_timestamp'] = np.nan
    metadata['future_label'] = np.nan
    metadata['future_avg_reflectivity'] = np.nan
    
    metadata = metadata.sort_values(by='timestamp').reset_index(drop=True)
    metadata.to_csv(f"metadata/metadata_{year}_{month}.csv", index=False)

def update_metadata():
  filenames = sorted(os.listdir("metadata"))
  metadata_list = [pd.read_csv(f"metadata/{name}") for name in filenames]
  metadata = pd.concat(metadata_list)
  metadata = metadata.sort_values(by='timestamp').reset_index(drop=True)
  
  duplicate_mask = metadata.duplicated(subset=['timestamp'], keep=False)
  duplicate_indices = metadata[duplicate_mask].index.tolist()
  metadata_cleaned = metadata.drop(duplicate_indices).reset_index(drop=True)
  
  metadata_cleaned.to_csv("metadata.csv", index=False)

if __name__ == '__main__':
  years = [2020, 2021, 2022, 2023]
  num_processes = len(years)
  
  if not os.path.exists("metadata"):
    os.makedirs("metadata")
  
  try:
      update_metadata()
      # Use multiprocessing to iterate over the metadata 
      with mp.Pool(processes=num_processes) as pool:
        start_time = time.time()
        pool.map(create_metadata, years)
        end_time = time.time() - start_time
        update_metadata()
        
        print(f"Time: {end_time}")
  except Exception as e:
      # If crash due to lack of memory, restart the process (progress is saved)
      update_metadata()
      print(e)
      logging.error(e, exc_info=True)
