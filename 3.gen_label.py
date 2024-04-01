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
import matplotlib.pyplot as plt
import pyart
from memory_profiler import profile

# Set ENV to be 'local', 'server' or 'colab'
ENV = "local".lower()

if ENV == "server":
  data_path = "data/NhaBe"
elif ENV == "server":
  data_path = "data"
elif ENV == "colab":
    from google.colab import drive
    drive.mount('/content/drive')
    # %cd drive/MyDrive/Coding/
    data_path = "data/NhaBe"

# View a sample radar image in raw and coordinate forms
def view_sample_image():
    metadata = pd.read_csv("metadata.csv")
    
    data = pyart.io.read_sigmet(f"{data_path}/{metadata['path'].tolist()[0]}")
    grid_data = pyart.map.grid_from_radars(
        data,
        grid_shape=(1, 1388, 500),
        grid_limits=((0, 1), (-300000, 300000), (-300000, 300000)),
    )

    print("Fields available:")
    print(grid_data.fields.keys(), "\n")

    print("Reflectivity values:")
    reflectivity = grid_data.fields['reflectivity']['data']
    print(reflectivity)
    print(reflectivity.shape, "\n")

    print("Non-masked reflectivity values:")
    # Use compress() to remove all masked values (white pixels)
    reflectivity = np.array(reflectivity.compressed())
    print(reflectivity, "\n")
    
    grid_display = pyart.graph.GridMapDisplay(grid_data)
    grid_display.plot_grid('reflectivity', cmap='pyart_HomeyerRainbow')
    plt.savefig("image/3.sample_image.jpg", dpi=150)
    plt.clf()
    
    plt.figure(figsize=(6, 6))
    _, _, _ = plt.hist(reflectivity, color='skyblue', edgecolor='black')
    plt.xlabel('DBz')
    plt.ylabel('Frequency')
    plt.title('Reflectivity value distribution')
    plt.savefig("image/3.reflectivity_value_distribution.jpg", dpi=150)

def find_future_images(interval=7200):
    metadata = pd.read_csv("metadata.csv")
    metadata['timestamp'] = pd.to_datetime(metadata['timestamp'], format="%Y-%m-%d %H-%M-%S")

    for idx, row in metadata.iterrows():
        current_time = row['timestamp']
        future_metadata = metadata[(metadata['timestamp'] - current_time > pd.Timedelta(interval, "s")) &
                                     (metadata['timestamp'] - current_time < pd.Timedelta(interval + 1800, "s"))].head(1)

        if future_metadata.empty:
            metadata.loc[idx, ['future_path']] = "NotAvail"
            metadata.loc[idx, ['future_timestamp']] = "NotAvail"
        else:
            metadata.loc[idx, ['future_path']] = future_metadata['path'].tolist()[0]
            metadata.loc[idx, ['future_timestamp']] = future_metadata['timestamp'].tolist()[0]

    metadata['timestamp'] = metadata['timestamp'].astype(str).str.replace(':', '-')
    metadata['future_timestamp'] = metadata['future_timestamp'].astype(str).str.replace(':', '-')
    metadata.to_csv("metadata.csv", index=False)
            
def image_labeling(metadata_chunk, weight_set=[0.001, 0.333, 0.333, 0.333]):
    metadata_chunk['timestamp'] = pd.to_datetime(metadata_chunk['timestamp'], format="%Y-%m-%d %H-%M-%S")
    
    for idx, row in metadata_chunk.iterrows():
        if type(row['future_label']) is str or row['future_path'] == "NotAvail":
            continue
        
        data = pyart.io.read_sigmet(f"{data_path}/{row['future_path']}")
        data.fields['reflectivity']['data'] = data.fields['reflectivity']['data'].astype(np.float16)
        
        grid_data = pyart.map.grid_from_radars(
            data,
            grid_shape=(1, 500, 500),
            grid_limits=((0, 1), (-250000, 250000), (-250000, 250000)),
        )
        
        reflectivity = np.array(grid_data.fields['reflectivity']['data'].compressed())
        weights = []
        for ele in reflectivity:
            if ele < 30:
                weights += [weight_set[0]]
            elif ele < 52:
                weights += [weight_set[1]]
            elif ele < 63:
                weights += [weight_set[2]]
            else:
                weights += [weight_set[3]]
        
        avg_reflectivity = np.average(reflectivity, weights=weights)
        if avg_reflectivity < 30:
            label = "clear"
        elif avg_reflectivity < 52:
            label = "light_rain"
        elif avg_reflectivity < 63:
            label = "heavy_rain"
        else:
            label = "storm"
            
        # print(f"{row['timestamp']} - Average reflectivity: {avg_reflectivity} | Label: {label}")
        
        metadata_chunk.loc[idx, ['future_avg_reflectivity']] = avg_reflectivity
        metadata_chunk.loc[idx, ['future_label']] = label
        
        # Close and delete to release memory
        del weights
        del grid_data
        del data
        
    metadata_chunk['timestamp'] = metadata_chunk['timestamp'].astype(str).str.replace(':', '-')
    return metadata_chunk
        
def update_metadata(new_metadata):
    updated_metadata = pd.read_csv("metadata.csv")
    
    updated_metadata.loc[new_metadata.index, 'future_avg_reflectivity'] = new_metadata['future_avg_reflectivity'].tolist()
    updated_metadata.loc[new_metadata.index, 'future_label'] = new_metadata['future_label'].tolist()
    
    updated_metadata.to_csv("metadata.csv", index=False)        
        
if __name__ == '__main__':
    # find_future_images(interval=7200)
    
    num_processes = 4
    chunk_size = 400
    
    counter = 0
    try:
        # Use multiprocessing to iterate over the metadata 
        with mp.Pool(processes=num_processes) as pool:
            metadata_chunks = pd.read_csv("metadata.csv", chunksize=chunk_size)
            for chunk in metadata_chunks:
                start_time = time.time()
                results = pool.map(image_labeling, np.array_split(chunk, num_processes))
                new_metadata = pd.concat(results)
                update_metadata(new_metadata)
                end_time = time.time() - start_time

                counter += 1
                print(f"### Chunk: {counter} | Time: {end_time} ###")
        
        print(f"Time: {end_time}")
    except Exception as e:
        # If crash due to lack of memory, restart the process (progress is saved)
        print(e)
        logging.error(e, exc_info=True)
        
        
        
        