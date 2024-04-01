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

# View a sample radar image in raw and coordinate forms
def view_sample_images():
    metadata = pd.read_csv("metadata.csv")
    
    # Raw data
    print("### Raw data ###")
    data = pyart.io.read_sigmet(f"{data_path}/{metadata['path'].tolist()[0]}")

    print(metadata['timestamp'].tolist()[0])

    print("Fields available:")
    print(data.fields.keys(), "\n")

    print("Reflectivity values:")
    reflectivity = data.fields['reflectivity']['data']
    print(reflectivity)
    print(reflectivity.shape, "\n")

    display = pyart.graph.RadarDisplay(data)
    display.plot_ppi("reflectivity")
    plt.savefig("image/2.sample_raw_image.jpg", dpi=150)
    plt.clf()
    
    # Convert raw data to grid data
    print("### Grid data ###")
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

    grid_display = pyart.graph.GridMapDisplay(grid_data)
    grid_display.plot_grid('reflectivity', cmap='pyart_HomeyerRainbow')
    plt.savefig("image/2.sample_grid_image.jpg", dpi=150)

# @profile
def image_generating(metadata_chunk):    
    for _, row in metadata_chunk.iterrows():
        if row['generated'] == True:
            continue
        
        data = pyart.io.read_sigmet(f"{data_path}/{row['path']}")
        data.fields['reflectivity']['data'] = data.fields['reflectivity']['data'].astype(np.float16)
        
        grid_data = pyart.map.grid_from_radars(
            data,
            grid_shape=(1, 500, 500),
            grid_limits=((0, 1), (-250000, 250000), (-250000, 250000)),
        )

        display = pyart.graph.GridMapDisplay(grid_data)
        display.plot_grid('reflectivity', cmap='pyart_HomeyerRainbow', colorbar_flag=False)

        plt.title('')
        plt.axis('off')
        plt.savefig(f"image/unlabeled/{row['timestamp']}.jpg", dpi=150, bbox_inches='tight')
        
        # print(f"{row['timestamp']} - Done")
        
        # Close and delete to release memory
        plt.close()
        del display, grid_data, data

def update_metadata():
    old_metadata = pd.read_csv("metadata.csv")
    
    if 'generated' in old_metadata.columns:
        old_metadata.drop(columns="generated", inplace=True)
    
    image_files = [file[:-4] for file in os.listdir(f"image/unlabeled") if file.endswith('.jpg')]
    generated = [True for _ in image_files]
    new_metadata = pd.DataFrame({'timestamp': image_files, 'generated': generated})
    
    updated_metadata = pd.merge(old_metadata, new_metadata, on='timestamp', how='outer')
    updated_metadata.to_csv("metadata.csv", index=False)

if __name__ == '__main__':    
    # view_sample_images()
    
    num_processes = 16
    chunk_size = 100 * num_processes
    
    counter = 0
    try:
        update_metadata()
        # Use multiprocessing to iterate over the metadata 
        with mp.Pool(processes=num_processes) as pool:
            metadata_chunks = pd.read_csv("metadata.csv", chunksize=chunk_size)
            for chunk in metadata_chunks:
                sub_metadata_chunks = np.array_split(chunk, num_processes)
                
                start_time = time.time()
                pool.map(image_generating, sub_metadata_chunks)
                end_time = time.time() - start_time
                
                counter += 1
                print(f"### Chunk: {counter} | Time: {end_time} ###")    
    except Exception as e:
        # If crash due to lack of memory, restart the process (progress is saved)
        print(e)
        logging.error(e, exc_info=True)
        
