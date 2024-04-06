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
def generate_image(metadata_chunk):    
    for _, row in metadata_chunk.iterrows():
        if row['generated'] == "True" or row['generated'] == "Error":
            continue
        timestamp = row['timestamp']
        try:
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
            plt.savefig(f"image/unlabeled1/{timestamp}.jpg", dpi=150, bbox_inches='tight')
            
            print(f"{timestamp} - Done")
            
            # Close and delete to release memory
            plt.close()
            del display, grid_data, data
        except Exception as e:
            with open(f'image/unlabeled1/{timestamp}.txt', 'w') as f: 
                f.write('error')
            logging.error(e, exc_info=True)
            continue

def update_metadata():
    old_metadata = pd.read_csv("metadata.csv")
    
    if 'generated' in old_metadata.columns:
        old_metadata.drop(columns="generated", inplace=True)
    
    files = [file for file in os.listdir("image/unlabeled1")]
    timestamps = [file[:-4] for file in files]
    generated = ["True" if file.endswith('.jpg') else "Error" for file in files]
    new_metadata = pd.DataFrame({'timestamp': timestamps, 'generated': generated})
    
    updated_metadata = pd.merge(old_metadata, new_metadata, on='timestamp', how='outer')
    updated_metadata.to_csv("metadata.csv", index=False)

def move_alternate_files():
    source_folder = "image/unlabeled1"
    destination_folder = "image/unlabeled2"
    
    files = os.listdir(source_folder)

    # Iterate over the files and copy every other file
    for i, file_name in enumerate(files):
        if i % 2 == 0:  # Every other file
            source_path = os.path.join(source_folder, file_name)
            destination_path = os.path.join(destination_folder, file_name)
            shutil.move(source_path, destination_path)
            print(f"Copied '{file_name}' to '{destination_folder}'")

if __name__ == '__main__':    
    # view_sample_images()
    
    num_processes = 20
    chunk_size = 100 * num_processes
    
    if os.path.exists("image/unlabeled1"):
        shutil.rmtree("image/unlabeled1")
        shutil.rmtree("image/unlabeled2")
    else:
        os.makedirs("image/unlabeled1")
        os.makedirs("image/unlabeled2")
    
    try:
        counter = 0
        update_metadata()
        # Use multiprocessing to iterate over the metadata 
        with mp.Pool(processes=num_processes) as pool:
            metadata_chunks = pd.read_csv("metadata.csv", chunksize=chunk_size)
            for chunk in metadata_chunks:
                sub_metadata_chunks = np.array_split(chunk, num_processes)
                
                start_time = time.time()
                pool.map(generate_image, sub_metadata_chunks)
                update_metadata()
                end_time = time.time() - start_time
                
                counter += 1
                print(f"### Chunk: {counter} | Time: {end_time} ###")    
    except Exception as e:
        # If crash due to lack of memory, restart the process (progress is saved)
        update_metadata()
        print(e)
        logging.error(e, exc_info=True)
        
    move_alternate_files()
        
