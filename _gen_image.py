import os, sys, platform, time
import multiprocessing as mp
import warnings, logging
warnings.filterwarnings('ignore')
logging.basicConfig(filename='errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

import pyart
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Set ENV to be 'local', 'server' or 'colab'
ENV = "server".lower()

if ENV == "local" or ENV == "server":
  data_path = "data"
elif ENV == "colab":
    from google.colab import drive
    drive.mount('/content/drive')
    # %cd drive/MyDrive/Coding/
    data_path = "data/NhaBe"

def generate_image(metadata_chunk):    
    for _, row in metadata_chunk.iterrows():
        if row['generated'] == "True" or row['generated'] == "Error":
            continue
        
        timestamp = row['timestamp_0']
        try:
            data = pyart.io.read_sigmet(f"{data_path}/{row['path']}")
            data.fields['reflectivity']['data'] = data.fields['reflectivity']['data'].astype(np.float16)
            
            if row['range'] == "120km":
                radar_range = 120000
            else:
                radar_range = 300000
            
            grid_data = pyart.map.grid_from_radars(data,
                                                   grid_shape=(1, 500, 500),
                                                   grid_limits=((0, 1), 
                                                                (-radar_range, radar_range), 
                                                                (-radar_range, radar_range)),)

            display = pyart.graph.GridMapDisplay(grid_data)
            display.plot_grid('reflectivity', cmap='pyart_HomeyerRainbow', colorbar_flag=False)

            plt.title('')
            plt.axis('off')
            plt.savefig(f"image/all/{timestamp}.jpg", dpi=150, bbox_inches='tight')
            
            print(f"{timestamp} - Done")
            
            # Close and delete to release memory
            plt.close()
            del display, grid_data, data
        except Exception as e:
            with open(f'image/all/{timestamp}.txt', 'w') as f: 
                f.write('error')
            logging.error(e, exc_info=True)
            continue

def update_metadata():
    old_metadata = pd.read_csv("metadata.csv")
    old_metadata = old_metadata[['path', 'timestamp_0']]
    
    files = [file for file in os.listdir("image/all")]
    timestamps = [file[:-4] for file in files]
    generated = ["True" if file.endswith('.jpg') else "Error" for file in files]
    new_metadata = pd.DataFrame({'timestamp_0': timestamps, 'generated': generated})
    
    updated_metadata = pd.merge(old_metadata, new_metadata, on='timestamp_0', how='outer')
    updated_metadata.to_csv("metadata.csv", index=False)

if __name__ == '__main__':
    print("Python version: ", sys.version)
    print("Ubuntu version: ", platform.release())
        
    # view_sample_images()
    
    num_processes = 16
    chunk_size = 100 * num_processes
    
    if not os.path.exists("image/all"):
        os.makedirs("image/all")
    
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
        update_metadata()
        print(e)
        logging.error(e, exc_info=True)
    
    metadata = pd.read_csv("metadata.csv")
    metadata = metadata[metadata['generated'] != 'Error']
    metadata.reset_index(drop=True, inplace=True)
    metadata.to_csv("metadata.csv", index=False)
        
