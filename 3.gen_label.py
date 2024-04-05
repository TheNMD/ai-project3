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
        if type(row['future_path']) is str or row['generated'] == 'Error':
            continue
                
        current_time = row['timestamp']
        future_metadata = metadata[(metadata['timestamp'] - current_time > pd.Timedelta(interval, "s")) &
                                   (metadata['timestamp'] - current_time < pd.Timedelta(interval + 1800, "s"))].head(1)
        
        if future_metadata.empty:
            metadata.loc[idx, ['future_path']] = "NotAvail"
            metadata.loc[idx, ['future_timestamp']] = "NotAvail"
            metadata.loc[idx, ['future_label']] = "NotAvail"
            metadata.loc[idx, ['future_avg_reflectivity']] = "NotAvail"
        else:
            metadata.loc[idx, ['future_path']] = future_metadata['path'].tolist()[0]
            metadata.loc[idx, ['future_timestamp']] = future_metadata['timestamp'].tolist()[0]
        
        print(current_time)

    metadata['timestamp'] = metadata['timestamp'].astype(str).str.replace(':', '-')
    metadata['future_timestamp'] = metadata['future_timestamp'].astype(str).str.replace(':', '-')
    metadata.to_csv("metadata.csv", index=False)

def calculate_avg_reflectivity(reflectivity):
    # calculate the percentage of each reflectivity value in each of 8 ranges
    # count the reflectivity value smaller than 30
    reflectivity_smaller_than_30 = len([ele for ele in reflectivity if ele < 30]) / len(reflectivity)
    reflectivity_30_to_35 = len([ele for ele in reflectivity if 30 <= ele < 35]) / len(reflectivity)
    reflectivity_35_to_40 = len([ele for ele in reflectivity if 35 <= ele < 40]) / len(reflectivity)
    reflectivity_40_to_45 = len([ele for ele in reflectivity if 40 <= ele < 45]) / len(reflectivity)
    reflectivity_45_to_50 = len([ele for ele in reflectivity if 45 <= ele < 50]) / len(reflectivity)
    reflectivity_50_to_55 = len([ele for ele in reflectivity if 50 <= ele < 55]) / len(reflectivity)
    reflectivity_55_to_60 = len([ele for ele in reflectivity if 55 <= ele < 60]) / len(reflectivity)
    reflectivity_bigger_than_60 = len([ele for ele in reflectivity if ele >= 60]) / len(reflectivity)

    # assign weight to each reflectivity range value
    weight_set = [pow(10, 1) * pow(1, 1 - reflectivity_smaller_than_30), 
                  pow(10, 2) * pow(3, 1 - reflectivity_30_to_35),
                  pow(10, 3) * pow(5, 1 - reflectivity_35_to_40),
                  pow(10, 4) * pow(7, 1 - reflectivity_40_to_45),
                  pow(10, 5) * pow(9, 1 - reflectivity_45_to_50), 
                  pow(10, 6) * pow(11, 1 - reflectivity_50_to_55),
                  pow(10, 7) * pow(13, 1 - reflectivity_55_to_60),
                  pow(10, 8) * pow(15, 1 - reflectivity_bigger_than_60)]

    # print(weight_set)
    
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
        
    return avg_reflectivity, label

def label_image(metadata_chunk):
    for idx, row in metadata_chunk.iterrows():
        if type(row['future_label']) is str or row['future_path'] == 'NotAvail' or row['generated'] == 'Error':
            continue
        try:
            data = pyart.io.read_sigmet(f"{data_path}/{row['future_path']}")
            data.fields['reflectivity']['data'] = data.fields['reflectivity']['data'].astype(np.float16)
            
            grid_data = pyart.map.grid_from_radars(
                data,
                grid_shape=(1, 500, 500),
                grid_limits=((0, 1), (-250000, 250000), (-250000, 250000)),
            )
            
            reflectivity = np.array(grid_data.fields['reflectivity']['data'].compressed())
            avg_reflectivity, label = calculate_avg_reflectivity(reflectivity)
                
            print(f"{row['timestamp']} - Average reflectivity: {avg_reflectivity} | Label: {label}")
            
            metadata_chunk.loc[idx, ['future_avg_reflectivity']] = avg_reflectivity
            metadata_chunk.loc[idx, ['future_label']] = label
            
            # Close and delete to release memory
            del grid_data
            del data
        except Exception as e:
            metadata_chunk.loc[idx, ['future_avg_reflectivity']] = np.nan
            metadata_chunk.loc[idx, ['future_label']] = "Error"
            logging.error(e, exc_info=True)
            continue
        
    return metadata_chunk
        
def update_metadata(new_metadata, idx):
    updated_metadata = pd.read_csv("metadata.csv")
    
    updated_metadata.loc[new_metadata.index, 'future_avg_reflectivity'] = new_metadata['future_avg_reflectivity'].tolist()
    updated_metadata.loc[new_metadata.index, 'future_label'] = new_metadata['future_label'].tolist()
    
    updated_metadata.to_csv(f"image/metadata_{idx}.csv", index=False)        

def move_to_label(metadata_chunk):
    for _, row in metadata_chunk.iterrows():
        timestamp = row['timestamp']
        
        future_label = row['future_label']
        current_label = row['current_label']
        
        if os.path.exists(f"image/unlabeled1/{timestamp}.jpg"):
            shutil.copy(f"image/unlabeled1/{timestamp}.jpg", f"image/labeled/{future_label}/{timestamp}.jpg")
            shutil.copy(f"image/unlabeled1/{timestamp}.jpg", f"image/labeled/{current_label}/{timestamp}.jpg")
        else:
            shutil.copy(f"image/unlabeled2/{timestamp}.jpg", f"image/labeled/{future_label}/{timestamp}.jpg")
            shutil.copy(f"image/unlabeled2/{timestamp}.jpg", f"image/labeled/{current_label}/{timestamp}.jpg")

def plot_distribution():
    metadata = pd.read_csv("metadata.csv")
    metadata = metadata[metadata['future_label' != "NotAvail"]]
    
    frequency = metadata['future_label'].value_counts()
    print(frequency)
    
    plt.figure(figsize=(6, 6))
    
    plt.bar(frequency.index, frequency.values, color='skyblue', edgecolor='black')
    plt.xlabel('Label')
    plt.ylabel('Frequency')
    plt.title('Label distribution')
    plt.savefig('image/3.Label distribution.png')
    plt.clf()
    
    _, _, _ = plt.hist(metadata['future_avg_reflectivity'], color='skyblue', edgecolor='black')
    plt.xlabel('Avg reflectivity')
    plt.ylabel('Frequency')
    plt.title('Avg reflectivity distribution')
    plt.savefig('image/3.Avg reflectivity distribution.png')
    plt.clf()
    
if __name__ == '__main__':
    # find_future_images(interval=7200)
    
    num_processes = 20
    chunk_size = 10 * num_processes 
    
    if not os.path.exists("image/labeled"):
        if not os.path.exists("image"):
            os.makedirs("image")
            
        os.makedirs("image/labeled")
        
        os.makedirs("image/labeled/future")
        os.makedirs("image/labeled/future/clear")
        os.makedirs("image/labeled/future/light_rain")
        os.makedirs("image/labeled/future/heavy_rain")
        os.makedirs("image/labeled/future/storm")
        
        os.makedirs("image/labeled/current")
        os.makedirs("image/labeled/current/clear")
        os.makedirs("image/labeled/current/light_rain")
        os.makedirs("image/labeled/current/heavy_rain")
        os.makedirs("image/labeled/current/storm")
    
    try:
        counter = 0
        # Use multiprocessing to iterate over the metadata 
        with mp.Pool(processes=num_processes) as pool:
            metadata_chunks = pd.read_csv("metadata.csv", chunksize=chunk_size)
            for chunk in metadata_chunks:
                sub_metadata_chunks = np.array_split(chunk, num_processes)
                
                start_time = time.time()
                results = pool.map(label_image, sub_metadata_chunks)
                update_metadata(pd.concat(results), counter + 1)
                end_time = time.time() - start_time

                counter += 1
                print(f"### Chunk: {counter} | Time: {end_time} ###")
    except Exception as e:
        # If crash due to lack of memory, restart the process (progress is saved)
        print(e)
        logging.error(e, exc_info=True)
    
    filenames = sorted(os.listdir("image"))
    metadata_list = [pd.read_csv(f"metadata/{name}") for name in filenames if name.endswith("csv")]
    metadata = pd.concat(metadata_list)
    metadata = metadata.sort_values(by='timestamp').reset_index(drop=True)
    metadata.to_csv("metadata.csv", index=False)
    
    # try:
    #     counter = 0
    #     # Use multiprocessing to iterate over the metadata 
    #     with mp.Pool(processes=num_processes) as pool:
    #         labeled_chunks = pd.read_csv("metadata.csv", chunksize=chunk_size)
    #         for chunk in labeled_chunks:
    #             start_time = time.time()
    #             pool.map(move_to_label, np.array_split(chunk, num_processes))
    #             end_time = time.time() - start_time

    #             counter += 1
    #             print(f"### Chunk: {counter} | Time: {end_time} ###")
    # except Exception as e:
    #     # If crash due to lack of memory, restart the process (progress is saved)
    #     print(e)
    #     logging.error(e, exc_info=True)
    
    # plot_distribution()
    
    metadata = pd.read_csv("metadata.csv")
    metadata_lite = metadata[metadata['generated'] != "Error"]
    metadata_lite = metadata_lite[metadata_lite['future_path'] != "NotAvail"]
    metadata_lite = metadata_lite[metadata_lite['future_label'] != "Error"]
    metadata_lite = metadata.drop(['path', 'future_path'], axis=1)
    metadata_lite.to_csv("metadata_lite.csv", index=False)
        

        
        
        
        