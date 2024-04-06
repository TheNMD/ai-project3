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

def calculate_avg_reflectivity(reflectivity):
    reflectivity = sorted(reflectivity)
    # Calculate the percentage of each reflectivity value in each of 8 ranges
    # Count the reflectivity value smaller than 30
    reflectivity_smaller_than_0 = len([ele for ele in reflectivity if ele < 0]) / len(reflectivity)
    reflectivity_0_to_5         = len([ele for ele in reflectivity if 0 <= ele < 5]) / len(reflectivity)
    reflectivity_5_to_10        = len([ele for ele in reflectivity if 5 <= ele < 10]) / len(reflectivity)
    reflectivity_10_to_15       = len([ele for ele in reflectivity if 10 <= ele < 15]) / len(reflectivity)
    reflectivity_15_to_20       = len([ele for ele in reflectivity if 15 <= ele < 20]) / len(reflectivity)
    reflectivity_20_to_25       = len([ele for ele in reflectivity if 20 <= ele < 25]) / len(reflectivity)
    reflectivity_25_to_30       = len([ele for ele in reflectivity if 25 <= ele < 30]) / len(reflectivity)
    reflectivity_30_to_35       = len([ele for ele in reflectivity if 30 <= ele < 35]) / len(reflectivity)
    reflectivity_35_to_40       = len([ele for ele in reflectivity if 35 <= ele < 40]) / len(reflectivity)
    reflectivity_40_to_45       = len([ele for ele in reflectivity if 40 <= ele < 45]) / len(reflectivity)
    reflectivity_45_to_50       = len([ele for ele in reflectivity if 45 <= ele < 50]) / len(reflectivity)
    reflectivity_50_to_55       = len([ele for ele in reflectivity if 50 <= ele < 55]) / len(reflectivity)
    reflectivity_55_to_60       = len([ele for ele in reflectivity if 55 <= ele < 60]) / len(reflectivity)
    reflectivity_bigger_than_60 = len([ele for ele in reflectivity if ele >= 60]) / len(reflectivity)
    
    # Assign weight to each reflectivity range value
    # weight_set = [pow(10, 1) * pow(10, 1 - reflectivity_smaller_than_30), 
    #               pow(10, 2) * pow(10, 1 - reflectivity_30_to_35),
    #               pow(10, 2) * 5 * pow(10, 2 - reflectivity_35_to_40),
    #               pow(10, 3) * pow(10, 1 - reflectivity_40_to_45),
    #               pow(10, 3) * 5 * pow(10, 1 - reflectivity_45_to_50), 
    #               pow(10, 4) * pow(10, 1 - reflectivity_50_to_55),
    #               pow(10, 4) * 5 * pow(10, 1 - reflectivity_55_to_60),
    #               pow(10, 6) * pow(10, 1 - reflectivity_bigger_than_60)]

    weight_set = [pow(10, 100 * (1 - reflectivity_smaller_than_0)),
                  pow(10, 100 * (1 - reflectivity_0_to_5)),
                  pow(10, 100 * (1 - reflectivity_5_to_10)),
                  pow(10, 100 * (1 - reflectivity_10_to_15)),
                  pow(10, 100 * (1 - reflectivity_15_to_20)),
                  pow(10, 100 * (1 - reflectivity_20_to_25)),
                  pow(10, 100 * (1 - reflectivity_25_to_30)), 
                  pow(10, 100 * (1 - reflectivity_30_to_35)),
                  pow(10, 100 * (1 - reflectivity_35_to_40)),
                  pow(10, 100 * (1 - reflectivity_40_to_45)),
                  pow(10, 100 * (1 - reflectivity_45_to_50)), 
                  pow(10, 100 * (1 - reflectivity_50_to_55)),
                  pow(10, 100 * (1 - reflectivity_55_to_60)) * 100,
                  pow(10, 100 * (1 - reflectivity_bigger_than_60)) * 100]

    # print(weight_set)

    weights = []
    for ele in reflectivity:
        if ele < 0:
            weights += [weight_set[0]]
        elif 0 <= ele < 5:
            weights += [weight_set[1]]
        elif 5 <= ele < 10:
            weights += [weight_set[2]]
        elif 10 <= ele < 15:
            weights += [weight_set[3]]
        elif 15 <= ele < 20:
            weights += [weight_set[4]]
        elif 20 <= ele < 25:
            weights += [weight_set[5]]
        elif 25 <= ele < 30:
            weights += [weight_set[6]]
        elif 30 <= ele < 35:
            weights += [weight_set[7]]
        elif 35 <= ele < 40:
            weights += [weight_set[8]]
        elif 40 <= ele < 45:
            weights += [weight_set[9]]
        elif 45 <= ele < 50:
            weights += [weight_set[10]]
        elif 50 <= ele < 55:
            weights += [weight_set[11]]
        elif 55 <= ele < 60:
            weights += [weight_set[12]]
        elif ele >= 60:
            weights += [weight_set[13]]

    avg_reflectivity = np.average(reflectivity, weights=weights)
    if avg_reflectivity < 30:
        label = "clear"
    elif 30 <= avg_reflectivity < 40:
        label = "light_rain"
    elif 40 <= avg_reflectivity < 50:
        label = "moderate_rain"
    elif 50 <= avg_reflectivity < 60:
        label = "heavy_rain"
    elif avg_reflectivity >= 60:
        label = "very_heavy_rain"
        
    # if avg_reflectivity < 25:
    #     label = "clear"
    # elif 25 <= avg_reflectivity < 35:
    #     label = "light_rain"
    # elif 35 <= avg_reflectivity < 45:
    #     label = "moderate_rain"
    # elif 45 <= avg_reflectivity < 55:
    #     label = "heavy_rain"
    # elif avg_reflectivity > 55:
    #     label = "very_heavy_rain"
        
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
        
def update_metadata(new_metadata):
    if not os.path.exists("metadata_temp.csv"):
        updated_metadata = pd.read_csv("metadata.csv")
    else:
        updated_metadata = pd.read_csv("metadata_temp.csv")
    
    updated_metadata.loc[new_metadata.index, 'future_avg_reflectivity'] = new_metadata['future_avg_reflectivity'].tolist()
    updated_metadata.loc[new_metadata.index, 'future_label'] = new_metadata['future_label'].tolist()
    
    updated_metadata.to_csv("metadata_temp.csv", index=False)        

def plot_distribution():
    metadata = pd.read_csv("metadata_lite.csv")
    
    frequency = metadata['future_label'].value_counts()
    print(frequency)
    
    plt.bar(frequency.index, frequency.values, color='skyblue', edgecolor='black')
    plt.xlabel('Label')
    plt.ylabel('Frequency')
    plt.title('Label Distribution')
    plt.savefig('image/3.label_dist.png')
    plt.clf()
    
    _, _, _ = plt.hist(metadata['future_avg_reflectivity'], color='skyblue', edgecolor='black')
    plt.xlabel('Avg Reflectivity')
    plt.ylabel('Frequency')
    plt.title('Avg Reflectivity Distribution')
    plt.savefig('image/3.avg_reflectivity_dist.png')
    plt.clf()

def move_to_label(metadata_chunk):
    for _, row in metadata_chunk.iterrows():
        timestamp = row['timestamp']
        future_label = row['future_label']
        if os.path.exists(f"image/unlabeled1/{timestamp}.jpg"):
            shutil.copy(f"image/unlabeled1/{timestamp}.jpg", f"image/labeled/{future_label}/{timestamp}.jpg")
        else:
            shutil.copy(f"image/unlabeled2/{timestamp}.jpg", f"image/labeled/{future_label}/{timestamp}.jpg")
    
if __name__ == '__main__':    
    num_processes = 20
    chunk_size = 100 * num_processes 
    
    if os.path.exists("image/labeled"):
        shutil.rmtree("image/labeled")
    else:
        os.makedirs("image/labeled") 
        os.makedirs("image/labeled/clear")
        os.makedirs("image/labeled/light_rain")
        os.makedirs("image/labeled/heavy_rain")
        os.makedirs("image/labeled/storm")
    
    # Label images
    try:
        counter = 0
        # Use multiprocessing to iterate over the metadata 
        with mp.Pool(processes=num_processes) as pool:
            metadata_chunks = pd.read_csv("metadata.csv", chunksize=chunk_size)
            for chunk in metadata_chunks:
                sub_metadata_chunks = np.array_split(chunk, num_processes)
                
                start_time = time.time()
                results = pool.map(label_image, sub_metadata_chunks)
                update_metadata(pd.concat(results))
                end_time = time.time() - start_time

                counter += 1
                print(f"### Chunk: {counter} | Time: {end_time} ###")
    except Exception as e:
        # If crash due to lack of memory, restart the process (progress is saved)
        print(e)
        logging.error(e, exc_info=True)
    
    updated_metadata = pd.read_csv("metadata_temp.csv")
    updated_metadata.to_csv("metadata.csv", index=False)
    
    # Make a metadata_lite.csv that contains only relevant info for model
    metadata_lite = pd.read_csv("metadata.csv")
    metadata_lite = metadata_lite[metadata_lite['generated'] != "Error"]
    metadata_lite = metadata_lite[metadata_lite['future_path'] != "NotAvail"]
    metadata_lite = metadata_lite[metadata_lite['future_label'] != "Error"]
    metadata_lite = metadata_lite.drop(['path', 'future_path'], axis=1)
    metadata_lite.to_csv("metadata_lite.csv", index=False)
    
    # Plot label and avg reflectivity distribution
    plot_distribution()
    
    # Move images from unlabeled to labeled folders
    try:
        counter = 0
        # Use multiprocessing to iterate over the metadata 
        with mp.Pool(processes=num_processes) as pool:
            labeled_chunks = pd.read_csv("metadata_lite.csv", chunksize=chunk_size)
            for chunk in labeled_chunks:
                sub_metadata_chunks = np.array_split(chunk, num_processes)
    
                start_time = time.time()
                pool.map(move_to_label, sub_metadata_chunks)
                end_time = time.time() - start_time

                counter += 1
                print(f"### Chunk: {counter} | Time: {end_time} ###")
    except Exception as e:
        # If crash due to lack of memory, restart the process (progress is saved)
        print(e)
        logging.error(e, exc_info=True)

        
        
        
        