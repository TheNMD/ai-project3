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

def calculate_avg_reflectivity(reflectivity):
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
                  pow(10, 100 * (1 - reflectivity_55_to_60)),
                  pow(10, 100 * (1 - reflectivity_bigger_than_60))]

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
    elif 40 <= avg_reflectivity < 47.5:
        label = "moderate_rain"
    elif 47.5 <= avg_reflectivity < 55:
        label = "heavy_rain"
    elif avg_reflectivity >= 55:
        label = "very_heavy_rain"
        
    return avg_reflectivity, label

def label_image(metadata_chunk):    
    for idx, row in metadata_chunk.iterrows():   
        if type(row['label_0h']) is str:
            continue
        try:
            img_path = row['path']
            timestamp = row['timestamp_0']
            radar_range = row['range']
            
            data = pyart.io.read_sigmet(f"{data_path}/{img_path}")
            data.fields['reflectivity']['data'] = data.fields['reflectivity']['data'].astype(np.float16)
            
            if radar_range == "120km":
                radar_range = 120000
            else:
                radar_range = 300000
                
            grid_data = pyart.map.grid_from_radars(data,
                                                    grid_shape=(1, 500, 500),
                                                    grid_limits=((0, 1), 
                                                                 (-radar_range, radar_range), 
                                                                 (-radar_range, radar_range)),)
            
            reflectivity = np.array(grid_data.fields['reflectivity']['data'].compressed())
            avg_reflectivity, label = calculate_avg_reflectivity(reflectivity)
                
            # print(f"{timestamp} - Done")
            
            metadata_chunk.loc[idx, ['avg_reflectivity_0h']] = avg_reflectivity
            metadata_chunk.loc[idx, ['label_0h']] = label
            
            # Close and delete to release memory
            del grid_data
            del data
        except Exception as e:
            metadata_chunk.loc[idx, ['avg_reflectivity_0h']] = "Error"
            metadata_chunk.loc[idx, ['label_0h']] = "Error"
            logging.error(e, exc_info=True)
            continue
        
    return metadata_chunk
        
def update_metadata(new_metadata):
    if not os.path.exists("metadata_temp.csv"):
        updated_metadata = pd.read_csv("metadata.csv")
    else:
        updated_metadata = pd.read_csv("metadata_temp.csv")
    
    updated_metadata.loc[new_metadata.index, 'avg_reflectivity_0h'] = new_metadata['avg_reflectivity_0h'].tolist()
    updated_metadata.loc[new_metadata.index, 'label_0h'] = new_metadata['label_0h'].tolist()
    
    updated_metadata.to_csv("metadata_temp.csv", index=False)        

def plot_distribution():
    metadata = pd.read_csv("metadata.csv")
    frequency = metadata['label_0h'].value_counts()
    
    with open('image/label_dist.txt', 'w') as file:
        file.write(f"{frequency}")
    
    plt.bar(frequency.index, frequency.values, color='skyblue', edgecolor='black')
    plt.xlabel('Label')
    plt.ylabel('Frequency')
    plt.title('Label distribution')
    plt.savefig('image/label_dist.png')
    plt.clf()
    
if __name__ == '__main__':
    print("Python version: ", sys.version)
    print("Ubuntu version: ", platform.release())
    
    # Label current images
    num_processes = 16
    chunk_size = 100 * num_processes 
    counter = 0
    try:
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
        print(e)
        logging.error(e, exc_info=True)
    
    # Update metadata
    metadata = pd.read_csv("metadata_temp.csv")
    metadata = metadata[metadata['label_0h'] != 'Error']
    metadata.reset_index(drop=True, inplace=True)
    metadata.to_csv("metadata.csv", index=False)
    
    # Plot label and avg reflectivity distribution
    plot_distribution()


        
        
        
        