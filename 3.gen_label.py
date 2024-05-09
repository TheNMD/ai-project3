import os, sys, platform, time
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(filename='errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

import pyart
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        path_col = 'path_0'
        label_col = 'label_0'
        avg_reflectivity_col = 'avg_reflectivity_0'
            
        if type(row[label_col]) is str:
            continue
        try:
            data = pyart.io.read_sigmet(f"{data_path}/{row[path_col]}")
            data.fields['reflectivity']['data'] = data.fields['reflectivity']['data'].astype(np.float16)
            
            grid_data = pyart.map.grid_from_radars(data,
                                                    grid_shape=(1, 500, 500),
                                                    grid_limits=((0, 1), (-250000, 250000), (-250000, 250000)),)
            
            reflectivity = np.array(grid_data.fields['reflectivity']['data'].compressed())
            avg_reflectivity, label = calculate_avg_reflectivity(reflectivity)
                
            print(f"{row['timestamp_0']} | Average reflectivity: {avg_reflectivity} | Label: {label}")
            
            metadata_chunk.loc[idx, [avg_reflectivity_col]] = avg_reflectivity
            metadata_chunk.loc[idx, [label_col]] = label
            
            # Close and delete to release memory
            del grid_data
            del data
        except Exception as e:
            metadata_chunk.loc[idx, [avg_reflectivity_col]] = "Error"
            metadata_chunk.loc[idx, [label_col]] = "Error"
            logging.error(e, exc_info=True)
            continue
        
    return metadata_chunk
        
def update_metadata(new_metadata):
    if not os.path.exists("metadata_temp.csv"):
        updated_metadata = pd.read_csv("metadata.csv")
    else:
        updated_metadata = pd.read_csv("metadata_temp.csv")
    
    updated_metadata.loc[new_metadata.index, 'avg_reflectivity_0'] = new_metadata['avg_reflectivity_0'].tolist()
    updated_metadata.loc[new_metadata.index, 'label_0'] = new_metadata['label_0'].tolist()
    
    updated_metadata.to_csv("metadata_temp.csv", index=False)        

def find_future_images(interval):
    metadata = pd.read_csv("metadata.csv")
    metadata['timestamp_0'] = pd.to_datetime(metadata['timestamp_0'], format="%Y-%m-%d %H-%M-%S")
    
    timestamp_col = f"timestamp_{interval}"
    label_col = f"label_{interval}"
    avg_reflectivity_col = f"avg_reflectivity_{interval}"

    if timestamp_col not in metadata.columns: metadata[timestamp_col] = np.nan
    if label_col not in metadata.columns: metadata[label_col] = np.nan
    if avg_reflectivity_col not in metadata.columns: metadata[avg_reflectivity_col] = np.nan

    for idx, row in metadata.iterrows():
        if type(row[timestamp_col]) is str:
            continue
                
        current_time = row['timestamp_0']
        future_metadata = metadata[(metadata['timestamp_0'] - current_time > pd.Timedelta(interval, "s")) &
                                   (metadata['timestamp_0'] - current_time < pd.Timedelta(interval + 1800, "s"))].head(1)
        
        if future_metadata.empty:
            metadata.loc[idx, [timestamp_col]] = "NotAvail"
            metadata.loc[idx, [avg_reflectivity_col]] = "NotAvail"
            metadata.loc[idx, [label_col]] = "NotAvail"
        else:
            future_timestamp = future_metadata['timestamp_0'].tolist()[0]
            metadata.loc[idx, [timestamp_col]] = future_timestamp
            metadata.loc[idx, [avg_reflectivity_col]] = metadata.loc[metadata['timestamp_0'] == future_timestamp, 'avg_reflectivity_0'].tolist()[0]
            metadata.loc[idx, [label_col]] = metadata.loc[metadata['timestamp_0'] == future_timestamp, 'label_0'].tolist()[0]
        
        print(current_time)

    metadata['timestamp_0'] = metadata['timestamp_0'].astype(str).str.replace(':', '-')
    metadata[timestamp_col] = metadata[timestamp_col].astype(str).str.replace(':', '-')
    metadata.to_csv(f"metadata_{interval}.csv", index=False)

def combine_metadata(interval):
    metadata = pd.read_csv("metadata.csv")
    
    new_metadata = pd.read_csv(f"metadata_{interval}.csv")
    metadata[f'timestamp_{interval}'] = new_metadata[f'timestamp_{interval}']
    metadata[f'avg_reflectivity_{interval}'] = new_metadata[f'avg_reflectivity_{interval}']
    metadata[f'label_{interval}'] = new_metadata[f'label_{interval}']
    
    metadata.to_csv("metadata.csv", index=False)

def plot_distribution(interval):
    metadata = pd.read_csv("metadata.csv")
    metadata = metadata[[f'avg_reflectivity_{interval}', f'label_{interval}']]
    metadata = metadata[(metadata[f'avg_reflectivity_{interval}'] != 'NotAvail') & (metadata[f'label_{interval}'] != 'NotAvail')]
    metadata[f'avg_reflectivity_{interval}'] = metadata[f'avg_reflectivity_{interval}'].astype('float64')
    metadata.reset_index(drop=True, inplace=True)
    
    frequency = metadata[f'label_{interval}'].value_counts()
    with open(f'image/labeled/{interval}_label_dist.txt', 'w') as file:
        file.write(f"{frequency}")
    
    plt.bar(frequency.index, frequency.values, color='skyblue', edgecolor='black')
    plt.xlabel('Label')
    plt.ylabel('Frequency')
    plt.title(f'Label Distribution - {interval}')
    plt.savefig(f'image/labeled/{interval}_label_dist.png')
    plt.clf()
        
    _, _, _ = plt.hist(metadata[f'avg_reflectivity_{interval}'], color='skyblue', edgecolor='black')
    plt.xlabel('Avg Reflectivity')
    plt.ylabel('Frequency')
    plt.title(f'Avg Reflectivity Distribution - {interval}')
    plt.savefig(f'image/labeled/{interval}_avg_reflectivity_dist.png')
    plt.clf()
    
if __name__ == '__main__':
    print("Python version: ", sys.version)
    print("Ubuntu version: ", platform.release())
    
    if not os.path.exists("image/labeled"):
        os.makedirs("image/labeled")
    
    # # Label images
    # num_processes = 16
    # chunk_size = 100 * num_processes 
    
    # try:
    #     counter = 0
    #     # Use multiprocessing to iterate over the metadata 
    #     with mp.Pool(processes=num_processes) as pool:
    #         metadata_chunks = pd.read_csv("metadata.csv", chunksize=chunk_size)
    #         for chunk in metadata_chunks:
    #             sub_metadata_chunks = np.array_split(chunk, num_processes)
                
    #             start_time = time.time()
    #             results = pool.map(label_image, sub_metadata_chunks)
    #             update_metadata(pd.concat(results))
    #             end_time = time.time() - start_time

    #             counter += 1
    #             print(f"### Chunk: {counter} | Time: {end_time} ###")
                
    #     metadata = pd.read_csv("metadata_temp.csv")
    #     metadata = metadata[metadata['label_0'] != 'Error']
    #     metadata.reset_index(drop=True, inplace=True)
    #     metadata.to_csv("metadata.csv", index=False)
    # except Exception as e:
    #     print(e)
    #     logging.error(e, exc_info=True)
    
    # Label future images
    try:
        # Use multiprocessing to iterate over the metadata
        # timestamps = [7200, 21600, 43200]
        # timestamps = [3600, 10800, 14400]
        timestamps = [1080]
        with mp.Pool(processes=len(timestamps)) as pool:
            start_time = time.time()
            pool.map(find_future_images, timestamps)
            end_time = time.time() - start_time

            print(f"Time: {end_time} ###")

    except Exception as e:
        print(e)
        logging.error(e, exc_info=True)
    
    # Combine all metadata
    combine_metadata(interval=1800)
    # combine_metadata(interval=3600)
    # combine_metadata(interval=7200)
    # combine_metadata(interval=10800)
    # combine_metadata(interval=14400)
    # combine_metadata(interval=21600)
    # combine_metadata(interval=43200)
    
    # Plot label and avg reflectivity distribution
    # plot_distribution(interval=0)
    plot_distribution(interval=1800)
    # plot_distribution(interval=3600)
    # plot_distribution(interval=7200)
    # plot_distribution(interval=10800)
    # plot_distribution(interval=14400)
    # plot_distribution(interval=21600)
    # plot_distribution(interval=43200)


        
        
        
        