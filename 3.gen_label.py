import os, sys, platform, shutil, time
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

def find_future_images(interval):
    metadata = pd.read_csv("metadata.csv")
    metadata['timestamp_0'] = pd.to_datetime(metadata['timestamp_0'], format="%Y-%m-%d %H-%M-%S")
    
    path_col = f"path_{interval}"
    timestamp_col = f"timestamp_{interval}"
    label_col = f"label_{interval}"
    avg_reflectivity_col = f"avg_reflectivity_{interval}"

    if path_col not in metadata.columns: metadata[path_col] = np.nan
    if timestamp_col not in metadata.columns: metadata[timestamp_col] = np.nan
    if label_col not in metadata.columns: metadata[label_col] = np.nan
    if avg_reflectivity_col not in metadata.columns: metadata[avg_reflectivity_col] = np.nan

    for idx, row in metadata.iterrows():
        if type(row[path_col]) is str or row['generated'] == 'Error':
            continue
                
        current_time = row['timestamp_0']
        future_metadata = metadata[(metadata['timestamp_0'] - current_time > pd.Timedelta(interval, "s")) &
                                   (metadata['timestamp_0'] - current_time < pd.Timedelta(interval + 1800, "s"))].head(1)
        
        if future_metadata.empty:
            metadata.loc[idx, [path_col]] = "NotAvail"
            metadata.loc[idx, [timestamp_col]] = "NotAvail"
            metadata.loc[idx, [label_col]] = "NotAvail"
            metadata.loc[idx, [avg_reflectivity_col]] = "NotAvail"
        else:
            metadata.loc[idx, [path_col]] = future_metadata['path_0'].tolist()[0]
            metadata.loc[idx, [timestamp_col]] = future_metadata['timestamp_0'].tolist()[0]
        
        print(current_time)

    metadata['timestamp_0'] = metadata['timestamp_0'].astype(str).str.replace(':', '-')
    metadata[timestamp_col] = metadata[timestamp_col].astype(str).str.replace(':', '-')
    metadata.to_csv("metadata.csv", index=False)

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
        for interval in [0, 7200, 21600, 43200]:
            path_col = f'path_{interval}'
            label_col = f'label_{interval}'
            avg_reflectivity_col = f'avg_reflectivity_{interval}'
                
            if type(row[label_col]) is str or row[path_col] == 'NotAvail' or row['generated'] == 'Error':
                continue
            try:
                data = pyart.io.read_sigmet(f"{data_path}/{row[path_col]}")
                data.fields['reflectivity']['data'] = data.fields['reflectivity']['data'].astype(np.float16)
                
                grid_data = pyart.map.grid_from_radars(
                    data,
                    grid_shape=(1, 500, 500),
                    grid_limits=((0, 1), (-250000, 250000), (-250000, 250000)),
                )
                
                reflectivity = np.array(grid_data.fields['reflectivity']['data'].compressed())
                avg_reflectivity, label = calculate_avg_reflectivity(reflectivity)
                    
                print(f"{interval} - {row['timestamp_0']} | Average reflectivity: {avg_reflectivity} | Label: {label}")
                
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
    updated_metadata.loc[new_metadata.index, 'avg_reflectivity_7200'] = new_metadata['avg_reflectivity_7200'].tolist()
    updated_metadata.loc[new_metadata.index, 'label_7200'] = new_metadata['label_7200'].tolist()
    updated_metadata.loc[new_metadata.index, 'avg_reflectivity_21600'] = new_metadata['avg_reflectivity_21600'].tolist()
    updated_metadata.loc[new_metadata.index, 'label_21600'] = new_metadata['label_21600'].tolist()
    updated_metadata.loc[new_metadata.index, 'avg_reflectivity_43200'] = new_metadata['avg_reflectivity_43200'].tolist()
    updated_metadata.loc[new_metadata.index, 'label_43200'] = new_metadata['label_43200'].tolist()
    
    updated_metadata.to_csv("metadata_temp.csv", index=False)        

def plot_distribution():
    metadata = pd.read_csv("metadata_lite.csv")
    
    frequency_0    = metadata['label_0'].value_counts()
    frequency_7200  = metadata['label_7200'].value_counts()
    frequency_21600 = metadata['label_21600'].value_counts()
    frequency_43200 = metadata['label_43200 '].value_counts()
    
    with open('image/label_summary.txt', 'w') as file:
        file.write("### 0 ###\n")
        file.write(f"{frequency_0}\n\n")
        file.write("### 7200 ###\n")
        file.write(f"{frequency_7200}\n\n")
        file.write("### 21600 ###\n")
        file.write(f"{frequency_21600}\n\n")
        file.write("### 43200 ###\n")
        file.write(f"{frequency_43200}\n\n")
    
    plt.bar(frequency_0.index, frequency_0.values, color='skyblue', edgecolor='black')
    plt.xlabel('Label')
    plt.ylabel('Frequency')
    plt.title('Label Distribution - Current')
    plt.savefig('image/labeled/label_dist_0.png')
    plt.clf()
    
    plt.bar(frequency_7200.index, frequency_7200.values, color='skyblue', edgecolor='black')
    plt.xlabel('Label')
    plt.ylabel('Frequency')
    plt.title('Label Distribution - 7200')
    plt.savefig('image/labeled/label_dist_7200.png')
    plt.clf()
    
    plt.bar(frequency_21600.index, frequency_21600.values, color='skyblue', edgecolor='black')
    plt.xlabel('Label')
    plt.ylabel('Frequency')
    plt.title('Label Distribution - 21600')
    plt.savefig('image/labeled/label_dist_21600.png')
    plt.clf()
    
    plt.bar(frequency_43200.index, frequency_43200.values, color='skyblue', edgecolor='black')
    plt.xlabel('Label')
    plt.ylabel('Frequency')
    plt.title('Label Distribution - 43200')
    plt.savefig('image/labeled/label_dist_43200.png')
    plt.clf()
    
    _, _, _ = plt.hist(metadata['avg_reflectivity_0'], color='skyblue', edgecolor='black')
    plt.xlabel('Avg Reflectivity')
    plt.ylabel('Frequency')
    plt.title('Avg Reflectivity Distribution - 0')
    plt.savefig('image/labeled/avg_reflectivity_dist_0.png')
    plt.clf()
    
    _, _, _ = plt.hist(metadata['avg_reflectivity_7200'], color='skyblue', edgecolor='black')
    plt.xlabel('Avg Reflectivity')
    plt.ylabel('Frequency')
    plt.title('Avg Reflectivity Distribution - 7200')
    plt.savefig('image/labeled/avg_reflectivity_dist_7200.png')
    plt.clf()
    
    _, _, _ = plt.hist(metadata['avg_reflectivity_21600'], color='skyblue', edgecolor='black')
    plt.xlabel('Avg Reflectivity')
    plt.ylabel('Frequency')
    plt.title('Avg Reflectivity Distribution - 21600')
    plt.savefig('image/labeled/avg_reflectivity_dist_21600.png')
    plt.clf()
    
    _, _, _ = plt.hist(metadata['avg_reflectivity_43200'], color='skyblue', edgecolor='black')
    plt.xlabel('Avg Reflectivity')
    plt.ylabel('Frequency')
    plt.title('Avg Reflectivity Distribution - 43200')
    plt.savefig('image/labeled/avg_reflectivity_dist_43200.png')
    plt.clf()
    
if __name__ == '__main__':
    print("Python version: ", sys.version)
    print("Ubuntu version: ", platform.release())
    
    find_future_images(interval=7200)
    find_future_images(interval=21600)
    find_future_images(interval=43200)
    
    # num_processes = 16
    # chunk_size = 100 * num_processes 
    
    # # Label images
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
    # except Exception as e:
    #     # If crash due to lack of memory, restart the process (progress is saved)
    #     print(e)
    #     logging.error(e, exc_info=True)
    
    # updated_metadata = pd.read_csv("metadata_temp.csv")
    # updated_metadata.to_csv("metadata.csv", index=False)
    
    # # Make a metadata_lite.csv that contains only relevant info for model
    # metadata_lite = pd.read_csv("metadata.csv")
    # metadata_lite = metadata_lite.drop(['path_0', 'path_7200', 'path_21600', 'path_43200', 'generated'], axis=1)
    # metadata_lite.to_csv("metadata_lite.csv", index=False)
    
    # # Plot label and avg reflectivity distribution
    # plot_distribution()


        
        
        
        