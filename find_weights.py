import os
import time
import multiprocessing as mp
from collections import Counter
import random

import warnings
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(filename='errors.log', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyart

# Given a list of dbz values
# Assign each value a weight according to how important it is
# Calculate weighted average
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
    elif 40 <= avg_reflectivity < 47.5:
        label = "moderate_rain"
    elif 47.5 <= avg_reflectivity < 55:
        label = "heavy_rain"
    elif avg_reflectivity >= 55:
        label = "very_heavy_rain"

    return avg_reflectivity, label


def run_processes(filenames):
    results = []
    for name in sorted(filenames):
        try:
            # Read data
            data = pyart.io.read_sigmet(f"sample_data/NhaBe/{name}")
            data.fields['reflectivity']['data'] = data.fields['reflectivity']['data'].astype(np.float16)

            # Convert to grid
            grid_data = pyart.map.grid_from_radars(
                data,
                grid_shape=(1, 500, 500),
                grid_limits=((0, 1), (-250000, 250000), (-250000, 250000)),
            )
            # Use timestamp to name the files, replace ":" with "-"
            timestamp = str(pyart.util.datetime_from_radar(data)).replace(':', '-')

            # Compress to remove masked elements
            reflectivity = np.array(grid_data.fields['reflectivity']['data'].compressed())

            # Plot reflectivity value distribution
            plot_distribution(sorted(reflectivity), "Reflectivity", f"NhaBe/{timestamp}-dist.png")
            
            # Calculate average reflectivity and label
            avg_reflectivity, label = calculate_avg_reflectivity(reflectivity)

            print(f"{timestamp} - {avg_reflectivity} - {label}")
            results += [(timestamp, avg_reflectivity, label)]
        except Exception as e:
            print(e)
            logging.error(e, exc_info=True)
    return results


# Write results to a txt file
def update_result(results):
    full_results = []
    for result in results:
        full_results += result
    full_results = sorted(full_results)
    df = pd.DataFrame(full_results, columns=['timestamp', 'avg_reflectivity', 'label'])
    df.to_csv('sample_data/results.csv', index=False)
    return full_results


# Plot value distribution
def plot_distribution(list, value_name, save_name):
    _, _, _ = plt.hist(list, color='skyblue', edgecolor='black')
    plt.xlabel(f'{value_name}')
    plt.ylabel('Frequency')
    plt.title(f'{value_name} Distribution')
    plt.savefig(f'sample_data/{save_name}')
    plt.clf()


if __name__ == "__main__":
    filenames = [file for file in os.listdir('sample_data/NhaBe') if not file.endswith('jpg') 
                                                                 and not file.endswith('png')]
    random.shuffle(filenames)
    # Only take n files
    # filenames = filenames[:1000]
    
    reflectivity_list = []
    label_list = []
    
    # Change num_processes to increase threads
    num_processes = 8
    try:
        # Use multiprocessing to iterate over the filename list 
        with mp.Pool(processes=num_processes) as pool:
            sub_list = np.array_split(filenames, num_processes)
            start_time = time.time()
            results = pool.map(run_processes, sub_list)

            # Update result to a txt file
            full_results = update_result(results)
            reflectivity_list = [result[1] for result in full_results]
            label_list = [result[2] for result in full_results]

            end_time = time.time() - start_time
            print(f"Time: {end_time}")
    except Exception as e:
        # If crash due to lack of memory, restart the process (progress is saved)
        print(e)
        logging.error(e, exc_info=True)
    print(f"Time taken: {end_time}")

    # Plot avg reflectivity value distribution
    plot_distribution(sorted(reflectivity_list), "Avg Reflectivity", "avg_reflectivity_dist.png")

    # Plot label distribution
    plot_distribution(sorted(label_list), "Label", "label_dist.png")
    
    frequency = Counter(label_list)
    for key, value in frequency.items():
        print(f"Label '{key}': {value}")
