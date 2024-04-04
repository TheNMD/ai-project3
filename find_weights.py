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

# Given a list of dbz values
# Assign each value a weight according to how important it is
# Calculate weighted average
def calculate_avg_reflectivity(reflectivity, 
                               weight_set=[0.000001, 0.00001, 0.0001, 0.001, 0.01, 10000, 100000, 1000000]):
    weights = []
    for ele in reflectivity:
        if ele < 30:
            weights += [weight_set[0]]
        elif ele >= 30 and ele < 35:
            weights += [weight_set[1]]
        elif ele >= 35 and ele < 40:
            weights += [weight_set[2]]
        elif ele >= 40 and ele < 45:
            weights += [weight_set[3]]
        elif ele >= 45 and ele < 50:
            weights += [weight_set[4]]
        elif ele >= 50 and ele < 55:
            weights += [weight_set[5]]
        elif ele >= 55 and ele < 60:
            weights += [weight_set[6]]
        elif ele > 60:
            weights += [weight_set[7]]
    
    avg_reflectivity = np.average(reflectivity, weights=weights)
    if avg_reflectivity < 30:
        label = "clear"
    elif avg_reflectivity >= 30 and avg_reflectivity < 52:
        label = "light_rain"
    elif avg_reflectivity >= 52 and avg_reflectivity < 63:
        label = "heavy_rain"
    elif avg_reflectivity > 63:
        label = "storm"
        
    return avg_reflectivity, label

def run_processes(filenames):
    results = []
    for name in filenames:
        try:
            # Read data
            data = pyart.io.read_sigmet(f"data/data_WF/NhaBe/{name}")
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
            continue
    return results

# Write results to a txt file
def update_result(results):
    full_results = []
    for result in results:
        full_results += result
    full_results = sorted(full_results)
    for result in full_results:
        with open('data/data_WF/results.txt', 'a') as file:
            file.write(f"{result[0]} - {result[1]} - {result[2]}" + '\n')
    return full_results

# Plot value distribution
def plot_distribution(list, value_name, save_name):
    _, _, _ = plt.hist(list, color='skyblue', edgecolor='black')
    plt.xlabel(f'{value_name}')
    plt.ylabel('Frequency')
    plt.title(f'{value_name} distribution')
    plt.savefig(f'data/data_WF/{save_name}')
    plt.clf()
    
if __name__ == "__main__":
    if os.path.exists('data/data_WF/results.txt'):      
        os.remove('data/data_WF/results.txt') 
    
    filenames = [file for file in os.listdir('data/data_WF/NhaBe') if not file.endswith('jpg')]
    reflectivity_list = []
    label_list = []
    
    # Change num_processes to increase threads
    num_processes = 4
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
    plot_distribution(sorted(reflectivity_list), "Avg reflectivity", "results-dist-avg_reflectivity.png")
    
    # Plot label distribution
    plot_distribution(sorted(label_list), "Label", "results-dist-label.png")