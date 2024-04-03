import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyart
import warnings
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(filename='errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


# Given a list of dbz values
# Assign each value a weight according to how important it is
# Calculate weighted average
def calculate_avg_reflectivity(reflectivity, weight_set=[0.00001, 0.001, 0.1, 100, 1000, 10000, 100000, 1000000]):
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

filenames = [file for file in os.listdir('data/data_WF/NhaBe') if not file.endswith('jpg') and not file.endswith('txt')]

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
        _, _, _ = plt.hist(reflectivity, color='skyblue', edgecolor='black')
        plt.xlabel('Avg reflectivity')
        plt.ylabel('Frequency')
        plt.title('Avg reflectivity distribution')
        plt.savefig(f'data/data_WF/NhaBe/{timestamp}-dist.png')
        plt.clf()
        
        # Calculate average reflectivity and label
        avg_reflectivity, label = calculate_avg_reflectivity(reflectivity)
        
        print(f"{timestamp} - {avg_reflectivity} - {label}")
        with open('data/data_WF/NhaBe/results.txt', 'a') as file:
            file.write(f"{timestamp} - {avg_reflectivity} - {label}" + '\n')
    except Exception as e:
        continue