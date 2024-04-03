import numpy as np
import pandas as pd
import pyart


def calculate_avg_reflectivity(reflectivity, weight_set):
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

timestamp = ''
data = pyart.io.read_sigmet(f"/data/data_WF/NhaBe/{timestamp}")
data.fields['reflectivity']['data'] = data.fields['reflectivity']['data'].astype(np.float16)

grid_data = pyart.map.grid_from_radars(
    data,
    grid_shape=(1, 500, 500),
    grid_limits=((0, 1), (-250000, 250000), (-250000, 250000)),
)

reflectivity = np.array(grid_data.fields['reflectivity']['data'].compressed())
avg_reflectivity, label = calculate_avg_reflectivity(reflectivity, [1, 20, 25, 30])

print(avg_reflectivity, label)