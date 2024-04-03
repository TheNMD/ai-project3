import pyart
import numpy as np

def calculate_avg_reflectivity(reflectivity, weight_set):
    weights = []
    
    num_clear = sum(1 for ele in reflectivity if ele < 30)
    num_light_rain = sum(1 for ele in reflectivity if ele >= 30 and ele < 52)
    num_heavy_rain = sum(1 for ele in reflectivity if ele >= 52 and ele < 63)
    num_storm = sum(1 for ele in reflectivity if ele > 63)
    
    for ele in reflectivity:
        if ele < 30:
            weights += [weight_set[0] / num_clear]
        elif ele < 52:
            weights += [weight_set[1] / num_light_rain]
        elif ele < 63:
            weights += [weight_set[2] / num_heavy_rain]
        else:
            weights += [weight_set[3] / num_storm]
    
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

# data = pyart.io.read_sigmet(r"/data/data_WF/NhaBe/2020/Pro-Raw(1-8)T7-2020/01/2020-07-01T02:03")
# data.fields['reflectivity']['data'] = data.fields['reflectivity']['data'].astype(np.float16)

# grid_data = pyart.map.grid_from_radars(
#     data,
#     grid_shape=(1, 500, 500),
#     grid_limits=((0, 1), (-250000, 250000), (-250000, 250000)),
# )

reflectivity = np.array(grid_data.fields['reflectivity']['data'].compressed())
avg_reflectivity, label = calculate_avg_reflectivity(reflectivity, [0.1, 0.3, 0.3, 0.3])