import os
import json
import pickle
import numpy as np

def find_provinces():
    provinces = [file[:-5] for file in os.listdir("coordinate") if file.endswith('.json')]
    provinces_info = {}

    for province in provinces:  
        with open(f'coordinate/{province}.json', 'r') as file:
            data = json.load(file)

        coordinates = data['coordinates'][0][0]
        longitude_array = np.array([coordinate[0] for coordinate in coordinates])
        latitude_array = np.array([coordinate[1] for coordinate in coordinates])

        provinces_info[province] = {'long_max': np.max(longitude_array),
                                    'long_min': np.min(longitude_array),
                                    'lat_max': np.max(latitude_array),
                                    'lat_min': np.min(latitude_array)}
    
        
    with open("coordinate/provinces_info.pkl", "wb") as f:
        pickle.dump(provinces_info, f)
    print(provinces_info)
    print(f"Num provinces: {len(provinces_info)}")
    
    return provinces_info
    
# provinces_info = find_provinces()

with open("coordinate/provinces_info.pkl", "rb") as f:
    provinces_info = pickle.load(f)
print(provinces_info)
print(f"Num provinces: {len(provinces_info)}")