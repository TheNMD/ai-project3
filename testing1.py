import os, sys, platform, shutil, time
import multiprocessing as mp
import warnings, logging
warnings.filterwarnings('ignore')
logging.basicConfig(filename='errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

import pyart
import numpy as np
import pandas as pd

data = pyart.io.read_sigmet("sample_data/2022-09-09 03-00-42")
data.fields['reflectivity']['data'] = data.fields['reflectivity']['data'].astype(np.float16)

grid_data = pyart.map.grid_from_radars(data, 
                                       grid_shape=(1, 500, 500),
                                       grid_limits=((0, 1), (-250000, 250000), (-250000, 250000)))

reflectivity_array = np.array(grid_data.fields['reflectivity']['data'])[0]
longitude_array = np.array(grid_data.point_longitude['data'])[0]
latitude_array = np.array(grid_data.point_latitude['data'])[0]

print(f"Long Max = {np.max(longitude_array)}")
print(f"Long Min = {np.min(longitude_array)}")
print(f"Lat Max = {np.max(latitude_array)}")
print(f"Lat Min = {np.min(latitude_array)}")