import os
import h5py
import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib.colors import Normalize
# from metpy.plots import ctables

metadata = pd.read_csv("dataset/metadata.csv", encoding="utf-8").set_index('id')

datetime = metadata['start_datetime']
datetime = pd.to_datetime(datetime, format="%Y-%m-%d %H:%M:%S")
datetime = datetime.dt.strftime('%Y%m%d_%H%M%S').tolist()

for filename in sorted(os.listdir('dataset/hdf5')):
    if filename == "all_data.hdf5":
        continue
    if f"{datetime[0]}.png" in os.listdir('dataset/image'):
        temp = datetime.pop(0)
        # print(temp)
        continue
    with h5py.File(f'dataset/hdf5/{filename}', "r") as file:
        for key in list(file.keys()):
            obj = file[key][()][0]
            obj[obj < 5] = 0
            save_name = datetime.pop(0)
            plt.imshow(obj, cmap='gist_ncar', interpolation='nearest')
            plt.axis('off')
            plt.savefig(f'dataset/image/{save_name}.png', bbox_inches='tight', pad_inches=0)
            plt.clf()

# with h5py.File(f'dataset/hdf5/20190102.hdf5', "r") as file:
#     for key in list(file.keys()):
#         obj = file[key][()][0]
#         obj[obj < 5] = 0
#         plt.imshow(obj, cmap='gist_ncar', interpolation='nearest')
#         # plt.axis('off')
#         plt.colorbar()
#         plt.savefig(f'test_{key}.png')
#         plt.clf()
