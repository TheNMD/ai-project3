import pyart
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

metadata = pd.read_csv("metadata.csv")
path = metadata['path'][3]
print(path)

data = pyart.io.read_sigmet(f"data/{path}")
data.fields['reflectivity']['data'] = data.fields['reflectivity']['data'].astype(np.float16)

# display = pyart.graph.RadarDisplay(data)
# display.plot_ppi("reflectivity")
# plt.savefig("testing/sample_radar_image1.jpg", dpi=150, bbox_inches='tight')
# plt.close()

            
grid_data = pyart.map.grid_from_radars(data,
                                        grid_shape=(1, 500, 500),
                                        grid_limits=((0, 1), (-250000, 250000), (-250000, 250000)),)

display = pyart.graph.GridMapDisplay(grid_data)
display.plot_grid('reflectivity', cmap='pyart_HomeyerRainbow', colorbar_flag=False)

plt.title('')
plt.axis('off')
plt.savefig("sample_radar_image.jpg", dpi=150, bbox_inches='tight')

print(dir(data))
print(data.fields.keys())
print(data.fields['reflectivity']['data'])

reflectivity = data.fields['reflectivity']['data']
print(f"Reflectivity data:\n{reflectivity}")
print(f"Shape: {reflectivity.shape}")