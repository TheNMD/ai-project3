import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import distinctipy
from _find_province import list_all_provinces

with open(f'coordinate/provinces_300km.pkl', 'rb') as file:
    data = pickle.load(file)
    data = data.astype(int)
    data = data[::-1]

province_list = list_all_provinces()
name_list = [ele['name'] for ele in province_list]
print(name_list)

num_colors = len(name_list) + 1
colors = distinctipy.get_colors(num_colors, colorblind_type="Deuteranomaly")
cmap = ListedColormap(colors)

plt.figure(figsize=(12, 10))
plt.imshow(data, cmap=cmap, interpolation='nearest')
plt.title('My Plot Title')
plt.axis('off')
cbar = plt.colorbar(ticks=np.arange(0, num_colors))
cbar_labels = ["-1 - NotAvailable"] + [f"{i} - {name_list[i]}" for i in range(num_colors - 1)]
cbar.ax.set_yticklabels(cbar_labels, fontsize=14)

plt.savefig("test.png")
