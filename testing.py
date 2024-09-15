import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import distinctipy

# Generate 50 distinct colors
colors = distinctipy.get_colors(100)

# Create the colormap
cmap = ListedColormap(colors)

# Create some example data
data = np.random.rand(10, 10)

# Plot with the custom colormap
plt.imshow(data, cmap=cmap)
plt.colorbar()
plt.title('Custom Colormap with 50 Colors')

plt.savefig("test.jpg")
