import os, pickle
import matplotlib.pyplot as plt
import distinctipy


with open('coordinate/provinces_300km.pkl', 'rb') as file:
    data = pickle.load(file)
    data = data.astype(int)
    data = data[::-1]

plt.imshow(data, cmap="hsv", interpolation='nearest')
plt.colorbar()
plt.savefig("testing/test_coordinate.png")
