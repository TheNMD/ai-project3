import numpy as np
import matplotlib.pyplot as plt
import random


magnitude = {'6': 0,
             '7' : 0, 
             '8': 0, 
             '9': 0, 
             '10' : 0, 
             '11': 0,
             '12': 0}

for i in range(200000):
    num = round(random.gauss(9, 0.5))
    magnitude[str(num)] += 1

for i in magnitude:
    print(magnitude[i])

plt.figure(figsize=(8, 6)) 
plt.bar(list(magnitude.keys()), list(magnitude.values()))

plt.xlabel('magnitude')
plt.ylabel('frequency')
plt.title('Magnitude distribution')

# Show the plot
plt.savefig("testing/magnitude1.png")
    

plt.clf()

plt.figure(figsize=(8, 6)) 
mean = 9
std = 0.5

# Create an array of x-values spanning 3 standard deviations below and above the mean
x = np.linspace(6, 12, 1000)

# Calculate the probability density function (PDF) of the Gaussian distribution
y = np.exp(-(x - mean)**2 / (2 * std**2))
plt.plot(x, y)

plt.xlabel('magnitude')
plt.ylabel('probability')
plt.title('Magnitude distribution')

# Show the plot
plt.savefig("testing/magnitude2.png")
    
