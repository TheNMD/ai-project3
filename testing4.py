import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Read the image and convert to grayscale
image = cv2.imread('image/labeled/heavy_rain/2023-06-29 06-20-32.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('testing1.png', gray)

# Step 2: Perform Gaussian blur
blur = cv2.GaussianBlur(gray,(3,3), 2)
cv2.imwrite('testing2.png', gray)

