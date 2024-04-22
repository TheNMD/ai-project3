import os
import cv2
import numpy as np

def calculate_color_percentage(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = gray_image.flatten()
    non_white_pixels = np.array(gray_image)
    non_white_pixels = non_white_pixels[non_white_pixels < 255]
    percentage = (len(non_white_pixels) / len(gray_image))
    return percentage

def list_images_with_low_color_percentage(folder_path, threshold=0.1):
    counter = 0
    for filename in os.listdir(folder_path)[:1000]:
        if filename.endswith('.jpg'):
            file_path = os.path.join(folder_path, filename)
            # Read the image
            image = cv2.imread(file_path)
            # Calculate color percentage
            color_percentage = calculate_color_percentage(image)
            # Check if color percentage is less than the threshold
            if color_percentage < threshold:
                print(f"Image '{filename}': Color Percentage = {color_percentage}")
                counter += 1
    print(f"Total images: {counter}")

folder_path = "image/unlabeled1"
list_images_with_low_color_percentage(folder_path, threshold=0.05)
