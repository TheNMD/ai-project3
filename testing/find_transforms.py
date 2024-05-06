import torch
from torchvision.transforms import v2
import random

import numpy as np
import cv2
from PIL import Image

image_size = 224

def median_blur(image, kernel_size=5):
    pil_image = v2.functional.to_pil_image(image)
    blurred_img = cv2.medianBlur(np.array(pil_image), kernel_size)
    return v2.functional.to_image(blurred_img)

transforms = v2.Compose([v2.ToImage(),
                         v2.Lambda(lambda image: median_blur(image, kernel_size=5)), 
                        #  v2.GaussianBlur(kernel_size=7, sigma=1.5),
                         v2.RandAugment(num_ops=2, magnitude=round(random.gauss(9, 0.5)), fill=255),
                        #  v2.RandomErasing(p=0.95, value=255),
                         v2.ToDtype(torch.float32, scale=True),
                         v2.Normalize(mean=[0.9844, 0.9930, 0.9632], std=[0.0641, 0.0342, 0.1163]),
                         v2.ToPILImage(),
                        ])

image = Image.open('image/labeled/7200/train/heavy_rain/2020-07-01 10-50-25.jpg')
image.save("test_transforms0.png")

image = transforms(image)
image.save("test_transforms3.png")
    
# import random

# num_6 = 0
# num_7 = 0
# num_8 = 0
# num_9 = 0
# num_10 = 0
# num_11 = 0
# num_12 = 0

# for i in range(200000):
#     num = round(random.gauss(9, 1))
#     if num == 6:
#         num_6 += 1
#     if num == 7:
#         num_7 += 1
#     if num == 8:
#         num_8 += 1
#     if num == 9:
#         num_9 += 1
#     if num == 10:
#         num_10 += 1
#     if num == 11:
#         num_11 += 1
#     if num == 12:
#         num_12 += 1

# print(num_6) 
# print(num_7)
# print(num_8)
# print(num_9)
# print(num_10)
# print(num_11)
# print(num_12)