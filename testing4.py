import torch
from torchvision.transforms import v2
from torchvision import datasets
from PIL import Image
import numpy as np

def black_to_white(image):
    print(image)

transforms = v2.Compose([v2.ToImage(), 
                        v2.Resize((256, 256)),
                        v2.RandomHorizontalFlip(p=0.1),
                        v2.RandomVerticalFlip(p=0.1),
                        v2.RandomAffine(degrees=(-180, 180), translate=(0.2, 0.2), fill=255),
                        v2.ToDtype(torch.float32, scale=True),
                        # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        v2.ToPILImage(),
                        ])

# Step 1: Read the image and convert to grayscale
image = Image.open('image/labeled/heavy_rain/2023-06-29 06-20-32.jpg')

image_array = np.array(image)

# print(image_array)

image = transforms(image)



image.save("testing3.png")