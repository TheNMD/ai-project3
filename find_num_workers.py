import os, time, random
import warnings, logging
warnings.filterwarnings('ignore')
logging.basicConfig(filename='errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

import torch, torchvision
from torchvision.transforms import v2
from torchvision.io import read_image
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import cv2 as cv

def median_blur(image, kernel_size):
    pil_image = v2.functional.to_pil_image(image)
    blurred_img = cv.medianBlur(np.array(pil_image), kernel_size)
    return v2.functional.to_image(blurred_img)

class CustomRandAugment(v2.RandAugment):
    def __init__(self, num_ops, magnitude, fill):
        super().__init__(num_ops=num_ops, magnitude=magnitude, fill=fill)
            
        # del self._AUGMENTATION_SPACE['Brightness']
        # del self._AUGMENTATION_SPACE['Color']
        # del self._AUGMENTATION_SPACE['Contrast']
        # del self._AUGMENTATION_SPACE['Sharpness']
        # del self._AUGMENTATION_SPACE['Posterize']
        # del self._AUGMENTATION_SPACE['Solarize']
        del self._AUGMENTATION_SPACE['AutoContrast']
        # del self._AUGMENTATION_SPACE['Equalize']

transforms = v2.Compose([v2.ToImage(),
                         v2.Resize((224, 224)),
                         CustomRandAugment(num_ops=2, magnitude=round(random.gauss(9, 0.5)), fill=255),
                         v2.Lambda(lambda image: median_blur(image, 5)),
                         v2.Lambda(lambda image: v2.functional.autocontrast(image)),
                         v2.ToDtype(torch.float32, scale=True),
                         v2.Normalize(mean=[0.9844, 0.9930, 0.9632], 
                                      std=[0.0641, 0.0342, 0.1163],
                                      inplace=True),
                        ])

class CustomImageDataset(Dataset):
    def __init__(self, img_labels, img_dir, transform, past_image_num, full_image_list):
        self.img_labels = img_labels
        self.img_dir = img_dir
        self.transform = transform
        self.past_image_num = past_image_num
        self.full_image_list = full_image_list

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        label = self.img_labels.iloc[idx, 2]
        
        img_name = self.img_labels.iloc[idx, 0]
        img_names = self.load_past_images(img_name) + [img_name]
        
        img_paths = [os.path.join(self.img_dir, img) for img in img_names]
        images = [read_image(path) for path in img_paths]
        images = [self.transform(img) for img in images]
        
        # Sum
        images = torch.stack(images, dim=0)
        if self.past_image_num > 0:
            images = torch.sum(images, dim=0)
            mean = torch.mean(images)
            std = torch.std(images)
            images = (images - mean) / std
        else:
            images = torch.squeeze(images, dim=0)
        
        # # Concat
        # images = torch.cat(tuple(images), dim=0)
        
        return images, label
    
    def load_past_images(self, img_name):
        idx = self.full_image_list.index[self.full_image_list == img_name][0]
        past_images = self.full_image_list[idx - self.past_image_num : idx].tolist()
        return past_images

label_file = pd.read_csv("image/sets/300km_0h_train.csv")
full_image_list = pd.read_csv("image/labels_300km.csv")
full_image_list = full_image_list[full_image_list['range'] == "300km"]['image_name']
dataset = CustomImageDataset(img_labels=label_file, 
                             img_dir="image/all", 
                             transform=transforms,
                             past_image_num=0,
                             full_image_list=full_image_list)

for num in [2, 4, 6, 8, 10, 12, 14, 16]:
    start_time = time.time()
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=128, 
                                             shuffle=True, 
                                             num_workers=num)

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # print(f'Number of workers: {num} | Batch {batch_idx + 1}')
        pass
    
    end_time = time.time() - start_time
    
    print(f"Number of workers: {num} | Time: {end_time}s")
    with open('testing_results.txt', 'a') as file:
        file.write(f"Number of workers: {num} | Time: {end_time}s\n")
    