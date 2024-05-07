import random, torch, torchvision
from torchvision.transforms import v2


import numpy as np
import cv2 as cv
from PIL import Image

image_size = 224

def median_blur(image, kernel_size):
    pil_image = v2.functional.to_pil_image(image)
    blurred_img = cv.medianBlur(np.array(pil_image), kernel_size)
    return v2.functional.to_image(blurred_img)

# "Identity"     : pass
# "ShearX"
# "ShearY"
# "TranslateX"
# "TranslateY"
# "Rotate"

# "Brightness"   : F.adjust_brightness(img, 1.0 + magnitude)
# "Contrast"     : F.adjust_contrast(img, 1.0 + magnitude)
# "Sharpness"    : F.adjust_sharpness(img, 1.0 + magnitude)
# "Posterize"    : F.posterize(img, int(magnitude))
# "Solarize"     : F.solarize(img, magnitude)
# "AutoContrast" : F.autocontrast(img)
# "Equalize"     : F.equalize(img)

class CustomRandAugment(v2.RandAugment):
    def __init__(self, num_ops, magnitude, fill):
        super().__init__(num_ops=num_ops, magnitude=magnitude, fill=fill)
            
        del self._AUGMENTATION_SPACE['Brightness']
        del self._AUGMENTATION_SPACE['Color']
        del self._AUGMENTATION_SPACE['Contrast']
        del self._AUGMENTATION_SPACE['Sharpness']
        del self._AUGMENTATION_SPACE['Posterize']
        del self._AUGMENTATION_SPACE['Solarize']
        del self._AUGMENTATION_SPACE['AutoContrast']
        del self._AUGMENTATION_SPACE['Equalize']


transforms = v2.Compose([v2.ToImage(),
                         v2.Lambda(lambda image: median_blur(image, 5)),
                         CustomRandAugment(num_ops=2, magnitude=round(random.gauss(9, 0.5)), fill=255),
                         v2.Lambda(lambda image: v2.functional.autocontrast(image)),
                        #  v2.RandAugment(num_ops=2, magnitude=round(random.gauss(9, 0.5)), fill=255),
                        #  v2.RandomErasing(p=0.95, value=255),
                         v2.ToDtype(torch.float32, scale=True),
                        #  v2.Normalize(mean=[0.9844, 0.9930, 0.9632], 
                        #               std=[0.0641, 0.0342, 0.1163]),
                         v2.ToPILImage(),
                        ])

image = Image.open('sample_data/SomeImages/2020-07-01 07-10-43.jpg')
print("Original image size (width, height):", image.size)
image.save("test_transforms0.png")

for i in range(1, 11):
    image_transformed = transforms(image)
    image_transformed.save(f"test_transforms{i}.png")


    