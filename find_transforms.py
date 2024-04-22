import torch
from torchvision.transforms import v2
from PIL import Image

image_size = 224

transforms = v2.Compose([v2.ToImage(), 
                         v2.Resize((image_size, image_size)),
                        #  v2.RandAugment(num_ops=2, magnitude=9, fill=255),
                        #  v2.RandomErasing(p=0.95, value=255),
                         v2.ToDtype(torch.float32, scale=True),
                         v2.Normalize(mean=[0.9844, 0.9930, 0.9632], std=[0.0641, 0.0342, 0.1163]),
                         v2.ToPILImage(),
                        ])

image = Image.open('image/unlabeled2/2020-07-01 10-50-25.jpg')
image.save("test_transforms0.png")
image = transforms(image)
image.save("test_transforms1.png")