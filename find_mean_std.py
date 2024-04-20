import torch
from torchvision import datasets
from torchvision.transforms import v2

# Define your dataset and data loader
transforms = v2.Compose([v2.ToImage(),
                         v2.ToDtype(torch.float32, scale=True),])
dataset = datasets.ImageFolder(root='image/labeled',
                               transform=transforms)

mean = torch.zeros(3)
std = torch.zeros(3)

counter = 0
for inputs, _ in dataset:
    mean += torch.mean(inputs, dim=(1, 2))
    std += torch.std(inputs, dim=(1, 2))
    counter += 1
    print(f"Image: {counter}")

mean /= len(dataset)
std /= len(dataset)

print(f"Mean: {mean}")
print(f"Std: {std}")
with open('image/mean_and_std.txt', 'w') as file:
    file.write(f"Mean: {mean}\n")
    file.write(f"Std: {std}")
