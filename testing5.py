import torch
from torchvision import datasets
from torchvision.transforms import v2

# Define your dataset and data loader
dataset = datasets.ImageFolder(root='image/sets/train',
                               transform=v2.ToTensor())

# Initialize variables to accumulate pixel values
mean = torch.zeros(3)
std = torch.zeros(3)

# Compute mean and standard deviation
for inputs, _ in dataset:
    mean += torch.mean(inputs, dim=(1, 2))
    std += torch.std(inputs, dim=(1, 2))

# Compute overall mean and standard deviation
mean /= len(dataset)
std /= len(dataset)

print("Mean:", mean)
print("Std Deviation:", std)
