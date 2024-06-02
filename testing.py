import timm, torchsummary
import torch
from safetensors.torch import load_file, load_model

# file_path = "result/model.safetensors"
# model = load_file(file_path)

# print(model)

# print(torchsummary.summary(model, (3, 224, 224)))

model = timm.create_model('convnext_large.fb_in22k', pretrained=True)
print(type(model))