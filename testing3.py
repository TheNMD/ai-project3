import timm
from torch import nn

model = timm.create_model('resnet50.a1_in1k', pretrained=True)
print(model)

num_feature = model.fc.in_features
model.fc = nn.Linear(in_features=num_feature, out_features=5)