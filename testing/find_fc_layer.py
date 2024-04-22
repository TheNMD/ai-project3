import timm
from torch import nn

model = timm.create_model('convnext_base.fb_in22k', pretrained=True)
print(model)

num_feature = model.head.fc.in_features
model.head.fc = nn.Linear(in_features=num_feature, out_features=5)
print(model.head.fc.weight)
model.head.fc.weight.data.mul_(0.001)
print(model.head.fc.weight)