import timm
from torch import nn

def add_stochastic_depth(model_name, model, drop_prob):
    if model_name == "convnext":
        for layer in model.modules():
            if isinstance(layer, timm.models.convnext.ConvNeXtBlock):
                layer.drop_path = timm.layers.DropPath(drop_prob)
    return model

model = timm.create_model('convnext_base.fb_in22k', pretrained=True)

num_feature = model.head.fc.in_features
model.head.fc = nn.Linear(in_features=num_feature, out_features=5)
model.head.fc.weight.data.mul_(0.001)
model = add_stochastic_depth("convnext", model, 0.2)

print(model)