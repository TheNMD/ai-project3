import timm
import torch

def add_stochastic_depth(model_name, model, drop_prob):
    if model_name == "convnext":
        for layer in model.modules():
            if isinstance(layer, timm.models.convnext.ConvNeXtBlock):
                layer.drop_path = timm.layers.DropPath(drop_prob)
    return model

model = timm.create_model('tf_efficientnetv2_m.in21k', pretrained=True)

print(model)

num_feature = model.classifier.in_features
model.classifier = torch.nn.Linear(in_features=num_feature, out_features=5)
model.classifier.weight.data.mul_(0.001)

print(model)

# model = timm.create_model('convnext_base.fb_in22k', pretrained=True)
# # print(model)

# num_feature = model.head.fc.in_features

# model.head.fc = torch.nn.Linear(in_features=num_feature, out_features=5)
# model.head.fc.weight.data.mul_(0.001)
# model = add_stochastic_depth("convnext", model, 0.2)

# # print(model)