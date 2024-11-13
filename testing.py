import torch
from torch.utils.tensorboard import SummaryWriter

from _init_model import load_model

# Hyperparameters
## For model
radar_range = "300km" # 120km | 300km
interval = "3h" # 0h | 1h | 2h | 3h | 4h | 5h | 6h
# convnext-s | convnext-b | convnext-l
# vit-s      | vit-b      | vit-l
# swin-s     | swin-b
# effnetv2-s | effnetv2-m
model_name = "convnext-b"
model_opt = "pretrained" # pretrained | custom
classes = 5
sdepth = 0.2 # 0.0 | 0.1 | 0.2 | 0.3
past_image_num = 6 # 0 | 6 | 12 | 18
combined_method = "concat" # sum | concat

model, image_size = load_model(
    model_name,
    model_opt,
    classes,
    sdepth,
    past_image_num,
    combined_method
)

# Initialize a TensorBoard writer
writer = SummaryWriter()

if combined_method == "sum":
    sample_input = torch.randn(1, 3, image_size['train_size'], image_size['train_size'])
else:
    sample_input = torch.randn(1, (past_image_num + 1) * 3, image_size['train_size'], image_size['train_size'])

# from torchviz import make_dot
# y = model(sample_input)
# graph = make_dot(y, params=dict(model.named_parameters()))
# graph.render("graph", format="png")

writer.add_graph(model, sample_input)
writer.close()

