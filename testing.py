import timm
import torch
from safetensors import safe_open

tensors = {}
with safe_open("result/model.safetensors", framework="pt", device=0) as f:
    safetensor_data = {k: torch.tensor(v) for k, v in f.items()}

# Initialize the model
model = timm.create_model('convnext_large.fb_in22k', pretrained=False)

# Load the weights into the model
model.load_state_dict(safetensor_data)

# Set the model to evaluation mode
model.eval()

