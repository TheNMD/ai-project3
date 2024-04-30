import timm

model = timm.create_model('convnext_base.fb_in22k', pretrained=True)

learning_rate = 1e-3
lr_decay = 0.8
weight_decay = 1e-8

# # layer-wise lr decay
# # layers = [layer.parameters() for layer in model.modules() if isinstance(layer, timm.models.convnext.ConvNeXtBlock)]
# layers = [param for _, param in model.named_parameters() if param.requires_grad]

# optimizer_settings = [{'params': layers[depth], 
#                        'lr': learning_rate * pow(lr_decay, depth), 
#                        'betas' : (0.9, 0.999), 
#                        'weight_decay' : weight_decay} for depth in range(len(layers))]

# print(len(optimizer_settings))

for name, param in model.named_parameters():
    print(name)