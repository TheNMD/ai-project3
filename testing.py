import timm

model = timm.create_model('convnext_base.fb_in22k', pretrained=True)

lr = 1e-3
lr_decay_factor = 0.8
weight_decay = 1e-8
counter = 0
optimizer_settings = []

for layer in model.modules():
    if isinstance(layer, timm.models.convnext.ConvNeXtBlock):
        if counter != 0: lr *= lr_decay_factor
        optimizer_settings += [{'params': layer.parameters(), 
                                'lr': lr, 
                                'betas' : (0.9, 0.999), 
                                'weight_decay' : weight_decay}]
        counter += 1
    
print(len(optimizer_settings))