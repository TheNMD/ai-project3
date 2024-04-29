import timm

model = timm.create_model('convnext_base.fb_in22k', pretrained=True)

lr = 1e-3
lr_decay = 0.8
weight_decay = 1e-8
layers = [layer for layer in model.modules() if isinstance(layer, timm.models.convnext.ConvNeXtBlock)]
optimizer_settings = [{'params': layers[i].parameters(), 
                        'lr': lr * pow(lr_decay, i), 
                        'betas' : (0.9, 0.999), 
                        'weight_decay' : weight_decay} for i in range(len(layers))]
    
print(len(optimizer_settings))