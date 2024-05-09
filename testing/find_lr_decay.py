import timm

model = timm.create_model('convnext_base.fb_in22k', pretrained=True)

learning_rate = 1e-3
lr_decay = 0.8
weight_decay = 1e-8

# # global lr
# optimizer_settings = [{'params': model.parameters(), 
#                        'lr': learning_rate, 
#                        'betas' : (0.9, 0.999), 
#                        'weight_decay' : weight_decay}]


# layer-wise lr decay
optimizer_settings = []

layer_names = [n for n, p in model.named_parameters()]
layer_names.reverse()

previous_group_name = layer_names[0].split('.')[0]

for idx, name in enumerate(layer_names):
    current_group_name = name.split('.')[0]
    
    if current_group_name == "stages":
        current_block_num = int(name.split('.')[3])
        if current_block_num % 3 == 1: current_block_num += 1 
        elif current_block_num % 3 == 0: current_block_num += 2
        current_group_name = f"{name.split('.')[0]}{name.split('.')[1]}{name.split('.')[2]}{current_block_num}"
    
    if current_group_name != previous_group_name:
        learning_rate *= lr_decay
    
    previous_group_name = current_group_name
    
    print(f'{idx}: lr = {learning_rate:.6f}, {name}')

    optimizer_settings += [{'params': [p for n, p in model.named_parameters() if n == name and p.requires_grad], 
                            'lr': learning_rate, 
                            'betas' : (0.9, 0.999), 
                            'weight_decay' : weight_decay}]


lr_list = sorted(list(set([ele['lr'] for ele in optimizer_settings])), reverse=True)
print(lr_list)

# print(len(optimizer_settings))