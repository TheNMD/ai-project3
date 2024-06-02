import pickle, timm, torchsummary

model = timm.create_model('convnext_large.fb_in22k', pretrained=True)

with open('model.pickle', 'wb') as f:
    pickle.dump(model, f)
    
with open('model.pickle', 'rb') as f:
    model = pickle.load(f)
    
print(model)