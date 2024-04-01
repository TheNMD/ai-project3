import os

image_files = [file for file in os.listdir("image/unlabeled")]
generated = ["True" if file.endswith('.jpg') else "Error" for file in image_files]

print(generated[:10])