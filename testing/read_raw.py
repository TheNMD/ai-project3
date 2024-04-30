import pyart
import numpy as np

data_path = "sample_data/NhaBe"
filename = "2020-07-01 03-20-16"

data = pyart.io.read_sigmet(f"{data_path}/{filename}")

# print(dir(data))
# print(data.fields.keys())
# print(data.fields['reflectivity']['data'])

reflectivity = data.fields['reflectivity']['data'].astype(np.float16)

print(f"Reflectivity data:\n{reflectivity}")
print(f"Shape: {reflectivity.shape}")
