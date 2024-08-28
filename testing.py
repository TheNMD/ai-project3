import os
import pandas as pd

old_metadata = pd.read_csv("metadata.csv")

files = [file for file in os.listdir("image/all")]
timestamps = [file[:-4] for file in files]
generated = ["True" if file.endswith('.jpg') else "Error" for file in files]

old_metadata.loc[:len(generated) - 1, 'generated'] = generated
old_metadata.to_csv("metadata_temp.csv", index=False)

print("Done")