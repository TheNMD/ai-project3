import os
import pandas as pd

metadata = pd.read_csv("metadata.csv")

metadata_odd = pd.DataFrame()
for file in os.listdir("image/unlabeled1"):
    if file.endswith("txt"): continue
    timestamp = file[:-4]
    row = metadata[metadata['timestamp_0'] == timestamp]
    metadata_odd = metadata_odd._append(row, ignore_index=True)
metadata_odd.to_csv("metadata_odd.csv", index=False)   
    
metadata_even = pd.DataFrame()
for file in os.listdir("image/unlabeled2"):
    if file.endswith("txt"): continue
    timestamp = file[:-4]
    row = metadata[metadata['timestamp_0'] == timestamp]
    metadata_even = metadata_even._append(row, ignore_index=True) 
metadata_even.to_csv("metadata_even.csv", index=False)   