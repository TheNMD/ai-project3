import os
import pandas as pd

def update_metadata():
    old_metadata = pd.read_csv("metadata.csv")
    
    if 'generated' in old_metadata.columns:
        old_metadata.drop(columns="generated", inplace=True)
    
    files = [file for file in os.listdir("image/unlabeled")]
    timestamps = [file[:-4] for file in files]
    generated = ["True" if file.endswith('.jpg') else "Error" for file in files]
    new_metadata = pd.DataFrame({'timestamp': timestamps, 'generated': generated})
    
    updated_metadata = pd.merge(old_metadata, new_metadata, on='timestamp', how='outer')
    updated_metadata.to_csv("metadata.csv", index=False)
    
    
update_metadata()