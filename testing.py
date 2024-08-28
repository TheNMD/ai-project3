import pandas as pd

metadata = pd.read_csv("metadata_temp.csv")
label_list = metadata['label_0h'].dropna()

print(len(label_list))