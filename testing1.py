import pandas as pd

metadata_chunks = pd.read_csv("metadata.csv", chunksize=2000)
for chunk in metadata_chunks:
    print(123)