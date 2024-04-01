import pandas as pd

# Sample DataFrame
df = pd.read_csv("metadata.csv")

# Column name to check for duplicates
column_name = 'timestamp'

duplicate_mask = df.duplicated(subset=[column_name], keep=False)
duplicate_indices = df[duplicate_mask].index.tolist()

print("Duplicate indices:", len(duplicate_indices))
