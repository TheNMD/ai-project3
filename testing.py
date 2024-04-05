import pandas as pd

df = pd.read_csv("metadata_temp.csv")

df = df['future_label']
df = df.dropna()

print(len(df))
