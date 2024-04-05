import pandas as pd

df = pd.read_csv("metadata.csv")

df = df['future_label']
df = df.dropna()

print(len(df))
