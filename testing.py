import pandas as pd

df = pd.read_csv("metadata.csv")

df = df[df['future_timestamp'] != "NotAvail"]

print(df)