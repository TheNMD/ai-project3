import pandas as pd

df = pd.read_csv("metadata.csv")

df = df[df['future_timestamp'] != "NotAvail"]

df.to_csv("metadata.csv", index=False)