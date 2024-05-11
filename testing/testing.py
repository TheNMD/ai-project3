import pandas as pd

# metadata = pd.read_csv("metadata_temp.csv")

# df_even = metadata.iloc[::2].copy()
# df_even.to_csv("metadata_even.csv", index=False)

# df_odd = metadata.iloc[1::2].copy()
# df_odd.to_csv("metadata_odd.csv", index=False)

df1 = pd.read_csv("metadata_odd_7200.csv")
df2 = pd.read_csv("metadata_even_7200.csv")

merged_df = pd.concat([df1, df2], ignore_index=True)
merged_df = merged_df.sort_values(by='timestamp_0').reset_index(drop=True)

merged_df.to_csv("metadata_7200.csv", index=False)