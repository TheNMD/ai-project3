import pandas as pd

metadata = pd.read_csv("metadata.csv")

for idx, row in metadata.iterrows():
    if row['avg_reflectivity_0'] < 30:
        metadata.loc[idx, ['label_0']] = "clear"
    elif 30 <= row['avg_reflectivity_0']  < 40:
        metadata.loc[idx, ['label_0']] = "light_rain"
    elif 40 <= row['avg_reflectivity_0']  < 42.5:
        metadata.loc[idx, ['label_0']] = "moderate_rain"
    elif 42.5 <= row['avg_reflectivity_0']  < 55:
        metadata.loc[idx, ['label_0']] = "heavy_rain"
    elif row['avg_reflectivity_0']  > 55:
        metadata.loc[idx, ['label_0']] = "very_heavy_rain"
    print(idx)
        
metadata.to_csv("metadata_temp.csv", index=False)