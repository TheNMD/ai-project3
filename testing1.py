import pandas as pd
import numpy as np

df = pd.read_csv("metadata.csv")
df['future_label'] = np.nan
df['future_avg_reflectivity'] = np.nan

df.to_csv('metadata_temp.csv', index=False)