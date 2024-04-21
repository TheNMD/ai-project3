import pandas as pd
import matplotlib.pyplot as plt



def plot_distribution(interval):
    metadata = pd.read_csv("metadata.csv")
    metadata = metadata[[f'avg_reflectivity_{interval}', f'label_{interval}']]
    metadata = metadata[metadata[f'label_{interval}'] != 'NotAvail']
    
    frequency = metadata[f'label_{interval}'].value_counts()
    with open(f'label_dist_{interval}.txt', 'w') as file:
        file.write(f"{frequency}")
    
    plt.bar(frequency.index, frequency.values, color='skyblue', edgecolor='black')
    plt.xlabel('Label')
    plt.ylabel('Frequency')
    plt.title(f'Label Distribution - {interval}')
    plt.savefig(f'label_dist_{interval}.png')
    plt.clf()
    
    _, _, _ = plt.hist(metadata[f'avg_reflectivity_{interval}'], color='skyblue', edgecolor='black')
    plt.xlabel('Avg Reflectivity')
    plt.ylabel('Frequency')
    plt.title(f'Avg Reflectivity Distribution - {interval}')
    plt.savefig(f'avg_reflectivity_dist_{interval}.png')
    plt.clf()

metadata = pd.read_csv("metadata.csv")

for idx, row in metadata.iterrows():
    if row['avg_reflectivity_0'] < 30:
        metadata.loc[idx, ['label_0']] = "clear"
    elif 30 <= row['avg_reflectivity_0']  < 40:
        metadata.loc[idx, ['label_0']] = "light_rain"
    elif 40 <= row['avg_reflectivity_0']  < 47.5:
        metadata.loc[idx, ['label_0']] = "moderate_rain"
    elif 47.5 <= row['avg_reflectivity_0']  < 55:
        metadata.loc[idx, ['label_0']] = "heavy_rain"
    elif row['avg_reflectivity_0']  >= 55:
        metadata.loc[idx, ['label_0']] = "very_heavy_rain"
    print(idx)
        
metadata.to_csv("metadata_temp.csv", index=False)

plot_distribution(0)