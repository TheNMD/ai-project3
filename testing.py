import pandas as pd

def find_future_label():
    metadata = pd.read_csv("metadata.csv")
    
    metadata_temp = metadata[['timestamp_0', 'avg_reflectivity_0', 'label_0']]
    
    for idx, row in metadata.iterrows():
        avg_reflectivity_7200 = metadata_temp[metadata_temp['timestamp_0'] == row['timestamp_7200'], 'avg_reflectivity_0'].tolist()[0]
        label_7200 = metadata_temp[metadata_temp['timestamp_0'] == row['timestamp_7200'], 'label_0'].tolist()[0]

        avg_reflectivity_21600 = metadata_temp[metadata_temp['timestamp_0'] == row['timestamp_21600'], 'avg_reflectivity_0'].tolist()[0]
        label_21600 = metadata_temp[metadata_temp['timestamp_0'] == row['timestamp_21600'], 'label_0'].tolist()[0]
        
        avg_reflectivity_43200 = metadata_temp[metadata_temp['timestamp_0'] == row['timestamp_43200'], 'avg_reflectivity_0'].tolist()[0]
        label_43200 = metadata_temp[metadata_temp['timestamp_0'] == row['timestamp_43200'], 'label_0'].tolist()[0]
        
        metadata.loc[idx, ['avg_reflectivity_7200']] = avg_reflectivity_7200
        metadata.loc[idx, ['label_7200']] = label_7200
        
        metadata.loc[idx, ['avg_reflectivity_21600']] = avg_reflectivity_21600
        metadata.loc[idx, ['label_21600']] = label_21600
        
        metadata.loc[idx, ['avg_reflectivity_43200']] = avg_reflectivity_43200
        metadata.loc[idx, ['label_43200']] = label_43200
        
        break