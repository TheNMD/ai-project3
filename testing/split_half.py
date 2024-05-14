import pandas as pd

metadata = pd.read_csv("testing/metadata_temp.csv")

subset1 = pd.DataFrame()
subset2 = pd.DataFrame()

for idx, row in metadata.iterrows():            
    current_time = row['timestamp_0']
    
    hour_minute_second = current_time.split(" ")[1]
    minute = hour_minute_second.split("-")[1]
    
    if minute in ["00", "01", "02", "10", "11", "12", "20", "21", "22", 
                  "30", "31", "32", "40", "41", "42", "50", "51", "52"]: 
        subset1 = subset1._append(row)
    elif minute in ["03", "04", "05", "13", "14", "15", "23", "24", "25",
                    "33", "34", "35", "43", "44", "45", "53", "54", "55"]: 
        subset2 = subset2._append(row)
    else:
        print(current_time)
    
subset1.to_csv("testing/medatada_odd_temp.csv", index=False)
subset2.to_csv("testing/medatada_even_temp.csv", index=False)
