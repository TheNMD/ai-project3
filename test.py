import pandas as pd

metadata = pd.read_csv("metadata_lite.csv")

frequency = metadata['future_label'].value_counts()
print(frequency)
with open('image/label_summary.txt', 'w') as file:
    # Write some text to the file
    file.write(str(frequency))