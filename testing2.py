import subprocess

import pandas as pd

df = pd.read_csv("data/data_WF/temp.csv")

raw_files = df['path'].tolist()

# Define the SCP command
srv = 'radaric@192.168.100.11'
source = '/data/data_WF/NhaBe'
destination = 'data/data_WF/temp'
password = 'abcd@1234'

for i in range(len(raw_files)): 
    command = f"sshpass -p {password} scp {srv}:'{source}/{raw_files[i]}' {destination}"
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Error executing SCP command:", e)
        break