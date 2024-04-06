import os
import subprocess
import shutil

import numpy as np
import pandas as pd
import pyart

class sample_data():
    def __init__(self):
        self.big_df = pd.read_csv("metadata.csv")
        
        self.srv = 'radaric@192.168.100.11'
        self.source = '/data/data_WF/NhaBe'
        self.destination = 'sample_data/data_WF/temp'
        self.password = 'abcd@1234'

    def choose_n_images(self, n=1000):
        df = self.big_df[(self.big_df['generated'] != "Error") & (self.big_df['future_path'] != "NotAvail")]
        df = df.sample(n=n)

        print(df.head())

        image_files = df['timestamp'].tolist()
        current_image_files = [file for file in os.listdir("sample_data/data_WF/NhaBe") if file.endswith('.jpg')]

        for file in image_files:
            if file in current_image_files:
                continue
            if os.path.exists(f"image/unlabeled1/{file}.jpg"):
                shutil.copy(f"image/unlabeled1/{file}.jpg", f"sample_data/data_WF/temp/{file}.jpg")
            elif os.path.exists(f"image/unlabeled2/{file}.jpg"):
                shutil.copy(f"image/unlabeled2/{file}.jpg", f"sample_data/data_WF/temp/{file}.jpg")

        df = df['path']
        df.to_csv("sample_data/data_WF/temp.csv", index=False)
        
    def copy_from_server(self):
        df = pd.read_csv("sample_data/data_WF/temp.csv")
        filenames = os.listdir("sample_data/data_WF/temp")
        
        for _, row in df.iterrows():
            filename = row['path'][-16:]
            if filename in filenames:
                continue
            command = f"sshpass -p {self.password} scp {self.srv}:'{self.source}/{row['path']}' {self.destination}"
            try:
                subprocess.run(command, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print("Error executing SCP command:", e)
                break
        
        df.to_csv("sample_data/data_WF/temp.csv", index=False)
            
    def rename_raw_files(self):
        filenames = [file for file in os.listdir('sample_data/data_WF/temp') if not file.endswith('jpg') 
                                                                             and not file.endswith('png')]

        for name in filenames:
            try:
                # Read data
                data = pyart.io.read_sigmet(f"sample_data/data_WF/temp/{name}")
                timestamp = str(pyart.util.datetime_from_radar(data)).replace(':', '-')
                os.rename(f"sample_data/data_WF/temp/{name}", f"sample_data/data_WF/temp/{timestamp}")
            except Exception as e:
                continue
            
if __name__ == "__main__":
    obj = sample_data()
    
    if not os.path.exists("sample_data/data_WF/temp"):
        os.makedirs("sample_data/data_WF/temp")
    
    # obj.choose_n_images(5000)
    obj.copy_from_server()
    # obj.rename_raw_files()