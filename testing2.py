import os
import shutil

def copy_alternate_files(source_folder, destination_folder):
    # List all files in the source folder
    files = os.listdir(source_folder)

    # Iterate over the files and copy every other file
    for i, file_name in enumerate(files):
        if i % 2 == 0:  # Every other file
            source_path = os.path.join(source_folder, file_name)
            destination_path = os.path.join(destination_folder, file_name)
            shutil.move(source_path, destination_path)
            print(f"Copied '{file_name}' to '{destination_folder}'")

# Example usage
source_folder = 'image/unlabeled1'
destination_folder = 'image/unlabeled2'

copy_alternate_files(source_folder, destination_folder)
