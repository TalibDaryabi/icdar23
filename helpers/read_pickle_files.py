import glob
import os
import bz2
import pickle

# Define your folder path
folder_path = '/tmp/uc46epev/icdar17_data/ouput_icdar_temp/my_script/all_patches_single_file'

# List all files in the directory with .pkl.bz2 extension
pkl_bz2_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pkl.bz2')]


for file_path in pkl_bz2_files:
    # Open the compressed file
    with bz2.BZ2File(file_path, "rb") as f:
        data = pickle.load(f)
