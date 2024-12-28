import glob
import os
import bz2
import pickle

# Define your folder path
folder_path = '/tmp/uc46epev/icdar17_data/output_icdar_real_/color/train_5000'

# List all files in the directory with .pkl.bz2 extension
pkl_bz2_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pkl.bz2')]

data = None
for file_path in pkl_bz2_files:
    # Open the compressed file
    with bz2.BZ2File(file_path, "rb") as f:
        data = pickle.load(f)

# Calculate the total number of patches
total_patches = sum(len(patches) for patches in data.values())
print("good ")
