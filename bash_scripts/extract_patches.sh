#!/bin/bash

# Job name for logs and tracking
JOB_NAME="extract_patches_job"


# Log file locations
LOG_FILE="/media/usb/icdar/icdar23/log/extract_patches"
ERROR_LOG="/media/usb/icdar/icdar23/log/extract_patches_error.txt"

# Python environment
VENV_PATH="/tmp/uc46epev/envs/icd"
PYTHON_PATH="/media/usb/icdar/icdar23/helpers/extract_patches.py"

# Parameters
# Paths to input and output directories
IN_DIR="/tmp/uc46epev/icdar17_data/input_icdar_real_/bin/train/icdar2017-training-binary"
OUT_DIR="/tmp/uc46epev/icdar17_data/output_icdar_real_/bin/train_5000"
WIN_SIZE=32
PATCHES_PER_PAGE=-1
SCALE=-1
NUM_CLUSTER=5000
BLACK_PIXEL_THRESH=-1
WHITE_PIXEL_THRESH=0.95
CENTERED=True
SIGMA=2.5

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Run the Python script with all required parameters
nohup python $PYTHON_PATH \
    --in_dir "$IN_DIR" \
    --out_dir "$OUT_DIR" \
    --win_size "$WIN_SIZE"\
    --patches_per_page $PATCHES_PER_PAGE \
    --scale "$SCALE"\
    --sigma $SIGMA \
    --edge_pixels "$EDGE_PIXELS"\
    > "$LOG_FILE" 2> "$ERROR_LOG" &


#python $PYTHON_PATH \
#    --in_dir "$IN_DIR" \
#    --out_dir "$OUT_DIR" \
#    --win_size "$WIN_SIZE"\
#    --num_of_clusters "$NUM_CLUSTER"\
#    --patches_per_page $PATCHES_PER_PAGE \
#    --scale "$SCALE"\
#    --sigma "$SIGMA" \
#    --black_pixel_thresh "$BLACK_PIXEL_THRESH"\
#    --white_pixel_thresh "$WHITE_PIXEL_THRESH"\
#    --centered "$CENTERED"
#    > "$LOG_FILE" 2> "$ERROR_LOG"

# Print the PID of the background process
#echo "Job $JOB_NAME is running in the background with PID $!"
