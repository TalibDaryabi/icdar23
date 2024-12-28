#!/bin/bash

# Job name for logs and tracking
JOB_NAME="extract_patches_job"

# Paths to input and output directories
IN_DIR="/tmp/uc46epev/icdar17_data/input_icdar_real_/color/train/icdar2017-training-color"
OUT_DIR="/tmp/uc46epev/icdar17_data/output_icdar_real_/color/train"

# Log file locations
LOG_FILE="/media/usb/icdar/icdar23/log/extract_patches_color"
ERROR_LOG="/media/usb/icdar/icdar23/log/extract_patches_error.txt"

# Python environment
VENV_PATH="/tmp/uc46epev/envs/icd"

PYTHON_PATH="/media/usb/icdar/icdar23/helpers/extract_patches_color.py"

# Parameters
NUM_CLUSTERS=1000
PATCHES_PER_PAGE=-1
SIGMA=2.5

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Run the Python script with all required parameters
#nohup python $PYTHON_PATH \
#    --in_dir "$IN_DIR" \
#    --out_dir "$OUT_DIR" \
#    --num_of_clusters $NUM_CLUSTERS \
#    --patches_per_page $PATCHES_PER_PAGE \
#    --sigma $SIGMA \
#    > "$LOG_FILE" 2> "$ERROR_LOG" &

$PYTHON_PATH \
    --in_dir "$IN_DIR" \
    --out_dir "$OUT_DIR" \
    --num_of_clusters $NUM_CLUSTERS \
    --patches_per_page $PATCHES_PER_PAGE \
    --sigma $SIGMA \
    > "$LOG_FILE" 2> "$ERROR_LOG" &


# Print the PID of the background process
#echo "Job $JOB_NAME is running in the background with PID $!"
