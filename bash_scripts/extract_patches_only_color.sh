#!/bin/bash

# Job name for logs and tracking
JOB_NAME="extract_patches_job"


# Log file locations
LOG_FILE="/media/usb/icdar/icdar23/log/extract_patches_only_color.txt"
ERROR_LOG="/media/usb/icdar/icdar23/log/extract_patches_error.txt"

# Python environment
VENV_PATH="/tmp/uc46epev/envs/icd"
PYTHON_PATH="/media/usb/icdar/icdar23/helpers/extract_patches_only_color.py"

# Parameters
# Paths to input and output directories
IN_DIR="/tmp/uc46epev/icdar17_data/input_icdar_real_/color/test"
OUT_DIR="/tmp/uc46epev/icdar17_data/output_icdar_real_/color/test"
WIN_SIZE=32
# config value is 2k , default value is -1
PATCHES_PER_PAGE=2000
SCALE=-1
# no value in Config
EDGE_PIXELS=0.1
# config value is 2.5 , default value is 1.6
SIGMA=2.5

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Run the Python script with all required parameters
#nohup python $PYTHON_PATH \
#    --in_dir "$IN_DIR" \
#    --out_dir "$OUT_DIR" \
#    --win_size "$WIN_SIZE"\
#    --patches_per_page $PATCHES_PER_PAGE \
#    --scale "$SCALE"\
#    --sigma $SIGMA \
#    --edge_pixels "$EDGE_PIXELS"\
#    > "$LOG_FILE" 2> "$ERROR_LOG" &


python $PYTHON_PATH \
    --in_dir "$IN_DIR" \
    --out_dir "$OUT_DIR" \
    --win_size "$WIN_SIZE"\
    --patches_per_page $PATCHES_PER_PAGE \
    --scale "$SCALE"\
    --sigma $SIGMA \
    --edge_pixels "$EDGE_PIXELS"\
    > "$LOG_FILE" 2> "$ERROR_LOG"

# Print the PID of the background process
#echo "Job $JOB_NAME is running in the background with PID $!"
