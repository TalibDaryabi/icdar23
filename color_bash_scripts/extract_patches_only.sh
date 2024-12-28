
#!/bin/bash

# Log file locations
LOG_FILE="/media/usb/icdar/icdar23/log/extract_patches"
ERROR_LOG="/media/usb/icdar/icdar23/log/extract_patches_error.txt"

# Python environment
VENV_PATH="/tmp/uc46epev/envs/icd"
PYTHON_PATH="/media/usb/icdar/icdar23/helpers/extract_patches_only.py"

# Parameters
# Paths to input and output directories
IN_DIR="/tmp/uc46epev/icdar17_data/input_icdar_real_/bin/test"
OUT_DIR="/tmp/uc46epev/icdar17_data/output_icdar_real_/bin/test"
WIN_SIZE=32
PATCHES_PER_PAGE=2000
SCALE=-1
BLACK_PIXEL_THRESH=-1
WHITE_PIXEL_THRESH=0.95
CENTERED=True
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
    --sigma "$SIGMA" \
    --black_pixel_thresh "$BLACK_PIXEL_THRESH"\
    --white_pixel_thresh "$WHITE_PIXEL_THRESH"\
    --centered "$CENTERED"
    > "$LOG_FILE" 2> "$ERROR_LOG"

# Print the PID of the background process
#echo "Job $JOB_NAME is running in the background with PID $!"









--in_dir "/tmp/uc46epev/icdar17_data/input_icdar_temp/binarized/test" --out_dir "/tmp/uc46epev/icdar17_data/ouput_icdar_temp/my_script/extract_patches_only" --patches_per_page 20 --sigma 2.5  --black_pixel_thresh -1 --white_pixel_thresh 0.95