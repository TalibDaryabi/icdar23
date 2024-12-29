#!/bin/bash

# Job name for logs and tracking
JOB_NAME="main"

echo "$pwd"
# Log file locations
LOG_FILE="/media/usb/icdar/icdar23/helpers/logs/main_log.txt"
ERROR_LOG="/media/usb/icdar/icdar23/log/extract_patches_error.txt"

# Python environment
VENV_PATH="/tmp/uc46epev/envs/icd"
PYTHON_PATH="/media/usb/icdar/icdar23/main.py"

# Add Python module search path
export PYTHONPATH="/media/usb/icdar/icdar23:$PYTHONPATH"

# Activate virtual environment
source "$VENV_PATH/bin/activate"

cd /media/usb/icdar/icdar23
# Run the Python script with all required parameters
nohup python $PYTHON_PATH > "$LOG_FILE" 2> "$ERROR_LOG" &
