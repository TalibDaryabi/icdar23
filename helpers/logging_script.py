import logging
import os

def setup_logging(file_name):
    """
    Sets up logging for the current script.
    - Ensures a `logs` folder exists in the script's directory.
    - Creates or overwrites a log file named after the script in the `logs` folder.
    - Returns a logger object.
    """
    # Get the directory of the current script
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Create `logs` folder in the script's directory
    log_folder = os.path.join(script_directory, "logs")
    os.makedirs(log_folder, exist_ok=True)  # Ensure the folder exists

    # Define log file path
    log_filename = os.path.join(log_folder, f"{os.path.splitext(file_name)[0]}.log")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename, mode='w'),  # Overwrite the log file each time
            logging.StreamHandler()  # Log to console
        ]
    )
    return logging.getLogger(file_name)
