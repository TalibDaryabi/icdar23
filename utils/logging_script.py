import logging
import os

def setup_logging(file_name):
    """
    Sets up logging for the current script.
    - Ensures a `logs` folder exists in the script's directory.
    - Creates or overwrites a log file named after the script in the `logs` folder.
    - Returns a logger object.
    """
    script_name = os.path.basename(__file__)
    root_directory = os.path.abspath(os.sep)

    # Create `logs` folder if it doesn't exist
    log_folder = os.path.join(root_directory, "logs")
    os.makedirs(log_folder, exist_ok=True)

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

    logger = logging.getLogger(__name__)  # Get a logger specific to this script
    logger.info(f"Logging initialized for {script_name}. Logs are being saved to {log_filename}")
    return logger