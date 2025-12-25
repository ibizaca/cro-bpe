from enum import Enum
from pathlib import Path
from shutil import rmtree
from datetime import datetime
import logging
from pydantic import BaseModel, Field
from collections.abc import Callable


### DIRECTORIES ###

class DataSubdirectories(str, Enum):
    RAW = "raw"
    PREPROCESSED = "preprocessed"
    FEATURES = "feature_engineered"
    MODELS = "models"


def create_data_directories() -> None:
    """
    Creates the downloads directory structure with subdirectories for raw data, preprocessed data, features, and models.
    If the directories already exist, they are removed and recreated to ensure a clean state.
    """

    data_directory = Path.cwd() / "data"

    for subdirectory in DataSubdirectories:
        new_directory = data_directory / subdirectory.value
        rmtree(path=new_directory, ignore_errors=True)
        new_directory.mkdir(parents=True, exist_ok=True)


def create_logging_directory() -> Path:
    """
    Creates the logging directory. If it already exists, it is removed and recreated to ensure a clean state.

    Returns:
        The path to the logging directory.
    """

    logging_directory = Path.cwd() / "logs"
    rmtree(path=logging_directory, ignore_errors=True)
    logging_directory.mkdir(exist_ok=True)

    return logging_directory


### LOGGING ###

def setup_logging(log_level: int = logging.INFO, log_to_file: bool = True) -> logging.Logger:
    """
    Sets up logging configuration for the project.

    Args:
        log_level: The logging level constant (logging.INFO, logging.DEBUG, logging.WARNING, logging.ERROR, logging.CRITICAL)
        log_to_file: Whether to also log to a file in addition to console

    Returns:
        The configured root logger
    """

    # Create logging directory
    logging_directory = create_logging_directory()

    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if enabled)
    if log_to_file:
        log_filename = logging_directory / f"cro_bpe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        root_logger.info(f"Logging to file: {log_filename}")

    root_logger.info(f"Logging configured with level: {logging.getLevelName(log_level)}")

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: The name of the logger (typically __name__ of the module)

    Returns:
        A configured logger instance
    """
    return logging.getLogger(name)


### Flow Control ###

class BaseFlowControl(BaseModel):
    run_step: bool = Field(default=True, description="Whether the flow step should be run or not")
    run_function: Callable = Field(description="The main function to execute for this flow step")


class DownloadControl(BaseFlowControl):
    pass


class PreprocessingControl(BaseFlowControl):
    pass


class FeatureEngineeringControl(BaseFlowControl):
    pass


class ModelTrainingControl(BaseFlowControl):
    pass


class Controls:
    DOWNLOAD = DownloadControl
    PREPROCESSING = PreprocessingControl
    FEATURE_ENGINEERING = FeatureEngineeringControl
    MODEL_TRAINING = ModelTrainingControl
