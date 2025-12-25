import logging

from downloading import run_downloading
from feature_engineering import run_feature_engineering
from model_training import run_model_training
from preprocessing import run_preprocessing
from utils.project import create_data_directories, setup_logging, WorkflowConfigs

logger = setup_logging(log_level=logging.INFO, log_to_file=True)

download_config = WorkflowConfigs.DOWNLOAD(run_step=True, run_function=run_downloading)
preprocessing_config = WorkflowConfigs.PREPROCESSING(run_step=True, run_function=run_preprocessing)
feature_engineering_config = WorkflowConfigs.FEATURE_ENGINEERING(run_step=True, run_function=run_feature_engineering)
model_training_config = WorkflowConfigs.MODEL_TRAINING(run_step=True, run_function=run_model_training)


def main():
    # Create necessary data directories
    logger.info("Creating data directories")
    create_data_directories()

    # Execute flow steps based on configuration
    logger.info("Starting workflow execution")
    if download_config.run_step:
        download_config.run_function()
    if preprocessing_config.run_step:
        preprocessing_config.run_function()
    if feature_engineering_config.run_step:
        feature_engineering_config.run_function()
    if model_training_config.run_step:
        model_training_config.run_function()


if __name__ == "__main__":
    main()
