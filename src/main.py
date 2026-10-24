import logging
import sys

from data_import import DataImportSettings, check_sources, run_downloading
from evaluation import EvaluationSettings, run_evaluation
from feature_engineering import FeatureEngineeringSettings, run_feature_engineering
from model_training import TrainingSettings, run_model_training
from preprocessing import PreprocessingSettings, run_preprocessing
from utils.errors import PipelineError
from utils.fileio import exclusive_run
from utils.project import WorkflowConfigs, create_data_directories, setup_logging

# CulturaX is gated and access has not been granted, so its import failed; continue without it.
download_settings = DataImportSettings(allow_missing=["culturax-hr"])

download_config = WorkflowConfigs.DOWNLOAD(run_step=False, run_function=run_downloading, settings=download_settings)
preprocessing_config = WorkflowConfigs.PREPROCESSING(
    run_step=True, run_function=run_preprocessing, settings=PreprocessingSettings()
)
feature_engineering_config = WorkflowConfigs.FEATURE_ENGINEERING(
    run_step=True, run_function=run_feature_engineering, settings=FeatureEngineeringSettings()
)
model_training_config = WorkflowConfigs.MODEL_TRAINING(
    run_step=True, run_function=run_model_training, settings=TrainingSettings()
)
evaluation_config = WorkflowConfigs.EVALUATION(
    run_step=True, run_function=run_evaluation, settings=EvaluationSettings()
)


def main() -> int:
    logger = setup_logging(log_level=logging.INFO, log_to_file=True)
    try:
        run_workflow(logger)
    except PipelineError as error:
        # Missing inputs or sources and export mismatches are data problems, not bugs: no traceback.
        logger.error("%s: %s", type(error).__name__, error)
        return 1
    return 0


def run_workflow(logger: logging.Logger) -> None:
    # Create necessary data directories
    logger.info("Creating data directories")
    create_data_directories(download_settings.data_root)

    # Execute flow steps based on configuration; one run at a time per data/ directory
    logger.info("Starting workflow execution")
    with exclusive_run(download_settings.data_root):
        if download_config.run_step:
            download_config.run()  # raises if a source is neither complete nor blocked
        elif preprocessing_config.run_step:
            # Stop before preprocessing silently drops a source whose download failed.
            check_sources(download_settings)
        if preprocessing_config.run_step:
            preprocessing_config.run()
        if feature_engineering_config.run_step:
            feature_engineering_config.run()
        if model_training_config.run_step:
            model_training_config.run()
        if evaluation_config.run_step:
            evaluation_config.run()


if __name__ == "__main__":
    sys.exit(main())
