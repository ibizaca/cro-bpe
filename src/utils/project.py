import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field


class DataSubdirectories(str, Enum):
    RAW = "raw"
    PREPROCESSED = "preprocessed"
    FEATURES = "feature_engineered"
    MODELS = "models"


@dataclass(frozen=True)
class DataPaths:
    """Every path the stages read or write, derived from one ``data/`` root."""

    root: Path

    # Data import
    @property
    def raw(self) -> Path:
        return self.root / DataSubdirectories.RAW.value

    def raw_source_root(self, source_id: str) -> Path:
        return self.raw / source_id

    def manifest(self, source_id: str) -> Path:
        return self.raw_source_root(source_id) / "manifest.json"

    def raw_parquet(self, source_id: str) -> Path:
        """Imported documents of *source_id* (``part-*.parquet``)."""
        return self.raw_source_root(source_id) / "parquet"

    def raw_source(self, source_id: str) -> Path:
        """Downloaded source files of *source_id*, deleted after conversion."""
        return self.raw_source_root(source_id) / "source"

    # Preprocessing
    @property
    def preprocessed(self) -> Path:
        return self.root / DataSubdirectories.PREPROCESSED.value

    def preprocessed_source(self, source_id: str) -> Path:
        return self.preprocessed / source_id

    def source_stats(self, source_id: str) -> Path:
        return self.preprocessed_source(source_id) / "stats.json"

    # Feature engineering
    @property
    def features(self) -> Path:
        return self.root / DataSubdirectories.FEATURES.value

    @property
    def train(self) -> Path:
        return self.features / "train"

    @property
    def holdout(self) -> Path:
        return self.features / "holdout"

    @property
    def mixture(self) -> Path:
        return self.features / "mixture.json"

    def minhash_source(self, source_id: str) -> Path:
        """Cached near-dedup band hashes of *source_id*'s preprocessed shards."""
        return self.features / "minhash" / source_id

    # Models
    @property
    def models(self) -> Path:
        return self.root / DataSubdirectories.MODELS.value

    def model_dir(self, vocab_size: int) -> Path:
        return self.models / f"cro-bpe-{vocab_size // 1000}k"

    @property
    def comparison(self) -> Path:
        return self.models / "comparison.json"

    @property
    def baselines(self) -> Path:
        """Hugging Face cache for the baseline tokenizers evaluation compares against."""
        return self.models / "baselines"

    @property
    def tiktoken_cache(self) -> Path:
        """Cache for tiktoken's own encodings (``TIKTOKEN_CACHE_DIR``), instead of the system temp directory."""
        return self.baselines / "tiktoken"

    @property
    def mlflow_db(self) -> Path:
        return self.models / "mlflow.db"

    @property
    def mlflow_artifacts(self) -> Path:
        return self.models / "mlartifacts"


class BaseWorkflowControl(BaseModel):
    run_step: bool = Field(default=True, description="Whether the flow step should be run or not")
    run_function: Callable = Field(description="The main function to execute for this flow step")
    settings: Any = Field(default=None, description="Settings passed to run_function; None uses the stage defaults")

    def run(self) -> Any:
        return self.run_function(self.settings)


class DownloadConfig(BaseWorkflowControl):
    pass


class PreprocessingConfig(BaseWorkflowControl):
    pass


class FeatureEngineeringConfig(BaseWorkflowControl):
    pass


class ModelTrainingConfig(BaseWorkflowControl):
    pass


class EvaluationConfig(BaseWorkflowControl):
    pass


class WorkflowConfigs:
    """Namespace of the per-step config classes; ``main.py`` instantiates each one."""

    DOWNLOAD: ClassVar[type[DownloadConfig]] = DownloadConfig
    PREPROCESSING: ClassVar[type[PreprocessingConfig]] = PreprocessingConfig
    FEATURE_ENGINEERING: ClassVar[type[FeatureEngineeringConfig]] = FeatureEngineeringConfig
    MODEL_TRAINING: ClassVar[type[ModelTrainingConfig]] = ModelTrainingConfig
    EVALUATION: ClassVar[type[EvaluationConfig]] = EvaluationConfig


def create_data_directories(data_root: Path | None = None) -> None:
    """
    Creates missing data directories without deleting resumable downloads or
    outputs from completed workflow stages.
    """

    data_directory = data_root or default_data_root()

    for subdirectory in DataSubdirectories:
        new_directory = data_directory / subdirectory.value
        new_directory.mkdir(parents=True, exist_ok=True)


def default_data_root() -> Path:
    """``./data``, resolved when called (not at import time) so the working directory can change first."""

    return Path.cwd() / "data"


def setup_logging(
    log_level: int = logging.INFO, log_to_file: bool = True, keep_logs: int | None = 50
) -> logging.Logger:
    """
    Sets up logging configuration for the project.

    Args:
        log_level: The logging level constant (logging.INFO, logging.DEBUG, logging.WARNING, logging.ERROR,
            logging.CRITICAL)
        log_to_file: Whether to also log to a file in addition to console
        keep_logs: Number of most recent log files to keep, including the new one (``None`` keeps all)

    Returns:
        The configured root logger
    """

    # Create logging directory
    logging_directory = create_logging_directory(keep_logs if log_to_file else None)

    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if enabled)
    if log_to_file:
        log_filename = logging_directory / f"cro_bpe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_filename, mode="a", encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        root_logger.info(f"Logging to file: {log_filename}")

    root_logger.info(f"Logging configured with level: {logging.getLevelName(log_level)}")

    return root_logger


def create_logging_directory(keep_logs: int | None = 50) -> Path:
    """
    Creates the logging directory if needed. Earlier log files are kept, except
    that only the newest *keep_logs* files survive (``None`` keeps all).

    Returns:
        The path to the logging directory.
    """

    logging_directory = Path.cwd() / "logs"
    logging_directory.mkdir(exist_ok=True)
    if keep_logs is not None:
        # Timestamped names sort chronologically; one slot is left for the new file.
        old_logs = sorted(logging_directory.glob("cro_bpe_*.log"))
        for old_log in old_logs[: max(0, len(old_logs) - max(0, keep_logs - 1))]:
            old_log.unlink(missing_ok=True)

    return logging_directory
