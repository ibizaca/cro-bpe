import logging
from pathlib import Path

import mlflow

from cro_bpe.config import RunConfig
from cro_bpe.downloading import run_downloading
from cro_bpe.feature_engineering import run_feature_engineering
from cro_bpe.model_training import run_model_training
from cro_bpe.preprocessing import run_preprocessing
from cro_bpe.utils import ProjectPaths, ensure_runtime_directories, read_json, setup_logging


def main() -> None:
    config = RunConfig.from_environment()
    paths = ProjectPaths.discover()
    ensure_runtime_directories(paths)
    logger = setup_logging(logs_dir=paths.logs_dir, log_level=logging.INFO, log_to_file=True)

    mlflow.set_tracking_uri(paths.mlruns_dir.as_uri())
    mlflow.set_experiment(config.mlflow_experiment_name)

    logger.info("Starting Croatian BPE pipeline")
    with mlflow.start_run(run_name=config.mlflow_run_name):
        _log_run_configuration(config)

        if config.workflow.run_download:
            download_manifest = run_downloading(config=config, paths=paths)
            _log_manifest(paths.raw_dir / config.artifacts.source_manifest_filename, "download")
            logger.info("Download stage complete: %s documents", download_manifest.statistics["documents_written"])

        if config.workflow.run_preprocessing:
            preprocessing_summary = run_preprocessing(config=config, paths=paths)
            _log_manifest(paths.preprocessed_dir / config.artifacts.preprocessing_summary_filename, "preprocessing")
            logger.info(
                "Preprocessing stage complete: %s documents retained",
                preprocessing_summary.statistics["kept_documents"],
            )

        if config.workflow.run_feature_engineering:
            diagnostics = run_feature_engineering(config=config, paths=paths)
            _log_manifest(paths.features_dir / config.artifacts.diagnostics_filename, "feature_engineering")
            logger.info("Feature engineering stage complete: %s documents analyzed", diagnostics.line_count)

        if config.workflow.run_model_training:
            tokenizer_manifest = run_model_training(config=config, paths=paths)
            _log_manifest(paths.models_dir / config.artifacts.tokenizer_manifest_filename, "model_training")
            logger.info("Model training stage complete: vocabulary size %s", tokenizer_manifest.vocab_size)

    logger.info("Croatian BPE pipeline finished successfully")


def _log_run_configuration(config: RunConfig) -> None:
    for key, value in _flatten_dictionary(config.model_dump(mode="json"), prefix="config").items():
        mlflow.log_param(key, value)


def _log_manifest(manifest_path: Path, stage_name: str) -> None:
    payload = read_json(manifest_path)
    flattened_payload = _flatten_dictionary(payload, prefix=stage_name)
    params: dict[str, str] = {}
    metrics: dict[str, float] = {}

    for key, value in flattened_payload.items():
        if isinstance(value, bool):
            params[key] = str(value)
        elif isinstance(value, int | float):
            metrics[key] = float(value)
        else:
            params[key] = _normalize_mlflow_param_value(value)

    if params:
        mlflow.log_params(params)
    if metrics:
        mlflow.log_metrics(metrics)
    mlflow.log_artifact(str(manifest_path), artifact_path=stage_name)


def _flatten_dictionary(payload: dict[str, object], prefix: str) -> dict[str, str | int | float | bool]:
    flattened_payload: dict[str, str | int | float | bool] = {}
    for key, value in payload.items():
        flattened_key = f"{prefix}.{key}"
        if isinstance(value, dict):
            flattened_payload.update(_flatten_dictionary(value, prefix=flattened_key))
        elif isinstance(value, list):
            flattened_payload[flattened_key] = ",".join(str(item) for item in value)
        elif value is None:
            flattened_payload[flattened_key] = "None"
        else:
            flattened_payload[flattened_key] = value
    return flattened_payload


def _normalize_mlflow_param_value(value: str) -> str:
    normalized = value.replace("\n", " ").strip()
    return normalized[:250]


if __name__ == "__main__":
    main()
