import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from cro_bpe.schemas import ArtifactDigest


class ProjectPaths(BaseModel):
    repository_root: Path = Field()
    data_dir: Path = Field()
    raw_dir: Path = Field()
    preprocessed_dir: Path = Field()
    features_dir: Path = Field()
    models_dir: Path = Field()
    logs_dir: Path = Field()
    mlruns_dir: Path = Field()

    @classmethod
    def discover(cls) -> "ProjectPaths":
        repository_root = Path(__file__).resolve().parents[3]
        data_dir = repository_root / "data"
        return cls(
            repository_root=repository_root,
            data_dir=data_dir,
            raw_dir=data_dir / "raw",
            preprocessed_dir=data_dir / "preprocessed",
            features_dir=data_dir / "feature_engineered",
            models_dir=data_dir / "models",
            logs_dir=repository_root / "logs",
            mlruns_dir=repository_root / "mlruns",
        )


def ensure_runtime_directories(paths: ProjectPaths) -> None:
    for path in [
        paths.data_dir,
        paths.raw_dir,
        paths.preprocessed_dir,
        paths.features_dir,
        paths.models_dir,
        paths.logs_dir,
        paths.mlruns_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def setup_logging(logs_dir: Path, log_level: int = logging.INFO, log_to_file: bool = True) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_to_file:
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(logs_dir / f"cro_bpe_{timestamp}.log", mode="a", encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped_line = line.strip()
            if stripped_line:
                yield json.loads(stripped_line)


def artifact_digest(path: Path) -> ArtifactDigest:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(65536):
            digest.update(chunk)
    return ArtifactDigest.from_path(path=path, sha256=digest.hexdigest(), bytes_size=path.stat().st_size)
