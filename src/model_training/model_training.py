"""Train the Croatian byte-level BPE tokenizer for every vocabulary size.

Reads ``data/feature_engineered/train`` and writes, for every size in
``TrainingSettings.vocab_sizes``::

    data/models/cro-bpe-<vocab>/
      tokenizer.json       ← load with model_training.tokenizer.load_tokenizer
      vocab.json, merges.txt
      cro-bpe-<vocab>.tiktoken, tiktoken.json ← load with model_training.export.load_tiktoken
      config.json          ← copy of tiktoken.json, counted by Hub download stats
      training_config.json ← settings, corpus summary and MLflow run id

BPE training is greedy and does not depend on the target size, so the merges of
a smaller vocabulary are a prefix of a larger one's. By default the largest size
is trained once and the smaller ones are truncated from it, which gives the
same models as separate runs (``tests/test_training.py`` checks this) at a
fraction of the cost. Set ``train_once=False`` to train every size separately.

Each model is built in ``cro-bpe-<vocab>.tmp`` and renamed into place only after
its tiktoken export has passed the equality check and ``training_config.json``
is written, so a failed run never leaves a model that looks complete. The check
compares tiktoken and ``tokenizer.json`` ids on documents sampled from every
training shard and on the whole holdout set, so all sources are covered.

Each size gets its own run in the experiment tracker (MLflow by default, see
:mod:`utils.tracking`); parameters, training time and the model directory are
logged there, and the evaluation stage adds its metrics to the same run.

Usage::

    uv run python -m model_training.model_training
"""

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from shutil import rmtree

import numpy as np
import pyarrow.parquet as pq
from pydantic import BaseModel, Field, field_validator
from tokenizers import Tokenizer

from model_training.export import export_tiktoken, load_tiktoken, verify_tiktoken
from model_training.tokenizer import (
    PRE_TOKENIZER_PATTERN,
    SPECIAL_TOKENS,
    build_tokenizer,
    build_trainer,
    load_tokenizer,
)
from utils.artifacts import MixtureSummary, Provenance, TrainingConfig
from utils.errors import MissingInputError
from utils.fileio import exclusive_run
from utils.project import DataPaths, default_data_root
from utils.tracking import MlflowTracker, Tracker

logger = getLogger(__name__)

# Special tokens plus the 256-byte alphabet: the vocabulary before the first merge.
BASE_VOCAB_SIZE = len(SPECIAL_TOKENS) + 256
_ROW_GROUPS_PER_SHARD = 4


class TrainingSettings(BaseModel):
    vocab_sizes: list[int] = Field(
        default=[32_000, 48_000, 64_000],
        description="Final vocabulary sizes including special tokens; one model is produced per size.",
    )
    train_once: bool = Field(
        default=True, description="Train the largest size once and truncate it to the smaller sizes."
    )
    min_frequency: int = Field(default=2, description="Minimum pair frequency for a merge.")
    max_token_length: int | None = Field(default=24, description="Maximum characters in a learned token.")
    batch_size: int = Field(default=1_000, description="Documents per batch fed to the trainer.")
    export_check_documents: int = Field(
        default=20_000,
        description="Training documents, spread over all shards, on which the tiktoken export must match "
        "tokenizer.json (the holdout set is always checked as well).",
    )
    experiment_name: str = Field(default="cro-bpe", description="MLflow experiment name.")
    data_root: Path = Field(default_factory=default_data_root, description="Root of the data/ directory.")

    @field_validator("vocab_sizes")
    @classmethod
    def _check_vocab_sizes(cls, sizes: list[int]) -> list[int]:
        if not sizes:
            raise ValueError("at least one vocabulary size is required")
        if len(set(sizes)) != len(sizes):
            raise ValueError(f"vocabulary sizes must be unique: {sizes}")
        for size in sizes:
            # Model directories are named cro-bpe-<size // 1000>k, so sizes must be whole thousands.
            if size % 1000 or size <= BASE_VOCAB_SIZE:
                raise ValueError(f"vocabulary size {size} must be a multiple of 1000 above {BASE_VOCAB_SIZE}")
        return sizes


@dataclass(frozen=True)
class TrainedModel:
    tokenizer: Tokenizer
    vocab_size: int
    trained_vocab_size: int  # size of the training run it was truncated from
    training_seconds: float


@dataclass(frozen=True)
class _Corpus:
    train_files: list[Path]
    train_documents: int
    summary: MixtureSummary
    check_texts: list[str]


def run(settings: TrainingSettings | None = None, tracker: Tracker | None = None) -> list[Path]:
    """Train and save every vocabulary size; returns the model directories."""

    settings = settings or TrainingSettings()
    paths = DataPaths(settings.data_root)
    tracker = tracker or MlflowTracker(settings.experiment_name, paths)
    logger.info("Starting model training for vocabulary sizes %s", settings.vocab_sizes)

    corpus = _load_corpus(paths, settings)
    logger.info(
        "Corpus: %d documents, %.2f GB; export check on %d documents",
        corpus.train_documents,
        corpus.summary.train_bytes / 1e9,
        len(corpus.check_texts),
    )

    sizes = sorted(settings.vocab_sizes, reverse=True)
    full = _train(sizes[0], corpus, settings) if settings.train_once else None
    model_dirs = []
    for vocab_size in sizes:
        model = _truncated(full, vocab_size) if full is not None else _train(vocab_size, corpus, settings)
        model_dirs.append(_save_model(model, corpus, paths, settings, tracker))
    return model_dirs


def _load_corpus(paths: DataPaths, settings: TrainingSettings) -> _Corpus:
    train_files = sorted(paths.train.glob("part-*.parquet"))
    if not train_files or not paths.mixture.is_file():
        raise MissingInputError(f"No training corpus in {paths.features}; run feature engineering first")
    holdout_files = sorted(paths.holdout.glob("part-*.parquet"))
    return _Corpus(
        train_files=train_files,
        train_documents=sum(pq.ParquetFile(path).metadata.num_rows for path in train_files),
        summary=MixtureSummary.load(paths.mixture),
        check_texts=export_check_texts(train_files, holdout_files, settings.export_check_documents),
    )


def export_check_texts(train_files: list[Path], holdout_files: list[Path], train_documents: int) -> list[str]:
    """About *train_documents* texts from evenly spaced row groups of every train shard, plus the holdout set.

    Train shards are written source by source, so taking documents from the start
    of the corpus would check only the first source (Wikipedia).
    """

    per_shard = -(-train_documents // len(train_files)) if train_files else 0
    texts: list[str] = []
    for path in train_files:
        parquet_file = pq.ParquetFile(path)
        groups = np.unique(np.linspace(0, parquet_file.num_row_groups - 1, _ROW_GROUPS_PER_SHARD).astype(int))
        per_group = -(-per_shard // len(groups))
        for group in groups.tolist():
            column = parquet_file.read_row_group(group, columns=["text"]).column("text")
            texts.extend(column.slice(0, per_group).to_pylist())
    for path in holdout_files:
        texts.extend(pq.read_table(path, columns=["text"]).column("text").to_pylist())
    return texts


def _train(vocab_size: int, corpus: _Corpus, settings: TrainingSettings) -> TrainedModel:
    logger.info("Training BPE (vocab %d) on %d documents", vocab_size, corpus.train_documents)
    tokenizer = build_tokenizer()
    trainer = build_trainer(vocab_size, settings.min_frequency, settings.max_token_length)
    started = time.monotonic()
    tokenizer.train_from_iterator(
        _iter_texts(corpus.train_files, settings.batch_size), trainer=trainer, length=corpus.train_documents
    )
    training_seconds = time.monotonic() - started
    logger.info("Training finished in %.1f min, vocabulary size %d", training_seconds / 60, tokenizer.get_vocab_size())
    return TrainedModel(tokenizer, vocab_size, vocab_size, training_seconds)


def _iter_texts(files: list[Path], batch_size: int) -> Iterator[list[str]]:
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=["text"]):
            yield batch.column("text").to_pylist()


def _truncated(full: TrainedModel, vocab_size: int) -> TrainedModel:
    return TrainedModel(
        truncate_tokenizer(full.tokenizer, vocab_size), vocab_size, full.vocab_size, full.training_seconds
    )


def truncate_tokenizer(tokenizer: Tokenizer, vocab_size: int) -> Tokenizer:
    """The tokenizer *tokenizer*'s training run would have produced with a target of *vocab_size*.

    Ids are assigned in merge order, so the smaller model keeps the ids below
    *vocab_size* and the merges that produced them.
    """

    config = json.loads(tokenizer.to_str())
    model = config["model"]
    if len(model["vocab"]) <= vocab_size:
        return Tokenizer.from_str(json.dumps(config))
    base_size = len(model["vocab"]) - len(model["merges"])
    model["vocab"] = {token: token_id for token, token_id in model["vocab"].items() if token_id < vocab_size}
    model["merges"] = model["merges"][: vocab_size - base_size]
    merged = ("".join(merge.split(" ") if isinstance(merge, str) else merge) for merge in model["merges"])
    if len(model["vocab"]) != vocab_size or any(token not in model["vocab"] for token in merged):
        raise ValueError("Token ids are not in merge order; the vocabulary cannot be truncated")
    return Tokenizer.from_str(json.dumps(config))


def _save_model(
    model: TrainedModel, corpus: _Corpus, paths: DataPaths, settings: TrainingSettings, tracker: Tracker
) -> Path:
    model_dir = paths.model_dir(model.vocab_size)
    staging_dir = model_dir.with_name(model_dir.name + ".tmp")
    rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True)
    run_settings = {
        **settings.model_dump(mode="json", exclude={"vocab_sizes", "data_root"}),
        "vocab_size": model.vocab_size,
        "trained_vocab_size": model.trained_vocab_size,
    }

    with tracker.run(model_dir.name) as run_id:
        tracker.log_params(run_settings)
        tracker.log_params(
            {
                "train_documents": corpus.train_documents,
                "train_bytes": corpus.summary.train_bytes,
                "pre_tokenizer_pattern": PRE_TOKENIZER_PATTERN,
            }
        )

        model.tokenizer.save(str(staging_dir / "tokenizer.json"))
        model.tokenizer.model.save(str(staging_dir))
        export_tiktoken(staging_dir, name=model_dir.name)
        verify_tiktoken(load_tiktoken(staging_dir), load_tokenizer(staging_dir), corpus.check_texts)
        TrainingConfig(
            settings=run_settings,
            special_tokens=SPECIAL_TOKENS,
            pre_tokenizer_pattern=PRE_TOKENIZER_PATTERN,
            corpus=corpus.summary,
            tracking_run_id=run_id,
            provenance=Provenance.current(),
        ).save(staging_dir / "training_config.json")
        rmtree(model_dir, ignore_errors=True)
        staging_dir.replace(model_dir)

        tracker.log_metrics({"training_minutes": model.training_seconds / 60})
        tracker.log_artifacts(model_dir, artifact_path="tokenizer")
        logger.info("Tokenizer saved to %s", model_dir)
    return model_dir


if __name__ == "__main__":
    with exclusive_run(default_data_root()):
        run()
