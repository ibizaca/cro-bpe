"""Typed contracts for the JSON files one stage writes and a later stage reads.

Every file is validated on load (unknown or missing keys raise) and written
atomically, so a renamed field fails loudly instead of being read as missing.
"""

from __future__ import annotations

import platform
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field

from utils.fileio import atomic_write_text

_PROVENANCE_PACKAGES = ("tokenizers", "tiktoken", "pyarrow", "numpy")


class Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        atomic_write_text(path, self.model_dump_json(indent=2))


class Provenance(Artifact):
    """Code and environment that produced an output."""

    git_commit: str | None = Field(description="HEAD of the repository, None outside a git checkout.")
    git_dirty: bool | None = Field(description="Whether the working tree had uncommitted changes.")
    python: str
    packages: dict[str, str]

    @classmethod
    def current(cls) -> Provenance:
        repository = Path(__file__).resolve().parents[2]
        commit, dirty = None, None
        try:
            commit = _git(repository, "rev-parse", "HEAD")
            dirty = bool(_git(repository, "status", "--porcelain"))
        except OSError, subprocess.CalledProcessError:
            pass
        return cls(
            git_commit=commit,
            git_dirty=dirty,
            python=platform.python_version(),
            packages={package: metadata.version(package) for package in _PROVENANCE_PACKAGES},
        )


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments], capture_output=True, text=True, check=True, timeout=10
    )
    return completed.stdout.strip()


# ─────────────────────────────── preprocessing ───────────────────────────────


class ShardSidecar(Artifact):
    """``data/preprocessed/<source>/part-*.stats.json``: one shard's counts and the fingerprint it was built with."""

    fingerprint: str
    counts: dict[str, int]


class SourceStats(Artifact):
    """``data/preprocessed/<source>/stats.json``: a source's totals; written once all its shards are current."""

    fingerprint: str = Field(description="Fingerprint over the source's shard fingerprints.")
    counts: dict[str, int]


# ─────────────────────────────── feature engineering ─────────────────────────


class SourceMixtureStats(Artifact):
    documents: int
    duplicates_removed: int
    near_duplicates_removed: int = 0
    holdout_documents: int
    available_bytes: int
    sample_fraction: float = 0.0
    train_documents: int = 0
    train_bytes: int = 0
    train_share: float = 0.0


class MixtureSummary(Artifact):
    """``data/feature_engineered/mixture.json``: how the training and holdout corpora were built."""

    fingerprint: str
    settings: dict[str, Any]
    mixture_weights: dict[str, float]
    train_bytes: int
    sources: dict[str, SourceMixtureStats]
    provenance: Provenance

    def train_shares(self) -> dict[str, float]:
        return {source: stats.train_share for source, stats in self.sources.items()}


# ─────────────────────────────── model training ──────────────────────────────


class TrainingConfig(Artifact):
    """``data/models/cro-bpe-<vocab>/training_config.json``; written last, so it marks a complete model."""

    settings: dict[str, Any]
    special_tokens: list[str]
    pre_tokenizer_pattern: str
    corpus: MixtureSummary
    tracking_run_id: str | None = Field(description="Experiment-tracking run that trained the model.")
    provenance: Provenance
