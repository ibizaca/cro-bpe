"""Per-source manifest types and persistence.

Each source in the catalog gets a manifest.json written to
``data/raw/<source-id>/manifest.json``.  The file is updated atomically
so a partial run never leaves a corrupt manifest on disk.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import ConfigDict, Field

from utils.artifacts import Artifact
from utils.fileio import sha256_of_file
from utils.project import DataPaths


class ManifestStatus(StrEnum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    INGESTING = "ingesting"
    COMPLETE = "complete"
    BLOCKED = "blocked"
    FAILED = "failed"
    EXCLUDED = "excluded"
    MISSING = "missing"  # no manifest on disk; only used in memory by check_sources


class FileStatus(StrEnum):
    PENDING = "pending"
    COMPLETE = "complete"
    FAILED = "failed"
    DELETED_AFTER_CONVERSION = "deleted_after_conversion"


class _ManifestPart(Artifact):
    # Stages update manifests field by field; validate each assignment so a status stays an enum.
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class FileRecord(_ManifestPart):
    """Tracks a single downloaded source file."""

    url: str
    local_path: str
    size_bytes: int | None = None
    sha256: str | None = None
    compression_validated: bool = False
    status: FileStatus = FileStatus.PENDING


class OutputFileRecord(_ManifestPart):
    """One Parquet shard written by the import."""

    path: str
    rows: int
    size_bytes: int
    uncompressed_bytes: int
    sha256: str


class SourceManifest(_ManifestPart):
    """Complete provenance and statistics record for one corpus source."""

    source_id: str
    status: ManifestStatus = ManifestStatus.PENDING
    version: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    files: list[FileRecord] = Field(default_factory=list)
    output_files: list[OutputFileRecord] = Field(default_factory=list)
    documents: int = 0
    output_size_bytes: int | None = None
    blocked_reason: str | None = None
    blocked_note: str | None = None
    error: str | None = None
    skipped_files: list[str] = Field(default_factory=list, description="Archive members that failed to parse.")


def load_manifest(source_id: str, data_root: Path) -> SourceManifest | None:
    path = DataPaths(data_root).manifest(source_id)
    return SourceManifest.load(path) if path.exists() else None


def save_manifest(manifest: SourceManifest, data_root: Path) -> None:
    manifest.save(DataPaths(data_root).manifest(manifest.source_id))


def is_complete(source_id: str, data_root: Path, *, verify_hashes: bool = False) -> bool:
    """Whether *source_id* finished and all its output shards exist with the recorded sizes.

    With *verify_hashes*, each shard's SHA-256 is also recomputed and compared,
    which reads the whole output (tens of GB for the web crawls).
    """
    m = load_manifest(source_id, data_root)
    if m is None or m.status != ManifestStatus.COMPLETE or m.documents <= 0 or not m.output_files:
        return False
    output_dir = DataPaths(data_root).raw_parquet(source_id)
    for output in m.output_files:
        path = output_dir / output.path
        if not path.is_file() or path.stat().st_size != output.size_bytes:
            return False
        if verify_hashes and sha256_of_file(path) != output.sha256:
            return False
    return True
