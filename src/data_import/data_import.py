"""Orchestration for the Croatian corpus downloading pipeline.

Iterates the full source catalog, downloads every accessible corpus,
streams each source into bounded Parquet shards, and writes a machine-readable
manifest.

Rerun behaviour
---------------
- A source whose manifest records ``status=complete`` and whose shards exist
  with the recorded sizes is skipped. Shard checksums are recomputed only with
  ``DataImportSettings.verify_existing``.
- An interrupted HTTP download is resumed from its ``.part`` file.
- Partial Parquet output is re-created from scratch on a second run.
- Downloaded source files, including Hugging Face files (which are downloaded
  into ``data/raw/<id>/source``, not the global Hub cache), are deleted after
  conversion unless ``DataImportSettings.keep_source_files`` is set.
- :func:`run` fails unless every catalog source ends up ``complete`` or
  ``blocked``; sources listed in ``DataImportSettings.allow_missing`` only log a
  warning. :func:`check_sources` applies the same check to the manifests on
  disk, which ``main.py`` uses when the download step is skipped.

Usage::

    uv run python -m data_import.data_import

Or via ``main.py`` which calls :func:`run`.
"""

import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from shutil import rmtree

from pydantic import BaseModel, Field

from data_import.catalog import (
    BLOCKED,
    EXCLUDED,
    SOURCES,
    ExcludedEntry,
    HfDownloadSource,
    HttpSource,
    HuggingFaceSource,
    SourceEntry,
)
from data_import.manifests import (
    FileRecord,
    FileStatus,
    ManifestStatus,
    OutputFileRecord,
    SourceManifest,
    is_complete,
    load_manifest,
    save_manifest,
)
from data_import.parquet import ParquetShardWriter
from data_import.readers import ReadReport, download_huggingface_files, source_filename, stream_source
from utils.downloaders import download_http
from utils.errors import MissingSourcesError
from utils.fileio import exclusive_run, sha256_of_file
from utils.project import DataPaths, default_data_root

logger = logging.getLogger(__name__)

# Final states of a source that let the later stages run.
_USABLE_STATUSES = (ManifestStatus.COMPLETE, ManifestStatus.BLOCKED)


class DataImportSettings(BaseModel):
    keep_source_files: bool = Field(
        default=False, description="Keep downloaded archives after conversion, so readers can be re-run offline."
    )
    download_retries: int = Field(default=8, description="Attempts per file; each retry resumes the .part file.")
    verify_existing: bool = Field(
        default=False, description="Recompute SHA-256 of completed sources' shards before skipping them."
    )
    allow_missing: list[str] = Field(
        default_factory=list,
        description="Source ids that may be missing or failed without stopping the pipeline (logged as a warning).",
    )
    data_root: Path = Field(default_factory=default_data_root, description="Root of the data/ directory.")


# ─────────────────────────── helpers ─────────────────────────────────────────


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_blocked_manifest(entry_id: str, reason: str, note: str, data_root: Path, version: str | None = None) -> None:
    m = SourceManifest(
        source_id=entry_id,
        status=ManifestStatus.BLOCKED,
        version=version,
        started_at=_now(),
        completed_at=_now(),
        blocked_reason=reason,
        blocked_note=note,
    )
    save_manifest(m, data_root)
    logger.info("[%s] recorded as blocked: %s", entry_id, reason)


def _write_excluded_manifest(entry: ExcludedEntry, data_root: Path) -> None:
    m = SourceManifest(
        source_id=entry.id,
        status=ManifestStatus.EXCLUDED,
        started_at=_now(),
        completed_at=_now(),
        blocked_reason=entry.reason.value,
        blocked_note=entry.note,
    )
    save_manifest(m, data_root)


# ─────────────────────── source ingestion ────────────────────────────────────


def _download_http_files(
    entry: HttpSource,
    s_dir: Path,
    manifest: SourceManifest,
    data_root: Path,
    retries: int,
) -> bool:
    """Download all HTTP source files for *entry* into *s_dir*.

    Populates *manifest.files* and returns ``True`` if all files succeeded.
    """
    s_dir.mkdir(parents=True, exist_ok=True)
    all_ok = True

    for url in entry.urls:
        filename = source_filename(url)
        dest = s_dir / filename

        file_rec = next((f for f in manifest.files if f.url == url), None)
        if file_rec is None:
            file_rec = FileRecord(url=url, local_path=str(dest))
            manifest.files.append(file_rec)

        has_matching_size = (
            file_rec.size_bytes is not None
            and file_rec.size_bytes > 0
            and dest.is_file()
            and dest.stat().st_size == file_rec.size_bytes
        )
        if file_rec.status == FileStatus.COMPLETE and has_matching_size:
            logger.debug("[%s] skipping already-downloaded %s", entry.id, filename)
            continue

        if file_rec.status == FileStatus.COMPLETE and dest.exists():
            logger.warning("[%s] discarding source file %s, its size changed", entry.id, filename)
            dest.unlink()
            dest.with_suffix(dest.suffix + ".part").unlink(missing_ok=True)
        elif dest.exists():
            # download_http only renames finished transfers to dest, so this file is complete:
            # the previous run stopped before recording it. download_http adopts it below.
            logger.info("[%s] adopting finished download %s", entry.id, filename)
        file_rec.size_bytes = None
        file_rec.sha256 = None
        file_rec.compression_validated = False
        file_rec.status = FileStatus.PENDING

        logger.info("[%s] downloading %s", entry.id, url)
        try:
            download_http(url, dest, retries=retries)
            file_rec.size_bytes = dest.stat().st_size
            file_rec.sha256 = sha256_of_file(dest)
            file_rec.status = FileStatus.COMPLETE
            save_manifest(manifest, data_root)
        except Exception as exc:
            logger.error("[%s] failed to download %s: %s", entry.id, url, exc)
            file_rec.status = FileStatus.FAILED
            all_ok = False
            save_manifest(manifest, data_root)

    return all_ok


def _download_huggingface_files(
    entry: HfDownloadSource, s_dir: Path, manifest: SourceManifest, data_root: Path
) -> None:
    """Download the entry's Hub Parquet files into *s_dir* and record them in *manifest.files*."""

    s_dir.mkdir(parents=True, exist_ok=True)
    files = download_huggingface_files(entry, s_dir)
    manifest.files = [
        FileRecord(
            url=f"hf://datasets/{entry.hf_repo}/{path.relative_to(s_dir).as_posix()}",
            local_path=str(path),
            size_bytes=path.stat().st_size,
            sha256=sha256_of_file(path),
            status=FileStatus.COMPLETE,
        )
        for path in files
    ]
    save_manifest(manifest, data_root)


def _is_gated_error(error: BaseException | None) -> bool:
    """Whether *error* (or an exception it was raised from) means the Hub denied access to a gated dataset."""

    from huggingface_hub.errors import GatedRepoError

    while error is not None:
        if isinstance(error, GatedRepoError) or "gated dataset" in str(error).lower():
            return True
        error = error.__cause__ or error.__context__
    return False


def _has_hf_token() -> bool:
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))


def _delete_converted_sources(entry: SourceEntry, manifest: SourceManifest, data_root: Path) -> None:
    rmtree(DataPaths(data_root).raw_source(entry.id), ignore_errors=True)
    for file_record in manifest.files:
        file_record.status = FileStatus.DELETED_AFTER_CONVERSION
    save_manifest(manifest, data_root)


def ingest_source(
    entry: SourceEntry, data_root: Path | None = None, settings: DataImportSettings | None = None
) -> SourceManifest:
    """Download and ingest a single catalog source.

    Returns the final :class:`SourceManifest` for the source.
    """
    settings = settings or DataImportSettings()
    data_root = data_root or settings.data_root

    completed = _skip_if_complete(entry, data_root, settings)
    if completed is not None:
        return completed
    manifest = _start_manifest(entry, data_root)
    stopped = _download(entry, manifest, data_root, settings)
    if stopped is not None:
        return stopped
    try:
        shards, report = _convert(entry, manifest, data_root)
    except Exception as exc:
        return _blocked_or_failed(entry, manifest, data_root, exc)
    return _finish(entry, manifest, data_root, settings, shards, report)


def _skip_if_complete(entry: SourceEntry, data_root: Path, settings: DataImportSettings) -> SourceManifest | None:
    """The manifest of an already complete source (cleaning up leftover source files), else ``None``."""

    if not is_complete(entry.id, data_root, verify_hashes=settings.verify_existing):
        return None
    logger.info("[%s] already complete — skipping", entry.id)
    completed = load_manifest(entry.id, data_root)
    if completed is None:
        raise RuntimeError(f"Complete manifest disappeared for {entry.id}")
    if not settings.keep_source_files and DataPaths(data_root).raw_source(entry.id).exists():
        _delete_converted_sources(entry, completed, data_root)  # left over from an interrupted run
    return completed


def _start_manifest(entry: SourceEntry, data_root: Path) -> SourceManifest:
    """The source's manifest, reset for a new attempt; file records are kept so downloads resume."""

    manifest = load_manifest(entry.id, data_root) or SourceManifest(source_id=entry.id)
    manifest.status = ManifestStatus.DOWNLOADING
    manifest.version = entry.version
    manifest.started_at = manifest.started_at or _now()
    manifest.completed_at = None
    manifest.error = None
    save_manifest(manifest, data_root)
    return manifest


def _download(
    entry: SourceEntry, manifest: SourceManifest, data_root: Path, settings: DataImportSettings
) -> SourceManifest | None:
    """Download the source files; returns the blocked or failed manifest if that failed, else ``None``.

    Streamed Hugging Face sources have nothing to download.
    """
    if isinstance(entry, HuggingFaceSource) and entry.requires_hf_token and not _has_hf_token():
        return _blocked(
            entry,
            manifest,
            data_root,
            reason="auth_required",
            note=(
                "HF_TOKEN environment variable not set. "
                f"Accept the access form at https://huggingface.co/datasets/{entry.hf_repo} "
                "then re-run with HF_TOKEN exported."
            ),
        )

    s_dir = DataPaths(data_root).raw_source(entry.id)
    try:
        if isinstance(entry, HttpSource) and not _download_http_files(
            entry, s_dir, manifest, data_root, settings.download_retries
        ):
            failed_urls = [f.url for f in manifest.files if f.status == FileStatus.FAILED]
            return _failed(entry, manifest, data_root, f"One or more source files failed to download: {failed_urls}")
        if isinstance(entry, HfDownloadSource):
            _download_huggingface_files(entry, s_dir, manifest, data_root)
    except Exception as exc:
        return _blocked_or_failed(entry, manifest, data_root, exc)
    return None


def _convert(
    entry: SourceEntry, manifest: SourceManifest, data_root: Path
) -> tuple[list[OutputFileRecord], ReadReport]:
    """Read the source into Parquet shards under ``parquet.tmp/`` and move them into place.

    Raises (leaving no partial output) if reading fails or the source has no documents.
    """
    manifest.status = ManifestStatus.INGESTING
    save_manifest(manifest, data_root)

    paths = DataPaths(data_root)
    out_path = paths.raw_parquet(entry.id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_path.with_name(f"{out_path.name}.tmp")
    rmtree(tmp_out, ignore_errors=True)
    writer = ParquetShardWriter(tmp_out)
    report = ReadReport()
    try:
        for document in stream_source(entry, paths.raw_source(entry.id), report):
            writer.write(document)
        shards = writer.close()
        if not shards:
            raise ValueError("Source contained no text documents")
    except BaseException:
        rmtree(tmp_out, ignore_errors=True)
        raise

    rmtree(out_path, ignore_errors=True)
    tmp_out.replace(out_path)
    return shards, report


def _finish(
    entry: SourceEntry,
    manifest: SourceManifest,
    data_root: Path,
    settings: DataImportSettings,
    shards: list[OutputFileRecord],
    report: ReadReport,
) -> SourceManifest:
    """Record the output shards, verify them and delete the converted source files."""

    manifest.status = ManifestStatus.COMPLETE
    manifest.documents = sum(shard.rows for shard in shards)
    manifest.output_files = shards
    manifest.output_size_bytes = sum(shard.size_bytes for shard in shards)
    manifest.skipped_files = report.skipped_files
    for file_record in manifest.files:
        file_record.compression_validated = True
    manifest.completed_at = _now()
    save_manifest(manifest, data_root)
    # Shards were hashed when written and then only renamed, so sizes suffice here.
    if not is_complete(entry.id, data_root):
        return _failed(entry, manifest, data_root, "Parquet output verification failed")
    if not settings.keep_source_files:
        _delete_converted_sources(entry, manifest, data_root)

    logger.info(
        "[%s] complete — wrote %d documents, %d unparseable files skipped",
        entry.id,
        manifest.documents,
        len(report.skipped_files),
    )
    return manifest


def _blocked_or_failed(
    entry: SourceEntry, manifest: SourceManifest, data_root: Path, error: Exception
) -> SourceManifest:
    if _is_gated_error(error):
        return _blocked(entry, manifest, data_root, reason="auth_required", note=str(error))
    return _failed(entry, manifest, data_root, str(error))


def _blocked(
    entry: SourceEntry,
    manifest: SourceManifest,
    data_root: Path,
    reason: str,
    note: str,
) -> SourceManifest:
    manifest.status = ManifestStatus.BLOCKED
    manifest.blocked_reason = reason
    manifest.blocked_note = note
    manifest.completed_at = _now()
    save_manifest(manifest, data_root)
    logger.warning("[%s] blocked: %s", entry.id, reason)
    return manifest


def _failed(
    entry: SourceEntry,
    manifest: SourceManifest,
    data_root: Path,
    error: str,
) -> SourceManifest:
    manifest.status = ManifestStatus.FAILED
    manifest.error = error
    manifest.completed_at = _now()
    save_manifest(manifest, data_root)
    logger.error("[%s] failed: %s", entry.id, error)
    return manifest


# ─────────────────────────── orchestration ───────────────────────────────────


def check_sources(settings: DataImportSettings | None = None) -> list[SourceManifest]:
    """Raise unless every catalog source's manifest is complete or blocked; returns the manifests.

    Sources in ``settings.allow_missing`` are reported as warnings instead. Without
    this check a failed source would silently drop out of preprocessing and its
    share of the mixture would go to the other sources.
    """
    settings = settings or DataImportSettings()
    manifests: list[SourceManifest] = []
    problems: list[str] = []
    for entry in SOURCES:
        manifest = load_manifest(entry.id, settings.data_root) or SourceManifest(
            source_id=entry.id, status=ManifestStatus.MISSING
        )
        manifests.append(manifest)
        if manifest.status == ManifestStatus.BLOCKED or (
            manifest.status == ManifestStatus.COMPLETE and is_complete(entry.id, settings.data_root)
        ):
            continue
        problem = f"{entry.id} ({manifest.status}{': ' + manifest.error if manifest.error else ''})"
        if entry.id in settings.allow_missing:
            logger.warning("Continuing without %s, it is listed in allow_missing", problem)
        else:
            problems.append(problem)
    if problems:
        raise MissingSourcesError(
            f"Sources not {' or '.join(_USABLE_STATUSES)}: {'; '.join(problems)}. "
            "Re-run the download step, or add the ids to DataImportSettings.allow_missing to continue without them."
        )
    return manifests


def run(settings: DataImportSettings | None = None) -> list[SourceManifest]:
    """Run the full catalog-driven downloading pipeline; returns the manifest of every catalog source.

    Raises if a source is not complete or blocked (see :func:`check_sources`).
    """
    settings = settings or DataImportSettings()
    data_root = settings.data_root
    logger.info("Starting data downloading")

    # Record blocked and excluded sources so manifests are always complete
    for blocked in BLOCKED:
        if not is_complete(blocked.id, data_root):
            _write_blocked_manifest(
                blocked.id,
                reason=blocked.reason.value,
                note=blocked.note,
                data_root=data_root,
                version=None,
            )

    for excluded in EXCLUDED:
        if not DataPaths(data_root).manifest(excluded.id).exists():
            _write_excluded_manifest(excluded, data_root)

    # Process active sources
    results: list[SourceManifest] = []
    for entry in SOURCES:
        try:
            m = ingest_source(entry, data_root, settings)
        except Exception as exc:
            logger.exception("[%s] unexpected error: %s", entry.id, exc)
            m = SourceManifest(source_id=entry.id, status=ManifestStatus.FAILED, error=str(exc))
            save_manifest(m, data_root)
        results.append(m)

    # Summary
    logger.info("─" * 60)
    logger.info("Downloading summary")
    logger.info("─" * 60)
    for m in results:
        if m.status == ManifestStatus.COMPLETE and m.skipped_files:
            logger.info("  ! %-30s  documents=%d, skipped files: %s", m.source_id, m.documents, m.skipped_files)
        elif m.status == ManifestStatus.COMPLETE:
            logger.info("  ✓ %-30s  documents=%d", m.source_id, m.documents)
        elif m.status == ManifestStatus.BLOCKED:
            logger.info("  ⊘ %-30s  blocked: %s", m.source_id, m.blocked_reason)
        else:
            logger.info("  ✗ %-30s  %s", m.source_id, m.status)
    for b in BLOCKED:
        logger.info("  ⊘ %-30s  blocked: %s", b.id, b.reason.value)
    logger.info("─" * 60)
    check_sources(settings)
    return results


if __name__ == "__main__":
    with exclusive_run(default_data_root()):
        run()
