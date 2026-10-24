"""Clean and quality-filter every downloaded source shard.

Reads ``data/raw/<source-id>/parquet/part-*.parquet`` and writes one cleaned
shard per input shard to ``data/preprocessed/<source-id>/part-*.parquet`` with
columns ``text``, ``source``, ``id``, ``hash`` (xxh3-64 of the cleaned text,
used for deduplication downstream) and ``n_bytes`` (UTF-8 length).

Rerun behaviour
---------------
- Each shard's ``.stats.json`` sidecar stores a fingerprint of the quality
  thresholds, the ``VERSION`` of this module, ``cleaning``, ``filters`` and
  ``charclasses``, and the input shard's name and size. A shard is skipped only
  when its output exists and the fingerprint matches; otherwise it is rebuilt.
  Bump the module's ``VERSION`` after a change that alters what is written;
  ``tests/test_versions.py`` fails until the version snapshot is updated, so a
  forgotten bump is noticed.
- Output shards whose input shard no longer exists are deleted.
- Output is written to a ``.tmp`` file and renamed atomically.
- The per-source ``stats.json`` records a source fingerprint over its shards,
  which feature engineering uses to detect changed inputs.

Usage::

    uv run python -m preprocessing.preprocessing
"""

import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from logging import getLogger
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import xxhash
from pydantic import BaseModel, Field, ValidationError

from data_import.manifests import is_complete
from preprocessing import charclasses, cleaning, filters
from preprocessing.cleaning import clean_wikitext_residue, normalize_text
from preprocessing.filters import QualityThresholds, rejection_reason
from utils.artifacts import ShardSidecar, SourceStats
from utils.fileio import atomic_path, exclusive_run
from utils.fingerprint import file_signatures, fingerprint
from utils.parquet_io import PREPROCESSED_SCHEMA, WRITE_OPTIONS
from utils.project import DataPaths, default_data_root

logger = getLogger(__name__)

# Bump when process_shard or clean_document change what is written.
VERSION = 1

_WIKIPEDIA_SOURCE = "hr-wikipedia"


class PreprocessingSettings(BaseModel):
    workers: int = Field(default=max(1, (os.cpu_count() or 2) - 2), description="Parallel shard workers.")
    batch_size: int = Field(default=16_384, description="Rows read per Parquet batch.")
    thresholds: QualityThresholds = Field(default_factory=QualityThresholds)
    data_root: Path = Field(default_factory=default_data_root, description="Root of the data/ directory.")


def run(settings: PreprocessingSettings | None = None) -> dict[str, dict[str, int]]:
    """Preprocess every completed source; returns the per-source statistics."""

    settings = settings or PreprocessingSettings()
    logger.info("Starting data preprocessing with %d workers", settings.workers)

    paths = DataPaths(settings.data_root)
    stage = stage_fingerprint(settings)
    sources = completed_sources(settings.data_root)
    source_stats: dict[str, Counter[str]] = {source_id: Counter() for source_id in sources}
    shard_prints: dict[str, list[str]] = {source_id: [] for source_id in sources}
    pending: list[tuple[Path, Path, str, str]] = []
    stale = 0

    for source_id in sources:
        output_dir = paths.preprocessed_source(source_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths.source_stats(source_id).unlink(missing_ok=True)  # rewritten below once every shard is current
        input_paths = sorted(paths.raw_parquet(source_id).glob("part-*.parquet"))
        input_names = {path.name for path in input_paths}
        for orphan in output_dir.glob("part-*.parquet"):
            if orphan.name not in input_names:
                logger.info("[%s] removing %s, its input shard no longer exists", source_id, orphan.name)
                orphan.unlink()
                _stats_path(orphan).unlink(missing_ok=True)
        for input_path in input_paths:
            output_path = output_dir / input_path.name
            shard_print = shard_fingerprint(stage, input_path)
            shard_prints[source_id].append(shard_print)
            sidecar = _read_sidecar(output_path)
            if sidecar is not None and sidecar.fingerprint == shard_print:
                source_stats[source_id].update(sidecar.counts)
            else:
                stale += sidecar is not None
                pending.append((input_path, output_path, source_id, shard_print))

    logger.info("%d sources, %d shards pending (%d with outdated fingerprints)", len(sources), len(pending), stale)

    with ProcessPoolExecutor(max_workers=settings.workers) as executor:
        futures = {
            executor.submit(process_shard, input_path, output_path, source_id, settings, shard_print): source_id
            for input_path, output_path, source_id, shard_print in pending
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            source_stats[futures[future]].update(future.result())
            if completed % 50 == 0 or completed == len(futures):
                logger.info("Preprocessed %d/%d shards", completed, len(futures))

    for source_id, stats in source_stats.items():
        SourceStats(fingerprint=fingerprint(shard_prints[source_id]), counts=dict(stats)).save(
            paths.source_stats(source_id)
        )
        kept = stats["documents_out"] / max(1, stats["documents_in"])
        rejections = {
            key.removeprefix("rejected_"): value for key, value in stats.items() if key.startswith("rejected_")
        }
        logger.info(
            "[%s] kept %d/%d documents (%.1f%%), %.2f GB; rejected: %s",
            source_id,
            stats["documents_out"],
            stats["documents_in"],
            100 * kept,
            stats["bytes_out"] / 1e9,
            rejections,
        )
    return {source_id: dict(stats) for source_id, stats in source_stats.items()}


def stage_fingerprint(settings: PreprocessingSettings) -> str:
    """Everything except the input that determines a shard's output."""

    return fingerprint(
        {
            "preprocessing": VERSION,
            "cleaning": cleaning.VERSION,
            "filters": filters.VERSION,
            "charclasses": charclasses.VERSION,
        },
        settings.thresholds.model_dump(),
        _WIKIPEDIA_SOURCE,
        str(PREPROCESSED_SCHEMA),
    )


def completed_sources(data_root: Path) -> list[str]:
    """Source ids whose import is complete, by the same check the data import stage uses."""

    manifests = sorted(DataPaths(data_root).raw.glob("*/manifest.json"))
    return [path.parent.name for path in manifests if is_complete(path.parent.name, data_root)]


def _stats_path(shard_path: Path) -> Path:
    return shard_path.with_suffix(".stats.json")


def shard_fingerprint(stage: str, input_path: Path) -> str:
    return fingerprint(stage, file_signatures([input_path]))


def _read_sidecar(output_path: Path) -> ShardSidecar | None:
    """The sidecar of a finished shard, or ``None`` if the shard or sidecar is missing or unreadable."""

    if not output_path.is_file() or not _stats_path(output_path).is_file():
        return None
    try:
        return ShardSidecar.load(_stats_path(output_path))
    except ValidationError:
        return None


def process_shard(
    input_path: Path, output_path: Path, source_id: str, settings: PreprocessingSettings, shard_print: str
) -> dict[str, int]:
    """Clean, filter and hash one raw shard. Returns the shard's statistics."""

    stats: Counter[str] = Counter()
    parquet_file = pq.ParquetFile(input_path)
    # The sidecar goes first and is rewritten last: a shard counts as finished only once its fingerprint is on disk.
    _stats_path(output_path).unlink(missing_ok=True)

    with (
        atomic_path(output_path) as temporary_path,
        pq.ParquetWriter(temporary_path, PREPROCESSED_SCHEMA, **WRITE_OPTIONS) as writer,
    ):
        for batch in parquet_file.iter_batches(batch_size=settings.batch_size, columns=["text", "id"]):
            texts, ids, hashes, sizes = [], [], [], []
            for raw_text, doc_id in zip(batch.column("text").to_pylist(), batch.column("id").to_pylist()):
                stats["documents_in"] += 1
                stats["bytes_in"] += len(raw_text.encode("utf-8"))
                text = clean_document(raw_text, source_id)
                reason = rejection_reason(text, settings.thresholds)
                if reason is not None:
                    stats[f"rejected_{reason}"] += 1
                    continue
                encoded = text.encode("utf-8")
                texts.append(text)
                ids.append(doc_id)
                hashes.append(xxhash.xxh3_64_intdigest(encoded))
                sizes.append(len(encoded))
            if texts:
                writer.write_table(
                    pa.Table.from_arrays(
                        [
                            pa.array(texts, type=pa.large_string()),
                            pa.array([source_id] * len(texts), type=pa.string()),
                            pa.array(ids, type=pa.string()),
                            pa.array(hashes, type=pa.uint64()),
                            pa.array(sizes, type=pa.int64()),
                        ],
                        schema=PREPROCESSED_SCHEMA,
                    )
                )
                stats["documents_out"] += len(texts)
                stats["bytes_out"] += sum(sizes)

    ShardSidecar(fingerprint=shard_print, counts=dict(stats)).save(_stats_path(output_path))
    return dict(stats)


def clean_document(text: str, source_id: str) -> str:
    if source_id == _WIKIPEDIA_SOURCE:
        text = clean_wikitext_residue(text)
    return normalize_text(text)


if __name__ == "__main__":
    with exclusive_run(default_data_root()):
        run()
