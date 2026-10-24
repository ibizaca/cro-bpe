"""Build the tokenizer training and evaluation corpora from preprocessed shards.

Steps
-----
1. Global exact deduplication on the ``hash`` column, keeping the first
   occurrence in :data:`~feature_engineering.mixture.SOURCE_PRIORITY` order.
2. Global near-duplicate removal with MinHash LSH
   (:mod:`~feature_engineering.near_dedup`), with the same priority rule: a
   document sharing a band hash with an earlier kept document is removed. The
   holdout is selected afterwards, so no training document is a near-copy of a
   holdout document.
3. A per-source holdout set for tokenizer evaluation, never used for training.
4. Weighted, deterministic sampling of each source towards a target corpus size
   according to :data:`~feature_engineering.mixture.MIXTURE_WEIGHTS`.

Output::

    data/feature_engineered/
      train/part-*.parquet     ← columns: text, source
      holdout/part-*.parquet
      mixture.json             ← settings and per-source statistics
      minhash/<source>/*.npy   ← cached band hashes per preprocessed shard

Rerun behaviour
---------------
``mixture.json`` stores a fingerprint of the settings, ``SOURCE_PRIORITY``,
``MIXTURE_WEIGHTS``, the ``VERSION`` of this module and of
:mod:`~feature_engineering.mixture`, and every preprocessed source (its
``stats.json`` fingerprint and shard names and sizes). The stage is skipped when
it matches; otherwise all outputs are rebuilt. ``mixture.json`` is written last,
so an interrupted build is never reused. Band hashes are cached per shard and
only computed for new or changed shards or MinHash settings.

Usage::

    uv run python -m feature_engineering.feature_engineering
"""

import os
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from logging import getLogger
from pathlib import Path
from shutil import rmtree

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, Field, ValidationError

from feature_engineering import mixture, near_dedup
from feature_engineering.mixture import allocate_budget
from utils.artifacts import MixtureSummary, Provenance, SourceMixtureStats, SourceStats
from utils.errors import MissingInputError
from utils.fileio import exclusive_run
from utils.fingerprint import file_signatures, fingerprint
from utils.parquet_io import CORPUS_SCHEMA, RollingParquetWriter
from utils.project import DataPaths, default_data_root

logger = getLogger(__name__)

# Part of the corpus fingerprint: bump when a change here alters the train or holdout output.
VERSION = 2

_MAX_OUTPUT_SHARD_BYTES = 256 * 1024 * 1024


class FeatureEngineeringSettings(BaseModel):
    target_train_bytes: int = Field(default=12 * 10**9, description="Target UTF-8 size of the training corpus.")
    holdout_docs_per_source: int = Field(default=2_000, description="Maximum evaluation documents per source.")
    holdout_max_fraction: float = Field(default=0.01, description="Maximum share of a source used for holdout.")
    near_dedup: bool = Field(default=True, description="Remove near-duplicates (MinHash LSH) after exact dedup.")
    minhash_ngram: int = Field(default=5, ge=1, description="Words per shingle.")
    minhash_bands: int = Field(default=14, ge=1, description="LSH bands; more bands catch less similar pairs.")
    minhash_rows: int = Field(default=8, ge=1, description="MinHash values per band; more rows need more similarity.")
    workers: int = Field(default=max(1, (os.cpu_count() or 2) - 2), description="Parallel band-hash workers.")
    data_root: Path = Field(default_factory=default_data_root, description="Root of the data/ directory.")


class _SourceIndex:
    """Hash and size columns of one source's preprocessed shards."""

    def __init__(self, source_id: str, paths: DataPaths) -> None:
        self.source_id = source_id
        self.shards = sorted(paths.preprocessed_source(source_id).glob("part-*.parquet"))
        tables = [pq.read_table(shard, columns=["hash", "n_bytes"]) for shard in self.shards]
        self.shard_rows = np.array([table.num_rows for table in tables], dtype=np.int64)
        self.hashes = np.concatenate([table.column("hash").to_numpy() for table in tables])
        self.sizes = np.concatenate([table.column("n_bytes").to_numpy() for table in tables])


def run(settings: FeatureEngineeringSettings | None = None) -> MixtureSummary:
    """Build the training and holdout corpora; returns the ``mixture.json`` summary."""

    settings = settings or FeatureEngineeringSettings()
    logger.info("Starting feature engineering")
    paths = DataPaths(settings.data_root)

    sources = [source for source in mixture.SOURCE_PRIORITY if paths.source_stats(source).is_file()]
    unknown = sorted({path.parent.name for path in paths.preprocessed.glob("*/stats.json")} - set(sources))
    if unknown:
        logger.warning("Preprocessed sources without a priority/weight are ignored: %s", unknown)

    if not sources:
        raise MissingInputError(
            f"No preprocessed sources found in {paths.preprocessed} (expected <source>/stats.json). "
            "Run the download and preprocessing steps first (download_config.run_step is False in main.py)."
        )

    corpus_print = corpus_fingerprint(settings, sources)
    previous = _previous_summary(paths.mixture)
    if previous is not None:
        if previous.fingerprint == corpus_print:
            logger.info("Training corpus already built from the same inputs and settings, skipping")
            return previous
        logger.info("Settings, mixture or preprocessed inputs changed since the last build, rebuilding")

    indexes = [_SourceIndex(source, paths) for source in sources]
    logger.info("Loaded %d document hashes from %d sources", sum(len(index.hashes) for index in indexes), len(indexes))

    # Global exact dedup: np.unique returns the first index of each hash in priority order.
    all_hashes = np.concatenate([index.hashes for index in indexes])
    _, first_occurrence = np.unique(all_hashes, return_index=True)
    global_keep = np.zeros(len(all_hashes), dtype=bool)
    global_keep[first_occurrence] = True
    del all_hashes, first_occurrence

    near_duplicate = np.zeros(len(global_keep), dtype=bool)
    if settings.near_dedup:
        files = _signature_files(indexes, settings)
        near_duplicate = near_dedup.near_duplicate_mask(_band_columns(files, settings.minhash_bands), global_keep)
        global_keep &= ~near_duplicate
        logger.info("Near-dedup removed %d documents", near_duplicate.sum())

    stats: dict[str, SourceMixtureStats] = {}
    masks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    offset = 0
    for index in indexes:
        keep = global_keep[offset : offset + len(index.hashes)]
        near = near_duplicate[offset : offset + len(index.hashes)]
        offset += len(index.hashes)
        holdout = _holdout_mask(index.hashes, keep, settings)
        candidates = keep & ~holdout
        masks[index.source_id] = (candidates, holdout)
        stats[index.source_id] = SourceMixtureStats(
            documents=len(index.hashes),
            duplicates_removed=int((~keep & ~near).sum()),
            near_duplicates_removed=int(near.sum()),
            holdout_documents=int(holdout.sum()),
            available_bytes=int(index.sizes[candidates].sum()),
        )

    fractions = allocate_budget(
        {source: stat.available_bytes for source, stat in stats.items()}, settings.target_train_bytes
    )

    paths.mixture.unlink(missing_ok=True)
    rmtree(paths.train, ignore_errors=True)
    rmtree(paths.holdout, ignore_errors=True)
    writers = {
        "train": RollingParquetWriter(paths.train, CORPUS_SCHEMA, _MAX_OUTPUT_SHARD_BYTES),
        "holdout": RollingParquetWriter(paths.holdout, CORPUS_SCHEMA, _MAX_OUTPUT_SHARD_BYTES),
    }
    with writers["train"], writers["holdout"]:
        for index in indexes:
            candidates, holdout = masks[index.source_id]
            train = _sample_mask(index.hashes, candidates, fractions[index.source_id])
            stat = stats[index.source_id]
            stat.sample_fraction = round(fractions[index.source_id], 6)
            stat.train_documents = int(train.sum())
            stat.train_bytes = int(index.sizes[train].sum())
            logger.info("[%s] writing %d train / %d holdout documents", index.source_id, train.sum(), holdout.sum())
            _write_selected(index, train, holdout, writers)

    train_bytes = sum(stat.train_bytes for stat in stats.values())
    for source, stat in stats.items():
        stat.train_share = round(stat.train_bytes / max(1, train_bytes), 4)
        logger.info(
            "[%s] dedup removed %d exact and %d near, sampled %.1f%%, %.2f GB (%.1f%% of corpus)",
            source,
            stat.duplicates_removed,
            stat.near_duplicates_removed,
            100 * stat.sample_fraction,
            stat.train_bytes / 1e9,
            100 * stat.train_share,
        )
    logger.info("Training corpus: %.2f GB in %d shards", train_bytes / 1e9, writers["train"].shard_count)

    summary = MixtureSummary(
        fingerprint=corpus_print,
        settings=settings.model_dump(mode="json", exclude={"data_root", "workers"}),
        mixture_weights=dict(mixture.MIXTURE_WEIGHTS),
        train_bytes=train_bytes,
        sources=stats,
        provenance=Provenance.current(),
    )
    summary.save(paths.mixture)
    return summary


def corpus_fingerprint(settings: FeatureEngineeringSettings, sources: list[str]) -> str:
    """Fingerprint of everything the training and holdout corpora depend on."""

    paths = DataPaths(settings.data_root)
    inputs = {
        source: {
            "stats": SourceStats.load(paths.source_stats(source)).fingerprint,
            "shards": file_signatures(sorted(paths.preprocessed_source(source).glob("part-*.parquet"))),
        }
        for source in sources
    }
    return fingerprint(
        {"feature_engineering": VERSION, "mixture": mixture.VERSION, "near_dedup": near_dedup.VERSION},
        settings.model_dump(exclude={"data_root", "workers"}),
        mixture.SOURCE_PRIORITY,
        mixture.MIXTURE_WEIGHTS,
        inputs,
    )


def _previous_summary(path: Path) -> MixtureSummary | None:
    if not path.is_file():
        return None
    try:
        return MixtureSummary.load(path)
    except ValidationError:
        logger.info("%s has an outdated format, rebuilding", path.name)
        return None


def _signature_files(indexes: list[_SourceIndex], settings: FeatureEngineeringSettings) -> list[Path]:
    """Band-hash cache file of every shard, in index order; missing ones are computed in parallel."""

    paths = DataPaths(settings.data_root)
    minhash = (settings.minhash_ngram, settings.minhash_bands, settings.minhash_rows)
    files, pending = [], []
    for index in indexes:
        cache_dir = paths.minhash_source(index.source_id)
        expected = [near_dedup.signature_path(shard, cache_dir, *minhash) for shard in index.shards]
        for stale in set(cache_dir.glob("*.npy")) - set(expected):
            stale.unlink()
        files += expected
        pending += [(shard, path) for shard, path in zip(index.shards, expected) if not path.is_file()]
    logger.info(
        "Band hashes: %d of %d shards cached, computing %d", len(files) - len(pending), len(files), len(pending)
    )
    with ProcessPoolExecutor(max_workers=settings.workers) as executor:
        futures = [executor.submit(near_dedup.shard_signatures, shard, path, *minhash) for shard, path in pending]
        for completed, future in enumerate(as_completed(futures), start=1):
            future.result()
            if completed % 50 == 0 or completed == len(futures):
                logger.info("Band hashes computed for %d/%d shards", completed, len(futures))
    return files


def _band_columns(files: list[Path], bands: int) -> Iterator[np.ndarray]:
    """One band's hashes for every document at a time, concatenated in index order."""

    for band in range(bands):
        yield np.concatenate([np.load(path, mmap_mode="r")[band] for path in files])


def _holdout_mask(hashes: np.ndarray, keep: np.ndarray, settings: FeatureEngineeringSettings) -> np.ndarray:
    """Pick the kept documents with the smallest split key (high hash bits)."""

    candidates = np.flatnonzero(keep)
    count = min(settings.holdout_docs_per_source, int(len(candidates) * settings.holdout_max_fraction))
    mask = np.zeros(len(hashes), dtype=bool)
    if count > 0:
        split_keys = hashes[candidates] >> np.uint64(32)
        mask[candidates[np.argpartition(split_keys, count - 1)[:count]]] = True
    return mask


def _sample_mask(hashes: np.ndarray, candidates: np.ndarray, fraction: float) -> np.ndarray:
    """Deterministic Bernoulli sample on the low 32 hash bits."""

    threshold = np.uint64(min(1.0, fraction) * 2**32)
    return candidates & ((hashes & np.uint64(0xFFFFFFFF)) < threshold)


def _write_selected(
    index: _SourceIndex, train: np.ndarray, holdout: np.ndarray, writers: dict[str, RollingParquetWriter]
) -> None:
    offsets = np.concatenate([[0], np.cumsum(index.shard_rows)])
    for shard, start, stop in zip(index.shards, offsets[:-1], offsets[1:]):
        split_masks = {"train": train[start:stop], "holdout": holdout[start:stop]}
        if not any(mask.any() for mask in split_masks.values()):
            continue
        table = pq.read_table(shard, columns=["text", "source"]).cast(CORPUS_SCHEMA)
        for split, mask in split_masks.items():
            if mask.any():
                writers[split].write(table.filter(pa.array(mask)))


if __name__ == "__main__":
    with exclusive_run(default_data_root()):
        run()
