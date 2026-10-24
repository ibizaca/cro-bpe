"""Evaluation data: the holdout set and word frequencies from the training corpus."""

import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from utils.artifacts import MixtureSummary
from utils.errors import MissingInputError
from utils.project import DataPaths

_EDGE_PUNCTUATION = re.compile(r"^\W+|\W+$")


def load_holdout(data_root: Path) -> tuple[list[str], list[str]]:
    """Holdout texts and their source ids."""

    holdout_dir = DataPaths(data_root).holdout
    files = sorted(holdout_dir.glob("part-*.parquet"))
    if not files:
        raise MissingInputError(f"No holdout shards in {holdout_dir}; run feature engineering first")
    texts: list[str] = []
    sources: list[str] = []
    for path in files:
        table = pq.read_table(path, columns=["text", "source"])
        texts.extend(table.column("text").to_pylist())
        sources.extend(table.column("source").to_pylist())
    return texts, sources


def mixture_weights(data_root: Path) -> dict[str, float]:
    """Each source's share of training bytes, from ``mixture.json``."""

    return MixtureSummary.load(DataPaths(data_root).mixture).train_shares()


def training_word_ranks(data_root: Path, sample_bytes: int, max_rank: int) -> dict[str, int]:
    """Frequency rank (0 = most frequent) of the *max_rank* most frequent words.

    Counted on about *sample_bytes* of training text, taken in equal parts from
    the start of every training shard.
    """

    train_dir = DataPaths(data_root).train
    files = sorted(train_dir.glob("part-*.parquet"))
    if not files:
        raise MissingInputError(f"No train shards in {train_dir}; run feature engineering first")
    counts: Counter[str] = Counter()
    per_file_bytes = sample_bytes // len(files)
    for path in files:
        read_bytes = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1_000, columns=["text"]):
            for text in batch.column("text").to_pylist():
                counts.update(key for key in map(word_key, text.split()) if key)
                read_bytes += len(text)
            if read_bytes >= per_file_bytes:
                break
    return {word: rank for rank, (word, _) in enumerate(counts.most_common(max_rank))}


def word_key(word: str) -> str:
    """Whitespace word without leading and trailing punctuation (case kept, as the tokenizer sees it)."""

    return _EDGE_PUNCTUATION.sub("", word)
