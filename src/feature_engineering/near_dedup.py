"""Near-duplicate detection with MinHash and locality-sensitive hashing (LSH).

Each document gets ``bands × rows`` MinHash values over its word n-gram
shingles (lowercased letter and digit runs). Each band's ``rows`` values are
hashed to one uint64, so a document is described by ``bands`` band hashes. Two
documents whose shingle sets have Jaccard similarity *s* share at least one band
hash with probability ``1 - (1 - s**rows)**bands``; with the defaults (5-grams,
14 bands of 8 rows, as used for FineWeb) that is 0.92 at s = 0.8, 0.77 at
s = 0.75 and 0.05 at s = 0.5.

Documents with fewer than ``ngram`` words have no shingles and get no
signature (all band hashes 0); they are never near-duplicates.

Band hashes are computed once per preprocessed shard and cached in
``data/feature_engineered/minhash/<source>/``. The cache file name carries a
fingerprint of the shard (name and size), the MinHash settings and
:data:`VERSION`, so a changed shard or setting is recomputed and the outdated
file deleted.
"""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import xxhash

from utils.fileio import atomic_path
from utils.fingerprint import file_signatures, fingerprint

# Part of the corpus fingerprint: bump when a change here alters the band hashes or the duplicate rule.
VERSION = 1

_WORD_SEPARATOR = r"[^\p{L}\p{N}]+"  # RE2: Unicode letters and digits, so č, ć, đ, š, ž stay inside words
_SEED = 20261023
_MIX = np.uint64(0x9E3779B97F4A7C15)  # odd multiplier for combining hashes
_BATCH_ROWS = 4_096
_BLOCK_SHINGLES = 1 << 16  # 512 KB of shingle hashes per MinHash block, within a core's L2 cache


def signature_path(shard: Path, cache_dir: Path, ngram: int, bands: int, rows: int) -> Path:
    """Cache file of *shard*'s band hashes; the name changes with the shard and the settings."""

    key = fingerprint(VERSION, ngram, bands, rows, file_signatures([shard]))[:16]
    return cache_dir / f"{shard.stem}.{key}.npy"


def shard_signatures(shard: Path, output: Path, ngram: int, bands: int, rows: int) -> int:
    """Compute and save *shard*'s ``(bands, documents)`` band hashes to *output*; returns the document count."""

    batches = [
        band_hashes(batch.column("text"), ngram, bands, rows)
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=_BATCH_ROWS, columns=["text"])
    ]
    signatures = np.concatenate(batches, axis=1) if batches else np.zeros((bands, 0), dtype=np.uint64)
    with atomic_path(output) as temporary, temporary.open("wb") as handle:
        np.save(handle, signatures)
    return int(signatures.shape[1])


def band_hashes(texts: pa.Array | pa.ChunkedArray, ngram: int, bands: int, rows: int) -> np.ndarray:
    """``(bands, len(texts))`` uint64 band hashes; a column of zeros for texts with fewer than *ngram* words."""

    if isinstance(texts, pa.ChunkedArray):
        texts = texts.combine_chunks()
    documents = len(texts)
    result = np.zeros((bands, documents), dtype=np.uint64)
    words = pc.split_pattern_regex(pc.utf8_lower(texts), pattern=_WORD_SEPARATOR)
    flat = pc.list_flatten(words)
    parents = pc.list_parent_indices(words)
    nonempty = pc.not_equal(flat, "")
    flat, parents = pc.filter(flat, nonempty), pc.filter(parents, nonempty).to_numpy()
    if len(flat) < ngram:
        return result

    # Hash each distinct word once, then look the hashes up for every occurrence.
    encoded = pc.dictionary_encode(flat)
    vocabulary = encoded.dictionary.to_pylist()
    word_hashes = np.fromiter((xxhash.xxh3_64_intdigest(word) for word in vocabulary), np.uint64, len(vocabulary))
    word_hashes = word_hashes[encoded.indices.to_numpy()]

    # Shingle starting at word i: its ngram words must all belong to the same document.
    starts = len(word_hashes) - ngram + 1
    with np.errstate(over="ignore"):
        shingles = word_hashes[:starts].copy()
        for k in range(1, ngram):
            shingles = shingles * _MIX + word_hashes[k : starts + k]
    valid = parents[:starts] == parents[ngram - 1 :]
    shingles, shingle_docs = _finalize(shingles[valid]), parents[:starts][valid]
    if len(shingles) == 0:
        return result

    # MinHash per document: one pass per hash function, minimum over each document's shingles. The passes
    # run over blocks of whole documents small enough for the CPU cache; over a whole batch they are bound
    # by memory bandwidth once several workers run.
    signed_docs, first_shingle = np.unique(shingle_docs, return_index=True)
    multipliers, offsets = _permutations(bands * rows)
    minima = np.empty((bands * rows, len(signed_docs)), dtype=np.uint64)
    bounds = np.append(first_shingle, len(shingles))  # shingle range of document i: bounds[i]:bounds[i + 1]
    edges = np.searchsorted(first_shingle, np.arange(0, len(shingles), _BLOCK_SHINGLES))
    edges = np.append(np.unique(edges[edges < len(signed_docs)]), len(signed_docs))
    with np.errstate(over="ignore"):
        for start, stop in zip(edges[:-1], edges[1:]):
            block = shingles[bounds[start] : bounds[stop]]
            block_starts = first_shingle[start:stop] - bounds[start]
            for index in range(bands * rows):
                values = (block * multipliers[index] + offsets[index]) >> np.uint64(32)
                minima[index, start:stop] = np.minimum.reduceat(values, block_starts)
        for band in range(bands):
            combined = np.zeros(len(signed_docs), dtype=np.uint64)
            for value in minima[band * rows : (band + 1) * rows]:
                combined = combined * _MIX + value
            result[band, signed_docs] = _finalize(combined) | np.uint64(1)  # never 0, which marks "no signature"
    return result


def _finalize(values: np.ndarray) -> np.ndarray:
    """splitmix64 finalizer, so combined hashes use all 64 bits."""

    values = values ^ (values >> np.uint64(30))
    values = values * np.uint64(0xBF58476D1CE4E5B9)
    values = values ^ (values >> np.uint64(27))
    values = values * np.uint64(0x94D049BB133111EB)
    finalized: np.ndarray = values ^ (values >> np.uint64(31))
    return finalized


def _permutations(count: int) -> tuple[np.ndarray, np.ndarray]:
    """Multipliers (odd) and offsets of *count* multiply-shift hash functions, fixed by the seed."""

    rng = np.random.default_rng(_SEED)
    multipliers = rng.integers(1, 2**63, size=count, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
    offsets = rng.integers(0, 2**63, size=count, dtype=np.uint64)
    return multipliers, offsets


def near_duplicate_mask(band_columns: Iterable[np.ndarray], keep: np.ndarray) -> np.ndarray:
    """Documents in *keep* that share a band hash with an earlier document in *keep*.

    *band_columns* yields one band's hashes for all documents at a time, in
    priority order (earlier documents win), so only one band is in memory.
    """

    duplicate = np.zeros(len(keep), dtype=bool)
    for column in band_columns:
        candidates = np.flatnonzero(keep & (column != 0))
        _, first = np.unique(column[candidates], return_index=True)
        later = np.ones(len(candidates), dtype=bool)
        later[first] = False
        duplicate[candidates[later]] = True
    return duplicate
