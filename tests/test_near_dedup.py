"""MinHash band hashes and the near-duplicate rule."""

from __future__ import annotations

import random

import numpy as np
import pyarrow as pa

from feature_engineering.near_dedup import band_hashes, near_duplicate_mask

_SETTINGS = {"ngram": 5, "bands": 14, "rows": 8}


def _document(seed: int, words: int = 300) -> list[str]:
    rng = random.Random(seed)
    return [f"riječ{rng.randrange(20_000)}" for _ in range(words)]


def test_band_hashes_find_near_copies_only() -> None:
    original = _document(0)
    edited = [*original[:150], "promijenjeno", *original[151:]]
    texts = [" ".join(original), " ".join(original), " ".join(edited), " ".join(_document(1))]
    bands = band_hashes(pa.array(texts), **_SETTINGS)
    assert bands.shape == (14, 4) and bands.dtype == np.uint64
    assert (bands[:, 0] == bands[:, 1]).all()
    assert (bands[:, 0] == bands[:, 2]).any()
    assert not (bands[:, 0] == bands[:, 3]).any()


def test_case_and_punctuation_are_ignored() -> None:
    words = _document(2, 40)
    plain = " ".join(words)
    styled = ", ".join(word.upper() for word in words) + "!"
    bands = band_hashes(pa.chunked_array([[plain], [styled]]), **_SETTINGS)
    assert (bands[:, 0] == bands[:, 1]).all()


def test_short_and_empty_texts_get_no_signature() -> None:
    bands = band_hashes(pa.array(["Dobar dan svima.", "", "   ", " ".join(_document(3, 5))]), **_SETTINGS)
    assert not bands[:, :3].any()
    assert bands[:, 3].all()  # exactly ngram words: one shingle


def test_near_duplicate_mask_keeps_the_earliest_kept_copy() -> None:
    #                  doc: 0  1  2  3  4  5
    bands = np.array([[7, 7, 9, 7, 0, 0], [1, 2, 3, 4, 0, 0]], dtype=np.uint64)
    keep = np.array([False, True, True, True, True, True])
    duplicate = near_duplicate_mask(bands, keep)
    # Doc 0 is not kept (an exact duplicate), so doc 1 is the first copy; docs without a signature never match.
    assert duplicate.tolist() == [False, False, False, True, False, False]
