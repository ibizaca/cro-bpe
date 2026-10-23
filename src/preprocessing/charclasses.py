"""Vectorized per-code-point character classes.

Counting character classes with ``re`` scans a document once per class in
Python-level match objects; a lookup table over the UTF-32 code points does
all classes in a few numpy operations.
"""

import re

import numpy as np

# Part of the preprocessing fingerprint: bump when a change to the character tables alters the output.
VERSION = 1

LETTER = np.uint8(1)
LATIN_LETTER = np.uint8(2)
WHITESPACE = np.uint8(4)
NEEDS_NORMALIZATION = np.uint8(8)

# Characters rewritten or removed by ``cleaning.normalize_text`` (other than "&",
# multiple spaces and newlines, which are cheap substring checks).
NORMALIZATION_CHARS = (
    "\r\x85  \x0b\x0c"  # line breaks
    "\t   -   　"  # spaces
    "\x00-\x08\x0e-\x1f\x7f-\x84\x86-\x9f­​-‏‪-‮⁠-⁤﻿￾￿"  # invisible
    "Ǆ-ǌﬀ-ﬄ"  # digraph letters and ligatures
)

_LATIN_RANGES = [(0x41, 0x5A), (0x61, 0x7A), (0xC0, 0x24F), (0x1E00, 0x1EFF)]


def _build_table() -> np.ndarray:
    table = np.zeros(0x110000, dtype=np.uint8)
    for code_point in range(0x110000):
        character = chr(code_point)
        if character.isalpha():
            table[code_point] |= LETTER
        if character.isspace():
            table[code_point] |= WHITESPACE
    for start, stop in _LATIN_RANGES:
        latin = np.arange(start, stop + 1)
        table[latin[(table[latin] & LETTER) != 0]] |= LATIN_LETTER
    normalization = re.compile(f"[{NORMALIZATION_CHARS}]")
    for code_point in range(0x10000):
        if normalization.match(chr(code_point)):
            table[code_point] |= NEEDS_NORMALIZATION
    return table


_TABLE = _build_table()


def classify(text: str) -> np.ndarray:
    """Class bit flags for every code point of *text*."""

    return _TABLE[np.frombuffer(text.encode("utf-32-le", "surrogatepass"), dtype=np.uint32)]


def count(flags: np.ndarray, character_class: np.uint8) -> int:
    return int(np.count_nonzero(flags & character_class))
