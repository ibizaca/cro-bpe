"""Document-level quality heuristics.

The corpus is Croatian-only, so no language identification is performed; the
filters remove extraction noise (markup, code, tables, repeated boilerplate)
and stray non-Latin-script documents that would otherwise waste vocabulary.
"""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, Field

from preprocessing.charclasses import LATIN_LETTER, LETTER, WHITESPACE, classify, count

# Part of the preprocessing fingerprint: bump when a change to the filters or their defaults alters the output.
VERSION = 1


class QualityThresholds(BaseModel):
    min_chars: int = Field(default=20, description="Minimum document length after cleaning.")
    min_words: int = Field(default=3, description="Minimum whitespace-separated words.")
    min_letter_ratio: float = Field(default=0.5, description="Minimum letters / non-whitespace characters.")
    max_non_latin_ratio: float = Field(default=0.15, description="Maximum non-Latin letters / letters.")
    min_mean_word_length: float = Field(default=2.0, description="Lower bound on mean word length.")
    max_mean_word_length: float = Field(default=14.0, description="Upper bound on mean word length.")
    max_duplicate_line_char_ratio: float = Field(
        default=0.3, description="Maximum share of characters in lines repeated within the document."
    )


def _duplicate_line_char_ratio(text: str) -> float:
    lines = [line for line in text.split("\n") if line]
    if len(lines) < 2:
        return 0.0
    counts = Counter(lines)
    duplicated = sum(len(line) * (count - 1) for line, count in counts.items() if count > 1)
    return duplicated / max(1, sum(len(line) for line in lines))


def rejection_reason(text: str, thresholds: QualityThresholds) -> str | None:
    """Return the first failed check's name, or ``None`` if the document is kept."""

    if len(text) < thresholds.min_chars:
        return "too_short"

    words = len(text.split())
    if words < thresholds.min_words:
        return "too_few_words"

    flags = classify(text)
    non_whitespace = len(flags) - count(flags, WHITESPACE)
    mean_word_length = non_whitespace / words
    if not thresholds.min_mean_word_length <= mean_word_length <= thresholds.max_mean_word_length:
        return "mean_word_length"

    letters = count(flags, LETTER)
    if letters < thresholds.min_letter_ratio * non_whitespace:
        return "low_letter_ratio"

    if letters - count(flags, LATIN_LETTER) > thresholds.max_non_latin_ratio * letters:
        return "non_latin_script"

    if "\n" in text and _duplicate_line_char_ratio(text) > thresholds.max_duplicate_line_char_ratio:
        return "repeated_lines"

    return None
