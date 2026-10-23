"""Tokenizer definition: special-token ids, pre-tokenization, round-trip and loading."""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from conftest import ROUNDTRIP_TEXTS, TINY_VOCAB_SIZE
from model_training.tokenizer import (
    CLS,
    END_OF_TEXT,
    MASK,
    PAD,
    RESERVED_TOKEN_COUNT,
    SEP,
    SPECIAL_TOKENS,
    build_tokenizer,
    load_tokenizer,
)


def test_special_token_list() -> None:
    assert SPECIAL_TOKENS[:5] == [END_OF_TEXT, PAD, MASK, CLS, SEP]
    assert SPECIAL_TOKENS[5:] == [f"<|reserved_{i}|>" for i in range(RESERVED_TOKEN_COUNT)]
    assert len(SPECIAL_TOKENS) == len(set(SPECIAL_TOKENS)) == 69


def test_special_tokens_get_the_lowest_ids(tiny_tokenizer: Tokenizer) -> None:
    assert [tiny_tokenizer.id_to_token(i) for i in range(len(SPECIAL_TOKENS))] == SPECIAL_TOKENS
    assert tiny_tokenizer.token_to_id(END_OF_TEXT) == 0
    assert tiny_tokenizer.token_to_id(PAD) == 1
    assert tiny_tokenizer.token_to_id(MASK) == 2
    # The byte alphabet follows directly; "!" is the first byte-level character.
    assert tiny_tokenizer.id_to_token(len(SPECIAL_TOKENS)) == "!"


def test_special_token_in_text_is_one_id(tiny_tokenizer: Tokenizer) -> None:
    ids = tiny_tokenizer.encode(f"prije {MASK} poslije", add_special_tokens=False).ids
    assert ids.count(2) == 1


def test_no_post_processor_template(tiny_tokenizer: Tokenizer) -> None:
    """tokenizer.json adds no <|cls|>/<|sep|> wrapping; encoder model code does that itself."""
    ids = tiny_tokenizer.encode("Dobar dan").ids
    assert not set(ids) & set(range(len(SPECIAL_TOKENS)))


@pytest.mark.parametrize(
    ("text", "pieces"),
    [
        # Byte-level notation: "Ġ" is a space, "Ċ" a newline, "Äį" the two UTF-8 bytes of "č".
        ("Godine 2024.", ["Godine", "Ġ202", "4", "."]),  # digit groups take the leading space
        ("u 10 sati", ["u", "Ġ10", "Ġsati"]),
        ("čćđ Džep", ["ÄįÄĩÄĳ", "ĠDÅ¾ep"]),
        ("riječ, riječ!", ["rijeÄį", ",", "ĠrijeÄį", "!"]),
        ("a\n\nb", ["a", "ĊĊ", "b"]),
        ("kraj   ", ["kraj", "ĠĠĠ"]),
    ],
)
def test_pre_tokenizer_splits(text: str, pieces: list[str]) -> None:
    assert [piece for piece, _ in build_tokenizer().pre_tokenizer.pre_tokenize_str(text)] == pieces


@pytest.mark.parametrize("text", ROUNDTRIP_TEXTS)
def test_roundtrip(tiny_tokenizer: Tokenizer, text: str) -> None:
    ids = tiny_tokenizer.encode(text, add_special_tokens=False).ids
    assert tiny_tokenizer.decode(ids, skip_special_tokens=False) == text


def test_nfc_normalization(tiny_tokenizer: Tokenizer) -> None:
    composed = "čaša"
    decomposed = unicodedata.normalize("NFD", composed)
    assert decomposed != composed
    assert tiny_tokenizer.encode(decomposed).ids == tiny_tokenizer.encode(composed).ids


@pytest.mark.parametrize("cache_capacity", [10, 10_000, 1_000_000])
def test_load_tokenizer_matches_from_file(tiny_model_dir: Path, tiny_tokenizer: Tokenizer, cache_capacity: int) -> None:
    loaded = load_tokenizer(tiny_model_dir, cache_capacity=cache_capacity)
    assert loaded.get_vocab_size() == TINY_VOCAB_SIZE
    for text in ROUNDTRIP_TEXTS:
        assert loaded.encode(text).ids == tiny_tokenizer.encode(text).ids
