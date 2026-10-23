"""Evaluation metrics on the tiny tokenizer."""

import math

from tokenizers import Tokenizer

from conftest import CORPUS
from evaluation.metrics import quality_metrics, rare_word_fertility, vocabulary_metrics


def test_quality_totals_and_roundtrip(tiny_tokenizer: Tokenizer) -> None:
    sources = ["a"] * 3 + ["b"] * (len(CORPUS) - 3)
    quality = quality_metrics(tiny_tokenizer, CORPUS, sources, batch_size=2)
    tokens = sum(len(tiny_tokenizer.encode(text, add_special_tokens=False).ids) for text in CORPUS)
    assert quality["all"].tokens == tokens == quality["a"].tokens + quality["b"].tokens
    assert quality["all"].bytes_per_token == sum(len(t.encode()) for t in CORPUS) / tokens
    assert quality["all"].roundtrip_failures == 0


def test_renyi_efficiency_is_between_zero_and_one(tiny_tokenizer: Tokenizer) -> None:
    efficiency = vocabulary_metrics(tiny_tokenizer, CORPUS, renyi_alpha=2.5).renyi_efficiency
    assert 0 < efficiency < 1


def test_renyi_efficiency_of_uniform_use() -> None:
    """Each of the 95 printable ASCII characters used once: Rényi entropy is log(95) at any order."""
    from tokenizers import models, pre_tokenizers

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tokenizer = Tokenizer(models.BPE({character: i for i, character in enumerate(alphabet)}, []))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    text = "".join(chr(code) for code in range(32, 127))
    ids = tokenizer.encode(text).ids
    assert len(ids) == len(set(ids)) == 95
    metrics = vocabulary_metrics(tokenizer, [text], renyi_alpha=2.5)
    assert math.isclose(metrics.renyi_efficiency, math.log(95) / math.log(256))


def test_rare_word_fertility_assigns_every_token(tiny_tokenizer: Tokenizer) -> None:
    ranks = {"je": 0, "i": 1, "Zakon": 5_000, "stanovnika": 50_000}
    fertility = rare_word_fertility(tiny_tokenizer, CORPUS, ranks, batch_size=3)
    assert list(fertility) == ["top-1k", "1k-10k", "10k-100k", "rest"]
    assert fertility["top-1k"].words == sum(text.split().count(w) for text in CORPUS for w in ("je", "i"))
    assert fertility["1k-10k"].words == sum(text.split().count("Zakon") for text in CORPUS)
    # Every whitespace word with letters or digits is counted exactly once.
    counted = sum(bucket.words for bucket in fertility.values())
    assert counted == sum(1 for text in CORPUS for word in text.split() if any(c.isalnum() for c in word))
