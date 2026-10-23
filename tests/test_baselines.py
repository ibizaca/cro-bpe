"""Baseline downloads and results without throughput, against a fake Hub (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from conftest import CORPUS, FAKE_REVISION, fake_hub_download
from evaluation import evaluation
from evaluation.metrics import ALL_SOURCES, MIXTURE, quality_metrics, rare_word_fertility, vocabulary_metrics
from evaluation.results import SummaryRow, TokenizerResult

_WORDPIECE_VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hrvatski", "jezik", ".", "##a"]


def test_snapshot_revision() -> None:
    assert evaluation.snapshot_revision(Path(f"/cache/models--a--b/snapshots/{FAKE_REVISION}/tokenizer.json")) == (
        FAKE_REVISION
    )
    assert evaluation.snapshot_revision(Path("/somewhere/tokenizer.json")) is None


def test_download_baselines(tmp_path: Path, tiny_model_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hub = fake_hub_download(
        {
            ("fake/bpe", "tokenizer.json"): (tiny_model_dir / "tokenizer.json").read_bytes(),
            ("fake/wordpiece", "vocab.txt"): "\n".join(_WORDPIECE_VOCAB).encode(),
            ("fake/wordpiece", "tokenizer_config.json"): json.dumps({"do_lower_case": False}).encode(),
        }
    )
    monkeypatch.setattr(evaluation, "hf_hub_download", hub)
    files = evaluation.download_baselines(
        {"bpe": "fake/bpe", "wordpiece": "fake/wordpiece", "offline": "fake/missing"}, tmp_path
    )

    assert set(files) == {"bpe", "wordpiece"}  # an unavailable baseline is skipped, not fatal
    assert files["bpe"].revision == files["wordpiece"].revision == FAKE_REVISION
    assert files["bpe"].path.read_bytes() == (tiny_model_dir / "tokenizer.json").read_bytes()
    assert files["wordpiece"].path.is_relative_to(tmp_path)
    wordpiece = Tokenizer.from_file(str(files["wordpiece"].path))
    # Cased, as tokenizer_config.json says: the capitalized word is not in the vocabulary.
    assert wordpiece.encode("hrvatski Hrvatski jezika.", add_special_tokens=False).tokens == [
        "hrvatski",
        "[UNK]",
        "jezik",
        "##a",
        ".",
    ]


def test_summary_row_without_throughput(tiny_tokenizer: Tokenizer) -> None:
    sources = ["a"] * len(CORPUS)
    result = TokenizerResult(
        engine="tokenizers",
        quality=quality_metrics(tiny_tokenizer, CORPUS, sources, weights={"a": 1.0}),
        vocabulary=vocabulary_metrics(tiny_tokenizer, CORPUS, renyi_alpha=2.5),
        fertility=rare_word_fertility(tiny_tokenizer, CORPUS, {}),
    )
    assert result.throughput is None
    assert TokenizerResult.model_validate_json(result.model_dump_json()) == result

    row = SummaryRow.of(result, ALL_SOURCES, MIXTURE)
    flat = row.flat()
    assert row.encode_batch_megabytes_per_second == {}
    for key in (
        "encode_batch.tokens_per_second",
        "encode_batch.spread",
        "encode_sequential.megabytes_per_second",
        "decode_batch.tokens_per_second",
    ):
        assert flat[key] is None
    assert flat["bytes_per_token"] == result.quality[ALL_SOURCES].bytes_per_token
