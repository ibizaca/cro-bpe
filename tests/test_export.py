"""tiktoken export: rank file contents, special tokens and the equality gate."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from conftest import CORPUS, ROUNDTRIP_TEXTS
from model_training.export import load_tiktoken, verify_tiktoken
from model_training.tokenizer import MASK, SPECIAL_TOKENS, build_tokenizer, build_trainer

_MODELS_ROOT = Path(__file__).resolve().parents[1] / "data" / "models"
_HOLDOUT = Path(__file__).resolve().parents[1] / "data" / "feature_engineered" / "holdout" / "part-00000.parquet"


def test_special_tokens_are_not_mergeable_ranks(tiny_model_dir: Path) -> None:
    encoding = load_tiktoken(tiny_model_dir)
    assert not any(token.startswith(b"<|") and token.endswith(b"|>") for token in encoding._mergeable_ranks)
    assert min(encoding._mergeable_ranks.values()) == len(SPECIAL_TOKENS)
    assert encoding.encode(MASK, allowed_special="all") == [2]
    assert encoding.encode("<|reserved_63|>", allowed_special="all") == [len(SPECIAL_TOKENS) - 1]


def test_config_uses_the_models_own_pattern(tiny_model_dir: Path) -> None:
    config = json.loads((tiny_model_dir / "tiktoken.json").read_text(encoding="utf-8"))
    tokenizer_config = json.loads((tiny_model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    split = tokenizer_config["pre_tokenizer"]["pretokenizers"][0]
    assert config["pattern"] == split["pattern"]["Regex"]
    assert config["special_tokens"] == {token: i for i, token in enumerate(SPECIAL_TOKENS)}


def test_ids_match_tokenizer_json(tiny_model_dir: Path, tiny_tokenizer: Tokenizer) -> None:
    encoding = load_tiktoken(tiny_model_dir)
    verify_tiktoken(encoding, tiny_tokenizer, ROUNDTRIP_TEXTS)  # raises on any difference
    for text in ROUNDTRIP_TEXTS:
        assert encoding.decode(encoding.encode_ordinary(text)) == text


def test_nfc_normalization_matches(tiny_model_dir: Path, tiny_tokenizer: Tokenizer) -> None:
    encoding = load_tiktoken(tiny_model_dir)
    decomposed = unicodedata.normalize("NFD", "čaša i žaba")
    assert encoding.encode_ordinary(decomposed) == tiny_tokenizer.encode(decomposed).ids
    assert encoding.encode_ordinary_batch([decomposed]) == [tiny_tokenizer.encode(decomposed).ids]


def test_gate_rejects_a_different_tokenizer(tiny_model_dir: Path) -> None:
    other = build_tokenizer()
    other.train_from_iterator(CORPUS, trainer=build_trainer(400, 1, 24))
    with pytest.raises(ValueError, match="differs from tokenizer.json"):
        verify_tiktoken(load_tiktoken(tiny_model_dir), other, CORPUS)


@pytest.mark.parametrize("model_dir", sorted(_MODELS_ROOT.glob("cro-bpe-*k")), ids=lambda path: path.name)
def test_trained_model(model_dir: Path) -> None:
    """Trained models (skipped when absent): special-token ids, round-trip and the gate on holdout text."""
    if not (model_dir / "tiktoken.json").is_file() or not _HOLDOUT.is_file():
        pytest.skip("model not exported or holdout set not built")
    import pyarrow.parquet as pq

    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    assert [tokenizer.id_to_token(i) for i in range(len(SPECIAL_TOKENS))] == SPECIAL_TOKENS
    texts = pq.read_table(_HOLDOUT, columns=["text"]).column("text").to_pylist()[::50]
    decoded = tokenizer.decode_batch([e.ids for e in tokenizer.encode_batch_fast(texts, add_special_tokens=False)])
    assert decoded == texts
    verify_tiktoken(load_tiktoken(model_dir), tokenizer, texts)
