"""Shared fixtures: a tiny tokenizer trained with the production definition, and its tiktoken export."""

import random
import sys
from collections.abc import Callable
from pathlib import Path

# Tests must never use MLflow: any `import mlflow` on a test path raises ImportError.
# Stages take a tracker argument, and tests pass utils.tracking.NullTracker.
sys.modules["mlflow"] = None  # type: ignore[assignment]

import pytest
from tokenizers import Tokenizer

from model_training.export import export_tiktoken
from model_training.tokenizer import SPECIAL_TOKENS, build_tokenizer, build_trainer

TINY_VOCAB_SIZE = 500  # special tokens (69) + byte alphabet (256) + 175 merges

CORPUS = [
    "Hrvatski jezik obuhvaća govoreni i pisani hrvatski standardni jezik.",
    "Zastupnici Hrvatskoga sabora izglasali su Zakon o izmjenama i dopunama Zakona o porezu.",
    "Đurđevac je 2024. godine imao 6.123 stanovnika, a Čakovec 27.104.",
    "Najneodgovornijima se pokazaše oni koji su se prvi javili.",
    "Dobar dan, kako ste? Ja sam dobro, hvala na pitanju!",
    "Šuma, žaba, džep, ljubav, njiva, ćevapi i čvarci.",
    "U 10 sati i 30 minuta počinje sjednica (vidi članak 5.).",
]

# Inputs that stress the byte-level path: diacritics, digraphs, emoji, CJK,
# CRLF, tabs, whitespace runs, leading/trailing spaces and special-token look-alikes.
ROUNDTRIP_TEXTS = [
    *CORPUS,
    "",
    " ",
    "   višestruki   razmaci   ",
    "red1\r\nred2\n\n\nred3\ttab",
    "Emoji 🇭🇷 i 😀, kineski 中文, ćirilica Здраво.",
    "<|endof text|> nije poseban token, <| mask |> također nije.",
    "0 00 000 0000 1234567 3.14159 -42 1e10",
]


def synthetic_corpus(documents: int, seed: int = 0) -> list[str]:
    """Croatian-looking words with enough variety for a few thousand merges."""

    rng = random.Random(seed)
    syllables = ["ka", "ri", "jel", "no", "ža", "šu", "ma", "vi", "ći", "dra", "go", "sti", "lje", "nje", "đu"]
    words = ["".join(rng.choices(syllables, k=rng.randint(1, 4))) for _ in range(3_000)]
    return [" ".join(rng.choices(words, k=60)) + "." for _ in range(documents)]


FAKE_REVISION = "0123456789abcdef0123456789abcdef01234567"


def fake_hub_download(files: dict[tuple[str, str], bytes]) -> Callable[..., str]:
    """``hf_hub_download`` stand-in serving *files* by (repository, filename) in the Hub cache layout."""

    from huggingface_hub.errors import EntryNotFoundError

    def hf_hub_download(repo_id: str, filename: str, cache_dir: Path) -> str:
        if not any(repository == repo_id for repository, _ in files):
            raise OSError(f"no network in tests: {repo_id}")
        if (repo_id, filename) not in files:
            raise EntryNotFoundError(f"{filename} not in {repo_id}")
        path = Path(cache_dir) / f"models--{repo_id.replace('/', '--')}" / "snapshots" / FAKE_REVISION / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(files[repo_id, filename])
        return str(path)

    return hf_hub_download


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Directory with tokenizer.json, vocab.json, merges.txt and the tiktoken export of a tiny model."""

    model_dir = tmp_path_factory.mktemp("cro-bpe-tiny")
    tokenizer = build_tokenizer()
    tokenizer.train_from_iterator(CORPUS * 50, trainer=build_trainer(TINY_VOCAB_SIZE, 1, 24))
    assert tokenizer.get_vocab_size() == TINY_VOCAB_SIZE
    tokenizer.save(str(model_dir / "tokenizer.json"))
    tokenizer.model.save(str(model_dir))
    export_tiktoken(model_dir)
    return model_dir


@pytest.fixture(scope="session")
def tiny_tokenizer(tiny_model_dir: Path) -> Tokenizer:
    return Tokenizer.from_file(str(tiny_model_dir / "tokenizer.json"))


__all__ = [
    "CORPUS",
    "FAKE_REVISION",
    "ROUNDTRIP_TEXTS",
    "SPECIAL_TOKENS",
    "TINY_VOCAB_SIZE",
    "fake_hub_download",
    "synthetic_corpus",
]
