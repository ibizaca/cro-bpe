"""Export a trained byte-level BPE tokenizer to tiktoken format.

Writes next to ``tokenizer.json``::

    cro-bpe-<vocab>.tiktoken ← base64 token bytes and rank per line
    tiktoken.json            ← pre-tokenizer pattern, special tokens, rank file name

Ranks equal the Hugging Face token ids. The trainer assigns ids after the
special tokens and the 256-byte alphabet in merge order, so id order is merge
priority. tiktoken has no normalizer; :func:`load_tiktoken` applies NFC before
encoding, as ``tokenizer.json`` does.

:func:`verify_tiktoken` is the equality gate: tiktoken ids must equal the
``encode_batch_fast`` ids for every document, otherwise it raises. Text that
spells a special token is encoded as that token by ``tokenizer.json``, so the
gate compares against ``encode(..., allowed_special="all")``; preprocessing
removes such strings from the training data.
"""

import base64
import json
import unicodedata
from logging import getLogger
from pathlib import Path
from typing import Any

import tiktoken
from tokenizers import Tokenizer

from utils.errors import ExportMismatchError

logger = getLogger(__name__)

_CONFIG_NAME = "tiktoken.json"


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte-to-character map used by the ``ByteLevel`` pre-tokenizer."""

    printable = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
    printable += list(range(ord("®"), ord("ÿ") + 1))
    characters = printable[:]
    offset = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            characters.append(256 + offset)
            offset += 1
    return dict(zip(printable, map(chr, characters)))


def _split_pattern(tokenizer_config: dict[str, Any]) -> str:
    """The ``Split`` regex of the saved pre-tokenizer (the model's own, not the current constant)."""

    pre_tokenizer = tokenizer_config["pre_tokenizer"]
    for step in pre_tokenizer.get("pretokenizers", [pre_tokenizer]):
        if step["type"] == "Split":
            return str(step["pattern"]["Regex"])
    raise ValueError("tokenizer.json has no Split pre-tokenizer")


class NFCEncoding(tiktoken.Encoding):
    """tiktoken encoding that NFC-normalizes text first, matching ``tokenizer.json``.

    The batch methods call these per document, so they are normalized too.
    """

    def encode_ordinary(self, text: str) -> list[int]:
        return super().encode_ordinary(unicodedata.normalize("NFC", text))

    def encode(self, text: str, **kwargs: Any) -> list[int]:  # type: ignore[override]
        return super().encode(unicodedata.normalize("NFC", text), **kwargs)


def export_tiktoken(model_dir: Path, name: str | None = None) -> Path:
    """Write the tiktoken rank file and config for ``model_dir/tokenizer.json``; return the rank file.

    *name* (default: the directory name) names the encoding and the rank file, so a
    model can be exported in a temporary directory before it is moved into place.
    """

    name = name or model_dir.name
    tokenizer_config = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    char_to_byte = {character: byte for byte, character in _bytes_to_unicode().items()}
    special_tokens = {token["content"]: token["id"] for token in tokenizer_config["added_tokens"]}
    # The BPE vocabulary also lists the special tokens; they belong only in special_tokens.
    ordinary = {
        token: token_id for token, token_id in tokenizer_config["model"]["vocab"].items() if token not in special_tokens
    }
    ranks = {bytes(char_to_byte[character] for character in token): token_id for token, token_id in ordinary.items()}
    if len(ranks) != len(ordinary):
        raise ValueError("Duplicate byte sequences in the vocabulary")

    rank_path = model_dir / f"{name}.tiktoken"
    rank_path.write_text(
        "".join(f"{base64.b64encode(token).decode()} {rank}\n" for token, rank in sorted(ranks.items(), key=_rank)),
        encoding="utf-8",
    )
    (model_dir / _CONFIG_NAME).write_text(
        json.dumps(
            {
                "name": name,
                "rank_file": rank_path.name,
                "pattern": _split_pattern(tokenizer_config),
                "special_tokens": special_tokens,
                "normalization": "NFC",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return rank_path


def _rank(item: tuple[bytes, int]) -> int:
    return item[1]


def load_tiktoken(model_dir: Path) -> NFCEncoding:
    """Load the tiktoken export in *model_dir* (see :func:`export_tiktoken`)."""

    config = json.loads((model_dir / _CONFIG_NAME).read_text(encoding="utf-8"))
    ranks = {}
    for line in (model_dir / config["rank_file"]).read_text(encoding="utf-8").splitlines():
        token, rank = line.split()
        ranks[base64.b64decode(token)] = int(rank)
    return NFCEncoding(
        name=config["name"], pat_str=config["pattern"], mergeable_ranks=ranks, special_tokens=config["special_tokens"]
    )


def tiktoken_mismatches(encoding: tiktoken.Encoding, tokenizer: Tokenizer, texts: list[str]) -> list[int]:
    """Indices of the texts for which *encoding* and *tokenizer* produce different ids."""

    expected = [e.ids for e in tokenizer.encode_batch_fast(texts, add_special_tokens=False)]
    actual = encoding.encode_batch(texts, allowed_special="all")
    return [index for index, (left, right) in enumerate(zip(expected, actual)) if left != right]


def verify_tiktoken(encoding: tiktoken.Encoding, tokenizer: Tokenizer, texts: list[str]) -> None:
    """Raise unless *encoding* and *tokenizer* produce identical ids for every text."""

    mismatches = tiktoken_mismatches(encoding, tokenizer, texts)
    if mismatches:
        raise ExportMismatchError(
            f"tiktoken export differs from tokenizer.json on {len(mismatches)} of {len(texts)} documents "
            f"(first: document {mismatches[0]})"
        )
    logger.info("tiktoken export matches tokenizer.json on all %d documents", len(texts))
