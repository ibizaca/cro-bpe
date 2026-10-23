"""Byte-level BPE tokenizer definition for Croatian."""

from __future__ import annotations

from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers

# GPT-4-style split without the English contraction branch: letter runs (with at
# most one leading non-letter, usually a space), digit groups of up to three
# (with an optional leading space, so " 2024" does not spend a token on the
# space), punctuation runs, and whitespace. \p{L} covers č, ć, dž, đ, š, ž natively.
PRE_TOKENIZER_PATTERN = "|".join(  # noqa: FLY002 - one alternative per line is easier to read
    [
        r"[^\r\n\p{L}\p{N}]?\p{L}+",
        r" ?\p{N}{1,3}",
        r" ?[^\s\p{L}\p{N}]+[\r\n]*",
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+",
    ]
)

# Special tokens get the lowest ids, in this order, in every vocabulary size:
# <|endoftext|>=0, <|pad|>=1, <|mask|>=2, <|cls|>=3, <|sep|>=4, <|reserved_0..63|>=5-68.
# mask/cls/sep serve encoder models (ModernBERT, ELECTRA); the reserved slots
# leave room for later chat, tool or fill-in-the-middle markers without resizing.
END_OF_TEXT = "<|endoftext|>"
PAD = "<|pad|>"
MASK = "<|mask|>"
CLS = "<|cls|>"
SEP = "<|sep|>"
RESERVED_TOKEN_COUNT = 64
SPECIAL_TOKENS = [END_OF_TEXT, PAD, MASK, CLS, SEP, *(f"<|reserved_{i}|>" for i in range(RESERVED_TOKEN_COUNT))]

DEFAULT_CACHE_CAPACITY = 10_000  # library default; 1M was slower for batched encoding


def build_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(PRE_TOKENIZER_PATTERN), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    return tokenizer


def build_trainer(vocab_size: int, min_frequency: int, max_token_length: int | None) -> trainers.BpeTrainer:
    return trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        max_token_length=max_token_length,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )


def load_tokenizer(model_dir: Path, cache_capacity: int = DEFAULT_CACHE_CAPACITY) -> Tokenizer:
    """Load ``model_dir/tokenizer.json`` with a BPE word cache of *cache_capacity* entries.

    The cache size is not serialized in ``tokenizer.json`` (the library default is
    10,000 words), so the model is rebuilt from ``vocab.json`` and ``merges.txt``.
    """

    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tokenizer.model = models.BPE.from_file(
        str(model_dir / "vocab.json"), str(model_dir / "merges.txt"), cache_capacity=cache_capacity
    )
    return tokenizer
