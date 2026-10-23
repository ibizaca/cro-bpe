"""Intrinsic tokenizer quality metrics."""

import re
from collections import Counter, defaultdict

import numpy as np
from tokenizers import Tokenizer

from evaluation.holdout import word_key
from evaluation.results import FertilityBucket, QualityMetrics, VocabularyMetrics

ALL_SOURCES = "all"
# Aggregate in which each source counts with its share of the training corpus.
MIXTURE = "mixture"

# Frequency-rank buckets (from training word counts) for rare-word fertility; unranked words fall into "rest".
FERTILITY_BUCKETS = {"top-1k": 1_000, "1k-10k": 10_000, "10k-100k": 100_000}
_REST_BUCKET = "rest"
_WORD = re.compile(r"\S+")


def _ratio(numerator: float, denominator: float) -> float:
    # Mixture counts are rescaled to fractions, so no max(1, ...) guard.
    return numerator / denominator if denominator else 0.0


def _ratios(counts: Counter[str] | dict[str, float], has_pre_tokenizer: bool) -> dict[str, float | None]:
    return {
        "bytes_per_token": _ratio(counts["bytes"], counts["tokens"]),
        "characters_per_token": _ratio(counts["characters"], counts["tokens"]),
        "tokens_per_word": _ratio(counts["tokens"], counts["words"]),
        "split_word_rate": _ratio(counts["split_pretokens"], counts["pretokens"]) if has_pre_tokenizer else None,
    }


def _quality(counts: Counter[str], has_pre_tokenizer: bool) -> QualityMetrics:
    return QualityMetrics.model_validate(
        {
            "documents": counts["documents"],
            "tokens": counts["tokens"],
            **_ratios(counts, has_pre_tokenizer),
            "roundtrip_failures": counts["roundtrip_failures"],
        }
    )


def _mixture_counts(totals: dict[str, Counter[str]], weights: dict[str, float]) -> dict[str, float]:
    """Per-source counts rescaled so each source contributes its weight's share of the bytes."""

    present = {
        source: weight
        for source, weight in weights.items()
        if source in totals and totals[source]["bytes"] > 0 and weight > 0
    }
    weighted: defaultdict[str, float] = defaultdict(float)
    for source, weight in present.items():
        scale = weight / sum(present.values()) / totals[source]["bytes"]
        for key, value in totals[source].items():
            weighted[key] += value * scale
    return weighted


def quality_metrics(
    tokenizer: Tokenizer,
    texts: list[str],
    sources: list[str],
    batch_size: int = 512,
    weights: dict[str, float] | None = None,
) -> dict[str, QualityMetrics]:
    """Compression, fertility and round-trip fidelity per source (plus ``all``).

    ``all`` sums the holdout documents, so every source counts with its holdout
    size. With *weights* (source -> share of the training corpus), ``mixture``
    adds the same ratios with each source weighted by its share of training bytes,
    which reflects text like the training mix.

    - ``bytes_per_token`` / ``characters_per_token``: compression, higher is better.
    - ``tokens_per_word``: fertility over whitespace-separated words, lower is better
      and comparable across tokenizers with different pre-tokenizers.
    - ``split_word_rate``: share of pre-tokenized words encoded as more than one token;
      ``None`` for tokenizers without a pre-tokenizer (e.g. SentencePiece-style
      Llama), where every document is a single "word".
    - ``roundtrip_failures``: documents for which ``decode(encode(text)) != text``.
    """

    has_pre_tokenizer = tokenizer.pre_tokenizer is not None
    totals: dict[str, Counter[str]] = defaultdict(Counter)
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encodings = tokenizer.encode_batch(batch_texts, add_special_tokens=False)
        decoded = tokenizer.decode_batch([encoding.ids for encoding in encodings], skip_special_tokens=False)
        for text, source, encoding, roundtrip in zip(
            batch_texts, sources[start : start + batch_size], encodings, decoded
        ):
            word_ids = np.fromiter((w for w in encoding.word_ids if w is not None), dtype=np.int64)
            tokens_per_pretoken = np.bincount(word_ids) if len(word_ids) else np.zeros(0, dtype=np.int64)
            tokens_per_pretoken = tokens_per_pretoken[tokens_per_pretoken > 0]
            document = Counter(
                documents=1,
                characters=len(text),
                bytes=len(text.encode("utf-8")),
                words=len(text.split()),
                tokens=len(encoding.ids),
                pretokens=len(tokens_per_pretoken),
                split_pretokens=int(np.count_nonzero(tokens_per_pretoken > 1)),
                roundtrip_failures=int(roundtrip != text),
            )
            totals[source].update(document)
            totals[ALL_SOURCES].update(document)

    results = {source: _quality(counts, has_pre_tokenizer) for source, counts in sorted(totals.items())}
    if weights:
        results[MIXTURE] = QualityMetrics.model_validate(_ratios(_mixture_counts(totals, weights), has_pre_tokenizer))
    return results


def vocabulary_metrics(tokenizer: Tokenizer, texts: list[str], renyi_alpha: float) -> VocabularyMetrics:
    """Vocabulary size, share used on *texts*, mean token length and Rényi efficiency.

    Rényi efficiency is the Rényi entropy (order *renyi_alpha*) of the token
    unigram distribution on *texts* divided by ``log(vocab_size)``. It penalizes
    vocabularies whose extra tokens are rarely used, and correlates with
    downstream model quality better than compression alone (Zouhar et al., 2023).
    """

    vocab_size = tokenizer.get_vocab_size()
    ids = np.concatenate(
        [np.asarray(e.ids, dtype=np.int64) for e in tokenizer.encode_batch_fast(texts, add_special_tokens=False)]
    )
    counts = np.bincount(ids, minlength=vocab_size)

    lengths = [len(tokenizer.decode([token_id], skip_special_tokens=True).strip()) for token_id in range(vocab_size)]
    probabilities = counts[counts > 0] / counts.sum()
    renyi_entropy = np.log(np.sum(probabilities**renyi_alpha)) / (1 - renyi_alpha)
    return VocabularyMetrics(
        vocab_size=vocab_size,
        used_fraction=float(np.count_nonzero(counts)) / vocab_size,
        mean_token_characters=float(np.mean([length for length in lengths if length > 0])),
        renyi_efficiency=float(renyi_entropy / np.log(vocab_size)),
    )


def _bucket(rank: int | None) -> str:
    if rank is not None:
        for bucket, upper in FERTILITY_BUCKETS.items():
            if rank < upper:
                return bucket
    return _REST_BUCKET


def rare_word_fertility(
    tokenizer: Tokenizer, texts: list[str], word_ranks: dict[str, int], batch_size: int = 512
) -> dict[str, FertilityBucket]:
    """Tokens per whitespace word, by the word's frequency rank in the training corpus.

    Tokens are assigned to the word their start offset falls into, or to the next
    word if they start in whitespace (a leading-space token such as ``Ġriječ``
    belongs to ``riječ``). Words are ranked without edge punctuation; pure
    punctuation "words" are left out.
    """

    tokens: Counter[str] = Counter()
    words: Counter[str] = Counter()
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        for text, encoding in zip(batch_texts, tokenizer.encode_batch(batch_texts, add_special_tokens=False)):
            spans = [(match.end(), word_key(match.group())) for match in _WORD.finditer(text)]
            if not spans or not encoding.offsets:
                continue
            ends = np.fromiter((end for end, _ in spans), dtype=np.int64, count=len(spans))
            token_starts = np.fromiter((offset[0] for offset in encoding.offsets), dtype=np.int64)
            word_index = np.searchsorted(ends, token_starts, side="right")
            per_word = np.bincount(word_index[word_index < len(spans)], minlength=len(spans))
            for (_, key), count in zip(spans, per_word.tolist()):
                if key:
                    bucket = _bucket(word_ranks.get(key))
                    tokens[bucket] += count
                    words[bucket] += 1
    return {
        bucket: FertilityBucket(words=words[bucket], tokens_per_word=tokens[bucket] / max(1, words[bucket]))
        for bucket in [*FERTILITY_BUCKETS, _REST_BUCKET]
    }
