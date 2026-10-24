"""Typed evaluation results and the reports written to ``data/models``.

The field names are the keys of ``evaluation.json`` and ``comparison.json``, so
the files keep their layout; :meth:`SummaryRow.flat` produces the summary
table's dotted keys (``encode_batch.4.megabytes_per_second``).
"""

import statistics
from typing import Any, Literal

from pydantic import Field, SerializerFunctionWrapHandler, model_serializer

from utils.artifacts import Artifact

_COUNT_FIELDS = ("documents", "tokens", "roundtrip_failures")


class QualityMetrics(Artifact):
    """Compression, fertility and round-trip fidelity of one holdout source or aggregate.

    The counts are absent (and not written) for the ``mixture`` aggregate, whose
    totals are reweighted fractions rather than document counts.
    """

    documents: int | None = None
    tokens: int | None = None
    bytes_per_token: float
    characters_per_token: float
    tokens_per_word: float
    split_word_rate: float | None = Field(description="None for tokenizers without a pre-tokenizer.")
    roundtrip_failures: int | None = None

    @model_serializer(mode="wrap")
    def _omit_absent_counts(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data: dict[str, Any] = handler(self)
        for key in _COUNT_FIELDS:
            if data.get(key) is None:
                data.pop(key, None)
        return data


class VocabularyMetrics(Artifact):
    vocab_size: int
    used_fraction: float
    mean_token_characters: float
    renyi_efficiency: float


class FertilityBucket(Artifact):
    words: int
    tokens_per_word: float


class Spread(Artifact):
    """One measurement over the benchmark rounds."""

    median: float
    min: float
    max: float
    spread: float = Field(description="(max - min) / median.")

    @classmethod
    def of(cls, values: list[float]) -> Spread:
        median = statistics.median(values)
        return cls(median=median, min=min(values), max=max(values), spread=(max(values) - min(values)) / median)


class EncodeBatchThroughput(Artifact):
    tokens_per_second: Spread
    megabytes_per_second: Spread
    burst_tokens_per_second: Spread
    burst_megabytes_per_second: Spread


class ThroughputSummary(Artifact):
    rounds: int
    min_seconds: float
    cache_capacity: int | None
    benchmark_tokens: int
    benchmark_bytes: int
    sequential_sample_bytes: int
    encode_batch: dict[str, EncodeBatchThroughput] = Field(description="Keyed by thread count.")
    encode_sequential_tokens_per_second: Spread
    encode_sequential_megabytes_per_second: Spread
    decode_batch_tokens_per_second: Spread

    def full_threads(self) -> str:
        """Key of the largest thread count, where the sequential and decode figures were measured."""
        return max(self.encode_batch, key=int)


class TokenizerResult(Artifact):
    engine: Literal["tokenizers", "tiktoken"]
    quality: dict[str, QualityMetrics] = Field(description="Per holdout source, plus 'all' and 'mixture'.")
    vocabulary: VocabularyMetrics
    fertility: dict[str, FertilityBucket] = Field(description="By the word's frequency rank in the training corpus.")
    throughput: ThroughputSummary | None = Field(default=None, description="None when not benchmarked.")


class SummaryRow(Artifact):
    """The headline figures of one tokenizer, as listed in the summary table and the log.

    The throughput figures are None (and the per-thread ones empty) for a tokenizer
    that was not benchmarked.
    """

    vocab_size: int
    bytes_per_token: float
    tokens_per_word: float
    mixture_bytes_per_token: float | None
    mixture_tokens_per_word: float | None
    renyi_efficiency: float
    used_fraction: float
    roundtrip_failures: int
    tokens_per_word_by_rank: dict[str, float]
    encode_batch_megabytes_per_second: dict[int, float] = Field(description="Median, by thread count.")
    encode_batch_tokens_per_second: float | None = Field(description="Median at the largest thread count.")
    encode_batch_spread: float | None
    encode_sequential_megabytes_per_second: float | None
    decode_batch_tokens_per_second: float | None

    @classmethod
    def of(cls, result: TokenizerResult, all_sources: str, mixture: str) -> SummaryRow:
        overall = result.quality[all_sources]
        weighted = result.quality.get(mixture)
        speed = result.throughput
        encode_batch = speed.encode_batch if speed else {}
        full = speed.encode_batch[speed.full_threads()] if speed else None
        sequential = speed.encode_sequential_megabytes_per_second if speed else None
        decode = speed.decode_batch_tokens_per_second if speed else None
        return cls(
            vocab_size=result.vocabulary.vocab_size,
            bytes_per_token=overall.bytes_per_token,
            tokens_per_word=overall.tokens_per_word,
            mixture_bytes_per_token=weighted.bytes_per_token if weighted else None,
            mixture_tokens_per_word=weighted.tokens_per_word if weighted else None,
            renyi_efficiency=result.vocabulary.renyi_efficiency,
            used_fraction=result.vocabulary.used_fraction,
            roundtrip_failures=overall.roundtrip_failures or 0,
            tokens_per_word_by_rank={bucket: values.tokens_per_word for bucket, values in result.fertility.items()},
            encode_batch_megabytes_per_second={
                int(threads): values.megabytes_per_second.median for threads, values in encode_batch.items()
            },
            encode_batch_tokens_per_second=full.tokens_per_second.median if full else None,
            encode_batch_spread=full.tokens_per_second.spread if full else None,
            encode_sequential_megabytes_per_second=sequential.median if sequential else None,
            decode_batch_tokens_per_second=decode.median if decode else None,
        )

    def flat(self) -> dict[str, int | float | None]:
        """The row with the summary table's dotted keys, as written to ``comparison.json``."""
        return {
            "vocab_size": self.vocab_size,
            "bytes_per_token": self.bytes_per_token,
            "tokens_per_word": self.tokens_per_word,
            "bytes_per_token.mixture": self.mixture_bytes_per_token,
            "tokens_per_word.mixture": self.mixture_tokens_per_word,
            "renyi_efficiency": self.renyi_efficiency,
            "used_fraction": self.used_fraction,
            "roundtrip_failures": self.roundtrip_failures,
            **{f"tokens_per_word.{bucket}": value for bucket, value in self.tokens_per_word_by_rank.items()},
            **{
                f"encode_batch.{threads}.megabytes_per_second": value
                for threads, value in self.encode_batch_megabytes_per_second.items()
            },
            "encode_batch.tokens_per_second": self.encode_batch_tokens_per_second,
            "encode_batch.spread": self.encode_batch_spread,
            "encode_sequential.megabytes_per_second": self.encode_sequential_megabytes_per_second,
            "decode_batch.tokens_per_second": self.decode_batch_tokens_per_second,
        }


class CpuState(Artifact):
    cpu_mhz: float | None
    package_temperature_c: float | None


class Environment(Artifact):
    cpu_count: int | None
    processor: str
    cpu_before_benchmark: CpuState
    cpu_after_benchmark: CpuState
    python: str
    tokenizers: str
    tiktoken: str


class HoldoutInfo(Artifact):
    documents: int
    sources: list[str]
    mixture_weights: dict[str, float]


class EvaluationReport(Artifact):
    """``data/models/cro-bpe-<vocab>/evaluation.json``: one model, its tiktoken export and the baselines."""

    model: str
    settings: dict[str, Any]
    environment: Environment
    holdout: HoldoutInfo
    tokenizers: dict[str, TokenizerResult]


class BaselineSource(Artifact):
    repository: str
    revision: str | None = Field(description="Commit of the downloaded tokenizer files in the Hub cache.")


class ComparisonReport(Artifact):
    """``data/models/comparison.json``: every evaluated tokenizer and the summary table."""

    settings: dict[str, Any]
    environment: Environment
    holdout: HoldoutInfo
    summary: dict[str, dict[str, int | float | None]] = Field(description="SummaryRow.flat() by tokenizer name.")
    tokenizers: dict[str, TokenizerResult]
    # Defaults keep reports from before these fields loadable.
    baselines: dict[str, BaselineSource] = Field(default={}, description="Baselines that loaded, by name.")
    tiktoken_mismatches: dict[str, int] = Field(
        default={},
        description="Holdout documents on which tiktoken's own encoding and its Hub mirror differ, by tiktoken name.",
    )
