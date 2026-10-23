"""Evaluate the trained tokenizers on the holdout set and against baselines.

Reads ``data/models/cro-bpe-<vocab>/`` for every size in
``EvaluationSettings.vocab_sizes`` and ``data/feature_engineered/holdout``, and
writes ``data/models/cro-bpe-<vocab>/evaluation.json`` (that model, its tiktoken
export and the baselines) plus ``data/models/comparison.json`` (all of them and
a summary table, the baselines' Hub revisions and the tiktoken mirror checks).
Per tokenizer:

- ``throughput`` (every model through both engines, but only the baselines in
  ``throughput_candidates``, because each one adds several minutes): sustained
  batched encoding speed per thread count, sequential encoding and batched
  decoding speed in tokens/s and MB/s, from interleaved
  subprocess runs (see :mod:`evaluation.throughput`). Measured first, before the
  CPU heats up from the quality metrics;
- ``quality``: bytes/token, characters/token, tokens/word, split-word rate and
  round-trip failures, per holdout source and overall (``all``, every holdout
  document counted equally), plus ``mixture``, the same ratios with each source
  weighted by its share of the training corpus;
- ``vocabulary``: size, share used on the holdout set, mean token length and
  Rényi efficiency;
- ``fertility``: tokens/word by the word's frequency rank in the training corpus.

Each model's tiktoken export must produce the same ids as ``tokenizer.json`` on
every holdout document (otherwise evaluation fails), so it shares the model's
quality metrics; it is listed as ``cro-bpe-<vocab> (tiktoken)`` when benchmarked.

Baselines are the tokenizers of current models, multilingual and Croatian/BCMS
ones included, plus Llama-2 and GPT-2: Hugging Face Hub repositories (ungated
mirrors, so no token is needed) with a ``tokenizer.json`` or a WordPiece
``vocab.txt``, cached in ``data/models/baselines``. A baseline that cannot be
loaded is skipped with a warning. Hub mirrors of OpenAI tokenizers can also be
benchmarked on tiktoken's own encoding (``tiktoken_builtin``); each such
encoding is checked against its mirror on the holdout set, and the number of
documents with different ids is recorded instead of failing the stage.

Metrics are added to each model's training run in the experiment tracker
(MLflow by default, see :mod:`utils.tracking`).

Only complete models (with ``training_config.json``) are evaluated.

Usage::

    uv run python -m evaluation.evaluation
"""

import json
import os
import platform
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Any

import tiktoken
import tokenizers
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from pydantic import BaseModel, Field
from tokenizers import Tokenizer
from tokenizers.implementations import BertWordPieceTokenizer

from evaluation.holdout import load_holdout, mixture_weights, training_word_ranks
from evaluation.metrics import (
    ALL_SOURCES,
    FERTILITY_BUCKETS,
    MIXTURE,
    quality_metrics,
    rare_word_fertility,
    vocabulary_metrics,
)
from evaluation.results import (
    BaselineSource,
    ComparisonReport,
    Environment,
    EvaluationReport,
    HoldoutInfo,
    SummaryRow,
    TokenizerResult,
)
from evaluation.throughput import BenchmarkCandidate, cpu_state, load_builtin_encoding, throughput_benchmark
from model_training.export import export_tiktoken, load_tiktoken, tiktoken_mismatches, verify_tiktoken
from model_training.model_training import TrainingSettings
from model_training.tokenizer import DEFAULT_CACHE_CAPACITY, load_tokenizer
from utils.artifacts import TrainingConfig
from utils.errors import MissingInputError
from utils.fileio import exclusive_run
from utils.project import DataPaths, default_data_root
from utils.tracking import MlflowTracker, Tracker

logger = getLogger(__name__)

TIKTOKEN_SUFFIX = " (tiktoken)"

_EXAMPLES = [
    "Hrvatski jezik obuhvaća govoreni i pisani hrvatski standardni jezik.",
    "Najneodgovornijima se pokazaše oni koji su se prvi javili.",
    "Đurđevac je 2024. godine imao 6.123 stanovnika, a Čakovec 27.104.",
    "Zastupnici Hrvatskoga sabora izglasali su Zakon o izmjenama i dopunama Zakona o porezu na dohodak.",
]


class EvaluationSettings(BaseModel):
    vocab_sizes: list[int] = Field(
        default=TrainingSettings().vocab_sizes, description="Selects the model directories cro-bpe-<vocab>."
    )
    baselines: dict[str, str] = Field(
        default={
            "llama-3-128k": "unsloth/Llama-3.2-1B",
            "gemma-3-262k": "unsloth/gemma-3-1b-it",
            "qwen-3-151k": "Qwen/Qwen3-0.6B",
            "gpt-4o-200k": "Xenova/gpt-4o",
            "mistral-nemo-131k": "unsloth/Mistral-Nemo-Base-2407",
            "eurollm-128k": "utter-project/EuroLLM-1.7B",
            "xlm-r-250k": "FacebookAI/xlm-roberta-base",
            "bertic-32k": "classla/bcms-bertic",
            "llama-2-32k": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "gpt-2-50k": "openai-community/gpt2",
        },
        description="Baseline name -> Hugging Face Hub repository providing tokenizer.json (or a WordPiece vocab.txt).",
    )
    tiktoken_builtin: dict[str, str] = Field(
        default={"gpt-4o-200k" + TIKTOKEN_SUFFIX: "o200k_base"},
        description="'<baseline> (tiktoken)' -> tiktoken encoding of that baseline, checked against its mirror.",
    )
    throughput_candidates: list[str] = Field(
        default=["llama-3-128k", "gpt-4o-200k", "gpt-4o-200k" + TIKTOKEN_SUFFIX],
        description="Baselines to benchmark, after every own model through both engines; the others get quality only.",
    )
    cache_capacity: int = Field(default=DEFAULT_CACHE_CAPACITY, description="BPE word cache size for own models.")
    benchmark_threads: list[int] = Field(
        default=sorted({1, 4, os.cpu_count() or 1}), description="Thread counts for batched encoding."
    )
    rounds: int = Field(default=5, description="Interleaved benchmark rounds (median, min and max are reported).")
    min_seconds: float = Field(default=10.0, description="Minimum sustained duration of each timed measurement.")
    sequential_max_bytes: int = Field(
        default=16 * 10**6, description="Holdout sample size for the single-threaded encode benchmark."
    )
    word_sample_bytes: int = Field(
        default=200 * 10**6, description="Training text sampled to rank words for rare-word fertility."
    )
    renyi_alpha: float = Field(default=2.5, description="Order of the Rényi entropy for Rényi efficiency.")
    experiment_name: str = Field(default=TrainingSettings().experiment_name, description="Tracking experiment name.")
    data_root: Path = Field(default_factory=default_data_root, description="Root of the data/ directory.")


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


@dataclass(frozen=True)
class BaselineFile:
    path: Path  # a tokenizer.json that loads
    revision: str | None  # commit of the downloaded files


def snapshot_revision(path: Path) -> str | None:
    """Commit hash of a file in the Hub cache (``.../snapshots/<sha>/<file>``)."""

    return path.parent.name if path.parent.parent.name == "snapshots" else None


def _wordpiece_tokenizer(repository: str, cache_dir: Path) -> Path:
    """``tokenizer.json`` built from a BERT-style ``vocab.txt``, saved next to the Hub cache."""

    vocab = Path(hf_hub_download(repository, "vocab.txt", cache_dir=cache_dir))
    config = json.loads(Path(hf_hub_download(repository, "tokenizer_config.json", cache_dir=cache_dir)).read_text())
    lowercase = config.get("do_lower_case", True)  # transformers' BertTokenizer default
    tokenizer = BertWordPieceTokenizer(str(vocab), lowercase=lowercase, strip_accents=config.get("strip_accents"))
    path = cache_dir / "wordpiece" / repository.replace("/", "--") / vocab.parent.name / "tokenizer.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))
    return path


def download_baselines(baselines: dict[str, str], cache_dir: Path) -> dict[str, BaselineFile]:
    """``tokenizer.json`` of each baseline, cached in *cache_dir* (under ``data/``, not ``~/.cache``).

    A repository without ``tokenizer.json`` but with a WordPiece ``vocab.txt`` (BERT-style
    encoders) gets one built from it.
    """

    files = {}
    for name, repository in baselines.items():
        try:
            try:
                path = Path(hf_hub_download(repository, "tokenizer.json", cache_dir=cache_dir))
                revision = snapshot_revision(path)
            except EntryNotFoundError:
                path = _wordpiece_tokenizer(repository, cache_dir)
                revision = path.parent.name
                logger.info("Built %s tokenizer.json from vocab.txt of %s", name, repository)
            Tokenizer.from_file(str(path))
            files[name] = BaselineFile(path, revision)
        except Exception as error:  # network, auth or missing-file errors must not fail the stage
            logger.warning("Skipping baseline %s (%s): %s", name, repository, error)
    return files


def model_directories(vocab_sizes: list[int], data_root: Path) -> list[Path]:
    model_dirs = []
    for vocab_size in vocab_sizes:
        model_dir = DataPaths(data_root).model_dir(vocab_size)
        # training_config.json is written last, after the export check, so it marks a complete model.
        if (model_dir / "tokenizer.json").is_file() and (model_dir / "training_config.json").is_file():
            model_dirs.append(model_dir)
        else:
            logger.warning("Skipping %s: no complete model found; run model training first", model_dir.name)
    if not model_dirs:
        raise MissingInputError(f"No trained model for vocabulary sizes {vocab_sizes}; run model training first")
    return model_dirs


def check_tiktoken(model_dir: Path, tokenizer: Tokenizer, texts: list[str]) -> None:
    """Export *model_dir* to tiktoken if needed, then require identical ids on *texts*."""

    if not (model_dir / "tiktoken.json").is_file():
        logger.info("Exporting %s to tiktoken", model_dir.name)
        export_tiktoken(model_dir)
    verify_tiktoken(load_tiktoken(model_dir), tokenizer, texts)


def _training_run_id(model_dir: Path, tracker: Tracker) -> str | None:
    """Tracking run that trained *model_dir*: the recorded id, else the latest run with its name."""

    return TrainingConfig.load(model_dir / "training_config.json").tracking_run_id or tracker.find_run(model_dir.name)


def _speed_summary(row: SummaryRow) -> str:
    if row.encode_batch_tokens_per_second is None:
        return "not benchmarked"
    throughput = ", ".join(
        f"{threads} threads {value:.1f} MB/s" for threads, value in row.encode_batch_megabytes_per_second.items()
    )
    return (
        f"encode {throughput} ({row.encode_batch_tokens_per_second / 1e6:.2f}M tokens/s at most threads, "
        f"spread {100 * (row.encode_batch_spread or 0):.0f}%), "
        f"sequential {row.encode_sequential_megabytes_per_second or 0:.1f} MB/s | "
        f"decode {(row.decode_batch_tokens_per_second or 0) / 1e6:.2f}M tokens/s"
    )


def log_summary(summary: dict[str, SummaryRow]) -> None:
    for name, row in summary.items():
        logger.info(
            "[%s] %.3f bytes/token (mixture %s), %.3f tokens/word (by rank %s), Rényi %.3f, %.1f%% vocab used, "
            "%d round-trip failures | %s",
            name,
            row.bytes_per_token,
            "n/a" if row.mixture_bytes_per_token is None else f"{row.mixture_bytes_per_token:.3f}",
            row.tokens_per_word,
            " / ".join(f"{value:.2f}" for value in row.tokens_per_word_by_rank.values()),
            row.renyi_efficiency,
            100 * row.used_fraction,
            row.roundtrip_failures,
            _speed_summary(row),
        )


def flatten(values: dict[str, Any], prefix: str) -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, dict):
            flat.update(flatten(value, f"{prefix}.{key}"))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[f"{prefix}.{key}"] = value
    return flat


def _builtin_encodings(
    settings: EvaluationSettings, baselines: dict[str, Tokenizer], texts: list[str]
) -> tuple[dict[str, str], dict[str, int]]:
    """tiktoken encodings whose mirror loaded, by name, and the documents on which each differs from its mirror."""

    encodings, mismatches = {}, {}
    for name, encoding_name in settings.tiktoken_builtin.items():
        mirror = name.removesuffix(TIKTOKEN_SUFFIX)
        if mirror not in baselines:
            logger.warning("Skipping %s: baseline %s is not available", name, mirror)
            continue
        encoding = load_builtin_encoding(encoding_name, settings.data_root)
        mismatches[name] = len(tiktoken_mismatches(encoding, baselines[mirror], texts))
        if mismatches[name]:
            logger.warning(
                "%s differs from its mirror %s on %d of %d holdout documents",
                encoding_name,
                settings.baselines[mirror],
                mismatches[name],
                len(texts),
            )
        else:
            logger.info("%s matches its mirror %s on all %d documents", encoding_name, mirror, len(texts))
        encodings[name] = encoding_name
    return encodings, mismatches


def _tracked_metrics(results: dict[str, TokenizerResult], model_name: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, result in results.items():
        data = result.model_dump()
        if name == model_name + TIKTOKEN_SUFFIX:
            metrics.update(flatten(data["throughput"], "eval.tiktoken"))
            continue
        # MLflow metric names cannot contain parentheses.
        prefix = "eval" if name == model_name else f"baseline.{name.replace(TIKTOKEN_SUFFIX, '.tiktoken')}"
        quality = (
            data["quality"]
            if name == model_name
            else {key: value for key, value in data["quality"].items() if key in (ALL_SOURCES, MIXTURE)}
        )
        metrics.update(flatten(quality, prefix))
        for section in ("vocabulary", "fertility", "throughput"):
            metrics.update(flatten(data[section] or {}, prefix if section == "vocabulary" else f"{prefix}.{section}"))
    return metrics


def run(settings: EvaluationSettings | None = None, tracker: Tracker | None = None) -> dict[str, SummaryRow]:
    """Evaluate every complete model; returns the summary rows by tokenizer name."""

    settings = settings or EvaluationSettings()
    logger.info("Starting evaluation")

    model_dirs = model_directories(settings.vocab_sizes, settings.data_root)
    texts, sources = load_holdout(settings.data_root)
    weights = mixture_weights(settings.data_root)
    logger.info("Evaluating on %d holdout documents from %d sources", len(texts), len(set(sources)))

    models = {model_dir.name: load_tokenizer(model_dir, settings.cache_capacity) for model_dir in model_dirs}
    for model_dir in model_dirs:
        check_tiktoken(model_dir, models[model_dir.name], texts)
    paths = DataPaths(settings.data_root)
    baseline_files = download_baselines(settings.baselines, paths.baselines)
    baselines = {name: Tokenizer.from_file(str(file.path)) for name, file in baseline_files.items()}
    builtin, mismatches = _builtin_encodings(settings, baselines, texts)

    candidates = []
    for model_dir in model_dirs:
        candidates.append(
            BenchmarkCandidate(model_dir.name, "hf", model_dir / "tokenizer.json", settings.cache_capacity)
        )
        candidates.append(BenchmarkCandidate(model_dir.name + TIKTOKEN_SUFFIX, "tiktoken", model_dir))
    available = {name: BenchmarkCandidate(name, "hf", file.path) for name, file in baseline_files.items()}
    available |= {name: BenchmarkCandidate(name, "tiktoken_builtin", encoding) for name, encoding in builtin.items()}
    for name in settings.throughput_candidates:
        if name not in available:
            logger.warning("Not benchmarking %s: no such baseline was loaded", name)
    candidates += [available[name] for name in settings.throughput_candidates if name in available]

    cpu_before = cpu_state()
    throughput = (
        throughput_benchmark(
            candidates,
            settings.benchmark_threads,
            settings.rounds,
            settings.min_seconds,
            settings.sequential_max_bytes,
            settings.data_root,
        )
        if candidates
        else {}
    )
    cpu_after = cpu_state()

    logger.info("Ranking training words for rare-word fertility")
    word_ranks = training_word_ranks(settings.data_root, settings.word_sample_bytes, max(FERTILITY_BUCKETS.values()))
    results: dict[str, TokenizerResult] = {}
    for name, tokenizer in {**models, **baselines}.items():
        logger.info("Quality metrics for %s", name)
        results[name] = TokenizerResult(
            engine="tokenizers",
            quality=quality_metrics(tokenizer, texts, sources, weights=weights),
            vocabulary=vocabulary_metrics(tokenizer, texts, settings.renyi_alpha),
            fertility=rare_word_fertility(tokenizer, texts, word_ranks),
            throughput=throughput.get(name),
        )
    # A tiktoken entry shares the quality of the tokenizer it encodes (identical ids for own models by the
    # equality gate, for mirrors up to the recorded mismatches); it is listed only when it was benchmarked.
    for name in throughput:
        tokenizer_name = name.removesuffix(TIKTOKEN_SUFFIX)
        if name != tokenizer_name:
            results[name] = results[tokenizer_name].model_copy(
                update={"engine": "tiktoken", "throughput": throughput[name]}
            )

    summary = {name: SummaryRow.of(result, ALL_SOURCES, MIXTURE) for name, result in results.items()}
    log_summary(summary)
    for name, tokenizer in models.items():
        for example in _EXAMPLES:
            logger.info("[%s] %s -> %s", name, example, tokenizer.encode(example, add_special_tokens=False).tokens)

    environment = Environment(
        cpu_count=os.cpu_count(),
        processor=cpu_model(),
        cpu_before_benchmark=cpu_before,
        cpu_after_benchmark=cpu_after,
        python=platform.python_version(),
        tokenizers=tokenizers.__version__,
        tiktoken=tiktoken.__version__,
    )
    holdout = HoldoutInfo(documents=len(texts), sources=sorted(set(sources)), mixture_weights=weights)
    settings_dump = settings.model_dump(mode="json")
    tracker = tracker or MlflowTracker(settings.experiment_name, paths)
    comparison_path = paths.comparison
    ComparisonReport(
        settings=settings_dump,
        environment=environment,
        holdout=holdout,
        summary={name: row.flat() for name, row in summary.items()},
        tokenizers=results,
        baselines={
            name: BaselineSource(repository=settings.baselines[name], revision=file.revision)
            for name, file in baseline_files.items()
        },
        tiktoken_mismatches=mismatches,
    ).save(comparison_path)

    for model_dir in model_dirs:
        name = model_dir.name
        model_results = {
            key: value
            for key, value in results.items()
            if key in (name, name + TIKTOKEN_SUFFIX) or key.removesuffix(TIKTOKEN_SUFFIX) in baseline_files
        }
        evaluation_path = model_dir / "evaluation.json"
        EvaluationReport(
            model=name, settings=settings_dump, environment=environment, holdout=holdout, tokenizers=model_results
        ).save(evaluation_path)
        with tracker.run(name, run_id=_training_run_id(model_dir, tracker)):
            tracker.log_metrics(_tracked_metrics(model_results, name))
            tracker.log_artifact(evaluation_path, artifact_path="evaluation")
            tracker.log_artifact(comparison_path, artifact_path="evaluation")
        logger.info("Evaluation written to %s", evaluation_path)
    logger.info("Comparison written to %s", comparison_path)
    return summary


if __name__ == "__main__":
    with exclusive_run(default_data_root()):
        run()
