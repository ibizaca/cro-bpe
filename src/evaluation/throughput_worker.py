"""One throughput measurement in a fresh process; prints a JSON result line.

Run by :func:`evaluation.throughput.throughput_benchmark`, which sets
``RAYON_NUM_THREADS`` (the Rust thread pool size is fixed once created, so a new
process is the only way to vary it) and interleaves candidates across rounds.

Usage::

    RAYON_NUM_THREADS=4 uv run python -m evaluation.throughput_worker \\
        hf data/models/cro-bpe-32k/tokenizer.json 4 --cache-capacity 1000000
    RAYON_NUM_THREADS=4 uv run python -m evaluation.throughput_worker tiktoken_builtin o200k_base 4
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

from evaluation.holdout import load_holdout
from utils.project import default_data_root


def _spread_sample(texts: list[str], max_bytes: int) -> list[str]:
    """Evenly strided subset of *texts* (so every source is represented) of at most *max_bytes*."""

    total_bytes = sum(len(text.encode("utf-8")) for text in texts)
    step = max(1, round(total_bytes / max_bytes))
    return texts[::step]


def _timed(function: Callable[[], object], min_seconds: float) -> tuple[float, float]:
    """Seconds of the first (burst) run, and mean seconds per run over repeated runs lasting *min_seconds*."""

    started = time.perf_counter()
    function()
    burst_seconds = time.perf_counter() - started
    runs = 0
    elapsed = 0.0
    while elapsed < min_seconds:
        started = time.perf_counter()
        function()
        elapsed += time.perf_counter() - started
        runs += 1
    return burst_seconds, elapsed / runs


def _functions(
    kind: str, path: str, threads: int, cache_capacity: int | None, data_root: Path
) -> tuple[Callable, Callable, Callable]:
    """Batch-encode, single-encode and batch-decode callables for one tokenizer."""

    if kind in ("tiktoken", "tiktoken_builtin"):
        from evaluation.throughput import load_builtin_encoding
        from model_training.export import load_tiktoken

        encoding = load_tiktoken(Path(path)) if kind == "tiktoken" else load_builtin_encoding(path, data_root)
        return (
            lambda texts: encoding.encode_ordinary_batch(texts, num_threads=threads),
            encoding.encode_ordinary,
            lambda ids: encoding.decode_batch(ids, num_threads=threads),
        )

    from tokenizers import Tokenizer

    from model_training.tokenizer import load_tokenizer

    tokenizer = load_tokenizer(Path(path).parent, cache_capacity) if cache_capacity else Tokenizer.from_file(path)
    return (
        lambda texts: [e.ids for e in tokenizer.encode_batch_fast(texts, add_special_tokens=False)],
        lambda text: tokenizer.encode(text, add_special_tokens=False).ids,
        lambda ids: tokenizer.decode_batch(ids, skip_special_tokens=False),
    )


def measure(
    kind: str,
    path: str,
    threads: int,
    cache_capacity: int | None,
    min_seconds: float,
    sequential_max_bytes: int | None,
    data_root: Path,
) -> dict[str, float]:
    """Batched encoding speed; with *sequential_max_bytes*, also sequential encoding and batched decoding."""

    texts, _ = load_holdout(data_root)
    encode_batch, encode, decode_batch = _functions(kind, path, threads, cache_capacity, data_root)
    total_bytes = sum(len(text.encode("utf-8")) for text in texts)
    ids = encode_batch(texts)
    total_tokens = sum(len(token_ids) for token_ids in ids)

    burst, sustained = _timed(lambda: encode_batch(texts), min_seconds)
    result = {
        "tokens": total_tokens,
        "bytes": total_bytes,
        "encode_batch_tokens_per_second": total_tokens / sustained,
        "encode_batch_megabytes_per_second": total_bytes / sustained / 1e6,
        "encode_batch_burst_tokens_per_second": total_tokens / burst,
        "encode_batch_burst_megabytes_per_second": total_bytes / burst / 1e6,
    }
    if sequential_max_bytes is None:
        return result

    sample = _spread_sample(texts, sequential_max_bytes)
    sample_bytes = sum(len(text.encode("utf-8")) for text in sample)
    sample_tokens = sum(len(encode(text)) for text in sample)
    _, sequential = _timed(lambda: [encode(text) for text in sample], min_seconds)
    _, decode = _timed(lambda: decode_batch(ids), min_seconds)
    return {
        **result,
        "sequential_sample_bytes": sample_bytes,
        "encode_sequential_tokens_per_second": sample_tokens / sequential,
        "encode_sequential_megabytes_per_second": sample_bytes / sequential / 1e6,
        "decode_batch_tokens_per_second": total_tokens / decode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kind", choices=["hf", "tiktoken", "tiktoken_builtin"])
    parser.add_argument("path", help="tokenizer.json (hf), model directory (tiktoken) or encoding name (builtin)")
    parser.add_argument("threads", type=int)
    parser.add_argument("--cache-capacity", type=int, default=None, help="BPE word cache size (hf, own models)")
    parser.add_argument("--min-seconds", type=float, default=10.0)
    parser.add_argument("--sequential-max-bytes", type=int, default=None, help="Also measure sequential/decode")
    parser.add_argument("--data-root", type=Path, default=None, help="data/ directory (default: ./data)")
    arguments = parser.parse_args()
    result = measure(
        arguments.kind,
        arguments.path,
        arguments.threads,
        arguments.cache_capacity,
        arguments.min_seconds,
        arguments.sequential_max_bytes,
        arguments.data_root or default_data_root(),
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
