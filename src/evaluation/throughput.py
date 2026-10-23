"""Encoding and decoding speed benchmarks.

Every measurement runs in a fresh subprocess (:mod:`evaluation.throughput_worker`)
with ``RAYON_NUM_THREADS`` set, so thread counts can vary and tokenizers do not
share caches or heat. Rounds are interleaved across candidates (round-robin), so
CPU throttling affects every candidate equally. Each measurement repeats the
workload for at least ``min_seconds`` after a first run, which is reported
separately as "burst" (turbo boost makes it unrepresentative). The median, min
and max over rounds are reported.

Candidates are own models and Hub tokenizers through ``tokenizers`` (``hf``), own
tiktoken exports (``tiktoken``) and tiktoken's own encodings such as
``o200k_base`` (``tiktoken_builtin``).

Measured per candidate:

- ``encode_batch``: batched encoding of the whole holdout set, per thread count
  (``encode_batch_fast`` / ``encode_ordinary_batch``), sustained and burst;
- ``encode_sequential``: one call per document on a sample, including Python
  call overhead (at the largest thread count's run; it is single-threaded);
- ``decode_batch``: batched decoding of the encoded holdout set, at the largest
  thread count.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from evaluation.results import CpuState, EncodeBatchThroughput, Spread, ThroughputSummary
from utils.project import DataPaths

if TYPE_CHECKING:
    import tiktoken

logger = getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkCandidate:
    name: str
    kind: Literal["hf", "tiktoken", "tiktoken_builtin"]
    # tokenizer.json for "hf", the model directory for "tiktoken", the encoding name for "tiktoken_builtin"
    path: Path | str
    cache_capacity: int | None = None  # rebuild the BPE model with this word cache ("hf", own models only)


def load_builtin_encoding(name: str, data_root: Path) -> tiktoken.Encoding:
    """tiktoken's own encoding *name* (e.g. ``o200k_base``), its rank file cached under ``data/``."""

    import tiktoken

    cache_dir = DataPaths(data_root).tiktoken_cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(cache_dir)  # read by tiktoken on every download
    return tiktoken.get_encoding(name)


def cpu_state() -> CpuState:
    """Current mean CPU clock and CPU package temperature, where the OS exposes them."""

    mhz = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        clocks = [float(line.split(":")[1]) for line in cpuinfo.read_text().splitlines() if line.startswith("cpu MHz")]
        mhz = statistics.mean(clocks) if clocks else None
    temperature = None
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        try:
            if (zone / "type").read_text().strip() == "x86_pkg_temp":
                temperature = int((zone / "temp").read_text()) / 1000
                break
        except OSError:
            continue
    return CpuState(cpu_mhz=mhz, package_temperature_c=temperature)


def _run_worker(
    candidate: BenchmarkCandidate, threads: int, min_seconds: float, sequential_max_bytes: int | None, data_root: Path
) -> dict[str, float]:
    command = [sys.executable, "-m", "evaluation.throughput_worker", candidate.kind, str(candidate.path), str(threads)]
    command += ["--min-seconds", str(min_seconds), "--data-root", str(data_root)]
    if candidate.cache_capacity:
        command += ["--cache-capacity", str(candidate.cache_capacity)]
    if sequential_max_bytes is not None:
        command += ["--sequential-max-bytes", str(sequential_max_bytes)]
    environment = {**os.environ, "RAYON_NUM_THREADS": str(threads)}
    completed = subprocess.run(command, env=environment, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Throughput worker for {candidate.name} ({threads} threads) failed:\n{completed.stderr}")
    result: dict[str, float] = json.loads(completed.stdout.strip().splitlines()[-1])
    return result


def _summarize(
    runs: dict[int, list[dict[str, float]]],
    full_threads: int,
    rounds: int,
    min_seconds: float,
    cache_capacity: int | None,
) -> ThroughputSummary:
    """Median/min/max over rounds per thread count, plus the single-threaded and decode figures."""

    full_runs = runs[full_threads]
    first = full_runs[0]
    return ThroughputSummary(
        rounds=rounds,
        min_seconds=min_seconds,
        cache_capacity=cache_capacity,
        benchmark_tokens=int(first["tokens"]),
        benchmark_bytes=int(first["bytes"]),
        sequential_sample_bytes=int(first["sequential_sample_bytes"]),
        encode_batch={
            str(threads): EncodeBatchThroughput.model_validate(
                {
                    metric.removeprefix("encode_batch_"): Spread.of([run[metric] for run in thread_runs])
                    for metric in thread_runs[0]
                    if metric.startswith("encode_batch_")
                }
            )
            for threads, thread_runs in sorted(runs.items())
        },
        encode_sequential_tokens_per_second=Spread.of(
            [run["encode_sequential_tokens_per_second"] for run in full_runs]
        ),
        encode_sequential_megabytes_per_second=Spread.of(
            [run["encode_sequential_megabytes_per_second"] for run in full_runs]
        ),
        decode_batch_tokens_per_second=Spread.of([run["decode_batch_tokens_per_second"] for run in full_runs]),
    )


def throughput_benchmark(
    candidates: list[BenchmarkCandidate],
    threads: list[int],
    rounds: int,
    min_seconds: float,
    sequential_max_bytes: int,
    data_root: Path,
) -> dict[str, ThroughputSummary]:
    """Interleaved, subprocess-isolated benchmark of every candidate; see the module docstring."""

    full_threads = max(threads)
    runs: dict[str, dict[int, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    for round_number in range(1, rounds + 1):
        logger.info("Throughput round %d/%d, CPU %s", round_number, rounds, cpu_state())
        for candidate in candidates:
            for thread_count in threads:
                result = _run_worker(
                    candidate,
                    thread_count,
                    min_seconds,
                    sequential_max_bytes if thread_count == full_threads else None,
                    data_root,
                )
                runs[candidate.name][thread_count].append(result)
                logger.info(
                    "  %s, %d threads: %.2fM tokens/s (%.1f MB/s) sustained, %.1f MB/s burst",
                    candidate.name,
                    thread_count,
                    result["encode_batch_tokens_per_second"] / 1e6,
                    result["encode_batch_megabytes_per_second"],
                    result["encode_batch_burst_megabytes_per_second"],
                )
    return {
        candidate.name: _summarize(runs[candidate.name], full_threads, rounds, min_seconds, candidate.cache_capacity)
        for candidate in candidates
    }
