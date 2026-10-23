"""Every stage after data import, on a tiny synthetic corpus, wired as ``main.py`` runs them.

No MLflow (``NullTracker``), no network (baselines come from a fake Hub), one
short benchmark round.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import FAKE_REVISION, fake_hub_download, synthetic_corpus
from data_import.catalog import SOURCES
from data_import.data_import import DataImportSettings, check_sources
from data_import.manifests import SourceManifest, save_manifest
from data_import.parquet import ParquetShardWriter
from data_import.readers import DocumentRecord
from evaluation import evaluation
from evaluation.evaluation import EvaluationSettings
from evaluation.results import ComparisonReport, EvaluationReport
from feature_engineering import feature_engineering
from feature_engineering.feature_engineering import FeatureEngineeringSettings
from model_training import model_training
from model_training.export import load_tiktoken
from model_training.model_training import TrainingSettings
from preprocessing import preprocessing
from preprocessing.preprocessing import PreprocessingSettings
from utils.artifacts import MixtureSummary, TrainingConfig
from utils.project import DataPaths
from utils.tracking import NullTracker

_SOURCES = {"hr-wikipedia": 300, "cc-100-hr": 900}


def _import_sources(data_root: Path) -> None:
    """Imported sources as data import leaves them; every other catalog source is blocked."""

    offset = 0
    for entry in SOURCES:
        if entry.id not in _SOURCES:
            save_manifest(SourceManifest(source_id=entry.id, status="blocked"), data_root)
            continue
        texts = synthetic_corpus(_SOURCES[entry.id], seed=offset)
        offset += 1
        writer = ParquetShardWriter(DataPaths(data_root).raw_parquet(entry.id), max_rows=250)
        for i, text in enumerate(texts):
            writer.write(DocumentRecord(text=text, source_id=entry.id, doc_id=str(i)))
        shards = writer.close()
        save_manifest(
            SourceManifest(
                source_id=entry.id,
                status="complete",
                documents=len(texts),
                output_files=shards,
            ),
            data_root,
        )


def test_pipeline_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = DataPaths(tmp_path)
    _import_sources(tmp_path)
    tracker = NullTracker()

    check_sources(DataImportSettings(data_root=tmp_path))
    stats = preprocessing.run(PreprocessingSettings(workers=1, data_root=tmp_path))
    assert {source: counts["documents_in"] for source, counts in stats.items()} == _SOURCES

    summary = feature_engineering.run(
        FeatureEngineeringSettings(data_root=tmp_path, target_train_bytes=10**9, holdout_max_fraction=0.05)
    )
    assert set(summary.sources) == set(_SOURCES)
    assert MixtureSummary.load(paths.mixture) == summary

    model_dirs = model_training.run(TrainingSettings(vocab_sizes=[1_000], data_root=tmp_path), tracker=tracker)
    assert [path.name for path in model_dirs] == ["cro-bpe-1k"]
    assert TrainingConfig.load(model_dirs[0] / "training_config.json").corpus == summary

    # Two "baselines" served by a fake Hub (the model itself); only one is benchmarked.
    tokenizer_json = (model_dirs[0] / "tokenizer.json").read_bytes()
    monkeypatch.setattr(
        evaluation,
        "hf_hub_download",
        fake_hub_download(
            {("fake/timed", "tokenizer.json"): tokenizer_json, ("fake/quality", "tokenizer.json"): tokenizer_json}
        ),
    )
    # tiktoken would download its own encodings; the model's export stands in for one.
    monkeypatch.setattr(evaluation, "load_builtin_encoding", lambda name, root: load_tiktoken(model_dirs[0]))
    rows = evaluation.run(
        EvaluationSettings(
            vocab_sizes=[1_000],
            baselines={"timed": "fake/timed", "quality-only": "fake/quality", "offline": "fake/missing"},
            tiktoken_builtin={"timed (tiktoken)": "stand-in"},  # checked against "timed", not benchmarked
            throughput_candidates=["timed", "offline"],
            benchmark_threads=[1],
            rounds=1,
            min_seconds=0.01,
            sequential_max_bytes=20_000,
            word_sample_bytes=100_000,
            data_root=tmp_path,
        ),
        tracker=tracker,
    )
    assert set(rows) == {"cro-bpe-1k", "cro-bpe-1k (tiktoken)", "timed", "quality-only"}
    row = rows["cro-bpe-1k"]
    assert row.roundtrip_failures == 0
    assert row.bytes_per_token > 1 and row.mixture_bytes_per_token is not None and row.mixture_bytes_per_token > 1
    assert row.encode_batch_tokens_per_second and set(row.encode_batch_megabytes_per_second) == {1}
    assert rows["timed"].encode_batch_tokens_per_second and rows["cro-bpe-1k (tiktoken)"].encode_batch_tokens_per_second
    assert rows["quality-only"].encode_batch_tokens_per_second is None
    assert rows["quality-only"].bytes_per_token == rows["timed"].bytes_per_token == row.bytes_per_token

    comparison = ComparisonReport.load(paths.comparison)
    assert set(comparison.holdout.mixture_weights) == set(_SOURCES)
    assert comparison.summary["cro-bpe-1k"] == row.flat()
    assert comparison.tokenizers["quality-only"].throughput is None
    assert comparison.summary["quality-only"]["encode_batch.tokens_per_second"] is None
    assert {name: source.revision for name, source in comparison.baselines.items()} == {
        "timed": FAKE_REVISION,
        "quality-only": FAKE_REVISION,
    }
    assert comparison.tiktoken_mismatches == {"timed (tiktoken)": 0}
    assert set(EvaluationReport.load(model_dirs[0] / "evaluation.json").tokenizers) == set(rows)
