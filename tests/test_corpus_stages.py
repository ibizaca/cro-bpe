"""Preprocessing and feature engineering on tiny sources: outputs, fingerprints and rebuilds."""

import json
import random
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from data_import.manifests import SourceManifest, save_manifest
from data_import.parquet import ParquetShardWriter
from data_import.readers import DocumentRecord
from feature_engineering import feature_engineering, mixture
from feature_engineering.feature_engineering import FeatureEngineeringSettings
from feature_engineering.mixture import allocate_budget
from preprocessing import charclasses, cleaning, filters, preprocessing
from preprocessing.filters import QualityThresholds
from preprocessing.preprocessing import PreprocessingSettings
from utils.artifacts import MixtureSummary, ShardSidecar, SourceStats
from utils.project import DataPaths

_SHARED = "Ovaj dokument postoji u oba izvora i mora ostati samo jednom u korpusu."


_SYLLABLES = ["ka", "ri", "jel", "no", "ža", "šu", "ma", "vi", "ći", "dra", "go", "sti", "lje", "nje", "đu"]


def test_preprocessing_cleans_filters_and_counts(data_root: Path) -> None:
    stats = _preprocess(data_root)
    assert stats["hr-wikipedia"]["documents_in"] == 102
    assert stats["hr-wikipedia"]["documents_out"] == 101
    assert stats["hr-wikipedia"]["rejected_too_short"] == 1
    summary = SourceStats.load(DataPaths(data_root).source_stats("hr-wikipedia"))
    assert summary.counts["documents_out"] == 101 and summary.fingerprint
    table = pq.read_table(DataPaths(data_root).preprocessed_source("cc-100-hr") / "part-00000.parquet")
    assert table.column_names == ["text", "source", "id", "hash", "n_bytes"]


def test_preprocessing_skips_current_shards(data_root: Path) -> None:
    first = _preprocess(data_root)
    mtimes = _sidecar_mtimes(data_root)
    assert _preprocess(data_root) == first
    assert _sidecar_mtimes(data_root) == mtimes


def test_threshold_change_rebuilds_every_shard(data_root: Path) -> None:
    _preprocess(data_root)
    mtimes = _sidecar_mtimes(data_root)
    stats = _preprocess(data_root, min_words=12)
    changed = _sidecar_mtimes(data_root)
    assert all(changed[name] != mtime for name, mtime in mtimes.items())
    assert stats["cc-100-hr"]["documents_out"] < 301


@pytest.mark.parametrize("module", [preprocessing, cleaning, filters, charclasses], ids=lambda module: module.__name__)
def test_version_bump_rebuilds(data_root: Path, monkeypatch: pytest.MonkeyPatch, module) -> None:
    _preprocess(data_root)
    mtimes = _sidecar_mtimes(data_root)
    monkeypatch.setattr(module, "VERSION", module.VERSION + 1)
    _preprocess(data_root)
    assert all(_sidecar_mtimes(data_root)[name] != mtime for name, mtime in mtimes.items())


def test_changed_input_shard_is_rebuilt_and_orphans_removed(data_root: Path) -> None:
    _preprocess(data_root)
    mtimes = _sidecar_mtimes(data_root)
    # Re-import cc-100-hr with fewer documents: fewer, different shards.
    raw_dir = DataPaths(data_root).raw_parquet("cc-100-hr")
    for shard in raw_dir.glob("*.parquet"):
        shard.unlink()
    _raw_source(data_root, "cc-100-hr", _texts("cc", 50), max_rows=40)
    stats = _preprocess(data_root)
    output_dir = DataPaths(data_root).preprocessed_source("cc-100-hr")
    assert sorted(path.name for path in output_dir.glob("part-*.parquet")) == [
        "part-00000.parquet",
        "part-00001.parquet",
    ]
    assert stats["cc-100-hr"]["documents_in"] == 50
    current = _sidecar_mtimes(data_root)
    assert all(current[name] == mtime for name, mtime in mtimes.items() if name.startswith("hr-wikipedia"))


def test_sidecar_without_fingerprint_is_rebuilt(data_root: Path) -> None:
    _preprocess(data_root)
    sidecar = DataPaths(data_root).preprocessed_source("hr-wikipedia") / "part-00000.stats.json"
    sidecar.write_text(json.dumps({"documents_in": 1}))  # format written before fingerprints
    _preprocess(data_root)
    assert ShardSidecar.load(sidecar).fingerprint


def test_budget_follows_weights_when_nothing_saturates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mixture, "MIXTURE_WEIGHTS", {"a": 0.25, "b": 0.75})
    fractions = allocate_budget({"a": 1_000, "b": 1_000}, target_bytes=400)
    assert fractions == pytest.approx({"a": 0.1, "b": 0.3})


def test_saturated_source_is_taken_whole_and_rest_redistributed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mixture, "MIXTURE_WEIGHTS", {"small": 0.5, "a": 0.25, "b": 0.25})
    fractions = allocate_budget({"small": 100, "a": 1_000, "b": 1_000}, target_bytes=1_000)
    assert fractions["small"] == 1.0
    # 900 bytes left, split equally between a and b.
    assert fractions["a"] == pytest.approx(0.45) and fractions["b"] == pytest.approx(0.45)


def test_budget_larger_than_corpus_takes_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mixture, "MIXTURE_WEIGHTS", {"a": 0.5, "b": 0.5})
    assert allocate_budget({"a": 10, "b": 20}, target_bytes=10_000) == {"a": 1.0, "b": 1.0}


def test_unweighted_and_empty_sources_get_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mixture, "MIXTURE_WEIGHTS", {"a": 1.0, "empty": 1.0})
    fractions = allocate_budget({"a": 1_000, "empty": 0, "unknown": 1_000}, target_bytes=500)
    assert fractions == pytest.approx({"a": 0.5, "empty": 0.0, "unknown": 0.0})


def test_feature_engineering_dedups_by_priority_and_splits(data_root: Path) -> None:
    _preprocess(data_root)
    summary = _features(data_root, target_train_bytes=10**9, holdout_max_fraction=0.05)
    assert summary.sources["cc-100-hr"].duplicates_removed == 1
    assert summary.sources["hr-wikipedia"].duplicates_removed == 0
    assert MixtureSummary.load(DataPaths(data_root).mixture) == summary
    assert summary.provenance.packages["tokenizers"]
    train, holdout = _split_texts(data_root, "train"), _split_texts(data_root, "holdout")
    assert [source for text, source in train + holdout if text == _SHARED] == ["hr-wikipedia"]
    assert not {text for text, _ in train} & {text for text, _ in holdout}
    assert {source for _, source in holdout} == {"hr-wikipedia", "cc-100-hr"}
    assert len(train) + len(holdout) == 101 + 300


def test_feature_engineering_skips_unchanged_and_rebuilds_on_changes(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _preprocess(data_root)
    summary_path = DataPaths(data_root).mixture
    first = _features(data_root, target_train_bytes=10_000)
    mtime = summary_path.stat().st_mtime_ns
    assert _features(data_root, target_train_bytes=10_000) == first
    assert summary_path.stat().st_mtime_ns == mtime

    # Mixture weights are not settings, but the corpus depends on them.
    weights = {**mixture.MIXTURE_WEIGHTS, "cc-100-hr": 0.9}
    monkeypatch.setattr(mixture, "MIXTURE_WEIGHTS", weights)
    rebuilt = _features(data_root, target_train_bytes=10_000)
    assert rebuilt.fingerprint != first.fingerprint
    assert rebuilt.mixture_weights["cc-100-hr"] == 0.9
    assert rebuilt.sources["cc-100-hr"].sample_fraction > first.sources["cc-100-hr"].sample_fraction

    # So do the preprocessed inputs.
    _preprocess(data_root, min_words=12)
    assert _features(data_root, target_train_bytes=10_000).fingerprint != rebuilt.fingerprint


def test_near_duplicate_in_lower_priority_source_is_removed(tmp_path: Path) -> None:
    original = _near_copy_sources(tmp_path)
    _preprocess(tmp_path)
    summary = _features(tmp_path, target_train_bytes=10**9, holdout_max_fraction=0.05)
    assert summary.sources["cc-100-hr"].near_duplicates_removed == 1
    assert summary.sources["hr-wikipedia"].near_duplicates_removed == 0
    assert summary.sources["cc-100-hr"].duplicates_removed == 0
    rows = _split_texts(tmp_path, "train") + _split_texts(tmp_path, "holdout")
    assert len(rows) == 101 + 300
    assert [source for text, source in rows if "promijenjeno" in text or text == original] == ["hr-wikipedia"]

    disabled = _features(tmp_path, target_train_bytes=10**9, holdout_max_fraction=0.05, near_dedup=False)
    assert disabled.sources["cc-100-hr"].near_duplicates_removed == 0
    assert len(_split_texts(tmp_path, "train") + _split_texts(tmp_path, "holdout")) == 101 + 301


def test_band_hash_cache_is_reused_and_follows_settings(data_root: Path) -> None:
    _preprocess(data_root)
    _features(data_root, target_train_bytes=10_000)
    cache = DataPaths(data_root).features / "minhash"
    files = {path: path.stat().st_mtime_ns for path in cache.glob("*/*.npy")}
    assert len(files) == len(list(DataPaths(data_root).preprocessed.glob("*/part-*.parquet")))

    _features(data_root, target_train_bytes=20_000)  # rebuilds the corpus, not the band hashes
    assert {path: path.stat().st_mtime_ns for path in cache.glob("*/*.npy")} == files

    _features(data_root, target_train_bytes=20_000, minhash_rows=4)
    changed = set(cache.glob("*/*.npy"))
    assert len(changed) == len(files) and not changed & set(files)


def test_mixture_version_bump_rebuilds(data_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _preprocess(data_root)
    first = _features(data_root)
    monkeypatch.setattr(mixture, "VERSION", mixture.VERSION + 1)
    assert _features(data_root).fingerprint != first.fingerprint


def test_outdated_mixture_json_is_rebuilt(data_root: Path) -> None:
    _preprocess(data_root)
    DataPaths(data_root).features.mkdir(parents=True)
    DataPaths(data_root).mixture.write_text(json.dumps({"settings": {}}))  # format written before fingerprints
    assert _features(data_root).fingerprint


def _preprocess(data_root: Path, **thresholds: float) -> dict[str, dict]:
    settings = PreprocessingSettings(workers=1, data_root=data_root, thresholds=QualityThresholds(**thresholds))
    return preprocessing.run(settings)


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    _raw_source(tmp_path, "hr-wikipedia", [*_texts("wiki", 100), _SHARED, "prekratko"])
    _raw_source(tmp_path, "cc-100-hr", [*_texts("cc", 300), _SHARED])
    return tmp_path


def _raw_source(data_root: Path, source_id: str, texts: list[str], max_rows: int = 40) -> None:
    writer = ParquetShardWriter(DataPaths(data_root).raw_parquet(source_id), max_rows=max_rows)
    for i, text in enumerate(texts):
        writer.write(DocumentRecord(text=text, source_id=source_id, doc_id=str(i)))
    shards = writer.close()
    save_manifest(
        SourceManifest(
            source_id=source_id,
            status="complete",
            documents=len(texts),
            output_files=shards,
        ),
        data_root,
    )


def _texts(source_id: str, count: int) -> list[str]:
    """Distinct 11-word documents (not near-duplicates of each other), the same for the same source id."""

    rng = random.Random(source_id)
    return [" ".join(_words(rng, 11)) + "." for _ in range(count)]


def _words(rng: random.Random, count: int) -> list[str]:
    return ["".join(rng.choices(_SYLLABLES, k=rng.randint(2, 4))) for _ in range(count)]


def _sidecar_mtimes(data_root: Path) -> dict[str, int]:
    root = DataPaths(data_root).preprocessed
    return {str(path.relative_to(root)): path.stat().st_mtime_ns for path in sorted(root.glob("*/*.stats.json"))}


def _features(data_root: Path, **settings: float) -> MixtureSummary:
    return feature_engineering.run(FeatureEngineeringSettings(data_root=data_root, **settings))


def _split_texts(data_root: Path, split: str) -> list[tuple[str, str]]:
    rows = []
    for path in sorted((DataPaths(data_root).features / split).glob("part-*.parquet")):
        table = pq.read_table(path)
        rows += list(zip(table.column("text").to_pylist(), table.column("source").to_pylist()))
    return rows


def _near_copy_sources(data_root: Path) -> str:
    """A 300-word document in hr-wikipedia and a copy with one word changed in cc-100-hr; returns the original."""

    original = _words(random.Random("long"), 300)
    edited = [*original[:150], "promijenjeno", *original[151:]]
    _raw_source(data_root, "hr-wikipedia", [*_texts("wiki", 100), " ".join(original)])
    _raw_source(data_root, "cc-100-hr", [*_texts("cc", 300), " ".join(edited)])
    return " ".join(original)
