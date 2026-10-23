"""Model training: truncation from one run, export-check sampling, atomic model directories."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError
from tokenizers import Tokenizer

from conftest import CORPUS, synthetic_corpus
from evaluation import evaluation
from evaluation.metrics import MIXTURE, quality_metrics
from model_training import model_training
from model_training.export import export_tiktoken, load_tiktoken, verify_tiktoken
from model_training.model_training import (
    TrainingSettings,
    export_check_texts,
    truncate_tokenizer,
)
from model_training.tokenizer import END_OF_TEXT, build_tokenizer, build_trainer
from utils.artifacts import MixtureSummary, Provenance, SourceMixtureStats, TrainingConfig
from utils.errors import ExportMismatchError, MissingInputError
from utils.project import DataPaths
from utils.tracking import NullTracker


def _train(corpus: list[str], vocab_size: int) -> Tokenizer:
    tokenizer = build_tokenizer()
    tokenizer.train_from_iterator(corpus, trainer=build_trainer(vocab_size, 2, 24))
    return tokenizer


def _model_config(tokenizer: Tokenizer) -> dict:
    return json.loads(tokenizer.to_str())["model"]


def test_truncation_equals_training_the_smaller_size() -> None:
    corpus = synthetic_corpus(2_000)
    full = _train(corpus, 2_000)
    assert full.get_vocab_size() == 2_000
    for size in (500, 1_000, 1_500):
        direct = _model_config(_train(corpus, size))
        truncated = truncate_tokenizer(full, size)
        assert _model_config(truncated)["vocab"] == direct["vocab"]
        assert _model_config(truncated)["merges"] == direct["merges"]
        assert (
            truncated.encode(corpus[0]).ids
            == Tokenizer.from_str(json.dumps({**json.loads(full.to_str()), "model": direct})).encode(corpus[0]).ids
        )


def test_truncation_to_a_larger_size_keeps_the_model() -> None:
    tokenizer = _train(CORPUS, 400)
    assert _model_config(truncate_tokenizer(tokenizer, 64_000)) == _model_config(tokenizer)


@pytest.mark.parametrize("sizes", [[32_000, 32_000], [32_500], [300], []])
def test_invalid_vocab_sizes(sizes: list[int]) -> None:
    with pytest.raises(ValidationError):
        TrainingSettings(vocab_sizes=sizes)


def _write_shards(directory: Path, sources: list[str], rows_per_shard: int, row_group_size: int) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, source in enumerate(sources):
        texts = [f"{source} dokument {i}" for i in range(rows_per_shard)]
        path = directory / f"part-{index:05d}.parquet"
        pq.write_table(pa.table({"text": texts, "source": [source] * len(texts)}), path, row_group_size=row_group_size)
        paths.append(path)
    return paths


def test_export_check_covers_every_shard_and_the_holdout(tmp_path: Path) -> None:
    train = _write_shards(tmp_path / "train", ["wiki", "eurlex", "web", "web"], 1_000, 100)
    holdout = _write_shards(tmp_path / "holdout", ["wiki", "web"], 10, 100)
    texts = export_check_texts(train, holdout, 400)
    assert {text.split()[0] for text in texts} == {"wiki", "eurlex", "web"}
    assert 400 <= len(texts) - 20 <= 440
    # Rows come from row groups across the whole shard, not only its start.
    assert max(int(text.split()[-1]) for text in texts if text.startswith("eurlex")) >= 900


def test_gate_handles_text_that_spells_a_special_token(tiny_model_dir: Path, tiny_tokenizer: Tokenizer) -> None:
    texts = [f"Kraj teksta{END_OF_TEXT}Novi tekst", f"{END_OF_TEXT}", "<|reserved_7|> i <|mask|>"]
    verify_tiktoken(load_tiktoken(tiny_model_dir), tiny_tokenizer, texts)
    assert tiny_tokenizer.encode(texts[1], add_special_tokens=False).ids == [0]


def test_export_name_is_independent_of_the_directory(tmp_path: Path, tiny_model_dir: Path) -> None:
    staging = tmp_path / "cro-bpe-32k.tmp"
    staging.mkdir()
    (staging / "tokenizer.json").write_bytes((tiny_model_dir / "tokenizer.json").read_bytes())
    export_tiktoken(staging, name="cro-bpe-32k")
    assert (staging / "cro-bpe-32k.tiktoken").is_file()
    assert json.loads((staging / "tiktoken.json").read_text())["name"] == "cro-bpe-32k"


# ─────────────────────────────── full run ────────────────────────────────────


@pytest.fixture
def features_root(tmp_path: Path) -> Path:
    features = tmp_path / "feature_engineered"
    corpus = synthetic_corpus(1_500)
    for split, texts in (("train", corpus[:1_400]), ("holdout", corpus[1_400:])):
        (features / split).mkdir(parents=True)
        pq.write_table(
            pa.table({"text": texts, "source": ["web"] * len(texts)}), features / split / "part-00000.parquet"
        )
    train_bytes = sum(map(len, corpus[:1_400]))
    MixtureSummary(
        fingerprint="test",
        settings={},
        mixture_weights={"web": 1.0},
        train_bytes=train_bytes,
        sources={
            "web": SourceMixtureStats(
                documents=1_500, duplicates_removed=0, holdout_documents=100, available_bytes=train_bytes
            )
        },
        provenance=Provenance.current(),
    ).save(DataPaths(tmp_path).mixture)
    return tmp_path


class _RecordingTracker(NullTracker):
    def __init__(self) -> None:
        self.runs: list[str] = []
        self.metrics: list[dict[str, float]] = []

    def run(self, name: str, run_id: str | None = None):  # type: ignore[override]
        self.runs.append(name)
        return super().run(name, run_id or f"run-{name}")

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.metrics.append(metrics)


def test_run_trains_once_and_writes_complete_models(features_root: Path) -> None:
    settings = TrainingSettings(vocab_sizes=[1_000, 2_000], min_frequency=2, data_root=features_root)
    tracker = _RecordingTracker()
    model_dirs = model_training.run(settings, tracker=tracker)
    assert [path.name for path in model_dirs] == ["cro-bpe-2k", "cro-bpe-1k"]
    assert tracker.runs == ["cro-bpe-2k", "cro-bpe-1k"]
    assert all("training_minutes" in metrics for metrics in tracker.metrics)
    for size, model_dir in zip([2_000, 1_000], model_dirs):
        config = TrainingConfig.load(model_dir / "training_config.json")
        assert config.settings["vocab_size"] == size and config.settings["trained_vocab_size"] == 2_000
        assert config.tracking_run_id == f"run-{model_dir.name}"
        assert config.provenance.git_commit is None or len(config.provenance.git_commit) == 40
        assert (model_dir / f"{model_dir.name}.tiktoken").is_file()
        assert Tokenizer.from_file(str(model_dir / "tokenizer.json")).get_vocab_size() == size
    assert not list((features_root / "models").glob("*.tmp"))
    assert [path.name for path in evaluation.model_directories([1_000, 2_000], features_root)] == [
        "cro-bpe-1k",
        "cro-bpe-2k",
    ]


def test_failed_export_check_leaves_no_model(features_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def reject(*args) -> None:
        raise ExportMismatchError("tiktoken export differs")

    monkeypatch.setattr(model_training, "verify_tiktoken", reject)
    with pytest.raises(ExportMismatchError, match="differs"):
        model_training.run(TrainingSettings(vocab_sizes=[1_000], data_root=features_root), tracker=NullTracker())
    assert not DataPaths(features_root).model_dir(1_000).exists()


def test_missing_corpus_is_a_missing_input(tmp_path: Path) -> None:
    with pytest.raises(MissingInputError, match="run feature engineering first"):
        model_training.run(TrainingSettings(vocab_sizes=[1_000], data_root=tmp_path), tracker=NullTracker())


def test_evaluation_skips_models_without_training_config(tmp_path: Path, tiny_model_dir: Path) -> None:
    incomplete = DataPaths(tmp_path).model_dir(32_000)
    incomplete.mkdir(parents=True)
    (incomplete / "tokenizer.json").write_bytes((tiny_model_dir / "tokenizer.json").read_bytes())
    with pytest.raises(MissingInputError, match="No trained model"):
        evaluation.model_directories([32_000], tmp_path)


# ─────────────────────────────── mixture weighting ───────────────────────────


def test_mixture_aggregate_weights_sources_by_training_share(tiny_tokenizer: Tokenizer) -> None:
    texts = ["Šuma, žaba, džep, ljubav."] * 10 + ["Emoji 😀 i kineski 中文."]
    sources = ["hr"] * 10 + ["web"]
    quality = quality_metrics(tiny_tokenizer, texts, sources, weights={"hr": 0.5, "web": 0.5})
    hr, web, mixture = quality["hr"], quality["web"], quality[MIXTURE]
    # Equal byte weights: tokens per byte is the mean of the two sources'.
    expected = 1 / ((1 / hr.bytes_per_token + 1 / web.bytes_per_token) / 2)
    assert mixture.bytes_per_token == pytest.approx(expected)
    assert quality["all"].bytes_per_token != pytest.approx(expected)  # the holdout sum is dominated by "hr"
    only_hr = quality_metrics(tiny_tokenizer, texts, sources, weights={"hr": 1.0, "web": 0.0})
    assert only_hr[MIXTURE].bytes_per_token == pytest.approx(hr.bytes_per_token)
