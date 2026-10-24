"""Shared utilities: data paths, atomic writes, rolling Parquet shards and artifact contracts."""

import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from data_import.manifests import FileRecord, FileStatus, ManifestStatus, SourceManifest, load_manifest, save_manifest
from utils.artifacts import ShardSidecar
from utils.errors import PipelineBusyError
from utils.fileio import atomic_path, atomic_write_text, exclusive_run
from utils.parquet_io import CORPUS_SCHEMA, RollingParquetWriter
from utils.project import DataPaths


def test_data_paths_layout(tmp_path: Path) -> None:
    paths = DataPaths(tmp_path)
    assert paths.manifest("x") == tmp_path / "raw" / "x" / "manifest.json"
    assert paths.raw_parquet("x") == tmp_path / "raw" / "x" / "parquet"
    assert paths.source_stats("x") == tmp_path / "preprocessed" / "x" / "stats.json"
    assert paths.mixture == tmp_path / "feature_engineered" / "mixture.json"
    assert paths.model_dir(48_000) == tmp_path / "models" / "cro-bpe-48k"


def test_atomic_write_replaces_only_on_success(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "file.json"
    atomic_write_text(target, "first")
    with pytest.raises(RuntimeError), atomic_path(target) as temporary:
        temporary.write_text("half written")
        raise RuntimeError("crash")
    assert target.read_text() == "first"
    assert sorted(path.name for path in target.parent.iterdir()) == ["file.json"]


def test_rolling_writer_rotates_and_renames_shards(tmp_path: Path) -> None:
    with RollingParquetWriter(tmp_path, CORPUS_SCHEMA, max_shard_bytes=100) as writer:
        for i in range(3):
            writer.write(_table([f"dokument {i} " * 10]))
    assert writer.shard_count == 3
    assert sorted(path.name for path in tmp_path.iterdir()) == [f"part-{i:05d}.parquet" for i in range(3)]
    assert pq.read_table(tmp_path / "part-00002.parquet").column("text").to_pylist() == ["dokument 2 " * 10]


def test_rolling_writer_leaves_no_partial_shard_on_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError), RollingParquetWriter(tmp_path, CORPUS_SCHEMA, max_shard_bytes=10**9) as writer:
        writer.write(_table(["dokument"]))
        raise RuntimeError("crash")
    assert list(tmp_path.iterdir()) == []


def test_artifacts_reject_unknown_and_missing_keys(tmp_path: Path) -> None:
    path = tmp_path / "part-00000.stats.json"
    ShardSidecar(fingerprint="abc", counts={"documents_in": 3}).save(path)
    assert ShardSidecar.load(path).counts == {"documents_in": 3}
    path.write_text(json.dumps({"fingerprint": "abc", "stats": {"documents_in": 3}}))  # renamed key
    with pytest.raises(ValidationError):
        ShardSidecar.load(path)


def test_second_run_on_the_same_data_root_is_refused(tmp_path: Path) -> None:
    with exclusive_run(tmp_path):
        with pytest.raises(PipelineBusyError, match=f"PID {os.getpid()}"), exclusive_run(tmp_path):
            pass
    with exclusive_run(tmp_path):  # released after the first run ends
        pass


def test_manifest_status_is_typed(tmp_path: Path) -> None:
    save_manifest(SourceManifest(source_id="x", status="complete"), tmp_path)
    manifest = load_manifest("x", tmp_path)
    assert manifest is not None and manifest.status is ManifestStatus.COMPLETE
    manifest.files.append(FileRecord(url="u", local_path="p"))
    assert manifest.files[0].status is FileStatus.PENDING
    with pytest.raises(ValidationError):
        manifest.status = "finished"  # type: ignore[assignment]
    invalid = DataPaths(tmp_path).manifest("y")
    invalid.parent.mkdir(parents=True)
    invalid.write_text(json.dumps({"source_id": "y", "status": "done"}))
    with pytest.raises(ValidationError):
        load_manifest("y", tmp_path)


def _table(texts: list[str]) -> pa.Table:
    return pa.table({"text": texts, "source": ["s"] * len(texts)}).cast(CORPUS_SCHEMA)
