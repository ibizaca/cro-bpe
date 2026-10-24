"""Parquet shard writing and manifest-based skipping of completed sources."""

from pathlib import Path
from unittest import mock

import pyarrow.parquet as pq
import pytest

from data_import import data_import, manifests
from data_import.catalog import SOURCES, HfStreamSource
from data_import.manifests import SourceManifest, is_complete, save_manifest
from data_import.parquet import ParquetShardWriter
from data_import.readers import DocumentRecord
from utils.fileio import sha256_of_file
from utils.project import DataPaths

_ENTRY = next(s for s in SOURCES if s.id == "multi-eurlex-hr")


def test_writer_rotates_shards_and_records_checksums(tmp_path: Path) -> None:
    writer = ParquetShardWriter(tmp_path, max_rows=4)
    for document in _documents(10):
        writer.write(document)
    shards = writer.close()
    assert [shard.rows for shard in shards] == [4, 4, 2]
    for shard in shards:
        assert sha256_of_file(tmp_path / shard.path) == shard.sha256
    texts = [t for shard in shards for t in pq.read_table(tmp_path / shard.path).column("text").to_pylist()]
    assert texts == [document.text for document in _documents(10)]
    assert not list(tmp_path.glob("*.tmp"))


def test_skip_path_does_not_hash(tmp_path: Path) -> None:
    _completed_source(tmp_path)
    with mock.patch.object(manifests, "sha256_of_file", wraps=manifests.sha256_of_file) as hashed:
        manifest = data_import.ingest_source(_ENTRY, tmp_path)
        assert manifest.status == "complete"
        assert hashed.call_count == 0
        data_import.ingest_source(_ENTRY, tmp_path, data_import.DataImportSettings(verify_existing=True))
        assert hashed.call_count == 2


def test_size_change_makes_source_incomplete(tmp_path: Path) -> None:
    shard = _completed_source(tmp_path)
    assert is_complete(_ENTRY.id, tmp_path)
    with open(shard, "ab") as handle:
        handle.write(b"x")
    assert not is_complete(_ENTRY.id, tmp_path)


def test_same_size_corruption_needs_hash_verification(tmp_path: Path) -> None:
    shard = _completed_source(tmp_path)
    data = bytearray(shard.read_bytes())
    data[len(data) // 2] ^= 0xFF
    shard.write_bytes(bytes(data))
    assert is_complete(_ENTRY.id, tmp_path)
    assert not is_complete(_ENTRY.id, tmp_path, verify_hashes=True)


@pytest.mark.parametrize("keep", [False, True])
def test_skip_path_cleans_leftover_sources_unless_kept(tmp_path: Path, keep: bool) -> None:
    _completed_source(tmp_path)
    leftover = DataPaths(tmp_path).raw_source(_ENTRY.id)
    leftover.mkdir(parents=True)
    (leftover / "multi_eurlex.tar.gz").write_bytes(b"archive")
    data_import.ingest_source(_ENTRY, tmp_path, data_import.DataImportSettings(keep_source_files=keep))
    assert leftover.exists() == keep


def test_gated_source_without_token_is_blocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    gated = next(s for s in SOURCES if isinstance(s, HfStreamSource) and s.requires_hf_token)
    manifest = data_import.ingest_source(gated, tmp_path)
    assert (manifest.status, manifest.blocked_reason) == ("blocked", "auth_required")


def _documents(count: int) -> list[DocumentRecord]:
    return [DocumentRecord(text=f"Dokument broj {i}.", source_id=_ENTRY.id, doc_id=str(i)) for i in range(count)]


def _completed_source(data_root: Path) -> Path:
    output_dir = DataPaths(data_root).raw_parquet(_ENTRY.id)
    writer = ParquetShardWriter(output_dir, max_rows=3)
    for document in _documents(5):
        writer.write(document)
    shards = writer.close()
    save_manifest(
        SourceManifest(
            source_id=_ENTRY.id,
            status="complete",
            documents=5,
            output_files=shards,
        ),
        data_root,
    )
    return output_dir / shards[0].path
