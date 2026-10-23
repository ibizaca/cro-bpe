"""HTTP downloader retries and resume, Hugging Face downloads and the pipeline's source check."""

import random
from collections.abc import Callable
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_import import data_import, readers
from data_import.catalog import SOURCES, HfDownloadSource, HfStreamSource, SourceFormat
from data_import.manifests import SourceManifest, load_manifest, save_manifest
from utils import downloaders
from utils.downloaders import download_http
from utils.errors import MissingSourcesError
from utils.fileio import sha256_of_file
from utils.project import DataPaths

_URL = "https://example.org/data/file.bin"
_PAYLOAD = b"0123456789" * 100


@pytest.fixture
def serve(monkeypatch: pytest.MonkeyPatch) -> Callable[[Callable[[httpx.Request], httpx.Response]], list]:
    """Route ``httpx.stream`` to *handler*; returns the list of received requests."""

    monkeypatch.setattr(downloaders.time, "sleep", lambda seconds: None)

    def install(handler: Callable[[httpx.Request], httpx.Response]) -> list[httpx.Request]:
        requests: list[httpx.Request] = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return handler(request)

        def stream(method: str, url: str, **kwargs):
            return httpx.Client(transport=httpx.MockTransport(respond)).stream(method, url, **kwargs)

        monkeypatch.setattr(downloaders.httpx, "stream", stream)
        return requests

    return install


def _range_server(request: httpx.Request) -> httpx.Response:
    start = int(request.headers["Range"].removeprefix("bytes=").rstrip("-")) if "Range" in request.headers else 0
    if start >= len(_PAYLOAD):
        return httpx.Response(416, headers={"Content-Range": f"bytes */{len(_PAYLOAD)}"})
    status = 206 if start else 200
    return httpx.Response(status, content=_PAYLOAD[start:], headers={"Content-Length": str(len(_PAYLOAD) - start)})


# ─────────────────────────────── download_http ───────────────────────────────


def test_download_and_resume(tmp_path: Path, serve) -> None:
    serve(_range_server)
    destination = tmp_path / "file.bin"
    destination.with_suffix(".bin.part").write_bytes(_PAYLOAD[:300])
    download_http(_URL, destination, retries=2)
    assert destination.read_bytes() == _PAYLOAD
    assert not destination.with_suffix(".bin.part").exists()


@pytest.mark.parametrize("status", [403, 404, 410])
def test_permanent_client_errors_are_not_retried(tmp_path: Path, serve, status: int) -> None:
    requests = serve(lambda request: httpx.Response(status))
    with pytest.raises(httpx.HTTPStatusError):
        download_http(_URL, tmp_path / "file.bin", retries=8)
    assert len(requests) == 1


@pytest.mark.parametrize("status", [408, 429, 500, 503])
def test_transient_errors_are_retried(tmp_path: Path, serve, status: int) -> None:
    attempts = iter([httpx.Response(status), httpx.Response(status)])
    requests = serve(lambda request: next(attempts, None) or _range_server(request))
    download_http(_URL, tmp_path / "file.bin", retries=4)
    assert len(requests) == 3
    assert (tmp_path / "file.bin").read_bytes() == _PAYLOAD


def test_416_for_a_finished_part_file_promotes_it(tmp_path: Path, serve) -> None:
    requests = serve(_range_server)
    destination = tmp_path / "file.bin"
    destination.with_suffix(".bin.part").write_bytes(_PAYLOAD)
    download_http(_URL, destination, retries=2, expected_sha256=sha256_of_file(destination.with_suffix(".bin.part")))
    assert destination.read_bytes() == _PAYLOAD
    assert len(requests) == 1


def test_416_for_an_oversized_part_file_restarts(tmp_path: Path, serve) -> None:
    requests = serve(_range_server)
    destination = tmp_path / "file.bin"
    destination.with_suffix(".bin.part").write_bytes(_PAYLOAD + b"extra")
    download_http(_URL, destination, retries=3)
    assert destination.read_bytes() == _PAYLOAD
    assert len(requests) == 2


# ─────────────────────────────── data_import ─────────────────────────────────


_WIKI = next(s for s in SOURCES if s.id == "hr-wikipedia")


def test_finished_download_without_manifest_record_is_adopted(tmp_path: Path, monkeypatch) -> None:
    """A crash between the rename and the manifest save must not discard the file."""

    s_dir = DataPaths(tmp_path).raw_source(_WIKI.id)
    s_dir.mkdir(parents=True)
    dest = s_dir / readers.source_filename(_WIKI.urls[0])
    dest.write_bytes(_PAYLOAD)
    monkeypatch.setattr(downloaders.httpx, "stream", pytest.fail)  # no network access expected
    manifest = SourceManifest(source_id=_WIKI.id)
    assert data_import._download_http_files(_WIKI, s_dir, manifest, tmp_path, retries=1)
    assert dest.read_bytes() == _PAYLOAD
    assert (manifest.files[0].status, manifest.files[0].size_bytes) == ("complete", len(_PAYLOAD))
    assert manifest.files[0].sha256 == sha256_of_file(dest)


def _write_parquet(path: Path, texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"text": texts, "id": [str(i) for i in range(len(texts))]}), path)


def test_huggingface_parquet_files_stay_under_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    entry = next(s for s in SOURCES if isinstance(s, HfDownloadSource))
    data_root = tmp_path / "data"
    calls = []

    def snapshot_download(repo_id: str, **kwargs) -> str:
        calls.append(kwargs)
        local_dir = Path(kwargs["local_dir"])
        _write_parquet(local_dir / entry.hf_data_files.replace("*", "000"), ["Prvi tekst.", "Drugi tekst."])
        return str(local_dir)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    manifest = data_import.ingest_source(entry, data_root)

    assert manifest.status == "complete" and manifest.documents == 2
    assert Path(calls[0]["local_dir"]) == DataPaths(data_root).raw_source(entry.id)
    assert [record.status for record in manifest.files] == ["deleted_after_conversion"]
    assert manifest.files[0].sha256 and manifest.files[0].size_bytes
    assert manifest.files[0].url.startswith(f"hf://datasets/{entry.hf_repo}/")
    assert not DataPaths(data_root).raw_source(entry.id).exists()
    assert {path.name for path in tmp_path.iterdir()} == {"data"}


class _FakeHubFilesystem:
    """``HfFileSystem`` stand-in serving local Parquet files and counting the bytes read."""

    def __init__(self, files: dict[str, Path], error: Exception | None = None) -> None:
        self.files, self.error, self.bytes_read = files, error, 0

    def glob(self, pattern: str) -> list[str]:
        if self.error is not None:
            raise self.error
        from fnmatch import fnmatch

        return [name for name in self.files if fnmatch(name, pattern)]

    def open(self, name: str, block_size: int):
        handle = open(self.files[name], "rb")
        original_read = handle.read

        def read(size: int = -1) -> bytes:
            data = original_read(size)
            self.bytes_read += len(data)
            return data

        handle.read = read  # type: ignore[method-assign]
        return handle


def test_streamed_source_reads_only_needed_columns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    entry = next(s for s in SOURCES if isinstance(s, HfStreamSource) and not s.requires_hf_token)
    texts = [f"Govor broj {i} u saboru." for i in range(200)]
    path = tmp_path / "train-00000.parquet"
    noise = [random.Random(i).randbytes(20_000) for i in range(len(texts))]  # incompressible, like audio
    pq.write_table(pa.table({entry.text_field: texts, "audio": noise}), path, row_group_size=50)
    remote = f"datasets/{entry.hf_repo}/{entry.hf_data_files.replace('*', '00000')}"
    filesystem = _FakeHubFilesystem({remote: path, f"datasets/{entry.hf_repo}/README.md": path})
    monkeypatch.setattr(huggingface_hub, "HfFileSystem", lambda: filesystem)

    documents = list(readers.stream_source(entry, tmp_path / "unused", readers.ReadReport()))

    assert [document.text for document in documents] == texts
    assert filesystem.bytes_read < path.stat().st_size / 10  # the audio column is never read
    assert {item.name for item in tmp_path.iterdir()} == {path.name}  # nothing written to disk


def test_gated_dataset_is_recorded_as_blocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub
    from huggingface_hub.errors import GatedRepoError

    entry = next(s for s in SOURCES if isinstance(s, HfStreamSource) and s.requires_hf_token)
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    response = httpx.Response(403, request=httpx.Request("GET", "https://huggingface.co"))
    filesystem = _FakeHubFilesystem({}, error=GatedRepoError("403 Client Error: gated", response=response))
    monkeypatch.setattr(huggingface_hub, "HfFileSystem", lambda: filesystem)
    manifest = data_import.ingest_source(entry, tmp_path)
    assert (manifest.status, manifest.blocked_reason) == ("blocked", "auth_required")


def test_every_http_format_has_a_reader() -> None:
    assert set(readers.HTTP_READERS) == set(SourceFormat)


# ─────────────────────────────── check_sources ───────────────────────────────


def _mark_all_blocked(data_root: Path) -> None:
    for entry in SOURCES:
        save_manifest(SourceManifest(source_id=entry.id, status="blocked"), data_root)


def test_check_sources_fails_on_a_failed_or_missing_source(tmp_path: Path) -> None:
    _mark_all_blocked(tmp_path)
    data_import.check_sources(data_import.DataImportSettings(data_root=tmp_path))
    save_manifest(SourceManifest(source_id=_WIKI.id, status="failed", error="boom"), tmp_path)
    with pytest.raises(MissingSourcesError, match="hr-wikipedia \\(failed: boom\\)"):
        data_import.check_sources(data_import.DataImportSettings(data_root=tmp_path))
    (tmp_path / "raw" / SOURCES[-1].id / "manifest.json").unlink()
    with pytest.raises(MissingSourcesError, match=f"{SOURCES[-1].id} \\(missing\\)"):
        data_import.check_sources(data_import.DataImportSettings(data_root=tmp_path, allow_missing=[_WIKI.id]))


def test_check_sources_allows_listed_sources(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _mark_all_blocked(tmp_path)
    save_manifest(SourceManifest(source_id=_WIKI.id, status="failed"), tmp_path)
    manifests = data_import.check_sources(data_import.DataImportSettings(data_root=tmp_path, allow_missing=[_WIKI.id]))
    assert len(manifests) == len(SOURCES)
    assert "Continuing without hr-wikipedia" in caplog.text


def test_complete_manifest_with_missing_shards_fails_the_check(tmp_path: Path) -> None:
    _mark_all_blocked(tmp_path)
    save_manifest(SourceManifest(source_id=_WIKI.id, status="complete", documents=5), tmp_path)
    assert load_manifest(_WIKI.id, tmp_path).status == "complete"
    with pytest.raises(MissingSourcesError, match="hr-wikipedia"):
        data_import.check_sources(data_import.DataImportSettings(data_root=tmp_path))
