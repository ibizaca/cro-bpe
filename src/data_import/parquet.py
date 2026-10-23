"""Memory-bounded Parquet output for normalized corpus documents."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from data_import.manifests import OutputFileRecord
from data_import.readers import DocumentRecord
from utils.fileio import atomic_path, sha256_of_file
from utils.parquet_io import RAW_SCHEMA, ROW_GROUP_ROWS, WRITE_OPTIONS

MAX_SHARD_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
MAX_SHARD_ROWS = 100_000


class ParquetShardWriter:
    """Write documents to atomic, bounded Parquet shards."""

    def __init__(
        self,
        output_dir: Path,
        *,
        max_uncompressed_bytes: int = MAX_SHARD_UNCOMPRESSED_BYTES,
        max_rows: int = MAX_SHARD_ROWS,
    ) -> None:
        self.output_dir = output_dir
        self.max_uncompressed_bytes = max_uncompressed_bytes
        self.max_rows = max_rows
        self.shards: list[OutputFileRecord] = []
        self._texts: list[str] = []
        self._sources: list[str] = []
        self._ids: list[str | None] = []
        self._urls: list[str | None] = []
        self._languages: list[str | None] = []
        self._uncompressed_bytes = 0

    def write(self, document: DocumentRecord) -> None:
        document_bytes = sum(
            len(value.encode("utf-8"))
            for value in (
                document.text,
                document.source_id,
                document.doc_id or "",
                document.url or "",
                document.language or "",
            )
        )
        if self._texts and (
            self._uncompressed_bytes + document_bytes > self.max_uncompressed_bytes or len(self._texts) >= self.max_rows
        ):
            self._flush()

        self._texts.append(document.text)
        self._sources.append(document.source_id)
        self._ids.append(document.doc_id)
        self._urls.append(document.url)
        self._languages.append(document.language)
        self._uncompressed_bytes += document_bytes

    def close(self) -> list[OutputFileRecord]:
        if self._texts:
            self._flush()
        return self.shards

    def _flush(self) -> None:
        shard_path = self.output_dir / f"part-{len(self.shards):05d}.parquet"
        table = pa.Table.from_arrays(
            [
                pa.array(self._texts, type=pa.large_string()),
                pa.array(self._sources, type=pa.string()),
                pa.array(self._ids, type=pa.string()),
                pa.array(self._urls, type=pa.string()),
                pa.array(self._languages, type=pa.string()),
            ],
            schema=RAW_SCHEMA,
        )
        with atomic_path(shard_path) as temporary_path:
            pq.write_table(
                table,
                temporary_path,
                **WRITE_OPTIONS,
                row_group_size=ROW_GROUP_ROWS,
                use_dictionary=["source", "language"],
                write_statistics=True,
            )
        self.shards.append(
            OutputFileRecord(
                path=shard_path.name,
                rows=table.num_rows,
                size_bytes=shard_path.stat().st_size,
                uncompressed_bytes=self._uncompressed_bytes,
                sha256=sha256_of_file(shard_path),
            )
        )
        self._texts.clear()
        self._sources.clear()
        self._ids.clear()
        self._urls.clear()
        self._languages.clear()
        self._uncompressed_bytes = 0
