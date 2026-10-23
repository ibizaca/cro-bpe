"""Schemas and writer options shared by every stage that writes Parquet."""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

ROW_GROUP_ROWS = 16_384
WRITE_OPTIONS: dict[str, Any] = {"compression": "zstd", "compression_level": 3}

# data/raw/<source>/parquet/part-*.parquet
RAW_SCHEMA = pa.schema(
    [
        pa.field("text", pa.large_string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("id", pa.string()),
        pa.field("url", pa.string()),
        pa.field("language", pa.string()),
    ]
)

# data/preprocessed/<source>/part-*.parquet; hash is xxh3-64 of the cleaned text.
PREPROCESSED_SCHEMA = pa.schema(
    [
        pa.field("text", pa.large_string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("id", pa.string()),
        pa.field("hash", pa.uint64(), nullable=False),
        pa.field("n_bytes", pa.int64(), nullable=False),
    ]
)

# data/feature_engineered/{train,holdout}/part-*.parquet
CORPUS_SCHEMA = pa.schema([pa.field("text", pa.large_string(), nullable=False), pa.field("source", pa.string())])


class RollingParquetWriter:
    """Write tables to ``part-NNNNN.parquet`` shards of about *max_shard_bytes* text each.

    Each shard is written to a ``.tmp`` file and renamed when it is closed, so a
    crash never leaves a truncated shard under its final name.
    """

    def __init__(self, output_dir: Path, schema: pa.Schema, max_shard_bytes: int) -> None:
        self.output_dir = output_dir
        self.schema = schema
        self.max_shard_bytes = max_shard_bytes
        self.shard_count = 0
        self._writer: pq.ParquetWriter | None = None
        self._paths: tuple[Path, Path] | None = None
        self._bytes = 0
        output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, table: pa.Table) -> None:
        if self._writer is None:
            final = self.output_dir / f"part-{self.shard_count:05d}.parquet"
            temporary = final.with_name(final.name + ".tmp")
            self._writer = pq.ParquetWriter(temporary, self.schema, **WRITE_OPTIONS)
            self._paths = (temporary, final)
            self.shard_count += 1
        self._writer.write_table(table, row_group_size=ROW_GROUP_ROWS)
        self._bytes += table.column("text").nbytes
        if self._bytes >= self.max_shard_bytes:
            self.close()

    def close(self) -> None:
        if self._writer is not None and self._paths is not None:
            self._writer.close()
            temporary, final = self._paths
            temporary.replace(final)
        self._writer, self._paths, self._bytes = None, None, 0

    def __enter__(self) -> RollingParquetWriter:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        if exc_type is None:
            self.close()
        elif self._writer is not None and self._paths is not None:
            self._writer.close()
            self._paths[0].unlink(missing_ok=True)
