"""File helpers: atomic writes (readers never see a half-written file), checksums and the run lock."""

import fcntl
import hashlib
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from utils.errors import PipelineBusyError

_LOCK_NAME = ".cro-bpe.lock"


@contextmanager
def atomic_path(path: Path) -> Iterator[Path]:
    """Yield a temporary sibling of *path*; it replaces *path* only if the block finishes.

    On an exception the temporary file is removed and *path* is left untouched.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    with atomic_path(path) as temporary:
        temporary.write_text(text, encoding="utf-8")


def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of *path* without loading it fully into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


@contextmanager
def exclusive_run(data_root: Path) -> Iterator[None]:
    """Hold an exclusive lock on *data_root* for the duration of a pipeline run.

    Two runs over the same ``data/`` would overwrite each other's shards and
    sidecars, so a second run fails at once with :class:`PipelineBusyError`
    naming the holder's PID. The OS releases the lock when the process exits,
    so a crashed run never leaves a stale lock.
    """

    data_root.mkdir(parents=True, exist_ok=True)
    with open(data_root / _LOCK_NAME, "a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.seek(0)
            holder = handle.read().strip() or "unknown"
            raise PipelineBusyError(
                f"Another cro-bpe run (PID {holder}) is using {data_root}; wait for it to finish"
            ) from None
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
