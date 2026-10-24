import re
import time
from logging import getLogger
from pathlib import Path

import httpx
from tqdm import tqdm

from utils.fileio import sha256_of_file

logger = getLogger(__name__)

_DEFAULT_RETRIES = 8
_MAX_BACKOFF_SECONDS = 60
_DEFAULT_TIMEOUT = 60.0
_DEFAULT_CHUNK = 1 << 20  # 1 MiB
_USER_AGENT = "cro-bpe/1.0 (Croatian tokenizer dataset downloader)"
# Statuses worth retrying; any other 4xx (404, 403, 410, …) will not change on retry.
_RETRYABLE_STATUS = frozenset({408, 425, 429})
_CONTENT_RANGE_TOTAL = re.compile(r"bytes \*/(\d+)")


def download_http(
    url: str,
    destination: Path,
    *,
    expected_sha256: str | None = None,
    retries: int = _DEFAULT_RETRIES,
    timeout: float = _DEFAULT_TIMEOUT,
    chunk_size: int = _DEFAULT_CHUNK,
) -> Path:
    """Download *url* to *destination* atomically with resume support and retries.

    - Parent directories are created automatically.
    - Existing zero-byte files are rejected; a non-empty file is reused only
      when its SHA-256 matches *expected_sha256* (if supplied).
    - A ``.part`` sibling file is used during transfer; on success it is
      renamed atomically so *destination* is never left in a partial state.
    - HTTP 206 Partial Content is used to resume an interrupted ``.part`` file.
      A 416 response whose ``Content-Range`` total equals the ``.part`` size means
      the transfer had already finished, so the file is promoted; any other 416
      discards the ``.part`` file and restarts.
    - Transient errors (408, 425, 429, 5xx, network and size errors) are retried
      with exponential back-off; other HTTP errors fail immediately.

    Returns the resolved path to the completed file.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        if (
            destination.is_file()
            and destination.stat().st_size > 0
            and (expected_sha256 is None or sha256_of_file(destination) == expected_sha256)
        ):
            logger.debug("Skipping %s — already present", destination.name)
            return destination
        logger.warning("Invalid existing download for %s; re-downloading", destination.name)
        destination.unlink()

    part = destination.with_suffix(destination.suffix + ".part")
    last_exc: Exception = RuntimeError("unreachable")

    for attempt in range(retries):
        try:
            resume_at = part.stat().st_size if part.exists() else 0
            headers = {
                "User-Agent": _USER_AGENT,
                "Accept-Encoding": "identity",
            }
            if resume_at > 0:
                headers["Range"] = f"bytes={resume_at}-"

            with httpx.stream("GET", url, headers=headers, timeout=timeout, follow_redirects=True) as resp:
                if resp.status_code == 416 and resume_at > 0:
                    match = _CONTENT_RANGE_TOTAL.fullmatch(resp.headers.get("Content-Range", ""))
                    if match is not None and int(match.group(1)) == resume_at:
                        # Interrupted after the last byte but before the rename.
                        return _promote(part, destination, url, expected_sha256)
                    part.unlink(missing_ok=True)
                    raise ValueError(f"Server rejected resume range for {url}; restarting transfer")

                resp.raise_for_status()

                # 200 means server ignored our Range header; restart from zero
                if resp.status_code == 200:
                    resume_at = 0

                open_mode = "ab" if (resume_at > 0 and resp.status_code == 206) else "wb"
                content_length = int(resp.headers.get("Content-Length", 0)) or None
                total = (content_length + resume_at) if (content_length and resume_at) else content_length

                with open(part, open_mode) as fh:
                    with tqdm(
                        total=total,
                        initial=resume_at,
                        unit="B",
                        unit_scale=True,
                        desc=destination.name,
                        leave=False,
                    ) as pbar:
                        for chunk in resp.iter_bytes(chunk_size=chunk_size):
                            fh.write(chunk)
                            pbar.update(len(chunk))

            downloaded_size = part.stat().st_size
            if downloaded_size == 0:
                raise ValueError(f"Download produced an empty file: {url}")
            if total is not None and downloaded_size != total:
                raise ValueError(f"Download size mismatch for {url}: expected {total} bytes, got {downloaded_size}")
            return _promote(part, destination, url, expected_sha256)

        except (httpx.HTTPStatusError, httpx.TransportError, OSError, ValueError) as exc:
            if not _is_retryable(exc):
                raise
            last_exc = exc
            if attempt < retries - 1:
                wait = min(2**attempt, _MAX_BACKOFF_SECONDS)
                logger.warning(
                    "Download attempt %d/%d failed for %s: %s — retrying in %ds", attempt + 1, retries, url, exc, wait
                )
                time.sleep(wait)

    raise last_exc


def _promote(part: Path, destination: Path, url: str, expected_sha256: str | None) -> Path:
    """Rename the finished *part* file to *destination*, checking its SHA-256 if one is expected."""

    part.rename(destination)
    if expected_sha256 is not None:
        actual = sha256_of_file(destination)
        if actual != expected_sha256:
            destination.unlink()
            raise ValueError(f"SHA-256 mismatch after download of {url}: expected {expected_sha256}, got {actual}")
    return destination


def _is_retryable(error: Exception) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        return status in _RETRYABLE_STATUS or status >= 500
    return True
