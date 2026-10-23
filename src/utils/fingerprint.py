"""Fingerprints that decide whether a stage's existing outputs are still valid.

A stage stores the fingerprint of everything its output depends on (settings,
module ``VERSION`` constants, input files) next to the output and rebuilds when
the stored value differs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def fingerprint(*parts: Any) -> str:
    """SHA-256 of the JSON encoding of *parts* (keys sorted, so dict order does not matter)."""

    encoded = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_signatures(paths: Iterable[Path]) -> list[tuple[str, int]]:
    """Name and size of every file, a cheap stand-in for hashing large inputs."""

    return [(path.name, path.stat().st_size) for path in paths]
