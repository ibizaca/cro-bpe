"""Catch a forgotten ``VERSION`` bump in the modules that stage fingerprints depend on.

Preprocessing and feature engineering rebuild their outputs when a module's
``VERSION`` changes, not when its source text changes, so a comment edit or a
reformat does not force a rebuild of the whole corpus. This test pins every
such module's source hash next to its version. When the code changes it fails:

- if the change alters what the stage writes, bump the module's ``VERSION``;
- either way, update ``tests/versions.json`` with the values the failure prints.
"""

import hashlib
import json
from pathlib import Path
from types import ModuleType

import pytest

from feature_engineering import feature_engineering, mixture, near_dedup
from preprocessing import charclasses, cleaning, filters, preprocessing

_SNAPSHOT = Path(__file__).with_name("versions.json")
_MODULES = [preprocessing, cleaning, filters, charclasses, feature_engineering, mixture, near_dedup]


def _current(module: ModuleType) -> dict[str, object]:
    source = Path(module.__file__ or "").read_bytes()
    return {"version": module.VERSION, "sha256": hashlib.sha256(source).hexdigest()}


@pytest.mark.parametrize("module", _MODULES, ids=lambda module: module.__name__)
def test_version_matches_snapshot(module: ModuleType) -> None:
    snapshot = json.loads(_SNAPSHOT.read_text(encoding="utf-8"))
    current = _current(module)
    expected = snapshot.get(module.__name__)
    if expected == current:
        return
    hint = (
        "the code changed but VERSION did not: bump VERSION if the output changes, then"
        if expected is not None and expected["version"] == current["version"]
        else "then"
    )
    pytest.fail(f"{module.__name__}: {hint} set its entry in {_SNAPSHOT.name} to {json.dumps(current)}", pytrace=False)
