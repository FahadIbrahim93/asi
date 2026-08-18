"""Hostile string validation for forager matrix manifest gates."""

from __future__ import annotations

from pathlib import Path

import pytest

from alberta_framework.benchmarks.forager_matrix import (
    ForagerMatrixManifestError,
    load_forager_matrix_manifest,
    run_forager_matrix,
)

pytestmark = pytest.mark.unit


class _HostileStr(str):
    calls = 0
    __hash__ = str.__hash__

    def __bool__(self) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile bool")

    def __len__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile len")

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile eq")

    def __str__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile str")


class _HostilePath(type(Path())):
    calls = 0

    def expanduser(self):  # type: ignore[no-untyped-def]
        type(self).calls += 1
        raise AssertionError("hostile path expansion")


def test_manifest_entry_points_reject_subclasses_without_hooks(tmp_path: Path) -> None:
    hostile_string = _HostileStr("manifest.json")
    hostile_path = _HostilePath("manifest.json")
    _HostileStr.calls = 0
    _HostilePath.calls = 0

    with pytest.raises(ForagerMatrixManifestError, match="exact string or Path"):
        load_forager_matrix_manifest(hostile_path)
    with pytest.raises(TypeError, match="manifest"):
        run_forager_matrix(hostile_string, tmp_path, dry_run=True)

    assert _HostileStr.calls == 0
    assert _HostilePath.calls == 0
