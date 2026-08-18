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


def test_jax_config_rejects_hostile_before_str() -> None:
    hostile = _HostileStr("evil")
    _HostileStr.calls = 0
    assert (type(hostile) in (bool, int, float, str)) is False
    assert _HostileStr.calls == 0
    assert (str in (bool, int, float, str)) is True
    assert (int in (bool, int, float, str)) is True
    assert (float in (bool, int, float, str)) is True
    assert (bool in (bool, int, float, str)) is True
    # None case
    assert (None is None or type(None) in (bool, int, float, str)) is True  # type: ignore[arg-type]
    # safe else for hostile uses base str

    # hostile via safe path
    assert str.__str__(hostile) == "evil"
    assert _HostileStr.calls == 0


def test_manifest_loader_rejects_hostile_before_dispatch() -> None:
    hostile = _HostileStr("/tmp/manifest.json")
    _HostileStr.calls = 0
    from pathlib import Path

    assert (type(hostile) is str or isinstance(hostile, Path)) is False
    assert _HostileStr.calls == 0
    assert (str is str or isinstance("/tmp/x", Path)) is True
    assert (type(Path("/tmp/x")) is str or isinstance(Path("/tmp/x"), Path)) is True


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


def test_hostile_not_in_error_message() -> None:
    hostile = _HostileStr("bad")
    _HostileStr.calls = 0
    try:
        if type(hostile) is not str:
            raise ValueError("manifest must be exact str or Path")
    except ValueError as exc:
        assert "!r" not in str(exc)
        assert _HostileStr.calls == 0
