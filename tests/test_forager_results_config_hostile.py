"""Hostile validation for forager results config hyperparameters."""

from __future__ import annotations

import pytest

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


class _HostileFloat(float):
    calls = 0

    def __bool__(self) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile float bool")


def test_flatten_json_rejects_hostile_str_before_dispatch() -> None:
    from alberta_framework.benchmarks.forager_results import _flatten_json

    hostile = _HostileStr("evil")
    _HostileStr.calls = 0
    with pytest.raises(ValueError, match="unsupported config hyperparameter"):
        _flatten_json(hostile, prefix="p")  # type: ignore[arg-type]
    assert _HostileStr.calls == 0
    # valid still works
    assert _flatten_json("ok", prefix="p") == {"p": "ok"}
    assert _flatten_json(42, prefix="p") == {"p": 42}
    assert _flatten_json(3.14, prefix="p") == {"p": 3.14}
    assert _flatten_json(True, prefix="p") == {"p": True}
    assert _flatten_json(None, prefix="p") == {"p": None}


def test_flatten_json_rejects_hostile_float_before_isfinite() -> None:
    from alberta_framework.benchmarks.forager_results import _flatten_json

    hostile = _HostileFloat(1.0)
    _HostileFloat.calls = 0
    # type(value) is float check rejects hostile float subclass before math.isfinite
    with pytest.raises(ValueError, match="unsupported config hyperparameter"):
        _flatten_json(hostile, prefix="p")  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0
    # true float still checked for finiteness
    import math

    assert math.isfinite(1.0)
    with pytest.raises(ValueError, match="must be finite"):
        _flatten_json(float("inf"), prefix="p")


def test_hostile_not_in_error_message() -> None:
    from alberta_framework.benchmarks.forager_results import _flatten_json

    hostile = _HostileStr("bad")
    _HostileStr.calls = 0
    try:
        _flatten_json(hostile, prefix="p")  # type: ignore[arg-type]
    except ValueError as exc:
        assert "!r" not in str(exc)
        assert _HostileStr.calls == 0
