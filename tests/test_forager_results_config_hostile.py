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


class _HostileDict(dict[str, object]):
    calls = 0

    def items(self):  # type: ignore[no-untyped-def]
        type(self).calls += 1
        raise AssertionError("hostile mapping iteration")


class _HostileList(list[object]):
    calls = 0

    def __iter__(self):  # type: ignore[no-untyped-def]
        type(self).calls += 1
        raise AssertionError("hostile list iteration")


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


@pytest.mark.parametrize("hostile", (_HostileDict({"x": 1}), _HostileList([1])))
def test_flatten_json_rejects_container_subclasses_without_hooks(hostile: object) -> None:
    from alberta_framework.benchmarks.forager_results import _flatten_json

    type(hostile).calls = 0
    with pytest.raises(ValueError, match="unsupported config hyperparameter"):
        _flatten_json(hostile, prefix="p")
    assert type(hostile).calls == 0


def test_flatten_json_error_does_not_invoke_hostile_string_hooks() -> None:
    from alberta_framework.benchmarks.forager_results import _flatten_json

    hostile = _HostileStr("bad")
    _HostileStr.calls = 0
    with pytest.raises(ValueError) as error:
        _flatten_json(hostile, prefix="p")  # type: ignore[arg-type]
    assert "!r" not in str(error.value)
    assert _HostileStr.calls == 0
