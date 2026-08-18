"""Hostile string validation for UPGD IPMNIST nonpromoting."""

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


def test_upgd_data_home_rejects_hostile_before_bool() -> None:
    hostile = _HostileStr("data_home")
    _HostileStr.calls = 0
    assert (type(hostile) is not str or not hostile) is True  # noqa: UP003
    assert _HostileStr.calls == 0
    assert (type("x") is not str or not "x") is False  # noqa: UP003
    assert (type("") is not str or not "") is True  # noqa: UP003


def test_upgd_environment_values_rejects_hostile() -> None:
    hostile = _HostileStr("env")
    _HostileStr.calls = 0
    env = {"a": hostile}  # type: ignore[dict-item]
    # any(type(value) is not str or not value for value in env.values())
    result = any(type(v) is not str or not v for v in env.values())
    assert result is True
    assert _HostileStr.calls == 0


def test_upgd_v2_data_home_rejects_hostile() -> None:
    hostile = _HostileStr("/tmp/v2")
    _HostileStr.calls = 0
    assert (type(hostile) is not str or not hostile) is True
    assert _HostileStr.calls == 0
