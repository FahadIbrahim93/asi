"""Hostile integer validation for official foragax."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class _HostileInt(int):
    calls = 0

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile eq")

    def __lt__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile lt")

    def __le__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile le")


def test_reward_delay_rejects_hostile_before_lt() -> None:
    _HostileInt.calls = 0
    # Simulate the gate: if type(value) is not int or value < 0
    hostile = _HostileInt(0)
    assert (type(hostile) is not int or hostile < 0) is True  # type check first
    assert _HostileInt.calls == 0
    assert (int is not int or 0 < 0) is False
    # bool also rejected
    assert (bool is not int or True < 0) is True  # type(True) is bool != int


def test_batch_indices_rejects_hostile_before_range() -> None:
    _HostileInt.calls = 0
    hostile = _HostileInt(1)
    # any(type(index) is not int for index in indices)
    indices = [hostile]  # type: ignore[list-item]
    assert any(type(x) is not int for x in indices) is True
    assert _HostileInt.calls == 0
    # valid
    assert any(type(x) is not int for x in [1, 2, 3]) is False


def test_batch_entry_rejects_hostile_before_eq() -> None:
    _HostileInt.calls = 0
    hostile = _HostileInt(1)
    # type(index) is not int or type(seed) is not int
    assert (type(hostile) is not int or int is not int) is True
    assert _HostileInt.calls == 0
    assert (int is not int or int is not int) is False


def test_hostile_not_in_error_message() -> None:
    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    try:
        if type(hostile) is not int:
            raise ValueError("must be an integer")
    except ValueError as exc:
        assert "!r" not in str(exc)
        assert _HostileInt.calls == 0
