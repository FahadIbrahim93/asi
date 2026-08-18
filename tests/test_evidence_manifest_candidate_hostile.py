"""Hostile string validation for evidence manifest candidate."""

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


def test_required_string_rejects_hostile_before_bool() -> None:
    from alberta_framework.evaluation.evidence_manifest import _required_string

    hostile = _HostileStr("candidate")
    _HostileStr.calls = 0
    with pytest.raises(RuntimeError, match="must be a non-empty string"):
        _required_string({"key": hostile}, "key", owner="test")  # type: ignore[dict-item]
    assert _HostileStr.calls == 0


def test_all_candidate_rejects_hostile_before_bool() -> None:
    hostile = _HostileStr("item")
    _HostileStr.calls = 0
    candidate = [hostile]  # type: ignore[list-item]
    result = all(type(item) is str and item for item in candidate)
    assert result is False
    assert _HostileStr.calls == 0
    # Builtin still True
    assert all(type(item) is str and item for item in ["a", "b"]) is True
