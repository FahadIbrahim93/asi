"""Hostile bytes/str for evaluation campaign schedule."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class _HostileBytes(bytes):
    calls = 0

    def decode(self, *args: object, **kwargs: object) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile decode")

    def __len__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile len")


class _HostileStr(str):
    calls = 0
    __hash__ = str.__hash__

    def encode(self, *args: object, **kwargs: object) -> bytes:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile encode")


def test_decode_schedule_rejects_hostile_dispatch() -> None:
    from alberta_framework.benchmarks.forager_matched_evaluation_campaign import (
        _decode_schedule,
    )

    hostile_b = _HostileBytes(b'{"schedule": 1}')
    _HostileBytes.calls = 0
    with pytest.raises(TypeError, match="mapping, bytes, or string"):
        _decode_schedule(hostile_b)  # type: ignore[arg-type]
    assert _HostileBytes.calls == 0

    hostile_s = _HostileStr('{"schedule": 1}')
    _HostileStr.calls = 0
    with pytest.raises(TypeError, match="mapping, bytes, or string"):
        _decode_schedule(hostile_s)  # type: ignore[arg-type]
    assert _HostileStr.calls == 0
