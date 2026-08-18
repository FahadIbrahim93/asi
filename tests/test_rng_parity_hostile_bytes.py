"""Hostile bytes/str for rng parity."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class _HostileBytes(bytes):
    calls = 0

    def decode(self, *args: object, **kwargs: object) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile decode")


class _HostileStr(str):
    calls = 0
    __hash__ = str.__hash__

    def encode(self, *args: object, **kwargs: object) -> bytes:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile encode")


def test_rng_parity_rejects_hostile_bytes() -> None:
    from alberta_framework.benchmarks.forager_rng_parity import validate_parity_result

    hostile = _HostileBytes(b'{"a": 1}')
    _HostileBytes.calls = 0
    with pytest.raises(TypeError, match="mapping, bytes, or str"):
        validate_parity_result(hostile)  # type: ignore[arg-type]
    assert _HostileBytes.calls == 0


def test_rng_parity_rejects_hostile_str() -> None:
    from alberta_framework.benchmarks.forager_rng_parity import validate_collector_result

    hostile = _HostileStr('{"a": 1}')
    _HostileStr.calls = 0
    with pytest.raises(TypeError, match="mapping, bytes, or str"):
        validate_collector_result(hostile)  # type: ignore[arg-type]
    assert _HostileStr.calls == 0


def test_parity_qualification_rejects_hostile() -> None:
    from alberta_framework.benchmarks.forager_rng_parity_qualification import (
        validate_host_qualification_receipt,
    )

    hostile_b = _HostileBytes(b'{"x": 1}')
    hostile_s = _HostileStr('{"x": 1}')
    _HostileBytes.calls = 0
    _HostileStr.calls = 0
    for hostile in (hostile_b, hostile_s):
        with pytest.raises(TypeError, match="mapping, bytes, or str"):
            validate_host_qualification_receipt(  # type: ignore[arg-type]
                hostile,
                None,
                {},
                {},
                expected_executor_qualification_receipt_sha256="0" * 64,
            )
    assert _HostileBytes.calls == 0
    assert _HostileStr.calls == 0


def test_canonical_json_bytes_not_hostile() -> None:
    # Ensure canonical_json_bytes still works for builtin
    from alberta_framework.benchmarks.forager_rng_parity import canonical_json_bytes

    data = {"a": 1}
    b = canonical_json_bytes(data)
    assert isinstance(b, bytes)
