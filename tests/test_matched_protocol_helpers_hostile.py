"""Hostile protocol identities for matched protocol helpers."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.forager_matched_protocol import (
    ForagerMatchedProtocol,
)

pytestmark = pytest.mark.unit


class _HostileProtocol(ForagerMatchedProtocol):  # type: ignore[type-arg]
    calls = 0

    def to_dict(self) -> dict[str, object]:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile to_dict")


def test_parse_protocol_instance_rejects_hostile_before_to_dict() -> None:
    from alberta_framework.benchmarks.forager_matched_protocol import (
        ForagerMatchedProtocolError,
        _parse_protocol_instance,
    )

    hostile = object.__new__(_HostileProtocol)
    _HostileProtocol.calls = 0
    with pytest.raises(ForagerMatchedProtocolError):
        _parse_protocol_instance(hostile)  # type: ignore[arg-type]
    assert _HostileProtocol.calls == 0


def test_canonical_json_bytes_rejects_hostile_before_to_dict() -> None:
    from alberta_framework.benchmarks.forager_matched_protocol import (
        ForagerMatchedProtocolError,
        canonical_json_bytes,
    )

    hostile = object.__new__(_HostileProtocol)
    _HostileProtocol.calls = 0
    with pytest.raises(ForagerMatchedProtocolError):
        canonical_json_bytes(hostile)  # type: ignore[arg-type]
    assert _HostileProtocol.calls == 0


def test_hostile_is_not_exact_protocol() -> None:
    hostile = object.__new__(_HostileProtocol)
    assert type(hostile) is not ForagerMatchedProtocol
