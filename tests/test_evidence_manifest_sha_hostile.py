"""Hostile string validation for evidence manifest sha helpers."""

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


def test_artifact_digest_rejects_hostile_before_len() -> None:
    from alberta_framework.evaluation.evidence_manifest import _artifact_digest

    hostile = _HostileStr("a" * 64)
    _HostileStr.calls = 0
    result = _artifact_digest({"scientific_digest": {"sha256": hostile}})  # type: ignore[dict-item]
    assert result is None
    assert _HostileStr.calls == 0
    # valid still works
    valid = "a" * 64
    assert _artifact_digest({"scientific_digest": {"sha256": valid}}) == valid
    assert _artifact_digest({"content_digest": {"sha256": valid}}) == valid
    assert _artifact_digest({"scientific_digest": {"sha256": "bad"}}) is None


def test_is_sha256_rejects_hostile_before_len() -> None:
    from alberta_framework.evaluation.evidence_manifest import _is_sha256

    hostile = _HostileStr("b" * 64)
    _HostileStr.calls = 0
    assert _is_sha256(hostile) is False
    assert _HostileStr.calls == 0
    assert _is_sha256("c" * 64) is True
    assert _is_sha256("notsha") is False
    assert _is_sha256(123) is False  # type: ignore[arg-type]
