"""Hostile integer validation for upgd ipmnist and matched protocol."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class _HostileInt(int):
    calls = 0

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile eq")

    def __float__(self) -> float:
        type(self).calls += 1
        raise AssertionError("hostile float")


class _HostileFloat(float):
    calls = 0

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile float eq")


def test_finite_number_rejects_hostile_before_float() -> None:
    from alberta_framework.evaluation.upgd_ipmnist_nonpromoting import _finite_number

    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    assert _finite_number(hostile) is None  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    # bool also None via exact type check (bool not in (int,float))
    assert _finite_number(True) is None  # type: ignore[arg-type]
    assert _finite_number(1) == 1.0  # type: ignore[arg-type]
    assert _finite_number(1.0) == 1.0  # type: ignore[arg-type]
    assert _finite_number(float("inf")) is None


def test_require_probability_rejects_hostile_before_float() -> None:
    from alberta_framework.benchmarks.forager_matched_protocol import (
        _require_probability,
    )

    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="must be a finite number"):
        _require_probability(hostile, "p")  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    # valid
    assert _require_probability(0.5, "p") == 0.5
