"""Hostile integer validation for forager."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class _HostileInt(int):
    calls = 0

    def __lt__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile lt")

    def __le__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile le")


class _HostileFloat(float):
    calls = 0

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile float eq")

    def __float__(self) -> float:
        type(self).calls += 1
        raise AssertionError("hostile float conversion")


def test_require_int_rejects_hostile_before_range() -> None:
    from alberta_framework.benchmarks.forager import _require_builtin_int

    hostile = _HostileInt(5)
    _HostileInt.calls = 0
    with pytest.raises(Exception, match="must be an integer"):
        _require_builtin_int(hostile, name="x", minimum=0, maximum=10)  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    assert _require_builtin_int(5, name="x", minimum=0, maximum=10) == 5
    with pytest.raises(Exception, match="must be an integer"):
        _require_builtin_int(True, name="x", minimum=0, maximum=10)  # type: ignore[arg-type]


def test_widths_rejects_hostile_before_lt() -> None:
    from alberta_framework.benchmarks.forager import AlbertaForagerConfig

    _HostileInt.calls = 0
    hostile = _HostileInt(1)
    with pytest.raises(ValueError, match="actor_hidden_sizes"):
        AlbertaForagerConfig(actor_hidden_sizes=(hostile,))  # type: ignore[arg-type]
    assert _HostileInt.calls == 0


def test_finite_rejects_hostile_before_isfinite() -> None:
    from alberta_framework.benchmarks.forager import ForagerFeatureConfig

    hostile = _HostileFloat(0.9)
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="reward_trace_decays"):
        ForagerFeatureConfig(reward_trace_decays=(hostile,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reward_scale"):
        ForagerFeatureConfig(reward_scale=hostile)  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0
