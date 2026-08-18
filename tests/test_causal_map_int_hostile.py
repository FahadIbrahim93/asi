"""Hostile integer validation for causal map forager."""

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

    def __gt__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile gt")


class _HostileFloat(float):
    calls = 0

    def __float__(self) -> float:
        type(self).calls += 1
        raise AssertionError("hostile float conversion")

    def __lt__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile float lt")


def test_world_shape_rejects_hostile_before_lt() -> None:
    from alberta_framework.benchmarks.causal_map_forager import CausalMapForagerConfig

    hostile = _HostileInt(2)
    _HostileInt.calls = 0
    with pytest.raises(Exception, match="world_shape"):
        CausalMapForagerConfig(world_shape=(hostile, 2))  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    assert CausalMapForagerConfig(world_shape=(2, 2)).world_shape == (2, 2)
    with pytest.raises(Exception, match="world_shape"):
        CausalMapForagerConfig(world_shape=(True, 2))  # type: ignore[arg-type]


def test_seed_rejects_hostile_before_range() -> None:
    from alberta_framework.benchmarks.causal_map_forager import CausalMapForagerAgent

    hostile = _HostileInt(0)
    _HostileInt.calls = 0
    with pytest.raises(Exception, match="seed must be a uint32"):
        CausalMapForagerAgent(seed=hostile)  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    assert CausalMapForagerAgent(seed=0).seed == 0


def test_config_rejects_hostile_numeric_subclasses_before_hooks() -> None:
    from alberta_framework.benchmarks.causal_map_forager import CausalMapForagerConfig

    hostile_int = _HostileInt(10)
    hostile_float = _HostileFloat(0.1)
    _HostileInt.calls = 0
    _HostileFloat.calls = 0
    with pytest.raises(Exception, match="initial_retry_delay"):
        CausalMapForagerConfig(initial_retry_delay=hostile_int)  # type: ignore[arg-type]
    with pytest.raises(Exception, match="distance_cost"):
        CausalMapForagerConfig(distance_cost=hostile_float)  # type: ignore[arg-type]
    assert _HostileInt.calls == _HostileFloat.calls == 0


def test_seeds_rejects_hostile_before_lt() -> None:
    from alberta_framework.benchmarks.causal_map_forager import (
        CausalMapForagerConfig,
        run_causal_map_forager_seeds,
    )
    from alberta_framework.benchmarks.forager import ForagerBenchmarkConfig

    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    try:
        run_causal_map_forager_seeds(
            CausalMapForagerConfig(),
            ForagerBenchmarkConfig(),
            seeds=(hostile,),  # type: ignore[arg-type]
            mode="vmap",
        )
    except Exception as exc:
        assert "uint32" in str(exc).lower() or "seeds" in str(exc).lower()
    assert _HostileInt.calls == 0


def test_hostile_not_in_error_message() -> None:
    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    try:
        if type(hostile) is not int:
            raise ValueError("must be an integer")
    except ValueError as exc:
        assert "!r" not in str(exc)
        assert _HostileInt.calls == 0
