"""Hostile integer validation for causal map forager."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.causal_map_forager import CausalMapForagerConfig

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


def test_world_shape_rejects_hostile_before_lt() -> None:
    hostile = _HostileInt(2)
    _HostileInt.calls = 0
    with pytest.raises(Exception, match="world_shape"):
        CausalMapForagerConfig(world_shape=(hostile, 2))  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    assert CausalMapForagerConfig(world_shape=(2, 2)).world_shape == (2, 2)
    with pytest.raises(Exception, match="world_shape"):
        CausalMapForagerConfig(world_shape=(True, 2))  # type: ignore[arg-type]


def test_world_shape_rejects_hostile_tuple_before_length_hooks() -> None:
    class HostileTuple(tuple[int, int]):
        calls = 0

        def __len__(self) -> int:
            type(self).calls += 1
            raise AssertionError("hostile len")

    hostile = HostileTuple((2, 2))
    with pytest.raises(ValueError, match="world_shape"):
        CausalMapForagerConfig(world_shape=hostile)
    assert HostileTuple.calls == 0


def test_seed_rejects_hostile_before_range() -> None:
    from alberta_framework.benchmarks.causal_map_forager import CausalMapForagerAgent

    hostile = _HostileInt(0)
    _HostileInt.calls = 0
    with pytest.raises(Exception, match="seed must be a uint32"):
        CausalMapForagerAgent(seed=hostile)  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    assert CausalMapForagerAgent(seed=0).seed == 0


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
