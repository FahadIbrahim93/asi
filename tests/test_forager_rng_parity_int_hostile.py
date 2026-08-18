"""Hostile integer validation for forager rng parity."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.forager_rng_parity import (
    FixedActionProbeConfig,
    ForagerRngParityError,
)

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

    def __gt__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile gt")

    def __ge__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile ge")


def test_seed_rejects_hostile_before_comparison() -> None:
    hostile = _HostileInt(0)
    _HostileInt.calls = 0
    with pytest.raises(ForagerRngParityError, match="must be an integer"):
        FixedActionProbeConfig(seed=hostile, actions=(0,))  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    # bool also rejected
    with pytest.raises(ForagerRngParityError, match="must be an integer"):
        FixedActionProbeConfig(seed=True, actions=(0,))  # type: ignore[arg-type]
    # valid still works
    cfg = FixedActionProbeConfig(seed=0, actions=(0,))
    assert cfg.seed == 0


def test_action_rejects_hostile_before_comparison() -> None:
    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    with pytest.raises(ForagerRngParityError, match="must be an integer"):
        FixedActionProbeConfig(seed=0, actions=(hostile,))  # type: ignore[arg-type]
    assert _HostileInt.calls == 0


def test_actions_rejects_hostile_tuple_before_container_hooks() -> None:
    class HostileTuple(tuple[int, ...]):
        calls = 0

        def __len__(self) -> int:
            type(self).calls += 1
            raise AssertionError("hostile len")

    hostile = HostileTuple((0,))
    with pytest.raises(ForagerRngParityError, match="immutable tuple"):
        FixedActionProbeConfig(seed=0, actions=hostile)
    assert HostileTuple.calls == 0
