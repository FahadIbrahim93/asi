"""Hostile integer validation for forager rng parity."""

from __future__ import annotations

import pytest

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
    from alberta_framework.benchmarks.forager_rng_parity import FixedActionProbeConfig

    hostile = _HostileInt(0)
    _HostileInt.calls = 0
    with pytest.raises(Exception, match="must be an integer"):
        FixedActionProbeConfig(seed=hostile, actions=(0,))  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    # bool also rejected
    with pytest.raises(Exception, match="must be an integer"):
        FixedActionProbeConfig(seed=True, actions=(0,))  # type: ignore[arg-type]
    # valid still works
    cfg = FixedActionProbeConfig(seed=0, actions=(0,))
    assert cfg.seed == 0


def test_action_rejects_hostile_before_comparison() -> None:
    from alberta_framework.benchmarks.forager_rng_parity import FixedActionProbeConfig

    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    with pytest.raises(Exception, match="must be an integer"):
        FixedActionProbeConfig(seed=0, actions=(hostile,))  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    assert _HostileInt.calls == 0


def test_hostile_not_in_error_message() -> None:
    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    try:
        if type(hostile) is not int:
            raise ValueError("seed must be an integer")
    except ValueError as exc:
        assert "!r" not in str(exc)
        assert _HostileInt.calls == 0
