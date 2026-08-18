"""Hostile integer validation for dual replay."""

from __future__ import annotations

import numpy as np
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

    def __float__(self) -> float:
        type(self).calls += 1
        raise AssertionError("hostile float")

    def __hash__(self) -> int:
        return int.__hash__(self)


class _HostileFloat(float):
    calls = 0

    def __float__(self) -> float:
        type(self).calls += 1
        raise AssertionError("hostile float")


class _HostileMeta(type):
    calls = 0

    def __eq__(cls, other: object) -> bool:
        del other
        cls.calls += 1
        raise AssertionError("hostile metaclass eq")


class _MetaclassHostileInt(int, metaclass=_HostileMeta):
    pass


def test_reservoir_capacity_rejects_hostile_before_lt() -> None:
    import jax.numpy as jnp
    import jax.random as jr

    from alberta_framework.core.dual_replay import reservoir_selection

    hostile = _HostileInt(5)
    _HostileInt.calls = 0
    key = jr.PRNGKey(0)
    with pytest.raises(ValueError, match="positive integer"):
        reservoir_selection(key, jnp.asarray(1, dtype=jnp.int32), hostile)
    assert _HostileInt.calls == 0
    # bool rejected
    with pytest.raises(ValueError, match="positive integer"):
        reservoir_selection(key, jnp.asarray(1, dtype=jnp.int32), True)
    assert _HostileInt.calls == 0


def test_checkpoint_array_real_rejects_hostile_before_float() -> None:
    from alberta_framework.core.dual_replay import DualReplayMemory

    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="must contain only JSON real numbers"):
        DualReplayMemory._checkpoint_array([[hostile]], name="x", shape=(1, 1), dtype=np.float32)
    assert _HostileInt.calls == 0
    # also hostile float subclass
    hf = _HostileFloat(1.0)
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="must contain only JSON real numbers"):
        DualReplayMemory._checkpoint_array([[hf]], name="x", shape=(1, 1), dtype=np.float32)
    assert _HostileFloat.calls == 0
    # valid
    arr = DualReplayMemory._checkpoint_array([[1.0]], name="x", shape=(1, 1), dtype=np.float32)
    assert arr.shape == (1, 1)


def test_checkpoint_restore_rejects_metaclass_hostile_real_without_hooks() -> None:
    import jax.random as jr

    from alberta_framework.core import dual_replay
    from alberta_framework.core.dual_replay import DualReplayConfig, DualReplayMemory

    memory = DualReplayMemory(
        DualReplayConfig(
            total_capacity=2,
            short_term_capacity=1,
            observation_dim=2,
            action_dim=2,
            short_term_sample_size=1,
            long_term_sample_size=1,
            long_term_policy="reservoir",
        )
    )
    payload = memory.checkpoint_payload(memory.init(jr.key(3)))
    state = payload["state"]
    assert isinstance(state, dict)
    short_term = state["short_term"]
    assert isinstance(short_term, dict)
    rewards = short_term["rewards"]
    assert isinstance(rewards, list)
    rewards[0] = _MetaclassHostileInt(1)
    payload["state_digest"] = dual_replay._payload_digest(state)

    _HostileMeta.calls = 0
    with pytest.raises(ValueError, match="must contain only JSON real numbers"):
        DualReplayMemory.from_checkpoint_payload(payload)
    assert _HostileMeta.calls == 0


def test_checkpoint_array_int_rejects_hostile_before_int_check() -> None:
    from alberta_framework.core.dual_replay import DualReplayMemory

    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="must contain only JSON integers"):
        DualReplayMemory._checkpoint_array([[hostile]], name="x", shape=(1, 1), dtype=np.int32)
    assert _HostileInt.calls == 0
    with pytest.raises(ValueError, match="must contain only JSON integers"):
        DualReplayMemory._checkpoint_array([[True]], name="x", shape=(1, 1), dtype=np.int32)
    arr = DualReplayMemory._checkpoint_array([[1]], name="x", shape=(1, 1), dtype=np.int32)
    assert arr.shape == (1, 1)


def test_checkpoint_counter_rejects_hostile_before_range() -> None:
    from alberta_framework.core.dual_replay import DualReplayMemory

    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="must be a JSON integer"):
        DualReplayMemory._checkpoint_counter(hostile, name="c")
    assert _HostileInt.calls == 0
    with pytest.raises(ValueError, match="must be a JSON integer"):
        DualReplayMemory._checkpoint_counter(True, name="c")
    assert DualReplayMemory._checkpoint_counter(1, name="c").shape == ()
