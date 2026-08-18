"""Complete update working-set preflight for on-policy SARSA."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.sarsa import SARSAAgent, SARSAConfig

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)
_FIRST_PERSIST_OVERFLOW = ((2**31 - 1) // 4 - 12) // 5 + 1
_FIRST_WORKING_SET_OVERFLOW = 53_687_089


def _direct_persist_bytes(
    feature_dim: int,
    n_heads: int = 2,
    hidden_sizes: tuple[int, ...] = (),
) -> int:
    layer_sizes = (feature_dim, *hidden_sizes)
    trunk_parameters = sum(
        fan_out * (fan_in + 1)
        for fan_in, fan_out in zip(layer_sizes, layer_sizes[1:], strict=False)
    )
    final_width = hidden_sizes[-1] if hidden_sizes else feature_dim
    head_parameters = n_heads * (final_width + 1)
    return 4 * (
        2 * (trunk_parameters + head_parameters)
        + sum(hidden_sizes)
        + 3
        + feature_dim
        + 5
    )


def _working_set_bytes(
    feature_dim: int,
    n_actions: int = 2,
    n_heads: int = 2,
    hidden_sizes: tuple[int, ...] = (),
) -> int:
    return (
        2 * _direct_persist_bytes(feature_dim, n_heads, hidden_sizes)
        + 12
        + 4 * n_actions
    )


def test_int32_wrap_forges_a_different_published_byte_identity() -> None:
    published_bytes = _INT32_MAX + 1
    wrapped_bytes = int(
        jnp.asarray(_INT32_MAX, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32)
    )
    assert wrapped_bytes == _INT32_MIN
    assert wrapped_bytes != published_bytes


def test_sarsa_persist_fits_while_update_working_set_does_not() -> None:
    persist = _direct_persist_bytes(_FIRST_WORKING_SET_OVERFLOW)
    working = _working_set_bytes(_FIRST_WORKING_SET_OVERFLOW)
    assert persist <= _INT32_MAX
    assert _working_set_bytes(_FIRST_WORKING_SET_OVERFLOW - 1) <= _INT32_MAX
    assert working > _INT32_MAX
    agent = SARSAAgent(SARSAConfig(n_actions=2), hidden_sizes=())
    with pytest.raises(ValueError, match="update working set byte count"):
        agent.init(feature_dim=_FIRST_WORKING_SET_OVERFLOW, key=jr.key(0))


def test_sarsa_persistent_aggregate_bound_still_fires_first() -> None:
    assert _direct_persist_bytes(_FIRST_PERSIST_OVERFLOW) > _INT32_MAX
    agent = SARSAAgent(SARSAConfig(n_actions=2), hidden_sizes=())
    with pytest.raises(ValueError, match="aggregate_direct_state_bytes"):
        agent.init(feature_dim=_FIRST_PERSIST_OVERFLOW, key=jr.key(0))


def test_legal_sarsa_init_and_update_identity_is_unchanged() -> None:
    agent = SARSAAgent(SARSAConfig(n_actions=2), hidden_sizes=())
    state = agent.init(feature_dim=4, key=jr.key(42))
    obs = jnp.ones(4, dtype=jnp.float32)
    action, new_key = agent.select_action(state, obs)
    state = state.replace(last_action=action, last_observation=obs, rng_key=new_key)
    next_obs = jnp.ones(4, dtype=jnp.float32) * 2.0
    next_action, new_key = agent.select_action(state, next_obs)
    state = state.replace(rng_key=new_key)
    result = agent.update(
        state,
        reward=jnp.array(1.0, dtype=jnp.float32),
        observation=next_obs,
        terminated=jnp.array(0.0, dtype=jnp.float32),
        next_action=next_action,
    )
    assert result.q_values.shape == (2,)
    assert result.action.shape == ()
    assert _direct_persist_bytes(4) <= _INT32_MAX
    assert _working_set_bytes(4) <= _INT32_MAX


def test_working_set_formula_uses_all_persisted_heads_but_only_returned_actions() -> None:
    persist = _direct_persist_bytes(7, n_heads=5, hidden_sizes=(3, 2))
    working = _working_set_bytes(
        7,
        n_actions=2,
        n_heads=5,
        hidden_sizes=(3, 2),
    )
    assert working == 2 * persist + 12 + 4 * 2
    assert working != 2 * persist + 12 + 4 * 5
