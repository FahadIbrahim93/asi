"""End-to-end contracts for bounded Dreamer-family sequence control."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.dreamer_sequence_control import (
    DreamerSequenceConfig,
    DreamerSequenceControl,
    DreamerTransition,
    lambda_returns,
)


def _config(*, imagination_enabled: bool = True) -> DreamerSequenceConfig:
    return DreamerSequenceConfig(
        observation_dim=2,
        n_actions=2,
        latent_dim=4,
        replay_capacity=8,
        sequence_length=2,
        imagination_horizon=3,
        model_learning_rate=0.01,
        actor_learning_rate=0.01,
        value_learning_rate=0.01,
        discount=0.9,
        lambda_=0.8,
        entropy_scale=0.001,
        imagination_enabled=imagination_enabled,
    )


def _transition(index: int, *, terminal: bool = False) -> DreamerTransition:
    observation = jnp.asarray([float(index), float(index % 2)], dtype=jnp.float32)
    next_observation = jnp.asarray([float(index + 1), float((index + 1) % 2)], dtype=jnp.float32)
    return DreamerTransition(
        observation=observation,
        action=jnp.asarray(index % 2, dtype=jnp.int32),
        reward=jnp.asarray(1.0 if index % 2 == 0 else -0.25, dtype=jnp.float32),
        discount=jnp.asarray(0.0 if terminal else 0.9, dtype=jnp.float32),
        next_observation=next_observation,
        terminated=jnp.asarray(terminal, dtype=jnp.bool_),
    )


def test_lambda_returns_have_exact_bootstrap_and_terminal_semantics() -> None:
    result = lambda_returns(
        rewards=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        discounts=jnp.asarray([0.5, 0.0], dtype=jnp.float32),
        next_values=jnp.asarray([10.0, 20.0], dtype=jnp.float32),
        bootstrap=jnp.asarray(30.0, dtype=jnp.float32),
        lambda_=0.25,
    )
    # G1 = 2; G0 = 1 + .5 * (.75 * 10 + .25 * 2) = 5.
    np.testing.assert_array_equal(result, np.asarray([5.0, 2.0], dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        lambda_returns(
            jnp.asarray([jnp.nan]),
            jnp.asarray([0.9]),
            jnp.asarray([0.0]),
            jnp.asarray(0.0),
            0.9,
        )


def test_real_sequence_replay_is_contiguous_and_never_crosses_a_boundary() -> None:
    agent = DreamerSequenceControl(_config())
    state = agent.init(jr.key(7, impl="threefry2x32"))
    for index in range(5):
        transition = _transition(index, terminal=index == 2)
        decision = agent.decide(state, transition.observation, action=transition.action)
        result = agent.learn(state, decision, transition)
        assert bool(result.applied)
        state = result.state

    for seed in range(12):
        sample = agent.sample_sequence(state.replay, jr.key(seed, impl="threefry2x32"))
        if bool(sample.valid):
            ids = np.asarray(sample.insertion_ids)
            np.testing.assert_array_equal(ids[1:], ids[:-1] + 1)
            assert np.all(np.asarray(sample.discounts[:-1]) > 0.0)


def test_sequence_training_and_imagined_actor_value_updates_are_real_and_accounted() -> None:
    agent = DreamerSequenceControl(_config())
    state = agent.init(jr.key(11, impl="threefry2x32"))
    initial_model = state.model_parameters
    initial_actor = state.actor_parameters
    initial_value = state.value_parameters
    final = None
    for index in range(6):
        transition = _transition(index)
        decision = agent.decide(state, transition.observation, action=transition.action)
        final = agent.learn(state, decision, transition)
        assert bool(final.applied)
        state = final.state

    assert final is not None
    assert int(state.real_transition_count) == 6
    assert int(state.replay_insert_count) == 6
    assert int(state.sequence_sample_count) == 5
    assert int(state.world_model_update_count) == 5
    assert int(state.imagination_rollout_count) == 5
    assert int(state.imagination_query_count) == 15
    assert int(state.actor_update_count) == int(state.value_update_count) == 5
    assert not jax.tree_util.tree_all(
        jax.tree_util.tree_map(jnp.array_equal, initial_model, state.model_parameters)
    )
    assert not jax.tree_util.tree_all(
        jax.tree_util.tree_map(jnp.array_equal, initial_actor, state.actor_parameters)
    )
    assert not jax.tree_util.tree_all(
        jax.tree_util.tree_map(jnp.array_equal, initial_value, state.value_parameters)
    )
    budget = agent.resource_budget(state)
    assert budget.persistent_bytes == sum(
        int(leaf.size * leaf.dtype.itemsize)
        for raw_leaf in jax.tree_util.tree_leaves(state)
        if (leaf := jnp.asarray(raw_leaf)) is not None
    )
    assert budget.replay_bytes > 0
    assert budget.update_working_set_bytes >= budget.persistent_bytes


def test_mechanism_off_preserves_model_and_replay_path_exactly() -> None:
    enabled = DreamerSequenceControl(_config(imagination_enabled=True))
    disabled = DreamerSequenceControl(_config(imagination_enabled=False))
    enabled_state = enabled.init(jr.key(17, impl="threefry2x32"))
    disabled_state = disabled.init(jr.key(17, impl="threefry2x32"))
    for index in range(6):
        transition = _transition(index)
        enabled_result = enabled.learn(
            enabled_state,
            enabled.decide(enabled_state, transition.observation, action=transition.action),
            transition,
        )
        disabled_result = disabled.learn(
            disabled_state,
            disabled.decide(disabled_state, transition.observation, action=transition.action),
            transition,
        )
        enabled_state = enabled_result.state
        disabled_state = disabled_result.state

    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            jnp.array_equal,
            enabled_state.model_parameters,
            disabled_state.model_parameters,
        )
    )
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(jnp.array_equal, enabled_state.replay, disabled_state.replay)
    )
    assert int(disabled_state.imagination_query_count) == 0
    assert int(disabled_state.actor_update_count) == 0
    assert int(disabled_state.value_update_count) == 0
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            jnp.array_equal,
            disabled_state.actor_parameters,
            disabled.init(jr.key(17, impl="threefry2x32")).actor_parameters,
        )
    )


def test_stale_cache_and_invalid_transition_fail_without_consuming_rng() -> None:
    agent = DreamerSequenceControl(_config())
    state = agent.init(jr.key(23, impl="threefry2x32"))
    transition = _transition(0)
    decision = agent.decide(state, transition.observation, action=transition.action)
    accepted = agent.learn(state, decision, transition)
    rejected = agent.learn(accepted.state, decision, _transition(1))
    assert not bool(rejected.applied)
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(jnp.array_equal, accepted.state, rejected.state)
    )

    bad = dataclasses.replace(transition, reward=jnp.asarray(jnp.nan, dtype=jnp.float32))
    rejected = agent.learn(state, decision, bad)
    assert not bool(rejected.applied)
    assert jax.tree_util.tree_all(jax.tree_util.tree_map(jnp.array_equal, state, rejected.state))
    forged_key = dataclasses.replace(
        decision, next_action_key=jr.key(999, impl="threefry2x32")
    )
    rejected = agent.learn(state, forged_key, transition)
    assert not bool(rejected.applied)
    assert jax.tree_util.tree_all(jax.tree_util.tree_map(jnp.array_equal, state, rejected.state))
    invalid_decision = agent.decide(
        state, transition.observation, action=jnp.asarray(9, dtype=jnp.int32)
    )
    assert not bool(invalid_decision.valid)
    assert jnp.array_equal(invalid_decision.next_action_key, state.action_key)


def test_semantically_forged_state_is_rejected_before_use() -> None:
    agent = DreamerSequenceControl(_config())
    state = agent.init(jr.key(29, impl="threefry2x32"))
    observation = jnp.zeros((2,), dtype=jnp.float32)

    forged_replay = dataclasses.replace(
        state.replay,
        size=jnp.asarray(1, dtype=jnp.int32),
        insertion_ids=state.replay.insertion_ids.at[0].set(7),
    )
    with pytest.raises(ValueError, match="state differs"):
        agent.decide(dataclasses.replace(state, replay=forged_replay), observation)

    forged_counters = dataclasses.replace(
        state,
        actor_update_count=jnp.asarray(1, dtype=jnp.int32),
    )
    with pytest.raises(ValueError, match="state differs"):
        agent.policy(forged_counters, observation)
    with pytest.raises(ValueError, match="state differs"):
        agent.policy(dataclasses.replace(state, actor_update_count=1), observation)

    transition = _transition(0)
    decision = agent.decide(state, transition.observation, action=transition.action)
    rejected = agent.learn(state, dataclasses.replace(decision, valid=True), transition)
    assert not bool(rejected.applied)
    assert jax.tree_util.tree_all(jax.tree_util.tree_map(jnp.array_equal, state, rejected.state))


def test_config_preflights_the_complete_update_working_set() -> None:
    # This configuration has a small persistent state (~26 MiB), but its
    # million-step recurrent training trace exceeds signed-int32 accounting.
    # Construction must reject it before init allocates the replay.
    with pytest.raises(ValueError, match="working-set"):
        DreamerSequenceConfig(
            observation_dim=1,
            n_actions=2,
            latent_dim=512,
            replay_capacity=1_048_576,
            sequence_length=1_048_576,
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"replay_capacity": 1},
        {"sequence_length": 9},
        {"lambda_": float("nan")},
        {"imagination_enabled": 1},
        {"latent_dim": 100_000},
    ],
)
def test_config_rejects_invalid_and_type_confused_values(changes: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        dataclasses.replace(_config(), **changes)
