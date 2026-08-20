"""End-to-end coverage for :class:`ModelReplayRehearsal`.

Every other ``model_replay_rehearsal`` test file only constructs and
validates the configuration, construction, or initialization boundary; none
of them ever call :meth:`ModelReplayRehearsal.step`. That left the runtime
transaction -- ``_step_jit``, ``_rehearse_batch``,
``encode_action``, and ``decode_action`` -- including both branches of the
documented ``action_encoding`` dispatch (``"scalar_index"`` and
``"one_hot"``), without a single execution in the suite.

These tests run the real update/record/rehearsal transaction for both
encodings and check that the two encodings of the same action stream produce
the same learning-relevant trajectory.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import tree_util

from alberta_framework.core.dual_replay import DualReplayConfig
from alberta_framework.core.learning_signals import LearningSignalEstimatorConfig
from alberta_framework.core.model_replay_rehearsal import (
    ModelReplayRehearsal,
    ModelReplayRehearsalConfig,
    RealModelReplayEvent,
    ReplayActionEncoding,
)
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig
from alberta_framework.core.world_model_ensemble import WorldModelEnsembleConfig

_OBSERVATION_DIM = 2
_N_ACTIONS = 3


def _ensemble_config(*, ensemble_size: int = 2) -> WorldModelEnsembleConfig:
    target_dim = _OBSERVATION_DIM + 2
    model = ActionConditionedWorldModelConfig(
        observation_dim=_OBSERVATION_DIM,
        n_actions=_N_ACTIONS,
        hidden_sizes=(),
        gamma=0.95,
        step_size=0.05,
        sparsity=0.0,
        use_layer_norm=False,
        error_decay=0.8,
    )
    signals = LearningSignalEstimatorConfig(
        ensemble_size=ensemble_size,
        target_dim=target_dim,
        progress_warmup_steps=2,
        change_calibration_steps=2,
        fast_loss_decay=0.5,
        slow_loss_decay=0.9,
        max_input_magnitude=100.0,
        max_predicted_variance=10_000.0,
        max_observed_loss=10_000.0,
    )
    return WorldModelEnsembleConfig(
        model=model,
        signal_estimator=signals,
        ensemble_size=ensemble_size,
        bootstrap_probability=0.5,
        residual_variance_decay=0.8,
        residual_variance_warmup_steps=1,
        residual_variance_floor=1e-6,
    )


def _replay_config(*, action_dim: int, total_capacity: int = 8) -> DualReplayConfig:
    return DualReplayConfig(
        total_capacity=total_capacity,
        short_term_capacity=4,
        observation_dim=_OBSERVATION_DIM,
        action_dim=action_dim,
        short_term_sample_size=2,
        long_term_sample_size=2,
    )


def _rehearsal(encoding: ReplayActionEncoding) -> ModelReplayRehearsal:
    action_dim = 1 if encoding == "scalar_index" else _N_ACTIONS
    config = ModelReplayRehearsalConfig(
        ensemble=_ensemble_config(),
        replay=_replay_config(action_dim=action_dim),
        action_encoding=encoding,
    )
    return ModelReplayRehearsal(config)


def _event(
    *, observation: jnp.ndarray, action: int, next_observation: jnp.ndarray
) -> RealModelReplayEvent:
    return RealModelReplayEvent(  # type: ignore[call-arg]
        observation=observation,
        action=jnp.asarray(action, dtype=jnp.int32),
        reward=jnp.asarray(0.5, dtype=jnp.float32),
        discount=jnp.asarray(0.99, dtype=jnp.float32),
        terminated=jnp.asarray(False),
        truncated=jnp.asarray(False),
        next_observation=next_observation,
        representation_version=jnp.asarray(0, dtype=jnp.int32),
        provenance_id=jnp.asarray(0, dtype=jnp.int32),
        source_id=jnp.asarray(0, dtype=jnp.int32),
        safety_cost=jnp.asarray(0.0, dtype=jnp.float32),
        safety_cost_available=jnp.asarray(True),
        valid=jnp.asarray(True),
    )


@pytest.mark.parametrize("encoding", ["scalar_index", "one_hot"])
def test_encode_decode_round_trips_every_action(encoding: ReplayActionEncoding) -> None:
    """``encode_action``/``decode_action`` recover every legal action exactly."""
    rehearsal = _rehearsal(encoding)
    for action in range(_N_ACTIONS):
        stored = rehearsal.encode_action(jnp.asarray(action, dtype=jnp.int32))
        conversion = rehearsal.decode_action(stored)
        assert bool(conversion.valid)
        assert int(conversion.action) == action


@pytest.mark.parametrize("encoding", ["scalar_index", "one_hot"])
def test_step_runs_the_full_transaction_end_to_end(encoding: ReplayActionEncoding) -> None:
    """The real update/record/rehearsal transaction commits every step.

    This is the first test in the suite that calls ``step`` -- for either
    encoding.
    """
    rehearsal = _rehearsal(encoding)
    state = rehearsal.init(jr.key(0))

    key = jr.key(1)
    for step in range(16):
        key, subkey = jr.split(key)
        obs_key, next_key, action_key = jr.split(subkey, 3)
        observation = jr.normal(obs_key, (_OBSERVATION_DIM,), dtype=jnp.float32)
        next_observation = jr.normal(next_key, (_OBSERVATION_DIM,), dtype=jnp.float32)
        action = int(jr.randint(action_key, (), 0, _N_ACTIONS))
        event = _event(observation=observation, action=action, next_observation=next_observation)

        result = rehearsal.step(state, event)

        assert bool(result.diagnostics.transaction_applied)
        assert not bool(result.diagnostics.rejected)
        assert bool(jnp.all(result.diagnostics.action_conversions_valid))
        assert int(result.state.accepted_real_event_count) == step + 1
        assert int(result.state.rejected_real_event_count) == 0
        for decoded_action in result.trace.actions.tolist():
            assert 0 <= decoded_action < _N_ACTIONS
        state = result.state

    rehearsal.validate_state(state)
    assert int(state.accepted_real_event_count) == 16


def test_scalar_index_and_one_hot_agree_on_the_same_action_stream() -> None:
    """The two ``action_encoding`` branches must be behaviorally equivalent.

    Feeding the identical observation/action/reward stream through the
    ``"scalar_index"`` and ``"one_hot"`` compositions must produce the same
    decoded action trace and the same real observed loss at every step: the
    storage encoding is not supposed to change what gets learned.
    """
    scalar_rehearsal = _rehearsal("scalar_index")
    one_hot_rehearsal = _rehearsal("one_hot")
    scalar_state = scalar_rehearsal.init(jr.key(0))
    one_hot_state = one_hot_rehearsal.init(jr.key(0))

    key = jr.key(7)
    for step in range(12):
        key, subkey = jr.split(key)
        obs_key, next_key, action_key = jr.split(subkey, 3)
        observation = jr.normal(obs_key, (_OBSERVATION_DIM,), dtype=jnp.float32)
        next_observation = jr.normal(next_key, (_OBSERVATION_DIM,), dtype=jnp.float32)
        action = int(jr.randint(action_key, (), 0, _N_ACTIONS))
        event = _event(observation=observation, action=action, next_observation=next_observation)

        scalar_result = scalar_rehearsal.step(scalar_state, event)
        one_hot_result = one_hot_rehearsal.step(one_hot_state, event)
        scalar_state = scalar_result.state
        one_hot_state = one_hot_result.state

        assert scalar_result.trace.actions.tolist() == one_hot_result.trace.actions.tolist()
        assert bool(
            jnp.array_equal(
                scalar_result.trace.observed_losses, one_hot_result.trace.observed_losses
            )
        )
        assert float(scalar_result.real_observed_loss) == pytest.approx(
            float(one_hot_result.real_observed_loss)
        )
        assert all(
            bool(jnp.array_equal(scalar_leaf, one_hot_leaf))
            for scalar_leaf, one_hot_leaf in zip(
                tree_util.tree_leaves(scalar_state.ensemble_state),
                tree_util.tree_leaves(one_hot_state.ensemble_state),
                strict=True,
            )
        )

    scalar_rehearsal.validate_state(scalar_state)
    one_hot_rehearsal.validate_state(one_hot_state)
