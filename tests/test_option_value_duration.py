"""Mechanism tests for conventional option value plus expected duration.

These are analytic and deterministic development tests, not held-out evidence
that Alberta Plan Step 5 is complete.
"""

import chex
import jax
import jax.numpy as jnp
import pytest

from alberta_framework.core.option_value_duration import (
    DURATION_HEAD,
    OptionValueDurationConfig,
    OptionValueDurationLearner,
)

pytestmark = pytest.mark.unit


def test_config_roundtrip_validation_and_fixed_parameter_count() -> None:
    config = OptionValueDurationConfig(
        reward_step_size=0.2,
        duration_step_size=0.3,
        duration_floor=1e-4,
    )
    learner = OptionValueDurationLearner.from_config(
        OptionValueDurationLearner(3, config).to_config()
    )

    assert learner.n_options == 3
    assert learner.config == config
    assert learner.trainable_parameter_count(feature_dim=5) == 3 * 2 * 5
    chex.assert_shape(learner.init(5).weights, (3, 2, 5))

    with pytest.raises(ValueError, match="n_options"):
        OptionValueDurationLearner(0)
    with pytest.raises(ValueError, match="reward_step_size"):
        OptionValueDurationConfig(reward_step_size=-0.1)
    with pytest.raises(ValueError, match="duration_step_size"):
        OptionValueDurationConfig(duration_step_size=-0.1)
    with pytest.raises(ValueError, match="duration_floor"):
        OptionValueDurationConfig(duration_floor=0.0)
    for invalid in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="reward_step_size"):
            OptionValueDurationConfig(reward_step_size=invalid)
        with pytest.raises(ValueError, match="duration_step_size"):
            OptionValueDurationConfig(duration_step_size=invalid)
        with pytest.raises(ValueError, match="duration_floor"):
            OptionValueDurationConfig(duration_floor=invalid)


def test_two_head_td_targets_and_updates_match_exact_analytic_values() -> None:
    learner = OptionValueDurationLearner(
        2,
        OptionValueDurationConfig(
            reward_step_size=0.1,
            duration_step_size=0.2,
        ),
    )
    initial_weights = jnp.array(
        [
            [[2.0, -1.0], [0.5, 1.5]],
            [[7.0, 8.0], [9.0, 10.0]],
        ],
        dtype=jnp.float32,
    )
    state = learner.init(2).replace(weights=initial_weights)  # type: ignore[attr-defined]

    result = jax.jit(learner.update)(
        state,
        jnp.array([1.0, 2.0], dtype=jnp.float32),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(3.0, dtype=jnp.float32),
        jnp.array([2.0, -1.0], dtype=jnp.float32),
        jnp.array(0.75, dtype=jnp.float32),
    )

    # predictions = [0, 3.5], next_predictions = [5, -0.5].
    # targets = [3, 1] + 0.75 * next_predictions = [6.75, 0.625].
    chex.assert_trees_all_close(
        result.predictions,
        jnp.array([0.0, 3.5], dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        result.next_predictions,
        jnp.array([5.0, -0.5], dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        result.td_targets,
        jnp.array([6.75, 0.625], dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        result.td_errors,
        jnp.array([6.75, -2.875], dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        result.state.weights[0],
        jnp.array([[2.675, 0.35], [-0.075, 0.35]], dtype=jnp.float32),
        atol=1e-6,
    )
    chex.assert_trees_all_close(result.state.weights[1], initial_weights[1])
    chex.assert_trees_all_equal(
        result.state.option_update_counts,
        jnp.array([1, 0], dtype=jnp.int32),
    )
    assert int(result.state.step_count) == 1


def test_applied_update_saturates_lifetime_counters_without_wrapping() -> None:
    learner = OptionValueDurationLearner(1)
    int32_max = jnp.iinfo(jnp.int32).max
    state = learner.init(1).replace(  # type: ignore[attr-defined]
        option_update_counts=jnp.array([int32_max], dtype=jnp.int32),
        step_count=jnp.array(int32_max, dtype=jnp.int32),
    )

    result = learner.update(
        state,
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
    )

    assert bool(result.update_applied)
    assert not bool(jnp.array_equal(result.state.weights, state.weights))
    assert int(result.state.option_update_counts[0]) == int32_max
    assert int(result.state.step_count) == int32_max


def test_termination_discount_zeros_bootstrap_and_no_average_reward_is_subtracted() -> None:
    learner = OptionValueDurationLearner(
        1,
        OptionValueDurationConfig(
            reward_step_size=0.0,
            duration_step_size=0.0,
        ),
    )
    state = learner.init(1).replace(  # type: ignore[attr-defined]
        weights=jnp.array([[[2.0], [7.0]]], dtype=jnp.float32)
    )

    result = learner.update(
        state,
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(5.0, dtype=jnp.float32),
        jnp.array([100.0], dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
    )

    # An arbitrarily large next prediction cannot leak through termination.
    # The conventional reward target is the raw reward, not reward minus rbar.
    chex.assert_trees_all_close(
        result.td_targets,
        jnp.array([5.0, 1.0], dtype=jnp.float32),
    )
    assert not hasattr(result.state, "average_reward")


def test_termination_does_not_multiply_inf_next_prediction() -> None:
    """gamma=0 * inf V(s') is 0*inf = NaN and would freeze both option heads."""
    learner = OptionValueDurationLearner(
        1,
        OptionValueDurationConfig(
            reward_step_size=0.1,
            duration_step_size=0.0,
        ),
    )
    state = learner.init(1).replace(  # type: ignore[attr-defined]
        weights=jnp.array([[[2.0], [7.0]]], dtype=jnp.float32)
    )
    next_obs = jnp.array([jnp.inf], dtype=jnp.float32)
    raw = jnp.asarray(0.0, dtype=jnp.float32) * (jnp.array([2.0], dtype=jnp.float32) @ next_obs)
    assert not bool(jnp.isfinite(raw))

    result = learner.update(
        state,
        jnp.array([1.0], dtype=jnp.float32),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(5.0, dtype=jnp.float32),
        next_obs,
        jnp.array(0.0, dtype=jnp.float32),
    )

    chex.assert_trees_all_close(
        result.td_targets,
        jnp.array([5.0, 1.0], dtype=jnp.float32),
    )
    chex.assert_tree_all_finite(result.state.weights)
    chex.assert_tree_all_finite(result.next_predictions)
    assert bool(result.update_applied)


def test_reward_rate_prediction_preserves_raw_duration_and_floors_only_score() -> None:
    learner = OptionValueDurationLearner(
        2,
        OptionValueDurationConfig(duration_floor=0.5),
    )
    state = learner.init(1).replace(  # type: ignore[attr-defined]
        weights=jnp.array(
            [
                [[6.0], [10.0]],
                [[4.0], [0.0]],
            ],
            dtype=jnp.float32,
        )
    )

    prediction = learner.predict(state, jnp.array([1.0], dtype=jnp.float32))

    chex.assert_trees_all_close(
        prediction.reward_values,
        jnp.array([6.0, 4.0], dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        prediction.durations,
        jnp.array([10.0, 0.0], dtype=jnp.float32),
    )
    chex.assert_trees_all_close(
        prediction.reward_rates,
        jnp.array([0.6, 8.0], dtype=jnp.float32),
    )


def test_infinite_reward_on_zero_feature_does_not_poison_duration_head() -> None:
    """Inf reward is 0*inf = NaN on a silent feature of the reward head.

    The duration head's TD error stays finite. Map NaN products back to the
    previous weight so that head can keep learning, and leave genuine infs.
    """
    learner = OptionValueDurationLearner(
        1,
        OptionValueDurationConfig(reward_step_size=0.1, duration_step_size=0.2),
    )
    state = learner.init(2)
    obs = jnp.array([0.0, 1.0], dtype=jnp.float32)
    nxt = jnp.array([0.0, 1.0], dtype=jnp.float32)
    option = jnp.array(0, dtype=jnp.int32)
    discount = jnp.array(0.0, dtype=jnp.float32)

    poisoned = learner.update(
        state, obs, option, jnp.array(jnp.inf, dtype=jnp.float32), nxt, discount
    )
    assert not bool(jnp.any(jnp.isnan(poisoned.state.weights)))
    chex.assert_tree_all_finite(poisoned.state.weights[0, DURATION_HEAD])
    chex.assert_trees_all_close(
        poisoned.state.weights[0, 0, 0],
        state.weights[0, 0, 0],
    )
    assert bool(poisoned.update_applied)
    chex.assert_trees_all_equal(
        poisoned.head_updates_applied,
        jnp.array([False, True]),
    )
    assert float(poisoned.td_errors[0]) == 0.0
    assert bool(jnp.isfinite(poisoned.td_errors[DURATION_HEAD]))

    recovered = learner.update(
        poisoned.state, obs, option, jnp.array(1.0, dtype=jnp.float32), nxt, discount
    )
    assert not bool(jnp.any(jnp.isnan(recovered.state.weights)))
    chex.assert_tree_all_finite(recovered.state.weights[0, DURATION_HEAD])
    assert bool(recovered.update_applied)
    assert bool(jnp.all(recovered.head_updates_applied))
