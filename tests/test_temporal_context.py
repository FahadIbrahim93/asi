# mypy: disable-error-code="call-arg,untyped-decorator"
"""Tests for causal temporal/context features."""

from typing import cast

import chex
import jax
import jax.numpy as jnp
import numpy as np

from alberta_framework.core.temporal_context import (
    TemporalContextConfig,
    TemporalContextFeaturizer,
    TemporalContextState,
    transform_temporal_context_arrays,
)


def test_temporal_context_shapes_and_roundtrip() -> None:
    config = TemporalContextConfig(input_dim=3, periods=(5.0, 10.0))
    featurizer = TemporalContextFeaturizer(config)
    state = featurizer.init()

    features = featurizer.features(state, jnp.ones(3))

    assert config.output_dim() == 13
    chex.assert_shape(features, (13,))
    chex.assert_tree_all_finite(features)
    assert TemporalContextConfig.from_config(config.to_config()) == config


def test_temporal_context_step_is_causal() -> None:
    config = TemporalContextConfig(input_dim=2, ema_decay=0.5, periods=())
    featurizer = TemporalContextFeaturizer(config)
    state = featurizer.init()
    observation = jnp.asarray([2.0, -2.0], dtype=jnp.float32)

    next_state, features = featurizer.step(state, observation)

    chex.assert_trees_all_close(features[:2], observation)
    chex.assert_trees_all_close(features[2:4], jnp.zeros(2))
    chex.assert_trees_all_close(features[4:6], observation)
    chex.assert_trees_all_close(next_state.observation_ema, observation * 0.5)
    assert int(next_state.step_count) == 1


def test_temporal_context_phase_products_expand_with_input() -> None:
    config = TemporalContextConfig(
        input_dim=2,
        include_phase_products=True,
        periods=(4.0,),
    )
    featurizer = TemporalContextFeaturizer(config)

    features = featurizer.features(featurizer.init(), jnp.asarray([3.0, -1.0]))

    assert config.output_dim() == 12
    chex.assert_shape(features, (12,))
    chex.assert_tree_all_finite(features)


def test_temporal_context_array_transform_is_jittable() -> None:
    config = TemporalContextConfig(input_dim=2, periods=(4.0,))
    featurizer = TemporalContextFeaturizer(config)
    observations = jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)

    @jax.jit
    def run(
        initial_state: TemporalContextState,
    ) -> tuple[TemporalContextState, jax.Array]:
        return transform_temporal_context_arrays(
            featurizer,
            observations,
            state=initial_state,
        )

    state, features = run(featurizer.init())

    chex.assert_shape(features, (2, config.output_dim()))
    assert int(state.step_count) == 2
    chex.assert_tree_all_finite(features)


def test_nonfinite_observation_holds_warm_state_and_recovers_under_jit() -> None:
    config = TemporalContextConfig(
        input_dim=3,
        ema_decay=0.5,
        periods=(4.0,),
        include_phase_products=True,
    )
    featurizer = TemporalContextFeaturizer(config)
    warm_state, _ = featurizer.step(
        featurizer.init(),
        jnp.asarray([2.0, -4.0, 6.0], dtype=jnp.float32),
    )

    @jax.jit
    def run(
        state: TemporalContextState,
        observation: jax.Array,
    ) -> tuple[TemporalContextState, jax.Array]:
        return cast(
            tuple[TemporalContextState, jax.Array],
            featurizer.step(state, observation),
        )

    held_state, invalid_features = run(
        warm_state,
        jnp.asarray([jnp.inf, 8.0, jnp.nan], dtype=jnp.float32),
    )

    chex.assert_tree_all_finite(invalid_features)
    chex.assert_trees_all_equal(held_state, warm_state)
    chex.assert_trees_all_equal(invalid_features[:3], jnp.asarray([0.0, 8.0, 0.0]))
    chex.assert_trees_all_equal(invalid_features[3:6], warm_state.observation_ema)
    # Invalid delta coordinates are exactly zero, not the negated warm EMA.
    chex.assert_trees_all_equal(invalid_features[6:9], jnp.asarray([0.0, 10.0, 0.0]))
    phase_products = invalid_features[11:].reshape(2, 3)
    chex.assert_trees_all_equal(phase_products[:, (0, 2)], jnp.zeros((2, 2)))

    recovered_state, recovered_features = run(
        held_state,
        jnp.asarray([4.0, 2.0, -2.0], dtype=jnp.float32),
    )

    chex.assert_tree_all_finite((recovered_state, recovered_features))
    chex.assert_trees_all_close(
        recovered_state.observation_ema,
        jnp.asarray([2.5, 0.0, 0.5], dtype=jnp.float32),
    )
    assert int(recovered_state.step_count) == 2


def test_finite_temporal_context_path_is_bitwise_unchanged() -> None:
    config = TemporalContextConfig(input_dim=3, ema_decay=0.75, periods=())
    featurizer = TemporalContextFeaturizer(config)
    state = TemporalContextState(
        observation_ema=jnp.asarray([0.25, -1.5, 3.0], dtype=jnp.float32),
        step_count=jnp.asarray(17, dtype=jnp.int32),
    )
    observation = jnp.asarray([1.5, 2.0, -0.5], dtype=jnp.float32)

    next_state, features = featurizer.step(state, observation)
    expected_features = jnp.concatenate(
        [observation, state.observation_ema, observation - state.observation_ema]
    )
    expected_ema = jnp.float32(0.75) * state.observation_ema + jnp.float32(
        0.25
    ) * observation

    assert np.asarray(features).tobytes() == np.asarray(expected_features).tobytes()
    assert np.asarray(next_state.observation_ema).tobytes() == np.asarray(
        expected_ema
    ).tobytes()
    assert int(next_state.step_count) == 18
