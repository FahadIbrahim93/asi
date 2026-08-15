"""Tests for online feature normalization."""

import chex
import jax.numpy as jnp
import pytest

from alberta_framework import (
    EMANormalizer,
    EMANormalizerState,
    Normalizer,
    StreamingBatchNormalizer,
    StreamingBatchNormalizerState,
    WelfordNormalizer,
    WelfordNormalizerState,
    normalizer_from_config,
)


class TestEMANormalizer:
    """Tests for the EMANormalizer class."""

    def test_init_creates_correct_state(self, feature_dim):
        """EMANormalizer init should create state with zero mean, unit variance."""
        normalizer = EMANormalizer()
        state = normalizer.init(feature_dim)

        assert isinstance(state, EMANormalizerState)
        chex.assert_shape(state.mean, (feature_dim,))
        chex.assert_shape(state.var, (feature_dim,))
        chex.assert_trees_all_close(state.mean, jnp.zeros(feature_dim))
        chex.assert_trees_all_close(state.var, jnp.ones(feature_dim))
        assert state.sample_count == 0.0

    def test_normalize_updates_statistics(self, sample_observation):
        """Normalizing should update mean and variance estimates."""
        normalizer = EMANormalizer()
        state = normalizer.init(len(sample_observation))

        normalized, new_state = normalizer.normalize(state, sample_observation)

        # Count should increase
        assert new_state.sample_count == 1.0

        # Mean should have moved toward the observation
        with pytest.raises(AssertionError):
            chex.assert_trees_all_close(new_state.mean, state.mean)

    def test_normalize_returns_finite_values(self, sample_observation):
        """Normalized output should always be finite."""
        normalizer = EMANormalizer()
        state = normalizer.init(len(sample_observation))

        normalized, _ = normalizer.normalize(state, sample_observation)

        chex.assert_tree_all_finite(normalized)

    def test_normalize_only_does_not_update_state(self, sample_observation):
        """normalize_only should not modify the state."""
        normalizer = EMANormalizer()
        state = normalizer.init(len(sample_observation))

        # First update state
        _, state = normalizer.normalize(state, sample_observation)
        original_count = state.sample_count

        # normalize_only should not change count
        _ = normalizer.normalize_only(state, sample_observation)

        assert state.sample_count == original_count

    def test_update_only_does_not_return_normalized(self, sample_observation):
        """update_only should only update state, returning new state."""
        normalizer = EMANormalizer()
        state = normalizer.init(len(sample_observation))

        new_state = normalizer.update_only(state, sample_observation)

        assert isinstance(new_state, EMANormalizerState)
        assert new_state.sample_count == 1.0

    def test_repeated_updates_converge(self, sample_observation):
        """Mean and variance should converge with repeated identical inputs."""
        normalizer = EMANormalizer(decay=0.9)
        state = normalizer.init(len(sample_observation))

        # Repeatedly normalize the same observation
        for _ in range(32):
            _, state = normalizer.normalize(state, sample_observation)

        # Mean should be close to the observation
        # (not exact due to decay and numerical issues)
        chex.assert_trees_all_close(state.mean, sample_observation, atol=0.5)

class TestWelfordNormalizer:
    """Tests for the WelfordNormalizer class."""

    def test_init_creates_correct_state(self, feature_dim):
        """WelfordNormalizer init should create state with zero mean, unit variance, zero p."""
        normalizer = WelfordNormalizer()
        state = normalizer.init(feature_dim)

        assert isinstance(state, WelfordNormalizerState)
        chex.assert_shape(state.mean, (feature_dim,))
        chex.assert_shape(state.var, (feature_dim,))
        chex.assert_shape(state.p, (feature_dim,))
        chex.assert_trees_all_close(state.mean, jnp.zeros(feature_dim))
        chex.assert_trees_all_close(state.var, jnp.ones(feature_dim))
        chex.assert_trees_all_close(state.p, jnp.zeros(feature_dim))
        assert state.sample_count == 0.0

    def test_var_is_one_when_count_less_than_two(self):
        """Variance should be 1.0 when fewer than 2 samples have been seen."""
        normalizer = WelfordNormalizer()
        state = normalizer.init(5)

        obs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        _, new_state = normalizer.normalize(state, obs)

        assert new_state.sample_count == 1.0
        chex.assert_trees_all_close(new_state.var, jnp.ones(5))

    def test_converges_to_true_mean_and_var(self):
        """After many samples, mean and var should match sample statistics."""
        normalizer = WelfordNormalizer()
        feature_dim = 5
        state = normalizer.init(feature_dim)

        true_mean = jnp.array([1.0, -2.0, 3.0, 0.5, -1.0])
        true_std = jnp.array([0.5, 1.0, 2.0, 0.3, 1.5])
        standardized = jnp.linspace(-1.5, 1.5, 32 * feature_dim).reshape(
            32, feature_dim
        )
        observations = true_mean + true_std * standardized
        for obs in observations:
            _, state = normalizer.normalize(state, obs)

        expected_mean = jnp.mean(observations, axis=0)
        expected_var = jnp.var(observations, axis=0, ddof=1)

        chex.assert_trees_all_close(state.mean, expected_mean, atol=1e-5)
        chex.assert_trees_all_close(state.var, expected_var, atol=1e-5)

    def test_normalize_returns_finite_values(self, sample_observation):
        """Normalized output should always be finite."""
        normalizer = WelfordNormalizer()
        state = normalizer.init(len(sample_observation))

        normalized, _ = normalizer.normalize(state, sample_observation)
        chex.assert_tree_all_finite(normalized)

    def test_normalize_only_does_not_update_state(self, sample_observation):
        """normalize_only should not modify the state."""
        normalizer = WelfordNormalizer()
        state = normalizer.init(len(sample_observation))

        _, state = normalizer.normalize(state, sample_observation)
        original_count = state.sample_count

        _ = normalizer.normalize_only(state, sample_observation)
        assert state.sample_count == original_count

    def test_update_only_returns_updated_state(self, sample_observation):
        """update_only should return updated state."""
        normalizer = WelfordNormalizer()
        state = normalizer.init(len(sample_observation))

        new_state = normalizer.update_only(state, sample_observation)

        assert isinstance(new_state, WelfordNormalizerState)
        assert new_state.sample_count == 1.0

    def test_p_field_has_correct_shape(self, feature_dim):
        """The p (M2) field should have shape (feature_dim,)."""
        normalizer = WelfordNormalizer()
        state = normalizer.init(feature_dim)
        chex.assert_shape(state.p, (feature_dim,))


class TestStreamingBatchNormalizer:
    """Tests for the StreamingBatchNormalizer class."""

    def test_init_creates_correct_state(self, feature_dim):
        """StreamingBatchNormalizer init should create running moments."""
        normalizer = StreamingBatchNormalizer(momentum=0.9)
        state = normalizer.init(feature_dim)

        assert isinstance(state, StreamingBatchNormalizerState)
        chex.assert_shape(state.mean, (feature_dim,))
        chex.assert_shape(state.var, (feature_dim,))
        chex.assert_trees_all_close(state.mean, jnp.zeros(feature_dim))
        chex.assert_trees_all_close(state.var, jnp.ones(feature_dim))
        assert state.sample_count == 0.0
        assert state.momentum == 0.9

    def test_normalize_updates_statistics(self, sample_observation):
        """Normalizing should update BatchNorm-style running moments."""
        normalizer = StreamingBatchNormalizer(momentum=0.5)
        state = normalizer.init(len(sample_observation))

        normalized, new_state = normalizer.normalize(state, sample_observation)

        chex.assert_tree_all_finite(normalized)
        assert new_state.sample_count == 1.0
        chex.assert_trees_all_close(new_state.mean, sample_observation)
        chex.assert_trees_all_close(new_state.var, jnp.ones_like(sample_observation))

    def test_roundtrip_config(self):
        """StreamingBatchNormalizer should serialize through the dispatcher."""
        normalizer = StreamingBatchNormalizer(momentum=0.75, epsilon=1e-4)
        config = normalizer.to_config()

        assert config["type"] == "StreamingBatchNormalizer"
        restored = normalizer_from_config(config)
        assert isinstance(restored, StreamingBatchNormalizer)
        assert restored._momentum == 0.75
        assert restored._epsilon == 1e-4


class TestNormalizerABC:
    """Tests that both normalizers satisfy the ABC contract."""

    @pytest.mark.parametrize(
        "normalizer_cls",
        [EMANormalizer, WelfordNormalizer, StreamingBatchNormalizer],
    )
    def test_is_normalizer_subclass(self, normalizer_cls):
        """All normalizers should be subclasses of Normalizer."""
        assert issubclass(normalizer_cls, Normalizer)

    @pytest.mark.parametrize(
        "normalizer",
        [EMANormalizer(), WelfordNormalizer(), StreamingBatchNormalizer()],
        ids=["EMA", "Welford", "StreamingBatch"],
    )
    def test_init_returns_state_with_required_fields(self, normalizer, feature_dim):
        """All normalizer states should have mean, var, and sample_count."""
        state = normalizer.init(feature_dim)

        assert hasattr(state, "mean")
        assert hasattr(state, "var")
        assert hasattr(state, "sample_count")
        chex.assert_shape(state.mean, (feature_dim,))
        chex.assert_shape(state.var, (feature_dim,))

    @pytest.mark.parametrize(
        "normalizer",
        [EMANormalizer(), WelfordNormalizer(), StreamingBatchNormalizer()],
        ids=["EMA", "Welford", "StreamingBatch"],
    )
    def test_normalize_returns_tuple(self, normalizer, sample_observation):
        """normalize should return (array, state) tuple."""
        state = normalizer.init(len(sample_observation))
        result = normalizer.normalize(state, sample_observation)

        assert isinstance(result, tuple)
        assert len(result) == 2

    @pytest.mark.parametrize(
        "normalizer",
        [EMANormalizer(), WelfordNormalizer(), StreamingBatchNormalizer()],
        ids=["EMA", "Welford", "StreamingBatch"],
    )
    def test_normalize_only_returns_array(self, normalizer, sample_observation):
        """normalize_only should return just an array."""
        state = normalizer.init(len(sample_observation))
        _, state = normalizer.normalize(state, sample_observation)
        result = normalizer.normalize_only(state, sample_observation)
        chex.assert_tree_all_finite(result)

    @pytest.mark.parametrize(
        "normalizer",
        [EMANormalizer(), WelfordNormalizer(), StreamingBatchNormalizer()],
        ids=["EMA", "Welford", "StreamingBatch"],
    )
    def test_update_only_increments_count(self, normalizer, sample_observation):
        """update_only should increment sample_count."""
        state = normalizer.init(len(sample_observation))
        new_state = normalizer.update_only(state, sample_observation)
        assert new_state.sample_count == 1.0

    @pytest.mark.parametrize(
        "normalizer",
        [EMANormalizer(), WelfordNormalizer(), StreamingBatchNormalizer()],
        ids=["EMA", "Welford", "StreamingBatch"],
    )
    def test_inf_observation_holds_finite_statistics(self, normalizer) -> None:
        """Inf observation minus an inf mean is inf-inf = NaN.

        Fail-closed: skip the statistics update and keep the previous finite
        mean/variance so later finite samples are not poisoned.
        """
        state = normalizer.init(2)
        inf_obs = jnp.array([jnp.inf, 1.0], dtype=jnp.float32)
        out, held = normalizer.normalize(state, inf_obs)
        assert bool(jnp.all(jnp.isfinite(out)))
        assert bool(jnp.all(jnp.isfinite(held.mean)))
        assert bool(jnp.all(jnp.isfinite(held.var)))
        chex.assert_trees_all_close(held.mean, state.mean)
        chex.assert_trees_all_close(held.var, state.var)
        assert int(held.sample_count) == int(state.sample_count)

        finite_obs = jnp.array([0.0, 2.0], dtype=jnp.float32)
        out2, recovered = normalizer.normalize(held, finite_obs)
        assert bool(jnp.all(jnp.isfinite(out2)))
        assert bool(jnp.all(jnp.isfinite(recovered.mean)))
        assert bool(jnp.all(jnp.isfinite(recovered.var)))
        assert int(recovered.sample_count) == 1

    @pytest.mark.parametrize(
        "normalizer",
        [EMANormalizer(), WelfordNormalizer(), StreamingBatchNormalizer()],
        ids=["EMA", "Welford", "StreamingBatch"],
    )
    def test_finite_overflow_is_not_rewritten_to_zero(self, normalizer) -> None:
        """Finite observation minus a finite opposite-max mean can overflow.

        That is a real arithmetic failure. Do not hide it as a zero feature.
        """
        state = normalizer.init(1)
        max_f32 = jnp.finfo(jnp.float32).max
        state = state.replace(mean=jnp.array([-max_f32], dtype=jnp.float32))
        out = normalizer.normalize_only(state, jnp.array([max_f32], dtype=jnp.float32))
        assert bool(jnp.isfinite(jnp.array([max_f32])))
        assert not bool(jnp.isfinite(out[0]))
        assert float(out[0]) != 0.0

