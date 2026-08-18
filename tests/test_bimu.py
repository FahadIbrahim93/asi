from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.benchmarks.bimu import (
    BIMU_PROTOCOL,
    bimu_update,
    bimu_update_transaction,
    late_window_mean,
    posterior_probability,
    posterior_probability_transaction,
)


def test_bimu_update_matches_equations_six_and_seven() -> None:
    state = jnp.array([0.0, 1.0], dtype=jnp.float32)
    prior = jnp.zeros(2, dtype=jnp.float32)
    gradient = jnp.array([2.0, -0.5], dtype=jnp.float32)
    updated = bimu_update(state, gradient, prior, memory_window=10, alpha_max=1.0)
    reciprocal = (
        1.0 / jnp.cosh(state) ** 2
        + 2.0 * jnp.tanh(state) * gradient
        + 1.0
        + 2.0 * jnp.abs(gradient)
    )
    eta = 1.0 / reciprocal
    expected = state - eta * (gradient + (state - prior) / (10 * jnp.cosh(state) ** 2))
    np.testing.assert_allclose(updated, expected, rtol=1e-6)


def test_mechanism_off_removes_forgetting_term() -> None:
    state = jnp.array([0.5], dtype=jnp.float32)
    result = bimu_update(state, jnp.zeros(1), jnp.zeros(1), memory_window=None, alpha_max=0.5)
    np.testing.assert_array_equal(result, state)


def test_posterior_probability_and_late_metric_are_distinct() -> None:
    np.testing.assert_allclose(posterior_probability(jnp.array([0.0])), [0.5])
    assert late_window_mean([0.1, 0.2, 0.8, 0.9], window=2) == pytest.approx(0.85)
    assert BIMU_PROTOCOL["primary_metric"] == "mean_test_accuracy_over_last_5_tasks"
    assert BIMU_PROTOCOL["whole_stream_online_accuracy_is_separate"] is True


def test_protocol_is_binary_bayesian_and_nonpromoting() -> None:
    assert BIMU_PROTOCOL["paper_revision"] == "arXiv:2605.30198v1"
    assert BIMU_PROTOCOL["weight_domain"] == (-1, 1)
    assert BIMU_PROTOCOL["development_only"] is True
    assert BIMU_PROTOCOL["scientific_promotion_allowed"] is False


def test_bimu_update_is_outer_jit_safe() -> None:
    update = jax.jit(
        lambda state, gradient, prior: bimu_update(
            state, gradient, prior, memory_window=10, alpha_max=1.0
        )
    )
    assert bool(jnp.all(jnp.isfinite(update(jnp.zeros(2), jnp.ones(2), jnp.zeros(2)))))
    transact = jax.jit(
        lambda state, gradient: bimu_update_transaction(
            state, gradient, jnp.zeros(2), memory_window=10, alpha_max=1.0
        )
    )
    for state, gradient in (
        (jnp.array([jnp.nan, 0.0]), jnp.ones(2)),
        (jnp.zeros(2), jnp.array([jnp.inf, 0.0])),
    ):
        safe, valid = transact(state, gradient)
        assert bool(jnp.all(jnp.isfinite(safe)))
        assert not bool(valid)


def test_bimu_rejects_array_protocol_object_without_calling_it() -> None:
    class Hostile:
        calls = 0

        def __array__(self) -> np.ndarray:
            self.calls += 1
            raise AssertionError("must not run")

    hostile = Hostile()
    with pytest.raises(ValueError, match="exact NumPy or JAX"):
        posterior_probability(hostile)
    assert hostile.calls == 0


def test_bimu_float32_overflow_is_invalid_not_laundered() -> None:
    maximum = jnp.finfo(jnp.float32).max
    transact = jax.jit(
        lambda gradient: bimu_update_transaction(
            jnp.zeros(2), gradient, jnp.zeros(2), memory_window=10, alpha_max=1.0
        )
    )
    safe, valid = transact(jnp.full((2,), maximum))
    assert bool(jnp.all(jnp.isfinite(safe)))
    assert not bool(valid)
    posterior, posterior_valid = jax.jit(posterior_probability_transaction)(jnp.array([jnp.inf]))
    np.testing.assert_array_equal(posterior, [0.5])
    assert not bool(posterior_valid)
    finite_posterior, finite_valid = jax.jit(posterior_probability_transaction)(
        jnp.array([maximum])
    )
    np.testing.assert_array_equal(finite_posterior, [0.5])
    assert not bool(finite_valid)
