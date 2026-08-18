from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.benchmarks.adalin import (
    ADALIN_PROTOCOL,
    adalin_relu,
    adalin_relu_transaction,
    adalin_tanh,
)


def test_relu_reduction_and_prelu_identity() -> None:
    x = jnp.array([-2.0, 3.0])
    np.testing.assert_array_equal(adalin_relu(x, jnp.zeros(2)), jax.nn.relu(x))
    np.testing.assert_array_equal(adalin_relu(x, jnp.array([0.25, 0.25])), [-0.5, 3.0])


def test_gate_is_stop_gradient() -> None:
    value, derivative = jax.value_and_grad(lambda z: adalin_tanh(z, jnp.array(0.2)))(jnp.array(2.0))
    gate = jnp.cos(0.5 * jnp.pi * jnp.abs(1.0 - jnp.tanh(2.0) ** 2))
    expected = (1.0 - jnp.tanh(2.0) ** 2) + 0.2 * gate
    assert jnp.isfinite(value)
    np.testing.assert_allclose(derivative, expected, rtol=1e-6)


def test_protocol_keeps_pmnist_difference_explicit() -> None:
    assert ADALIN_PROTOCOL["paper_revision"] == "arXiv:2505.09486v1"
    assert ADALIN_PROTOCOL["paper_pmnist_tasks"] == 400
    assert ADALIN_PROTOCOL["asi_target_tasks"] == 200
    assert ADALIN_PROTOCOL["mechanism_off"] == "alpha_zero_exact_base_activation"
    assert ADALIN_PROTOCOL["scientific_promotion_allowed"] is False


def test_adalin_is_outer_jit_safe() -> None:
    transformed = jax.jit(adalin_relu)(jnp.array([-1.0, 1.0]), jnp.array([0.2, 0.2]))
    np.testing.assert_allclose(transformed, [-0.2, 1.0])
    invalid = jax.jit(adalin_relu)(jnp.array([jnp.nan]), jnp.array([0.2]))
    assert bool(jnp.all(jnp.isnan(invalid)))
    safe, valid = jax.jit(adalin_relu_transaction)(jnp.array([jnp.nan]), jnp.array([0.2]))
    np.testing.assert_array_equal(safe, jnp.zeros(1))
    assert not bool(valid)


def test_adalin_preflights_cross_broadcast_and_hostile_array() -> None:
    with pytest.raises(ValueError, match="broadcast output"):
        adalin_relu(jnp.ones((1, 1_000_000)), jnp.ones((1_000_000, 1)))

    class Hostile:
        calls = 0

        def __array__(self) -> np.ndarray:
            self.calls += 1
            raise AssertionError("must not run")

    hostile = Hostile()
    with pytest.raises(ValueError, match="exact NumPy or JAX"):
        adalin_relu(hostile, jnp.ones(1))  # type: ignore[arg-type]
    assert hostile.calls == 0
