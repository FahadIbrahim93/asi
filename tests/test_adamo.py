from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.adamo import (
    AdamO,
    AdamOConfig,
    gram_working_bytes,
    isometry_gradient,
    isometry_penalty,
    state_persistent_bytes,
)
from alberta_framework.core.baseline_optimizers import Adam


@pytest.mark.parametrize("shape", [(4, 2), (2, 4), (3, 3)])
def test_penalty_matches_singular_value_definition(shape: tuple[int, int]) -> None:
    weight = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape) / 10.0
    singular_values = jnp.linalg.svd(weight, compute_uv=False)
    expected = jnp.sum(jnp.square(jnp.square(singular_values) - 1.0))
    np.testing.assert_allclose(isometry_penalty(weight), expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("shape", [(4, 2), (2, 4), (3, 3)])
def test_closed_form_gradient_matches_autodiff(shape: tuple[int, int]) -> None:
    weight = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape) / 7.0
    expected = jax.grad(isometry_penalty)(weight)
    np.testing.assert_allclose(isometry_gradient(weight), expected, rtol=1e-5, atol=1e-5)


def test_zero_strength_reduces_exactly_to_adam_task_step() -> None:
    config = AdamOConfig(
        step_size=1e-3, beta1=0.9, beta2=0.99, eps=1e-8, isometry_strength=0.0
    )
    optimizer = AdamO(config)
    adam = Adam(step_size=1e-3, beta1=0.9, beta2=0.99, eps=1e-8)
    weight = jnp.asarray([[0.2, -0.1], [0.4, 0.3]], dtype=jnp.float32)
    gradient = jnp.asarray([[0.5, -0.2], [0.1, 0.7]], dtype=jnp.float32)
    result = optimizer.update_from_gradient_checked(
        optimizer.init_for_shape(weight.shape), gradient, weight, regularize=True
    )
    adam_result = adam.update_from_gradient_checked(
        adam.init_for_shape(weight.shape), gradient, param=weight
    )
    np.testing.assert_array_equal(result.step, adam_result.step)
    np.testing.assert_array_equal(result.new_state.m, adam_result.new_state.m)
    np.testing.assert_array_equal(result.new_state.v, adam_result.new_state.v)


def test_isometry_gradient_is_decoupled_from_moments() -> None:
    weight = jnp.asarray([[2.0, 0.0], [0.0, 0.5]], dtype=jnp.float32)
    gradient = jnp.ones_like(weight)
    off = AdamO(AdamOConfig(isometry_strength=0.0))
    on = AdamO(AdamOConfig(isometry_strength=1e-3))
    off_result = off.update_from_gradient_checked(
        off.init_for_shape(weight.shape), gradient, weight, regularize=True
    )
    on_result = on.update_from_gradient_checked(
        on.init_for_shape(weight.shape), gradient, weight, regularize=True
    )
    np.testing.assert_array_equal(off_result.new_state.m, on_result.new_state.m)
    np.testing.assert_array_equal(off_result.new_state.v, on_result.new_state.v)
    np.testing.assert_array_equal(
        on_result.isometry_step,
        on.config.isometry_step_size * on.config.isometry_strength * isometry_gradient(weight),
    )
    np.testing.assert_allclose(
        on_result.step - off_result.step, on_result.isometry_step, atol=4e-12
    )


def test_bias_leaf_is_never_regularized() -> None:
    optimizer = AdamO(AdamOConfig(isometry_strength=1.0))
    bias = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    result = optimizer.update_from_gradient_checked(
        optimizer.init_for_shape(bias.shape), jnp.ones_like(bias), bias, regularize=False
    )
    np.testing.assert_array_equal(result.isometry_step, jnp.zeros_like(bias))


def test_nonfinite_candidate_fails_closed() -> None:
    optimizer = AdamO(AdamOConfig(isometry_strength=1.0))
    weight = jnp.full((2, 2), jnp.finfo(jnp.float32).max)
    state = optimizer.init_for_shape(weight.shape)
    result = jax.jit(optimizer.update_from_gradient_checked, static_argnames="regularize")(
        state, jnp.ones_like(weight), weight, regularize=True
    )
    assert not bool(result.update_applied)
    np.testing.assert_array_equal(result.step, jnp.zeros_like(weight))
    np.testing.assert_array_equal(result.new_state.m, state.m)


def test_resource_accounting_names_state_and_gram() -> None:
    state = AdamO().init_for_shape((3, 5))
    assert state_persistent_bytes(state) == (2 * 15 + 5) * 4
    assert gram_working_bytes((3, 5)) == 3 * 3 * 4


def test_update_is_jittable() -> None:
    optimizer = AdamO(AdamOConfig(isometry_strength=1e-3))
    weight = jnp.eye(2, dtype=jnp.float32)
    state = optimizer.init_for_shape(weight.shape)
    result = jax.jit(optimizer.update_from_gradient_checked, static_argnames="regularize")(
        state, jnp.ones_like(weight), weight, regularize=True
    )
    assert bool(result.update_applied)


def test_screening_inert_arm_reduces_to_adamw() -> None:
    from alberta_framework.benchmarks.ipmnist_screening import (
        _make_adamo_raw_learner,
        _make_adamw_learner,
        screening_spec,
    )

    params = {
        "w1": jnp.asarray([[0.2, -0.1], [0.4, 0.3]], dtype=jnp.float32),
        "b1": jnp.asarray([0.1, -0.2], dtype=jnp.float32),
    }
    grads = {name: jnp.ones_like(value) for name, value in params.items()}
    inert = screening_spec("adamo_inert")
    control = screening_spec("adamw_control")
    inert_init, inert_step = _make_adamo_raw_learner(inert.hyperparameters)
    control_init, control_step = _make_adamw_learner(control.hyperparameters)
    key = jax.random.key(0)
    inert_params, inert_state = inert_step(params, inert_init(params), grads, key)
    control_result = control_step(params, control_init(params), grads, key)
    control_params, control_state = control_result
    for name in params:
        np.testing.assert_array_equal(inert_params[name], control_params[name])
        np.testing.assert_array_equal(inert_state[name].m, control_state[name].m)
        np.testing.assert_array_equal(inert_state[name].v, control_state[name].v)
