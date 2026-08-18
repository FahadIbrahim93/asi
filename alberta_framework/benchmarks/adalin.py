"""AdaLin equation primitives and non-comparable PMNIST protocol declaration."""

from __future__ import annotations

import math
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

ADALIN_PROTOCOL = MappingProxyType(
    {
        "schema": "asi.adalin.protocol.v1",
        "paper_revision": "arXiv:2505.09486v1",
        "paper_revision_date": "2025-05-14",
        "paper_pmnist_tasks": 400,
        "paper_examples_per_task": 10_000,
        "paper_batch_size": 16,
        "paper_hidden_widths": (100, 100),
        "asi_target_tasks": 200,
        "asi_examples_per_task": 5_000,
        "asi_batch_size": 1,
        "asi_hidden_widths": (300, 150),
        "learner_observes_task_boundary": False,
        "mechanism_off": "alpha_zero_exact_base_activation",
        "finite_kernel_preflight_required": True,
        "matched_axes": ("seed", "updates", "observations"),
        "persistent_bytes_accounting_required": True,
        "environment_steps_accounting_required": True,
        "model_queries_accounting_required": True,
        "timing_is_telemetry_only": True,
        "development_only": True,
        "scientific_promotion_allowed": False,
    }
)

_MAX_ARRAY_ELEMENTS = 1_000_000


def _trusted_float_array(value: object, *, name: str) -> Array:
    actual_type = type(value)
    if not (actual_type is np.ndarray or issubclass(actual_type, (jax.Array, jax.core.Tracer))):
        raise ValueError(f"{name} must be an exact NumPy or JAX array")
    array = jnp.asarray(value)
    if (
        array.size < 1
        or array.size > _MAX_ARRAY_ELEMENTS
        or not jnp.issubdtype(array.dtype, jnp.floating)
    ):
        raise ValueError(f"{name} must be a bounded floating array")
    return array


def _adalin_transaction(
    x: Array, alpha: Array, *, activation: Array, derivative: Array
) -> tuple[Array, Array]:
    try:
        output_shape = np.broadcast_shapes(x.shape, alpha.shape, activation.shape, derivative.shape)
    except ValueError as error:
        raise ValueError("x and alpha must have compatible shapes") from error
    if math.prod(output_shape) > _MAX_ARRAY_ELEMENTS:
        raise ValueError("broadcast output exceeds the 1000000-element limit")
    gate = jax.lax.stop_gradient(jnp.cos(0.5 * jnp.pi * jnp.abs(derivative)))
    candidate = activation + alpha * x * gate
    valid = (
        jnp.all(jnp.isfinite(x)) & jnp.all(jnp.isfinite(alpha)) & jnp.all(jnp.isfinite(candidate))
    )
    safe = jnp.where(valid, candidate, jnp.zeros_like(candidate))
    return safe, valid


def _unwrap_transaction(result: tuple[Array, Array]) -> Array:
    safe, valid = result
    if isinstance(valid, jax.core.Tracer):
        return jnp.where(valid, safe, jnp.full_like(safe, jnp.nan))
    if not bool(valid):
        raise ValueError("AdaLin activation must be finite")
    return safe


def adalin_relu(x: Array, alpha: Array) -> Array:
    """AdaLin equation 2 for ReLU (algebraically PReLU)."""
    value = _trusted_float_array(x, name="x")
    coefficient = _trusted_float_array(alpha, name="alpha")
    return _unwrap_transaction(
        _adalin_transaction(
            value,
            coefficient,
            activation=jax.nn.relu(value),
            derivative=(value > 0).astype(value.dtype),
        )
    )


def adalin_relu_transaction(x: object, alpha: object) -> tuple[Array, Array]:
    """Return a finite ReLU-AdaLin value and caller-visible validity bit."""
    value = _trusted_float_array(x, name="x")
    coefficient = _trusted_float_array(alpha, name="alpha")
    return _adalin_transaction(
        value,
        coefficient,
        activation=jax.nn.relu(value),
        derivative=(value > 0).astype(value.dtype),
    )


def adalin_tanh(x: Array, alpha: Array) -> Array:
    """AdaLin equation 2 for tanh, whose Lipschitz constant is one."""
    value = _trusted_float_array(x, name="x")
    coefficient = _trusted_float_array(alpha, name="alpha")
    activation = jnp.tanh(value)
    return _unwrap_transaction(
        _adalin_transaction(
            value,
            coefficient,
            activation=activation,
            derivative=1.0 - activation**2,
        )
    )


def adalin_tanh_transaction(x: object, alpha: object) -> tuple[Array, Array]:
    """Return a finite tanh-AdaLin value and caller-visible validity bit."""
    value = _trusted_float_array(x, name="x")
    coefficient = _trusted_float_array(alpha, name="alpha")
    activation = jnp.tanh(value)
    return _adalin_transaction(
        value,
        coefficient,
        activation=activation,
        derivative=1.0 - activation**2,
    )
