"""Equation-level BiMU primitives for a separate binary/Bayesian lane."""

from __future__ import annotations

import math
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

BIMU_PROTOCOL = MappingProxyType(
    {
        "schema": "asi.bimu.protocol.v1",
        "paper_revision": "arXiv:2605.30198v1",
        "paper_revision_date": "2026-05-28",
        "lane": "binary_bayesian_permuted_mnist",
        "weight_domain": (-1, 1),
        "primary_metric": "mean_test_accuracy_over_last_5_tasks",
        "whole_stream_online_accuracy_is_separate": True,
        "paper_axes": (("tasks", 1000), ("hidden_units", 100), ("batch_size", 1)),
        "adaptation_difference": "ASI implementation exposes equation primitives; no run is frozen",
        "learner_observes_task_boundary": False,
        "finite_kernel_preflight_required": True,
        "matched_axes": ("seed", "updates", "observations", "label_queries"),
        "persistent_bytes_accounting_required": True,
        "environment_steps_accounting_required": True,
        "model_queries_accounting_required": True,
        "timing_is_telemetry_only": True,
        "development_only": True,
        "scientific_promotion_allowed": False,
    }
)

_INT32_MAX = 2**31 - 1
_MAX_VECTOR_ELEMENTS = 1_000_000


def _finite_vector(value: object, *, name: str) -> Array:
    actual_type = type(value)
    if not (actual_type is np.ndarray or issubclass(actual_type, (jax.Array, jax.core.Tracer))):
        raise ValueError(f"{name} must be an exact NumPy or JAX array")
    array = jnp.asarray(value)
    if (
        array.ndim != 1
        or array.size < 1
        or array.size > _MAX_VECTOR_ELEMENTS
        or not jnp.issubdtype(array.dtype, jnp.floating)
    ):
        raise ValueError(f"{name} must be a non-empty floating vector")
    valid = jnp.all(jnp.isfinite(array))
    if not isinstance(valid, jax.core.Tracer) and not bool(valid):
        raise ValueError(f"{name} must contain only finite values")
    return array


def posterior_probability_transaction(natural_parameter: object) -> tuple[Array, Array]:
    """Return a finite posterior probability and caller-visible validity bit."""
    state = _finite_vector(natural_parameter, name="natural_parameter")
    logit = 2.0 * state
    result = jax.nn.sigmoid(logit)
    valid = (
        jnp.all(jnp.isfinite(state)) & jnp.all(jnp.isfinite(logit)) & jnp.all(jnp.isfinite(result))
    )
    return jnp.where(valid, result, jnp.full_like(result, 0.5)), valid


def posterior_probability(natural_parameter: object) -> Array:
    """Return ``P(weight=+1) = sigmoid(2 lambda)`` (paper equation 2)."""
    safe, valid = posterior_probability_transaction(natural_parameter)
    if isinstance(valid, jax.core.Tracer):
        return jnp.where(valid, safe, jnp.full_like(safe, jnp.nan))
    if not bool(valid):
        raise ValueError("posterior probability must be finite")
    return safe


def bimu_update_transaction(
    natural_parameter: object,
    loss_gradient: object,
    prior_natural_parameter: object,
    *,
    memory_window: int | None,
    alpha_max: float,
) -> tuple[Array, Array]:
    """Apply BiMU equations 6--7 to one flat natural-parameter vector.

    ``memory_window=None`` is the predeclared mechanism-off reduction: it
    removes controlled forgetting while retaining the bounded metaplastic step.
    """
    state = _finite_vector(natural_parameter, name="natural_parameter")
    gradient = _finite_vector(loss_gradient, name="loss_gradient")
    prior = _finite_vector(prior_natural_parameter, name="prior_natural_parameter")
    if state.shape != gradient.shape or state.shape != prior.shape:
        raise ValueError("state, gradient, and prior must have identical shapes")
    if type(alpha_max) is not float or not math.isfinite(alpha_max) or alpha_max <= 0.0:
        raise ValueError("alpha_max must be a finite positive float")
    if memory_window is not None and (
        type(memory_window) is not int or memory_window < 1 or memory_window > _INT32_MAX
    ):
        raise ValueError("memory_window must be None or a positive integer")
    uncertainty = 1.0 / jnp.cosh(state) ** 2
    reciprocal_eta = (
        uncertainty + 2.0 * jnp.tanh(state) * gradient + 1.0 / alpha_max + 2.0 * jnp.abs(gradient)
    )
    eta = 1.0 / reciprocal_eta
    forgetting = 0.0 if memory_window is None else (state - prior) * uncertainty / memory_window
    summed_update = gradient + forgetting
    candidate = state - eta * summed_update
    source_valid = (
        jnp.all(jnp.isfinite(state))
        & jnp.all(jnp.isfinite(gradient))
        & jnp.all(jnp.isfinite(prior))
    )
    candidate_valid = jnp.all(jnp.isfinite(candidate))
    intermediate_valid = (
        jnp.all(jnp.isfinite(uncertainty))
        & jnp.all(jnp.isfinite(reciprocal_eta))
        & jnp.all(jnp.isfinite(eta))
        & jnp.all(jnp.isfinite(forgetting))
        & jnp.all(jnp.isfinite(summed_update))
    )
    valid = source_valid & intermediate_valid & candidate_valid
    fallback = jnp.where(source_valid, state, jnp.zeros_like(state))
    safe = jnp.where(valid, candidate, fallback)
    return safe, valid


def bimu_update(
    natural_parameter: object,
    loss_gradient: object,
    prior_natural_parameter: object,
    *,
    memory_window: int | None,
    alpha_max: float,
) -> Array:
    """Compatibility wrapper; traced callers use NaN to expose invalid transactions."""
    safe, valid = bimu_update_transaction(
        natural_parameter,
        loss_gradient,
        prior_natural_parameter,
        memory_window=memory_window,
        alpha_max=alpha_max,
    )
    if isinstance(valid, jax.core.Tracer):
        return jnp.where(valid, safe, jnp.full_like(safe, jnp.nan))
    if not bool(valid):
        raise ValueError("BiMU update must produce only finite values")
    return safe


def late_window_mean(task_accuracies: list[float] | tuple[float, ...], *, window: int = 5) -> float:
    """Compute BiMU's late-task metric without conflating whole-stream accuracy."""
    if type(window) is not int or window < 1 or window > _MAX_VECTOR_ELEMENTS:
        raise ValueError("window must be an integer in [1, 1000000]")
    if (
        (type(task_accuracies) is not list and type(task_accuracies) is not tuple)
        or len(task_accuracies) > _MAX_VECTOR_ELEMENTS
    ):
        raise ValueError("task_accuracies must be an exact bounded list or tuple")
    if any(type(value) is not int and type(value) is not float for value in task_accuracies):
        raise ValueError("task_accuracies must contain exact real numbers")
    values = np.asarray(task_accuracies)
    if values.ndim != 1 or values.size < window or values.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("task_accuracies must be a numeric vector at least window long")
    resolved = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(resolved)) or np.any((resolved < 0.0) | (resolved > 1.0)):
        raise ValueError("task_accuracies must be finite and in [0, 1]")
    return float(np.mean(resolved[-window:]))
