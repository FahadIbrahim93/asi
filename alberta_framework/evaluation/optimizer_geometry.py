"""Small-matrix primitives for staging continual optimizer geometry controls."""

from __future__ import annotations

from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

GEOMETRY_PROTOCOL = MappingProxyType(
    {
        "schema": "asi.optimizer-geometry.protocol.v1",
        "paper_revisions": (
            "arXiv:2605.08949v2",
            "arXiv:2606.10406v1",
            "arXiv:2601.07636v1",
        ),
        "stage": "small_streaming_matrix_pre_ipmnist",
        "protocol_difference": "equation primitives only; no LLM or batch-CL claim",
        "mechanism_off": "empty_basis_or_zero_gradient_exact_reduction",
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

_MAX_MATRIX_ELEMENTS = 1_000_000


def _trusted_array(value: object, *, name: str) -> Array:
    actual_type = type(value)
    if not (actual_type is np.ndarray or issubclass(actual_type, (jax.Array, jax.core.Tracer))):
        raise ValueError(f"{name} must be an exact NumPy or JAX array")
    result = jnp.asarray(value)
    if result.size > _MAX_MATRIX_ELEMENTS or not jnp.issubdtype(result.dtype, jnp.floating):
        raise ValueError(f"{name} must be a bounded floating array")
    return result


def orthogonal_correction_transaction(update: Array, protected_basis: Array) -> tuple[Array, Array]:
    """Return a finite orthogonal correction and caller-visible validity bit."""
    vector = _trusted_array(update, name="update")
    basis = _trusted_array(protected_basis, name="protected_basis")
    if vector.ndim != 1 or basis.ndim != 2 or basis.shape[1] != vector.shape[0]:
        raise ValueError("update must be a vector and basis rows must match its width")
    coordinates = basis @ vector
    projection = basis.T @ coordinates
    candidate = vector - projection
    valid = (
        jnp.all(jnp.isfinite(vector))
        & jnp.all(jnp.isfinite(basis))
        & jnp.all(jnp.isfinite(coordinates))
        & jnp.all(jnp.isfinite(projection))
        & jnp.all(jnp.isfinite(candidate))
    )
    safe = jnp.where(valid, candidate, jnp.zeros_like(candidate))
    return safe, valid


def _unwrap_transaction(result: tuple[Array, Array], *, name: str) -> Array:
    safe, valid = result
    if not isinstance(valid, jax.core.Tracer) and not bool(valid):
        raise ValueError(f"{name} must be finite")
    return safe


def orthogonal_correction(update: Array, protected_basis: Array) -> Array:
    """Project a vector away from row-wise orthonormal protected directions."""
    return _unwrap_transaction(
        orthogonal_correction_transaction(update, protected_basis),
        name="orthogonal correction",
    )


def spectral_matrix_sign_transaction(matrix: Array, *, steps: int = 5) -> tuple[Array, Array]:
    """Return a finite matrix-sign approximation and caller-visible validity bit."""
    value = _trusted_array(matrix, name="matrix")
    if (
        value.ndim != 2
        or value.size == 0
        or not jnp.issubdtype(value.dtype, jnp.floating)
        or type(steps) is not int
        or steps < 1
        or steps > 32
    ):
        raise ValueError("matrix must be non-empty and steps a positive integer")
    norm = jnp.linalg.norm(value)
    valid = jnp.all(jnp.isfinite(value)) & jnp.isfinite(norm)
    x = value / jnp.maximum(norm, jnp.asarray(1e-12, dtype=value.dtype))
    if x.shape[0] > x.shape[1]:
        x = x.T
        transposed = True
    else:
        transposed = False
    for _ in range(steps):
        a = x @ x.T
        next_x = 3.4445 * x - 4.7750 * a @ x + 2.0315 * a @ a @ x
        valid = valid & jnp.all(jnp.isfinite(a)) & jnp.all(jnp.isfinite(next_x))
        x = next_x
    candidate = x.T if transposed else x
    valid = valid & jnp.all(jnp.isfinite(candidate))
    safe = jnp.where(valid, candidate, jnp.zeros_like(candidate))
    return safe, valid


def spectral_matrix_sign(matrix: Array, *, steps: int = 5) -> Array:
    """Muon-style Newton--Schulz matrix-sign approximation for a small matrix."""
    return _unwrap_transaction(
        spectral_matrix_sign_transaction(matrix, steps=steps), name="matrix sign"
    )


def flad_noise_component_transaction(perturbation: Array, gradient: Array) -> tuple[Array, Array]:
    """Return a finite FLAD noise component and caller-visible validity bit."""
    delta = _trusted_array(perturbation, name="perturbation")
    direction = _trusted_array(gradient, name="gradient")
    if delta.shape != direction.shape or delta.ndim != 1 or delta.size < 1:
        raise ValueError("perturbation and gradient must be non-empty equal-width vectors")
    squared_norm = jnp.vdot(direction, direction).real
    numerator = jnp.vdot(direction, delta).real
    denominator = jnp.where(squared_norm > 0.0, squared_norm, 1.0)
    coefficient = numerator / denominator
    projection = jnp.where(squared_norm > 0.0, direction * coefficient, jnp.zeros_like(delta))
    candidate = delta - projection
    valid = (
        jnp.all(jnp.isfinite(delta))
        & jnp.all(jnp.isfinite(direction))
        & jnp.isfinite(squared_norm)
        & jnp.isfinite(numerator)
        & jnp.isfinite(coefficient)
        & jnp.all(jnp.isfinite(projection))
        & jnp.all(jnp.isfinite(candidate))
    )
    safe = jnp.where(valid, candidate, jnp.zeros_like(candidate))
    return safe, valid


def flad_noise_component(perturbation: Array, gradient: Array) -> Array:
    """Remove FLAD's gradient-aligned perturbation component."""
    return _unwrap_transaction(
        flad_noise_component_transaction(perturbation, gradient),
        name="FLAD decomposition",
    )
