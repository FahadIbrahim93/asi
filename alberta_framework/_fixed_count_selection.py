"""Deterministic fixed-count selection shared by synthetic stream generators."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def require_positive_builtin_int(value: object, *, name: str) -> int:
    """Return an exact built-in positive integer or fail before JAX tracing."""

    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive built-in integer")
    return value


def stable_smallest_mask(scores: Array, count: int) -> Array:
    """Select exactly ``count`` entries per row, breaking ties by source index.

    Selection runs along the last axis. Non-finite scores are invalid for the
    uniform-score callers, but mapping them to one shared trailing sort key keeps
    this primitive bounded and deterministic if an upstream generator is
    corrupted. ``count`` is static host configuration and may be zero here;
    public stream constructors impose their stricter positive-count contract.
    """

    if type(count) is not int or not 0 <= count <= scores.shape[-1]:
        raise ValueError("count must be a built-in integer within the candidate axis")
    finite_scores = jnp.where(jnp.isfinite(scores), scores, jnp.inf)
    order = jnp.argsort(finite_scores, axis=-1, stable=True)
    ranks = jnp.argsort(order, axis=-1, stable=True)
    return ranks < count


__all__ = ["require_positive_builtin_int", "stable_smallest_mask"]
