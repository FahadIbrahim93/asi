"""Deterministic fixed-count selection shared by synthetic stream generators."""

from __future__ import annotations

from typing import Final

import jax
import jax.numpy as jnp
from jax import Array

_INT32_MAX: Final[int] = (1 << 31) - 1


def _require_exact_str(name: str, value: object) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be an exact string")
    return value


def require_positive_builtin_int(value: object, *, name: str) -> int:
    """Return an exact positive built-in integer in the JAX int32 domain."""

    _require_exact_str("name", name)
    if type(value) is not int or not 1 <= value <= _INT32_MAX:
        raise ValueError(
            f"{name} must be a positive built-in integer at most int32 max ({_INT32_MAX})"
        )
    return value


def stable_smallest_mask(scores: Array, count: int) -> Array:
    """Select exactly ``count`` entries per row, breaking ties by source index.

    Selection runs along the last axis. Non-finite scores are invalid for the
    uniform-score callers, but mapping them to one shared trailing sort key keeps
    this primitive bounded and deterministic if an upstream generator is
    corrupted. ``count`` is static host configuration and may be zero here;
    public stream constructors impose their stricter positive-count contract.
    """

    if type(count) is not int or count < 0:
        raise ValueError("count must be a built-in integer within the candidate axis")
    score_type = type(scores)
    if not (
        issubclass(score_type, jax.Array)
        or issubclass(score_type, jax.core.Tracer)
    ):
        raise ValueError("scores must be a JAX array")
    if scores.ndim < 1:
        raise ValueError("scores must have a candidate axis")
    if not jnp.issubdtype(scores.dtype, jnp.floating):
        raise ValueError("scores must have a floating dtype")
    if count > scores.shape[-1]:
        raise ValueError("count must be a built-in integer within the candidate axis")
    finite_scores = jnp.where(jnp.isfinite(scores), scores, jnp.inf)
    order = jnp.argsort(finite_scores, axis=-1, stable=True)
    ranks = jnp.argsort(order, axis=-1, stable=True)
    return ranks < count


__all__ = ["require_positive_builtin_int", "stable_smallest_mask"]
