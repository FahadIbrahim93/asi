"""Shared JAX helpers for fail-closed numerical update transactions."""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool


def safe_discrete_action(
    action: Array | int,
    n_actions: int,
    *,
    allow_unset: bool = False,
) -> tuple[Array, Bool[Array, ""]]:
    """Return a safe scalar action code and its exact discrete-domain verdict.

    Casting a floating action to ``int32`` before validation can turn ``NaN``,
    infinity, fractions, and out-of-range values into an innocuous all-zero
    one-hot vector.  Validate the floating scalar first, then expose only a
    safe code to downstream one-hot arithmetic.  ``allow_unset`` admits the
    conventional ``-1`` episode-start sentinel.
    """

    if n_actions < 0:
        raise ValueError("n_actions must be non-negative")
    if n_actions == 0:
        return (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(True, dtype=jnp.bool_),
        )
    raw = jnp.asarray(action, dtype=jnp.float32).reshape(())
    lower = -1.0 if allow_unset else 0.0
    valid = (
        jnp.isfinite(raw)
        & (raw == jnp.floor(raw))
        & (raw >= lower)
        & (raw < float(n_actions))
    )
    fallback = -1 if allow_unset else 0
    safe = jnp.where(valid, raw, jnp.asarray(fallback, dtype=jnp.float32))
    return safe.astype(jnp.int32), valid


def floating_tree_is_finite(tree: object) -> Bool[Array, ""]:
    """Return whether every floating or complex leaf in ``tree`` is finite."""

    valid = jnp.asarray(True, dtype=jnp.bool_)
    for leaf in jax.tree.leaves(tree):
        array = jnp.asarray(leaf)
        if jnp.issubdtype(array.dtype, jnp.inexact):
            valid = valid & jnp.all(jnp.isfinite(array))
    return valid


def select_transaction[T](applied: Array, candidate: T, current: T) -> T:
    """Commit ``candidate`` iff a scalar JAX transaction predicate is true."""

    return cast(
        T,
        jax.lax.cond(
            applied,
            lambda: candidate,
            lambda: current,
        ),
    )


def neutralize_array(applied: Array, candidate: Array) -> Array:
    """Return a finite zero diagnostic/update for a rejected transaction."""

    return jnp.where(applied, candidate, jnp.zeros_like(candidate))


def neutralize_metrics(
    applied: Array,
    metrics: dict[str, Array],
) -> dict[str, Array]:
    """Neutralize every metric while preserving the static dictionary shape."""

    return {name: neutralize_array(applied, value) for name, value in metrics.items()}


def zero_if_collapsed_infinity(
    product: Array,
    infinite_input: Array,
    collapsed: Array,
) -> Array:
    """Replace only a bound-induced ``0 * inf`` NaN with exact zero.

    An unrelated NaN remains visible.  The repair applies only when the
    original input was infinite and the bound/clip scale actually collapsed.
    """

    return jnp.where(
        jnp.isnan(product) & jnp.isinf(infinite_input) & collapsed,
        jnp.zeros_like(product),
        product,
    )
