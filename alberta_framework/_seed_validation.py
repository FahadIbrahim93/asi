"""Canonical host-side seed validation for JAX benchmark boundaries.

JAX's default scalar key constructor consumes one unsigned 32-bit seed word.
Benchmark seed IDs therefore have to be validated before any ``int`` or
``uint32`` conversion; otherwise distinct caller values can execute as the
same random stream while being recorded as different identities.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

JAX_KEY_SEED_MAX: Final[int] = (1 << 32) - 1


def require_jax_seed(value: object, *, name: str = "seed") -> int:
    """Return one exact built-in integer in JAX's scalar-key seed domain."""

    if type(value) is not int:
        raise ValueError(
            f"{name} must be a built-in integer in [0, {JAX_KEY_SEED_MAX}] "
            "(the uint32 JAX key domain)"
        )
    seed = value
    if not 0 <= seed <= JAX_KEY_SEED_MAX:
        raise ValueError(
            f"{name} must be a built-in integer in [0, {JAX_KEY_SEED_MAX}] "
            "(the uint32 JAX key domain)"
        )
    return seed


def require_unique_jax_seeds(
    values: Sequence[object], *, name: str = "seeds"
) -> tuple[int, ...]:
    """Validate a non-empty, ordered sequence of unique canonical JAX seeds."""

    raw = tuple(values)
    if not raw:
        raise ValueError(f"{name} must be non-empty")
    seeds = tuple(
        require_jax_seed(value, name=f"{name}[{index}]")
        for index, value in enumerate(raw)
    )
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{name} must contain unique seeds")
    return seeds


__all__ = ["JAX_KEY_SEED_MAX", "require_jax_seed", "require_unique_jax_seeds"]
