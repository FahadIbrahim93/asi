"""Canonical host-side seed validation for JAX benchmark boundaries.

JAX's default scalar key constructor consumes one unsigned 32-bit seed word.
Benchmark seed IDs therefore have to be validated before any ``int`` or
``uint32`` conversion; otherwise distinct caller values can execute as the
same random stream while being recorded as different identities.
"""

from __future__ import annotations

from typing import Final, cast

JAX_KEY_SEED_MAX: Final[int] = (1 << 32) - 1
JAX_SEED_SEQUENCE_MAX_LENGTH: Final[int] = 4_096


def _require_exact_str(name: str, value: object) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be an exact string")
    return value


def require_jax_seed(value: object, *, name: str = "seed") -> int:
    """Return one exact built-in integer in JAX's scalar-key seed domain."""

    _require_exact_str("name", name)
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
    values: object, *, name: str = "seeds"
) -> tuple[int, ...]:
    """Validate one bounded ordered list/tuple of unique canonical JAX seeds."""

    _require_exact_str("name", name)
    if type(values) not in (list, tuple):
        raise ValueError(f"{name} must be an exact list or tuple of seeds")
    raw = cast(list[object] | tuple[object, ...], values)
    count = len(raw)
    if count == 0:
        raise ValueError(f"{name} must be non-empty")
    if count > JAX_SEED_SEQUENCE_MAX_LENGTH:
        raise ValueError(
            f"{name} must contain at most {JAX_SEED_SEQUENCE_MAX_LENGTH} seeds"
        )

    seen: set[int] = set()
    checked: list[int] = []
    for index in range(count):
        value = raw[index]
        if type(value) is not int or not 0 <= value <= JAX_KEY_SEED_MAX:
            raise ValueError(
                f"{name}[{index}] must be a built-in integer in "
                f"[0, {JAX_KEY_SEED_MAX}] (the uint32 JAX key domain)"
            )
        seed = value
        if seed in seen:
            raise ValueError(f"{name} must contain unique seeds")
        seen.add(seed)
        checked.append(seed)

    if type(raw) is tuple:
        return cast(tuple[int, ...], raw)
    return tuple(checked)


__all__ = [
    "JAX_KEY_SEED_MAX",
    "JAX_SEED_SEQUENCE_MAX_LENGTH",
    "require_jax_seed",
    "require_unique_jax_seeds",
]
