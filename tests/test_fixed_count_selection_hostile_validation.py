"""Hostile-safe validation for fixed-count selection."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework._fixed_count_selection import (
    require_positive_builtin_int,
    stable_smallest_mask,
)


class _HostileInt(int):
    def __index__(self) -> int:  # pragma: no cover
        raise AssertionError("index hook executed")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("repr hook")


class _StringSubclass(str):
    pass


def test_require_positive_rejects_bool() -> None:
    with pytest.raises(ValueError, match="positive built-in integer"):
        require_positive_builtin_int(True, name="n")
    with pytest.raises(ValueError, match="positive built-in integer"):
        require_positive_builtin_int(False, name="n")


def test_require_positive_rejects_hostile_int() -> None:
    with pytest.raises(ValueError, match="positive built-in integer"):
        require_positive_builtin_int(_HostileInt(1), name="n")


def test_require_positive_rejects_string_subclass_name() -> None:
    with pytest.raises(ValueError, match="exact string"):
        require_positive_builtin_int(1, name=_StringSubclass("n"))


def test_require_positive_does_not_invoke_hostile_name_repr() -> None:
    class EvilStr(str):
        def __repr__(self) -> str:  # pragma: no cover
            raise RuntimeError("repr hook")

        def __str__(self) -> str:  # pragma: no cover
            raise RuntimeError("str hook")

    with pytest.raises(ValueError, match="exact string"):
        require_positive_builtin_int(1, name=EvilStr("n"))


def test_require_positive_rejects_zero_and_negative() -> None:
    with pytest.raises(ValueError, match="positive built-in integer"):
        require_positive_builtin_int(0, name="n")
    with pytest.raises(ValueError, match="positive built-in integer"):
        require_positive_builtin_int(-1, name="n")


def test_require_positive_valid() -> None:
    assert require_positive_builtin_int(3, name="n") == 3


@pytest.mark.parametrize("value", [np.int64(1), np.uint32(1), 1.0, "1", 2**31])
def test_require_positive_rejects_nonbuiltin_or_out_of_int32_values(value: object) -> None:
    with pytest.raises(ValueError, match="positive built-in integer"):
        require_positive_builtin_int(value, name="n")


def test_require_positive_accepts_exact_int32_endpoint() -> None:
    assert require_positive_builtin_int(2**31 - 1, name="n") == 2**31 - 1


def test_stable_smallest_rejects_bool_count() -> None:
    scores = jnp.array([[0.1, 0.2, 0.3]])
    with pytest.raises(ValueError, match="built-in integer"):
        stable_smallest_mask(scores, True)
    with pytest.raises(ValueError, match="built-in integer"):
        stable_smallest_mask(scores, False)


def test_stable_smallest_rejects_hostile_int_count() -> None:
    scores = jnp.array([[0.1, 0.2, 0.3]])
    with pytest.raises(ValueError, match="built-in integer"):
        stable_smallest_mask(scores, _HostileInt(1))


def test_stable_smallest_rejects_out_of_bounds() -> None:
    scores = jnp.array([[0.1, 0.2]])
    with pytest.raises(ValueError, match="built-in integer"):
        stable_smallest_mask(scores, 5)
    with pytest.raises(ValueError, match="built-in integer"):
        stable_smallest_mask(scores, -1)


def test_stable_smallest_valid_zero_and_full() -> None:
    scores = jnp.array([[0.3, 0.1, 0.2]])
    mask0 = stable_smallest_mask(scores, 0)
    assert bool(jnp.all(mask0 == jnp.array([[False, False, False]])))
    mask2 = stable_smallest_mask(scores, 2)
    # smallest two are 0.1, 0.2 at indices 1,2
    assert bool(jnp.all(mask2 == jnp.array([[False, True, True]])))


def test_stable_smallest_tie_break_by_index() -> None:
    scores = jnp.array([[1.0, 1.0, 1.0]])
    mask = stable_smallest_mask(scores, 2)
    assert bool(jnp.all(mask == jnp.array([[True, True, False]])))


def test_stable_smallest_does_not_invoke_hostile_name_repr_via_require() -> None:
    class EvilStr(str):
        def __repr__(self) -> str:  # pragma: no cover
            raise RuntimeError("repr hook")

    with pytest.raises(ValueError, match="exact string"):
        require_positive_builtin_int(1, name=EvilStr("count"))


def test_stable_smallest_rejects_hostile_scores_before_attribute_access() -> None:
    class HostileScores:
        def __getattribute__(self, name: str) -> object:  # pragma: no cover
            raise AssertionError(f"attribute hook executed: {name}")

    with pytest.raises(ValueError, match="JAX array"):
        stable_smallest_mask(HostileScores(), 0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "scores",
    [
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray([1, 2], dtype=jnp.int32),
        jnp.asarray([True, False], dtype=jnp.bool_),
        jnp.asarray([1.0 + 1.0j], dtype=jnp.complex64),
    ],
)
def test_stable_smallest_rejects_missing_axis_or_nonfloating_scores(scores: object) -> None:
    with pytest.raises(ValueError, match="candidate axis|floating dtype"):
        stable_smallest_mask(scores, 0)  # type: ignore[arg-type]


def test_stable_smallest_output_formula_is_exact_across_leading_axes() -> None:
    scores = jnp.asarray(
        [
            [[0.3, 0.1, 0.2], [0.5, 0.4, 0.6]],
            [[0.0, -1.0, 2.0], [3.0, 1.0, 2.0]],
        ],
        dtype=jnp.float32,
    )
    mask = stable_smallest_mask(scores, 2)
    assert mask.shape == scores.shape
    assert mask.dtype == jnp.bool_
    assert bool(jnp.all(jnp.sum(mask, axis=-1) == 2))
