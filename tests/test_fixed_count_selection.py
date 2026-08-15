"""Contracts for deterministic exact-count ranking shared by stream generators."""

import chex
import jax
import jax.numpy as jnp
import pytest

from alberta_framework._fixed_count_selection import stable_smallest_mask

pytestmark = pytest.mark.unit


def test_stable_smallest_mask_handles_zero_boundary_ties_and_full_capacity() -> None:
    scores = jnp.array(
        [
            [0.5, 0.5, 0.5, 0.5],
            [0.1, 0.3, 0.3, 0.2],
        ],
        dtype=jnp.float32,
    )
    compiled = jax.jit(stable_smallest_mask, static_argnums=1)

    chex.assert_trees_all_equal(
        compiled(scores, 0),
        jnp.zeros_like(scores, dtype=jnp.bool_),
    )
    chex.assert_trees_all_equal(
        compiled(scores, 2),
        jnp.array(
            [
                [True, True, False, False],
                [True, False, False, True],
            ]
        ),
    )
    chex.assert_trees_all_equal(
        compiled(scores, scores.shape[-1]),
        jnp.ones_like(scores, dtype=jnp.bool_),
    )


def test_stable_smallest_mask_matches_legacy_threshold_for_unique_finite_scores() -> None:
    scores = jnp.array(
        [
            [0.7, 0.1, 0.9, 0.4, 0.2],
            [0.3, 0.8, 0.6, 0.05, 0.95],
        ],
        dtype=jnp.float32,
    )
    count = 3
    threshold = jnp.sort(scores, axis=-1)[..., count - 1 : count]
    legacy_mask = scores <= threshold

    chex.assert_trees_all_equal(stable_smallest_mask(scores, count), legacy_mask)


def test_stable_smallest_mask_is_scan_compatible_and_deterministic() -> None:
    batches = jnp.array(
        [
            [[0.5, 0.5, 0.5], [0.2, 0.1, 0.1]],
            [[0.3, 0.3, 0.3], [0.4, 0.4, 0.2]],
        ],
        dtype=jnp.float32,
    )

    def body(carry, scores):
        mask = stable_smallest_mask(scores, 2)
        return carry + jnp.sum(mask), mask

    eager_count, eager_masks = jax.lax.scan(body, jnp.array(0), batches)
    compiled_count, compiled_masks = jax.jit(
        lambda values: jax.lax.scan(body, jnp.array(0), values)
    )(batches)

    assert int(eager_count) == 8
    chex.assert_trees_all_equal(compiled_count, eager_count)
    chex.assert_trees_all_equal(compiled_masks, eager_masks)


def test_stable_smallest_mask_deprioritizes_nonfinite_scores_but_stays_bounded() -> None:
    scores = jnp.array([jnp.nan, -jnp.inf, 0.2, jnp.inf, 0.1, jnp.nan])
    expected = jnp.array([False, False, True, False, True, False])

    chex.assert_trees_all_equal(stable_smallest_mask(scores, 2), expected)
    chex.assert_trees_all_equal(
        jax.jit(stable_smallest_mask, static_argnums=1)(scores, 2),
        expected,
    )


@pytest.mark.parametrize("count", [-1, True, 1.0, 4])
def test_stable_smallest_mask_rejects_invalid_static_counts(count: object) -> None:
    with pytest.raises(ValueError, match="count must be a built-in integer"):
        stable_smallest_mask(jnp.ones(3), count)  # type: ignore[arg-type]
