"""Regression: floor_and_renormalize_probabilities must return a simplex.

For zero-mass and float32-underflowing inputs the floored normalizer yields an
all-zero ``normalized`` vector, so the affine step returned ``min_probability``
in every slot (summing to ``n * min_probability``) instead of a valid simplex.
Issue #2238.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from alberta_framework.core.behavior_model import floor_and_renormalize_probabilities

pytestmark = pytest.mark.unit


def test_zero_mass_returns_simplex() -> None:
    out = floor_and_renormalize_probabilities(jnp.asarray([0.0, 0.0, 0.0]))
    assert float(out.sum()) == pytest.approx(1.0, abs=1e-6)
    # Degenerate mass falls back to uniform.
    assert jnp.allclose(out, jnp.ones(3) / 3, atol=1e-6)


def test_float32_underflow_returns_simplex() -> None:
    out = floor_and_renormalize_probabilities(jnp.asarray([1e38, 1.0, 1.0]))
    assert float(out.sum()) == pytest.approx(1.0, abs=1e-6)


def test_well_formed_input_unchanged() -> None:
    probs = jnp.asarray([0.2, 0.3, 0.5])
    out = floor_and_renormalize_probabilities(probs)
    assert float(out.sum()) == pytest.approx(1.0, abs=1e-6)
    assert float(jnp.min(out)) >= 1e-6 - 1e-9
    # Ordering preserved.
    assert float(out[2]) > float(out[1]) > float(out[0])


def test_negative_containing_input_still_simplex() -> None:
    out = floor_and_renormalize_probabilities(jnp.asarray([-0.1, 0.4, 0.7]))
    assert float(out.sum()) == pytest.approx(1.0, abs=1e-6)


def test_single_action_stays_one() -> None:
    out = floor_and_renormalize_probabilities(jnp.asarray([1.0]))
    assert float(out.sum()) == pytest.approx(1.0, abs=1e-6)


def test_already_floored_stays_simplex() -> None:
    out = floor_and_renormalize_probabilities(jnp.asarray([1e-6, 1e-6, 1e-6]))
    assert float(out.sum()) == pytest.approx(1.0, abs=1e-6)
