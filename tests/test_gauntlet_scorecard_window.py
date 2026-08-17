"""Tests for gauntlet scorecard and window scalar validation."""

import jax.numpy as jnp
import pytest

from alberta_framework.streams.gauntlet import (
    GauntletConfig,
    early_window_mse,
    lifetime_scorecard,
    savings_ratio,
    segment_slice,
)


def test_early_window_mse_and_savings_ratio_reject_boolean_and_invalid_window() -> None:
    errors = jnp.ones((400,), dtype=jnp.float32)

    # Boolean window
    with pytest.raises(ValueError, match="window"):
        early_window_mse(errors, segment=0, segment_length=100, window=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="window"):
        early_window_mse(errors, segment=0, segment_length=100, window=False)  # type: ignore[arg-type]

    # Non-integer / out-of-range window
    with pytest.raises(ValueError, match="window"):
        early_window_mse(errors, segment=0, segment_length=100, window=0)

    with pytest.raises(ValueError, match="window"):
        early_window_mse(errors, segment=0, segment_length=100, window=101)

    # Invalid segment / segment_length
    with pytest.raises(ValueError, match="segment"):
        early_window_mse(errors, segment=True, segment_length=100, window=50)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="segment"):
        early_window_mse(errors, segment=-1, segment_length=100, window=50)

    with pytest.raises(ValueError, match="segment_length"):
        early_window_mse(errors, segment=0, segment_length=0, window=50)

    # savings_ratio propagates window validation
    with pytest.raises(ValueError, match="window"):
        savings_ratio(errors, first_segment=0, revisit_segment=2, segment_length=100, window=True)  # type: ignore[arg-type]

    # Valid evaluation
    res = early_window_mse(errors, segment=0, segment_length=100, window=50)
    assert float(res) == pytest.approx(1.0)


def test_lifetime_scorecard_rejects_boolean_and_invalid_arguments() -> None:
    config = GauntletConfig(segment_length=50)
    errors = jnp.ones((2, 400), dtype=jnp.float32)

    # Boolean / invalid n_cycles
    with pytest.raises(ValueError, match="n_cycles"):
        lifetime_scorecard(errors, config, n_cycles=True, window=20)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="n_cycles"):
        lifetime_scorecard(errors, config, n_cycles=0, window=20)

    # Boolean / invalid window
    with pytest.raises(ValueError, match="window"):
        lifetime_scorecard(errors, config, n_cycles=2, window=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="window"):
        lifetime_scorecard(errors, config, n_cycles=2, window=0)

    with pytest.raises(ValueError, match="window"):
        lifetime_scorecard(errors, config, n_cycles=2, window=51)

    card = lifetime_scorecard(errors, config, n_cycles=2, window=20)
    assert "fresh_early" in card
    assert "savings_c" in card


def test_segment_slice_rejects_boolean_and_negative_indices() -> None:
    errors = jnp.ones((100,), dtype=jnp.float32)
    with pytest.raises(ValueError, match="segment"):
        segment_slice(errors, segment=True, segment_length=50)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="segment_length"):
        segment_slice(errors, segment=0, segment_length=False)  # type: ignore[arg-type]
