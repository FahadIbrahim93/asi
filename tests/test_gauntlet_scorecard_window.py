"""Host-boundary identities for gauntlet scorecard windows."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.streams.gauntlet import (
    GauntletConfig,
    early_window_mse,
    lifetime_scorecard,
    savings_ratio,
)


def _lifetime_errors(*, n_cycles: int = 2, segment_length: int = 8) -> jnp.ndarray:
    return jnp.arange(n_cycles * 4 * segment_length, dtype=jnp.float32)


def test_legal_windows_stay_accepted() -> None:
    errors = jnp.asarray([1.0, 3.0, 5.0, 7.0], dtype=jnp.float32)
    assert float(early_window_mse(errors, 0, 4, window=2)) == 2.0
    assert float(savings_ratio(errors, 0, 0, 4, window=2)) == 1.0
    config = GauntletConfig(segment_length=8, relevant_dim=2, irrelevant_dim=0)
    card = lifetime_scorecard(_lifetime_errors(), config, n_cycles=2, window=3)
    assert card["fresh_early"].shape == (2,)


@pytest.mark.parametrize("invalid", [True, False, np.bool_(True), 0, -1, 1.0, 9])
def test_early_window_mse_rejects_non_builtin_or_oob_window(invalid: object) -> None:
    errors = jnp.arange(8, dtype=jnp.float32)
    with pytest.raises(ValueError, match="window"):
        early_window_mse(errors, segment=0, segment_length=8, window=invalid)


@pytest.mark.parametrize("invalid", [True, False, np.bool_(True), 0, -1, 1.0, 9])
def test_savings_ratio_rejects_non_builtin_or_oob_window(invalid: object) -> None:
    errors = jnp.arange(16, dtype=jnp.float32)
    with pytest.raises(ValueError, match="window"):
        savings_ratio(errors, 0, 1, 8, window=invalid)


@pytest.mark.parametrize("invalid", [True, False, np.bool_(True), 0, -1, 1.0, 9])
def test_lifetime_scorecard_rejects_non_builtin_or_oob_window(invalid: object) -> None:
    config = GauntletConfig(segment_length=8, relevant_dim=2, irrelevant_dim=0)
    with pytest.raises(ValueError, match="window"):
        lifetime_scorecard(_lifetime_errors(), config, n_cycles=2, window=invalid)


@pytest.mark.parametrize("invalid", [True, False, np.bool_(True), 0, -1, 1.0])
def test_lifetime_scorecard_rejects_non_builtin_n_cycles(invalid: object) -> None:
    config = GauntletConfig(segment_length=8, relevant_dim=2, irrelevant_dim=0)
    with pytest.raises(ValueError, match="n_cycles"):
        lifetime_scorecard(_lifetime_errors(), config, n_cycles=invalid, window=2)
