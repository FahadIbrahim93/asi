"""Hostile integer gate for gauntlet early_window_mse."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from alberta_framework.streams.gauntlet import early_window_mse

pytestmark = pytest.mark.unit


class _HostileInt(int):
    calls = 0

    def __repr__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile repr must not run")

    def __str__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile str must not run")

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile eq must not run")

    def __lt__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile lt must not run")

    def __le__(self, other: object) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile le must not run")

    def __bool__(self) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile bool must not run")

    def __int__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile int must not run")

    def __hash__(self) -> int:  # type: ignore[override]
        return int.__hash__(self)


def test_early_window_mse_rejects_hostile_before_repr() -> None:
    errors = jnp.ones((10,), dtype=jnp.float32)
    hostile = _HostileInt(5)
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="window must be an integer"):
        early_window_mse(errors, segment=0, segment_length=10, window=hostile)  # type: ignore[arg-type]
    assert _HostileInt.calls == 0

    # bool is subclass of int but must be rejected
    with pytest.raises(ValueError, match="window must be an integer"):
        early_window_mse(errors, segment=0, segment_length=10, window=True)  # type: ignore[arg-type]

    # valid still works
    result = early_window_mse(errors, segment=0, segment_length=10, window=5)
    assert float(result) == pytest.approx(1.0)

    # out of range without hostile still generic message
    with pytest.raises(ValueError, match="window must be an integer in"):
        early_window_mse(errors, segment=0, segment_length=10, window=20)
