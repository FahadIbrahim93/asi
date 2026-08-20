from __future__ import annotations

import jax.numpy as jnp
import pytest

from alberta_framework._scan_resources import (
    ScanBudget,
    require_jax_leading_length,
    require_matching_jax_leading_length,
    require_parallel_count,
    require_scan_steps,
    require_step_units,
)


class _HostileArray:
    calls = 0

    @property
    def shape(self) -> tuple[int, ...]:
        type(self).calls += 1
        raise AssertionError("hostile shape hook ran")


def test_budget_rejects_hostile_and_oversized_axis_identities_without_coercion() -> None:
    budget = ScanBudget("test", maximum_steps=4, maximum_parallel=2, maximum_step_units=6)
    for value in (True, 1.0, "1", 5):
        with pytest.raises(ValueError):
            require_scan_steps("steps", value, budget)
    for value in (True, 3):
        with pytest.raises(ValueError):
            require_parallel_count("workers", value, budget)
    assert require_scan_steps("steps", 3, budget) == 3
    assert require_parallel_count("workers", 2, budget) == 2
    require_step_units(3, 2, budget)
    with pytest.raises(ValueError, match="step-units"):
        require_step_units(4, 2, budget)


def test_array_gate_checks_identity_before_shape_metadata() -> None:
    budget = ScanBudget("test", maximum_steps=4)
    hostile = _HostileArray()
    _HostileArray.calls = 0
    with pytest.raises(TypeError, match="JAX array"):
        require_jax_leading_length("values", hostile, budget)
    assert _HostileArray.calls == 0


def test_array_gate_binds_rank_budget_and_matching_length() -> None:
    budget = ScanBudget("test", maximum_steps=4)
    primary = jnp.zeros((3, 2), dtype=jnp.float32)
    matching = jnp.ones((3,), dtype=jnp.float32)
    assert require_jax_leading_length("primary", primary, budget, ranks=(2,)) == 3
    require_matching_jax_leading_length("matching", matching, expected=3)
    with pytest.raises(ValueError, match="primary sequence length"):
        require_matching_jax_leading_length(
            "mismatch", jnp.zeros((2,), dtype=jnp.float32), expected=3
        )
