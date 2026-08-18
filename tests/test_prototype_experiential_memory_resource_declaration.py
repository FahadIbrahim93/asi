"""Leftover-identity gates for prototype experiential-memory resource declarations."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, cast

import pytest

from alberta_framework.core.prototype_agent import (
    PrototypeExperientialMemoryResourceDeclaration,
)


def _legal_declaration() -> PrototypeExperientialMemoryResourceDeclaration:
    return PrototypeExperientialMemoryResourceDeclaration(
        memory_capacity=1,
        memory_observation_dim=1,
        memory_key_dim=1,
        memory_action_dim=3,
        memory_outcome_dim=2,
        persistent_state_bytes=108,
        categorical_policy_queries=1,
        causal_step_queries=0,
        total_deterministic_prestate_queries=1,
        writes_attempted=1,
        random_draws=0,
        score_mass_values_interpreted=3,
        hard_safety_values_interpreted=3,
    )


def test_prototype_experiential_memory_resource_declaration_rejects_leftover_identities() -> None:
    """Public resource declarations must not keep leftover bool/int identities."""

    with pytest.raises(ValueError, match="persistent_state_bytes"):
        replace(_legal_declaration(), persistent_state_bytes=True)
    with pytest.raises(ValueError, match="random_draws"):
        replace(_legal_declaration(), random_draws=True)
    with pytest.raises(ValueError, match="score_mass_values_interpreted"):
        replace(
            _legal_declaration(),
            score_mass_values_interpreted=cast(Any, float("nan")),
        )

    legal = _legal_declaration()
    dumped = json.dumps(legal.to_config(), allow_nan=False)
    assert '"persistent_state_bytes": 108' in dumped
    assert '"random_draws": 0' in dumped
    assert '"score_mass_values_interpreted": 3' in dumped
    assert '"persistent_state_bytes": true' not in dumped
    assert '"random_draws": true' not in dumped
    assert '"score_mass_values_interpreted": true' not in dumped


@pytest.mark.parametrize(
    "field",
    [
        "persistent_state_bytes",
        "categorical_policy_queries",
        "causal_step_queries",
        "total_deterministic_prestate_queries",
        "writes_attempted",
        "random_draws",
        "score_mass_values_interpreted",
        "hard_safety_values_interpreted",
    ],
)
def test_resource_declaration_rejects_mutated_derived_identities(field: str) -> None:
    declaration = _legal_declaration()
    with pytest.raises(ValueError, match=field):
        replace(declaration, **{field: getattr(declaration, field) + 1})


def test_resource_declaration_rejects_subclass_before_attribute_hooks() -> None:
    class HostileDeclaration(PrototypeExperientialMemoryResourceDeclaration):
        calls = 0

        def __getattribute__(self, name: str) -> Any:
            type(self).calls += 1
            raise AssertionError(f"attribute hook must not run: {name}")

    value = object.__new__(HostileDeclaration)
    HostileDeclaration.calls = 0
    with pytest.raises(TypeError, match="exact PrototypeExperientialMemoryResourceDeclaration"):
        PrototypeExperientialMemoryResourceDeclaration.__post_init__(cast(Any, value))
    assert HostileDeclaration.calls == 0
