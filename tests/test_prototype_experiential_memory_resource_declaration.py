"""Leftover-identity gates for prototype experiential-memory resource declarations."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from alberta_framework.core.prototype_agent import (
    PrototypeExperientialMemoryResourceDeclaration,
)


def _legal_declaration() -> PrototypeExperientialMemoryResourceDeclaration:
    return PrototypeExperientialMemoryResourceDeclaration(
        persistent_state_bytes=16,
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
        replace(_legal_declaration(), score_mass_values_interpreted=float("nan"))

    legal = _legal_declaration()
    dumped = json.dumps(legal.to_config(), allow_nan=False)
    assert '"persistent_state_bytes": 16' in dumped
    assert '"random_draws": 0' in dumped
    assert '"score_mass_values_interpreted": 3' in dumped
    assert '"persistent_state_bytes": true' not in dumped
    assert '"random_draws": true' not in dumped
    assert '"score_mass_values_interpreted": true' not in dumped
