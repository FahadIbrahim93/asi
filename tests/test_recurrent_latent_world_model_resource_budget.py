"""Leftover-identity gates for recurrent-latent resource-budget records."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from alberta_framework.core.recurrent_latent_world_model_ensemble import (
    RecurrentLatentWorldModelResourceBudget,
)


def _legal_budget() -> RecurrentLatentWorldModelResourceBudget:
    return RecurrentLatentWorldModelResourceBudget(
        ensemble_size=2,
        observation_dim=3,
        latent_dim=4,
        target_dim=5,
        trainable_scalars_per_member=6,
        total_trainable_scalars=12,
        persistent_float32_scalars=8,
        persistent_int32_scalars=4,
        persistent_uint32_scalars=4,
        persistent_bool_scalars=2,
        persistent_state_scalars=18,
        persistent_state_bytes=72,
        bootstrap_prng_keys=1,
        bootstrap_prng_uint32_scalars=2,
        start_cache_logical_scalars=3,
        start_cache_logical_bytes=12,
        decision_cache_logical_scalars=4,
        decision_cache_logical_bytes=16,
        update_result_logical_scalars=10,
        update_result_logical_bytes=40,
        member_gradient_candidates_per_event=2,
        max_member_parameter_updates_per_event=2,
        recurrent_advances_per_accepted_event=1,
        max_event_count=100,
        max_member_update_count=100,
        replay_capacity=0,
    )


def test_recurrent_latent_resource_budget_rejects_leftover_identities() -> None:
    """Public resource-budget records must not keep leftover bool/int identities."""

    with pytest.raises(ValueError, match="ensemble_size"):
        replace(_legal_budget(), ensemble_size=True)
    with pytest.raises(ValueError, match="replay_capacity"):
        replace(_legal_budget(), replay_capacity=True)
    with pytest.raises(ValueError, match="persistent_state_bytes"):
        replace(_legal_budget(), persistent_state_bytes=float("nan"))

    legal = _legal_budget()
    dumped = json.dumps(legal.to_config(), allow_nan=False)
    assert '"ensemble_size": 2' in dumped
    assert '"replay_capacity": 0' in dumped
    assert '"persistent_state_bytes": 72' in dumped
    assert '"ensemble_size": true' not in dumped
    assert '"replay_capacity": true' not in dumped
    assert '"persistent_state_bytes": true' not in dumped
