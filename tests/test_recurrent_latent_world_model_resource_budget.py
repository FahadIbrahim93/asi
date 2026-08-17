"""Leftover-identity gates for recurrent-latent resource-budget records."""

from __future__ import annotations

import json
from dataclasses import fields, replace

import numpy as np
import pytest

from alberta_framework.core.recurrent_latent_world_model_ensemble import (
    RecurrentLatentWorldModelEnsemble,
    RecurrentLatentWorldModelEnsembleConfig,
    RecurrentLatentWorldModelResourceBudget,
)


class _HostileInt(int):
    def __index__(self) -> int:
        raise AssertionError("hostile index hook executed")


def _legal_budget() -> RecurrentLatentWorldModelResourceBudget:
    return RecurrentLatentWorldModelResourceBudget(
        ensemble_size=2,
        observation_dim=2,
        latent_dim=3,
        target_dim=4,
        trainable_scalars_per_member=104,
        total_trainable_scalars=208,
        persistent_float32_scalars=214,
        persistent_int32_scalars=5,
        persistent_uint32_scalars=2,
        persistent_bool_scalars=2,
        persistent_state_scalars=223,
        persistent_state_bytes=886,
        bootstrap_prng_keys=1,
        bootstrap_prng_uint32_scalars=2,
        start_cache_logical_scalars=10,
        start_cache_logical_bytes=37,
        decision_cache_logical_scalars=88,
        decision_cache_logical_bytes=334,
        update_result_logical_scalars=344,
        update_result_logical_bytes=1280,
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
    assert '"persistent_state_bytes": 886' in dumped
    assert '"ensemble_size": true' not in dumped
    assert '"replay_capacity": true' not in dumped
    assert '"persistent_state_bytes": true' not in dumped


@pytest.mark.parametrize(
    "field",
    [
        field.name
        for field in fields(RecurrentLatentWorldModelResourceBudget)
        if field.name
        not in {
            "ensemble_size",
            "observation_dim",
            "latent_dim",
            "trainable_scalars_per_member",
            "max_event_count",
        }
    ],
)
def test_recurrent_latent_budget_rejects_every_derived_field_mutation(field: str) -> None:
    budget = _legal_budget()
    with pytest.raises(ValueError, match=field):
        replace(budget, **{field: getattr(budget, field) + 1})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ensemble_size", 1),
        ("observation_dim", 0),
        ("latent_dim", 0),
        ("trainable_scalars_per_member", 0),
        ("max_event_count", 0),
    ],
)
def test_recurrent_latent_budget_requires_positive_provenance(
    field: str, value: int
) -> None:
    with pytest.raises(ValueError, match=field):
        replace(_legal_budget(), **{field: value})


@pytest.mark.parametrize("field", ["ensemble_size", "observation_dim", "latent_dim"])
def test_recurrent_latent_budget_binds_provenance_dimensions(field: str) -> None:
    budget = _legal_budget()
    with pytest.raises(ValueError):
        replace(budget, **{field: getattr(budget, field) + 1})


def test_recurrent_latent_budget_requires_attainable_action_dimension() -> None:
    with pytest.raises(ValueError, match="positive action dimension"):
        replace(_legal_budget(), trainable_scalars_per_member=103)


@pytest.mark.parametrize("value", [_HostileInt(2), 2**31, -1])
def test_recurrent_latent_budget_rejects_hostile_wide_or_negative_dimensions(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="ensemble_size"):
        replace(_legal_budget(), ensemble_size=value)  # type: ignore[arg-type]


def test_recurrent_latent_budget_canonicalizes_numpy_integer_dimensions() -> None:
    budget = replace(
        _legal_budget(),
        ensemble_size=np.int32(2),
        observation_dim=np.uint16(2),
        latent_dim=np.int64(3),
    )
    assert type(budget.ensemble_size) is int
    assert type(budget.observation_dim) is int
    assert type(budget.latent_dim) is int


@pytest.mark.parametrize(
    "config",
    [
        RecurrentLatentWorldModelEnsembleConfig(
            ensemble_size=2, observation_dim=2, n_actions=2, latent_dim=3, max_updates=17
        ),
        RecurrentLatentWorldModelEnsembleConfig(
            ensemble_size=3, observation_dim=4, n_actions=2, latent_dim=5, max_updates=31
        ),
        RecurrentLatentWorldModelEnsembleConfig(
            ensemble_size=4, observation_dim=1, n_actions=3, latent_dim=2, max_updates=1
        ),
    ],
)
def test_recurrent_latent_producer_matches_all_exact_formulas(
    config: RecurrentLatentWorldModelEnsembleConfig,
) -> None:
    budget = RecurrentLatentWorldModelEnsemble(config).resource_budget()
    assert budget == RecurrentLatentWorldModelResourceBudget(**budget.to_config())
