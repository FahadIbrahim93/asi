"""Leftover-identity gates for prototype-lifecycle resource-budget records."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, cast

import pytest

from alberta_framework.core.prototype_feature_lifecycle import (
    PrototypeFeatureLifecycleResourceBudget,
)


def _legal_budget() -> PrototypeFeatureLifecycleResourceBudget:
    return PrototypeFeatureLifecycleResourceBudget(
        mechanism_status="development_mechanism_only",
        scientific_promotion_allowed=False,
        base_feature_slots=4,
        active_pair_slots=1,
        candidate_pair_slots=1,
        n_tasks=1,
        n_options=1,
        n_primitive_actions=1,
        managed_oak_feature_width=5,
        learner_persistent_state_nbytes=8,
        router_persistent_state_nbytes=8,
        lifecycle_counter_nbytes=16,
        lifecycle_state_nbytes=32,
        consumer_binding_persistent_nbytes=12,
        internal_learner_template_nbytes=8,
        internal_oak_template_nbytes=8,
        internal_template_nbytes=16,
        owned_persistent_state_nbytes=48,
        managed_oak_consumer_nbytes=260,
        rebuilt_base_cache_nbytes=20,
        input_route_feature_groups=12,
        output_route_feature_groups=5,
        router_calls_per_observe=2,
        router_calls_per_committed_curation=2,
        max_active_pair_products_per_observe=5,
        max_candidate_pair_products_per_observe=1,
        max_observations=10,
    )


def test_prototype_lifecycle_resource_budget_rejects_leftover_identities() -> None:
    """Public resource-budget records must not keep leftover bool/int identities."""

    with pytest.raises(ValueError, match="base_feature_slots"):
        replace(_legal_budget(), base_feature_slots=True)
    with pytest.raises(ValueError, match="max_observations"):
        replace(_legal_budget(), max_observations=True)
    with pytest.raises(ValueError, match="lifecycle_state_nbytes"):
        replace(_legal_budget(), lifecycle_state_nbytes=cast(Any, float("nan")))
    with pytest.raises(ValueError, match="scientific_promotion_allowed"):
        replace(_legal_budget(), scientific_promotion_allowed=cast(Any, 1))
    with pytest.raises(ValueError, match="mechanism_status"):
        replace(_legal_budget(), mechanism_status=cast(Any, True))

    legal = _legal_budget()
    dumped = json.dumps(legal.to_config(), allow_nan=False)
    assert '"base_feature_slots": 4' in dumped
    assert '"max_observations": 10' in dumped
    assert '"lifecycle_state_nbytes": 32' in dumped
    assert '"scientific_promotion_allowed": false' in dumped
    assert '"base_feature_slots": true' not in dumped
    assert '"max_observations": true' not in dumped
    assert '"lifecycle_state_nbytes": true' not in dumped
    assert '"scientific_promotion_allowed": 1' not in dumped


@pytest.mark.parametrize(
    "field",
    [
        "managed_oak_feature_width",
        "lifecycle_counter_nbytes",
        "lifecycle_state_nbytes",
        "consumer_binding_persistent_nbytes",
        "internal_template_nbytes",
        "owned_persistent_state_nbytes",
        "managed_oak_consumer_nbytes",
        "rebuilt_base_cache_nbytes",
        "input_route_feature_groups",
        "output_route_feature_groups",
        "router_calls_per_observe",
        "router_calls_per_committed_curation",
        "max_active_pair_products_per_observe",
        "max_candidate_pair_products_per_observe",
    ],
)
def test_prototype_lifecycle_resource_budget_rejects_mutated_derived_identities(
    field: str,
) -> None:
    budget = _legal_budget()
    with pytest.raises(ValueError, match=field):
        replace(budget, **{field: getattr(budget, field) + 1})


def test_prototype_lifecycle_resource_budget_requires_fixed_mechanism_identity() -> None:
    with pytest.raises(ValueError, match="mechanism_status"):
        replace(_legal_budget(), mechanism_status="production")
    with pytest.raises(ValueError, match="scientific_promotion_allowed"):
        replace(_legal_budget(), scientific_promotion_allowed=True)


def test_prototype_lifecycle_resource_budget_rejects_subclass_before_hooks() -> None:
    class HostileBudget(PrototypeFeatureLifecycleResourceBudget):
        calls = 0

        def __getattribute__(self, name: str) -> Any:
            type(self).calls += 1
            raise AssertionError(f"attribute hook must not run: {name}")

    value = object.__new__(HostileBudget)
    HostileBudget.calls = 0
    with pytest.raises(TypeError, match="exact PrototypeFeatureLifecycleResourceBudget"):
        PrototypeFeatureLifecycleResourceBudget.__post_init__(cast(Any, value))
    assert HostileBudget.calls == 0
