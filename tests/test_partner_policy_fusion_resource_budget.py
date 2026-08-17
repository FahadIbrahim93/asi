"""Leftover-identity gates for partner-fusion resource-budget records."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from alberta_framework.core.partner_policy_fusion import (
    PartnerPolicyFusion,
    PartnerPolicyFusionConfig,
    PartnerPolicyFusionResourceBudget,
)


class _HostileInt(int):
    def __index__(self) -> int:
        raise AssertionError("hostile index hook executed")


def _legal_budget() -> PartnerPolicyFusionResourceBudget:
    return PartnerPolicyFusionResourceBudget(
        max_partners=3,
        context_dim=2,
        n_actions=3,
        model_feature_dim=4,
        trainable_float32_scalars=12,
        persistent_float32_scalars=17,
        persistent_int32_scalars=15,
        persistent_bool_scalars=2,
        persistent_state_scalars=34,
        persistent_state_bytes=130,
        max_messages_per_decision=3,
        max_model_scores_per_decision=3,
        partner_id_pairwise_equality_comparisons_per_decision=9,
        max_trainable_scalars_touched_per_feedback=4,
        decision_input_float32_scalars=10,
        decision_input_int32_scalars=33,
        decision_input_bool_scalars=7,
        feedback_input_float32_scalars=1,
        feedback_input_int32_scalars=4,
        feedback_input_bool_scalars=4,
        max_parameter_updates_per_feedback=1,
        rng_state_bytes=0,
        replay_capacity=0,
        dynamic_partner_capacity=0,
    )


def test_partner_fusion_resource_budget_rejects_leftover_identities() -> None:
    """Public resource-budget records must not keep leftover bool/int identities."""

    with pytest.raises(ValueError, match="max_partners"):
        PartnerPolicyFusionResourceBudget(
            max_partners=True,
            context_dim=2,
            n_actions=3,
            model_feature_dim=4,
            trainable_float32_scalars=12,
            persistent_float32_scalars=17,
            persistent_int32_scalars=15,
            persistent_bool_scalars=2,
            persistent_state_scalars=34,
            persistent_state_bytes=130,
            max_messages_per_decision=3,
            max_model_scores_per_decision=3,
            partner_id_pairwise_equality_comparisons_per_decision=9,
            max_trainable_scalars_touched_per_feedback=4,
            decision_input_float32_scalars=10,
            decision_input_int32_scalars=33,
            decision_input_bool_scalars=7,
            feedback_input_float32_scalars=1,
            feedback_input_int32_scalars=4,
            feedback_input_bool_scalars=4,
            max_parameter_updates_per_feedback=1,
            rng_state_bytes=0,
            replay_capacity=0,
            dynamic_partner_capacity=0,
        )
    with pytest.raises(ValueError, match="replay_capacity"):
        PartnerPolicyFusionResourceBudget(
            max_partners=3,
            context_dim=2,
            n_actions=3,
            model_feature_dim=4,
            trainable_float32_scalars=12,
            persistent_float32_scalars=17,
            persistent_int32_scalars=15,
            persistent_bool_scalars=2,
            persistent_state_scalars=34,
            persistent_state_bytes=130,
            max_messages_per_decision=3,
            max_model_scores_per_decision=3,
            partner_id_pairwise_equality_comparisons_per_decision=9,
            max_trainable_scalars_touched_per_feedback=4,
            decision_input_float32_scalars=10,
            decision_input_int32_scalars=33,
            decision_input_bool_scalars=7,
            feedback_input_float32_scalars=1,
            feedback_input_int32_scalars=4,
            feedback_input_bool_scalars=4,
            max_parameter_updates_per_feedback=1,
            rng_state_bytes=0,
            replay_capacity=True,
            dynamic_partner_capacity=0,
        )
    with pytest.raises(ValueError, match="persistent_state_bytes"):
        PartnerPolicyFusionResourceBudget(
            max_partners=3,
            context_dim=2,
            n_actions=3,
            model_feature_dim=4,
            trainable_float32_scalars=12,
            persistent_float32_scalars=17,
            persistent_int32_scalars=15,
            persistent_bool_scalars=2,
            persistent_state_scalars=34,
            persistent_state_bytes=float("nan"),
            max_messages_per_decision=3,
            max_model_scores_per_decision=3,
            partner_id_pairwise_equality_comparisons_per_decision=9,
            max_trainable_scalars_touched_per_feedback=4,
            decision_input_float32_scalars=10,
            decision_input_int32_scalars=33,
            decision_input_bool_scalars=7,
            feedback_input_float32_scalars=1,
            feedback_input_int32_scalars=4,
            feedback_input_bool_scalars=4,
            max_parameter_updates_per_feedback=1,
            rng_state_bytes=0,
            replay_capacity=0,
            dynamic_partner_capacity=0,
        )

    legal = _legal_budget()
    dumped = json.dumps(legal.to_config(), allow_nan=False)
    assert '"max_partners": 3' in dumped
    assert '"replay_capacity": 0' in dumped
    assert '"persistent_state_bytes": 130' in dumped
    assert '"max_partners": true' not in dumped
    assert '"replay_capacity": true' not in dumped
    assert '"persistent_state_bytes": true' not in dumped


@pytest.mark.parametrize(
    "field",
    [
        "model_feature_dim",
        "trainable_float32_scalars",
        "persistent_float32_scalars",
        "persistent_int32_scalars",
        "persistent_bool_scalars",
        "persistent_state_scalars",
        "persistent_state_bytes",
        "max_messages_per_decision",
        "max_model_scores_per_decision",
        "partner_id_pairwise_equality_comparisons_per_decision",
        "max_trainable_scalars_touched_per_feedback",
        "decision_input_float32_scalars",
        "decision_input_int32_scalars",
        "decision_input_bool_scalars",
        "feedback_input_float32_scalars",
        "feedback_input_int32_scalars",
        "feedback_input_bool_scalars",
        "max_parameter_updates_per_feedback",
        "rng_state_bytes",
        "replay_capacity",
        "dynamic_partner_capacity",
    ],
)
def test_partner_fusion_resource_budget_requires_exact_formulas(field: str) -> None:
    budget = _legal_budget()
    with pytest.raises(ValueError, match=field):
        dataclasses.replace(budget, **{field: getattr(budget, field) + 1})


@pytest.mark.parametrize("field", ["max_partners", "context_dim", "n_actions"])
def test_partner_fusion_resource_budget_requires_positive_dimensions(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        dataclasses.replace(_legal_budget(), **{field: 0})


@pytest.mark.parametrize("field", ["max_partners", "context_dim", "n_actions"])
def test_partner_fusion_resource_budget_binds_provenance_dimensions(field: str) -> None:
    budget = _legal_budget()
    with pytest.raises(ValueError, match="partner-fusion implementation"):
        dataclasses.replace(budget, **{field: getattr(budget, field) + 1})


@pytest.mark.parametrize("value", [_HostileInt(3), 2**31, -1])
def test_partner_fusion_resource_budget_rejects_hostile_or_wide_dimensions(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="max_partners"):
        dataclasses.replace(_legal_budget(), max_partners=value)  # type: ignore[arg-type]


def test_partner_fusion_resource_budget_canonicalizes_numpy_integer_dimensions() -> None:
    budget = dataclasses.replace(
        _legal_budget(),
        max_partners=np.int32(3),
        context_dim=np.uint16(2),
        n_actions=np.int64(3),
    )
    assert type(budget.max_partners) is int
    assert type(budget.context_dim) is int
    assert type(budget.n_actions) is int


def test_partner_fusion_producer_budget_is_total_at_configured_maxima() -> None:
    fusion = PartnerPolicyFusion(
        PartnerPolicyFusionConfig(
            max_partners=1024,
            context_dim=65_536,
            n_actions=65_536,
        )
    )
    budget = fusion.resource_budget
    assert budget == PartnerPolicyFusionResourceBudget(**budget.to_config())
    assert all(
        type(value) is int and 0 <= value <= 2**31 - 1
        for value in budget.to_config().values()
    )
