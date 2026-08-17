"""Leftover-identity gates for partner-fusion resource-budget records."""

from __future__ import annotations

import json

import pytest

from alberta_framework.core.partner_policy_fusion import PartnerPolicyFusionResourceBudget


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
