from __future__ import annotations

import pytest

import alberta_framework.evaluation.bimu_matched_nonpromoting as bimu_plan
from alberta_framework.evaluation.bimu_matched_nonpromoting import (
    FROZEN_BIMU_MATCHED_PLAN,
    FROZEN_PLAN_SHA256,
    INVALID_PRIOR_ATTEMPT,
    _plan_payload,
)


def test_frozen_bimu_plan_is_matched_and_prospective() -> None:
    plan = FROZEN_BIMU_MATCHED_PLAN
    assert plan.seeds == (157001, 157002, 157003)
    assert plan.arm_names == ("memory_off", "bimu")
    assert plan.control_config.memory_window is None
    assert plan.candidate_config.memory_window == 128
    control = plan.control_config.to_protocol_payload()
    candidate = plan.candidate_config.to_protocol_payload()
    assert {key for key in control if control[key] != candidate[key]} == {"memory_window"}
    assert plan.dataset_sha256 == "85c681c2f5fc5c274870b30c9accb3d2a6e9eb90a4575a2bf1ccca64f58b6227"
    assert FROZEN_PLAN_SHA256 == "182632b37c3a8598a30fb943742605374a846d965c5602f4db039f27f78754c1"
    assert bimu_plan.frozen_plan_payload()["seed_status"] == {
        "consumed_for_promotion": True,
        "retained_matched_result_exists": False,
        "reason": "the literal development roster is publicly exposed",
    }
    assert INVALID_PRIOR_ATTEMPT["pull_request"] == 1686
    assert INVALID_PRIOR_ATTEMPT["seed"] == 23
    payload = _plan_payload(plan)
    assert payload["expected_counters_per_arm"]["observations"] == 1280
    assert payload["expected_counters_per_arm"]["model_forward_queries"] == 10240
    assert payload["expected_counters_per_arm"]["optimizer_updates"] == 1280
    assert payload["expected_resources_per_arm"] == {
        "trainable_scalar_count": 25408,
        "parameter_numeric_bytes": 101632,
        "optimizer_state_numeric_bytes": 8,
        "initial_persistent_numeric_bytes": 101640,
        "final_persistent_numeric_bytes": 101640,
        "dataset_numeric_bytes": 1607680,
        "timing_qualified": False,
        "aggregate_working_set_bytes_claimed": False,
        "numeric_resource_ceiling_bytes": 256 * 1024 * 1024,
    }
    assert payload["comparison_scope"]["paper_comparable"] is False
    assert payload["authorization"] == {
        "execution_authorized": False,
        "authorization_transition_approved": False,
    }
    assert payload["output_namespace"] == "outputs/bimu_matched/development.v1"
    assert payload["paired_outcome_rule"] == {
        "schema": "asi.bimu.paired-outcome-rule.v1",
        "metric": "paper_late_five_test_accuracy",
        "supported": "all_three_paired_deltas_strictly_positive",
        "rejected": "all_three_paired_deltas_nonpositive",
        "otherwise": "inconclusive",
        "ties_are_positive": False,
        "secondary_metric_affects_outcome": False,
    }
    with pytest.raises(TypeError):
        INVALID_PRIOR_ATTEMPT["seed"] = 157001  # type: ignore[index]


def test_literal_plan_digest_fails_closed_on_unreviewed_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bimu_plan, "EXECUTION_AUTHORIZED", True)
    with pytest.raises(RuntimeError, match="literal digest"):
        bimu_plan.frozen_plan_payload()
