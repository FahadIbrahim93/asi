from __future__ import annotations

import dataclasses

import jax
import pytest

from alberta_framework.benchmarks.ftl_online_agent_development import (
    ACTION_DELTAS,
    ARM_IDS,
    FROZEN_SEEDS,
    OFFICIAL_CODE_REVISION,
    run_development_lane,
    validate_result,
)

pytestmark = pytest.mark.integration


def test_end_to_end_lane_has_matched_axes_controls_and_exact_receipts() -> None:
    result = run_development_lane(seed=FROZEN_SEEDS[0], steps_per_task=2, planning_horizon=1)
    assert tuple(arm.arm_id for arm in result.arms) == ARM_IDS
    assert result.development_only and not result.scientific_promotion_allowed
    assert result.negative_results_must_be_retained and not result.historical_ftl_claim_reused
    assert result.allowed_boundary_information == ()
    assert result.allowed_task_information == ("current_goal",)
    assert OFFICIAL_CODE_REVISION.endswith("a4fdb3b94a07a40d76e28d3aeab0f8ca97519dad")
    for arm in result.arms:
        assert arm.receipt.environment_steps == 6
        assert arm.receipt.planner_candidates == 24
    assert result.arms[0].receipt.model_queries == result.arms[1].receipt.model_queries == 30
    assert result.arms[2].receipt.model_queries == 24
    assert result.arms[0].receipt.model_updates == 6
    assert result.arms[1].receipt.model_updates == result.arms[2].receipt.model_updates == 0
    assert result.arms[0].receipt.logical_compute_units == 66
    assert result.arms[1].receipt.logical_compute_units == 60
    assert result.arms[2].receipt.logical_compute_units == 54
    assert result.arms[2].candidate_eligible is False


def test_sparse_model_predict_and_update_are_jit_eager_parity() -> None:
    # The end-to-end lane invokes the model's jitted methods; disabling JIT checks its eager path.
    compiled = run_development_lane(seed=FROZEN_SEEDS[1], steps_per_task=1, planning_horizon=1)
    with jax.disable_jit():
        eager = run_development_lane(seed=FROZEN_SEEDS[1], steps_per_task=1, planning_horizon=1)
    for compiled_arm, eager_arm in zip(compiled.arms, eager.arms, strict=True):
        assert compiled_arm.task_returns == eager_arm.task_returns
        assert compiled_arm.prequential_squared_errors == pytest.approx(
            eager_arm.prequential_squared_errors, rel=1e-6, abs=1e-10
        )


@pytest.mark.parametrize("seed", [True, -1, 0, FROZEN_SEEDS[-1] + 1])
def test_runner_rejects_hostile_or_unfrozen_seeds(seed: object) -> None:
    with pytest.raises(ValueError):
        run_development_lane(seed=seed, steps_per_task=1, planning_horizon=1)


def test_validator_rejects_promotion_counter_forgery_and_extra_fields() -> None:
    result = run_development_lane(seed=FROZEN_SEEDS[0], steps_per_task=1, planning_horizon=1)
    with pytest.raises(ValueError, match="nonpromoting"):
        validate_result(dataclasses.replace(result, scientific_promotion_allowed=True))
    receipt = dataclasses.replace(result.arms[0].receipt, model_queries=1)
    forged = dataclasses.replace(
        result,
        arms=(dataclasses.replace(result.arms[0], receipt=receipt), *result.arms[1:]),
    )
    with pytest.raises(ValueError, match="planner receipt"):
        validate_result(forged)
    payload = dataclasses.asdict(result)
    payload["unexpected"] = 1
    with pytest.raises(ValueError, match="exact mapping"):
        validate_result(payload)

    bad_bytes = dataclasses.replace(result.arms[0].receipt, persistent_bytes=1)
    forged_bytes = dataclasses.replace(
        result,
        arms=(dataclasses.replace(result.arms[0], receipt=bad_bytes), *result.arms[1:]),
    )
    with pytest.raises(ValueError, match="persistent-byte"):
        validate_result(forged_bytes)

    missing_metric = dataclasses.replace(
        result.arms[0], prequential_squared_errors=result.arms[0].prequential_squared_errors[:-1]
    )
    forged_metrics = dataclasses.replace(result, arms=(missing_metric, *result.arms[1:]))
    with pytest.raises(ValueError, match="metric count"):
        validate_result(forged_metrics)

    forged_identity = dataclasses.replace(result.identity, paper_registry_sha256="0" * 64)
    with pytest.raises(ValueError, match="current source/runtime/registries"):
        validate_result(dataclasses.replace(result, identity=forged_identity))


def test_mechanism_off_is_causal_and_does_not_adopt_updates() -> None:
    result = run_development_lane(seed=FROZEN_SEEDS[2], steps_per_task=3, planning_horizon=1)
    enabled, disabled, _ = result.arms
    assert enabled.receipt.persistent_bytes == disabled.receipt.persistent_bytes
    assert enabled.prequential_squared_errors != disabled.prequential_squared_errors
    assert enabled.task_returns != disabled.task_returns


def test_workload_action_registry_is_read_only() -> None:
    with pytest.raises(ValueError):
        ACTION_DELTAS[0, 0] = 7
