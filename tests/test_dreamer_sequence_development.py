"""Actual-consumer validation for the Dreamer-family sequence lane."""

from __future__ import annotations

import dataclasses
import tomllib
from pathlib import Path

import pytest

from alberta_framework.benchmarks.dreamer_sequence_development import (
    ARM_IDS,
    FROZEN_SEEDS,
    run_development_lane,
    validate_result,
)

pytestmark = pytest.mark.integration


def test_lane_consumes_recurrent_replay_actor_value_and_lambda_returns() -> None:
    result = run_development_lane(seed=FROZEN_SEEDS[0], steps_per_task=3)
    assert tuple(arm.arm_id for arm in result.arms) == ARM_IDS
    candidate, off, oracle = result.arms
    assert candidate.receipt.environment_steps == off.receipt.environment_steps == 18
    assert candidate.receipt.sequence_samples == off.receipt.sequence_samples == 8
    assert candidate.receipt.world_model_updates == off.receipt.world_model_updates == 8
    assert candidate.receipt.imagination_queries == 8 * result.imagination_horizon
    assert candidate.receipt.lambda_return_targets == 8 * result.imagination_horizon
    assert candidate.receipt.actor_updates == candidate.receipt.value_updates == 8
    assert off.receipt.imagination_queries == 0
    assert off.receipt.actor_updates == off.receipt.value_updates == 0
    assert candidate.model_digest == off.model_digest
    assert candidate.replay_digest == off.replay_digest
    assert oracle.candidate_eligible is False
    assert result.development_only and not result.scientific_promotion_allowed
    assert not result.dreamerv3_parity_claimed


def test_lane_replays_exactly_from_explicit_threefry_seed() -> None:
    left = run_development_lane(seed=FROZEN_SEEDS[1], steps_per_task=2)
    right = run_development_lane(seed=FROZEN_SEEDS[1], steps_per_task=2)
    assert validate_result(left) is left
    for left_arm, right_arm in zip(left.arms, right.arms, strict=True):
        left_receipt = dataclasses.replace(left_arm.receipt, elapsed_ns=0)
        right_receipt = dataclasses.replace(right_arm.receipt, elapsed_ns=0)
        assert dataclasses.replace(left_arm, receipt=left_receipt) == dataclasses.replace(
            right_arm, receipt=right_receipt
        )


def test_one_step_tasks_do_not_forge_sequences_across_boundaries() -> None:
    result = run_development_lane(seed=FROZEN_SEEDS[2], steps_per_task=1)
    candidate, off, _ = result.arms
    assert candidate.receipt.sequence_samples == off.receipt.sequence_samples == 0
    assert candidate.receipt.world_model_updates == off.receipt.world_model_updates == 0
    assert candidate.receipt.imagination_queries == 0


def test_validator_rejects_resource_and_identity_forgery() -> None:
    result = run_development_lane(seed=FROZEN_SEEDS[0], steps_per_task=2)
    arm = result.arms[0]
    forged_receipt = dataclasses.replace(arm.receipt, imagination_queries=1)
    forged_arm = dataclasses.replace(arm, receipt=forged_receipt)
    with pytest.raises(ValueError, match="imagination"):
        validate_result(
            dataclasses.replace(result, arms=(forged_arm, *result.arms[1:])),
            replay_execution=False,
        )

    identity = dataclasses.replace(result.identity, lane_source_sha256="0" * 64)
    with pytest.raises(ValueError, match="current source/runtime/registries"):
        validate_result(dataclasses.replace(result, identity=identity), replay_execution=False)

    with pytest.raises(ValueError, match="nonpromoting"):
        validate_result(
            dataclasses.replace(result, scientific_promotion_allowed=True),
            replay_execution=False,
        )


def test_cli_is_registered_as_a_package_entrypoint() -> None:
    project = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert project["project"]["scripts"]["asi-dreamer-sequence-development"] == (
        "alberta_framework.benchmarks.dreamer_sequence_development:main"
    )
