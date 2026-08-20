from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest

from alberta_framework.benchmarks.continual_world_qualification import (
    CW20_TASKS,
    ContinualWorldSmokePlan,
    IsolatedRuntimeIdentity,
    build_smoke_receipt,
    protocol_gap_record,
    validate_smoke_payload,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def runtime() -> IsolatedRuntimeIdentity:
    return IsolatedRuntimeIdentity(
        image_digest="sha256:" + "1" * 64,
        mujoco_archive_sha256="2" * 64,
        python_version="3.6.9",
        tensorflow_version="2.6.2",
        mujoco_py_version="2.0.2.13",
        gym_version="0.21.0",
        numpy_version="1.19.5",
        platform="linux-x86_64",
    )


@pytest.fixture
def receipt(runtime: IsolatedRuntimeIdentity):
    plan = ContinualWorldSmokePlan(runtime=runtime)
    horizon = 20 * plan.steps_per_task
    return build_smoke_receipt(
        plan,
        actions=np.zeros((horizon, 4), dtype=np.float32),
        observations=np.zeros((horizon, 32), dtype=np.float32),
        rewards=np.zeros((horizon,), dtype=np.float32),
        successes=np.zeros((horizon,), dtype=np.bool_),
        task_indices=np.repeat(np.arange(20, dtype=np.int32), plan.steps_per_task),
        persistent_environment_numeric_bytes=4096,
        timing_ns=10,
        outcome="inconclusive",
    )


def test_protocol_pins_official_sequence_sources_and_nonpromotion(runtime) -> None:
    plan = ContinualWorldSmokePlan(runtime=runtime)
    payload = plan.payload()
    assert len(CW20_TASKS) == 20
    assert CW20_TASKS[:10] == CW20_TASKS[10:]
    assert payload["paper_revision"] == "arXiv:2105.10919v3"
    assert payload["qualification_issue"] == 1580
    assert payload["qualification_lane_id"] == "continual-world-cw20"
    assert payload["official_repository"] == "https://github.com/awarelab/continual_world.git"
    assert payload["official_commit"] == "73f63bb4fa0b5d00bda973e20dfb783bfcf1b8aa"
    assert payload["metaworld_commit"] == "0875192baaa91c43523708f55866d98eaf3facaf"
    assert payload["learner_boundary_information"] == []
    assert payload["scientific_promotion_allowed"] is False


def test_fixed_action_receipt_is_exact_mechanism_off_and_round_trips(receipt) -> None:
    checked = validate_smoke_payload(receipt.payload())
    assert checked == receipt
    assert checked.environment_steps == 40
    assert checked.persistent_mechanism_bytes == 16
    assert checked.action_bytes == 40 * 4 * np.dtype(np.float32).itemsize
    assert checked.observation_bytes == 40 * 32 * np.dtype(np.float32).itemsize
    assert checked.reward_bytes == 40 * np.dtype(np.float32).itemsize
    assert checked.success_bytes == 40 * np.dtype(np.bool_).itemsize
    assert checked.task_index_bytes == 40 * np.dtype(np.int32).itemsize
    assert checked.data_steps == checked.learner_updates == checked.model_queries == 0
    assert checked.mechanism_off is True
    assert checked.negative_outcome_retained is True
    assert len(checked.identity.lane_source_sha256) == 64
    assert dict(checked.identity.dependency_source_sha256).keys() >= {
        "alberta_framework.benchmarks.external_qualification",
        "alberta_framework.benchmarks.qualification_provenance",
    }


def test_trace_builder_snapshots_and_rejects_wrong_boundary_schedule(runtime) -> None:
    plan = ContinualWorldSmokePlan(runtime=runtime)
    horizon = 40
    observations = np.zeros((horizon, 32), dtype=np.float32)
    task_indices = np.repeat(np.arange(20, dtype=np.int32), 2)
    receipt = build_smoke_receipt(
        plan,
        actions=np.zeros((horizon, 4), dtype=np.float32),
        observations=observations,
        rewards=np.zeros(horizon, dtype=np.float32),
        successes=np.zeros(horizon, dtype=np.bool_),
        task_indices=task_indices,
        persistent_environment_numeric_bytes=1,
        timing_ns=0,
        outcome="rejected",
    )
    observations[0, 0] = 99.0
    checked = validate_smoke_payload(receipt.payload())
    assert receipt.observation_sha256 == checked.observation_sha256
    task_indices[0] = 1
    with pytest.raises(ValueError, match="boundary schedule"):
        build_smoke_receipt(
            plan,
            actions=np.zeros((horizon, 4), dtype=np.float32),
            observations=np.zeros((horizon, 32), dtype=np.float32),
            rewards=np.zeros(horizon, dtype=np.float32),
            successes=np.zeros(horizon, dtype=np.bool_),
            task_indices=task_indices,
            persistent_environment_numeric_bytes=1,
            timing_ns=0,
            outcome="rejected",
        )
    hostile_actions = np.zeros((horizon, 4), dtype=np.float32)
    hostile_actions[0, 0] = 0.25
    with pytest.raises(ValueError, match="fixed-action"):
        build_smoke_receipt(
            plan,
            actions=hostile_actions,
            observations=np.zeros((horizon, 32), dtype=np.float32),
            rewards=np.zeros(horizon, dtype=np.float32),
            successes=np.zeros(horizon, dtype=np.bool_),
            task_indices=np.repeat(np.arange(20, dtype=np.int32), 2),
            persistent_environment_numeric_bytes=1,
            timing_ns=0,
            outcome="rejected",
        )


def test_validator_rejects_hostile_expansion_resources_and_promotion(receipt) -> None:
    expanded = receipt.payload()
    expanded["extra"] = 1
    with pytest.raises(ValueError, match="fields differ"):
        validate_smoke_payload(expanded)
    forged = copy.deepcopy(receipt.payload())
    forged["persistent_mechanism_bytes"] = 15
    with pytest.raises(ValueError, match="fixed float32 action"):
        validate_smoke_payload(forged)
    promoted = copy.deepcopy(receipt.payload())
    promoted["scientific_promotion_allowed"] = True
    with pytest.raises(ValueError, match="nonpromoting"):
        validate_smoke_payload(promoted)
    forged_bytes = copy.deepcopy(receipt.payload())
    forged_bytes["observation_bytes"] -= 1
    with pytest.raises(ValueError, match="trace byte receipt"):
        validate_smoke_payload(forged_bytes)
    forged_identity = copy.deepcopy(receipt.payload())
    forged_identity["identity"]["lane_source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="current tree/runtime"):
        validate_smoke_payload(forged_identity)
    with pytest.raises(ValueError, match="plan_sha256"):
        dataclasses.replace(receipt, plan_sha256="0" * 64)
    with pytest.raises(ValueError, match="outcome scope"):
        dataclasses.replace(receipt, outcome_scope="paper_comparison")
    with pytest.raises(ValueError, match="environment byte scope"):
        dataclasses.replace(receipt, persistent_environment_byte_scope="whole_process_rss")


def test_runtime_and_plan_fail_closed(runtime) -> None:
    with pytest.raises(ValueError, match="image_digest"):
        dataclasses.replace(runtime, image_digest="latest")
    with pytest.raises(ValueError, match="development seed"):
        ContinualWorldSmokePlan(runtime=runtime, seed=0)
    assert len(protocol_gap_record()) >= 8


def test_direct_receipt_revalidates_mutated_nested_plan_and_runtime(receipt) -> None:
    hostile_runtime = dataclasses.replace(receipt.plan.runtime)
    hostile_plan = dataclasses.replace(receipt.plan, runtime=hostile_runtime)
    object.__setattr__(hostile_runtime, "python_version", "")
    with pytest.raises(ValueError, match="python_version"):
        dataclasses.replace(
            receipt,
            plan=hostile_plan,
            plan_sha256=hostile_plan.sha256,
        )

    hostile_plan = dataclasses.replace(receipt.plan)
    object.__setattr__(hostile_plan, "steps_per_task", 3)
    with pytest.raises(ValueError, match="smoke steps_per_task is frozen"):
        dataclasses.replace(
            receipt,
            plan=hostile_plan,
            plan_sha256=hostile_plan.sha256,
            environment_steps=60,
            action_bytes=60 * 4 * np.dtype(np.float32).itemsize,
            observation_bytes=60 * 32 * np.dtype(np.float32).itemsize,
            reward_bytes=60 * np.dtype(np.float32).itemsize,
            success_bytes=60 * np.dtype(np.bool_).itemsize,
            task_index_bytes=60 * np.dtype(np.int32).itemsize,
        )


def test_payload_validator_enforces_receipt_ceiling_before_identity_reconstruction(receipt) -> None:
    oversized = receipt.payload()
    oversized["identity"]["runtime_identity"].append(["zz", "x" * (1 << 20)])

    with pytest.raises(ValueError, match="limit|ceiling"):
        validate_smoke_payload(oversized)


def test_payload_method_rejects_post_construction_mutation(receipt) -> None:
    object.__setattr__(receipt, "development_only", False)

    with pytest.raises(ValueError, match="permanently nonpromoting"):
        receipt.payload()
