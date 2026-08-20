"""Fail-closed tests for the COOM qualification smoke."""

from __future__ import annotations

import copy
import dataclasses
import sys

import jax
import numpy as np
import pytest

from alberta_framework.benchmarks.coom_qualification import (
    CD8_TASKS,
    CO8_TASKS,
    COOM_COMMIT,
    COOM_PAPER,
    COOM_SMOKE_SCHEMA,
    FROZEN_ARMS,
    COOMCatalogEntry,
    COOMSmokeProtocol,
    COOMSmokeResult,
    _feature,
    _run_arm,
    run_coom_qualification_smoke,
    validate_coom_smoke_payload,
)
from alberta_framework.benchmarks.external_qualification import qualification_plan

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def result() -> COOMSmokeResult:
    return run_coom_qualification_smoke(COOMSmokeProtocol(steps_per_task=2))


def test_catalog_matches_external_pin_paper_sequences_metrics_and_isolation() -> None:
    catalog = COOMCatalogEntry()
    plan = qualification_plan(1582)
    assert plan.paper_revisions == (COOM_PAPER,)
    assert plan.code_revisions[0].commit == COOM_COMMIT
    assert catalog.core_sequence_tasks == {"CD8": CD8_TASKS, "CO8": CO8_TASKS}
    assert catalog.metrics == ("average_performance", "forgetting", "forward_transfer")
    assert catalog.paper_seeds == tuple(range(10))
    assert catalog.default_steps_per_task == 200_000
    assert catalog.default_replay_capacity == 50_000
    assert (catalog.default_frame_height, catalog.default_frame_width) == (84, 84)
    assert catalog.default_frame_stack == catalog.default_frame_skip == 4
    assert catalog.task_id_visible_by_default is True
    assert catalog.integration == "isolated"
    assert catalog.status == "scaffolded"


def test_end_to_end_smoke_is_strict_nonpromoting_and_round_trips(
    result: COOMSmokeResult,
) -> None:
    assert result.schema == COOM_SMOKE_SCHEMA
    assert result.synthetic_contract_trace is True
    assert result.development_only is True
    assert result.scientific_promotion_allowed is False
    assert result.benchmark_result_claimed is False
    assert validate_coom_smoke_payload(result.to_payload()) == result
    assert tuple(arm.arm_id for arm in result.arms[:4]) == FROZEN_ARMS
    assert all(arm.negative_outcome_retained for arm in result.arms)
    assert all(not arm.performance_metrics_computed for arm in result.arms)


def test_dependency_probe_never_imports_or_executes_external_runtime(
    result: COOMSmokeResult,
) -> None:
    assert result.dependencies.imports_attempted == 0
    assert result.dependencies.external_runtime_executed is False
    assert result.dependencies.assets_downloaded is False
    assert "COOM" not in sys.modules
    assert "vizdoom" not in sys.modules
    assert "tensorflow" not in sys.modules
    assert result.qualification_blockers == qualification_plan(1582).required_gates


def test_resources_are_exact_and_mechanism_off_has_full_parity(
    result: COOMSmokeResult,
) -> None:
    for offset in range(0, len(result.arms), len(FROZEN_ARMS)):
        cyclic, sarsa, mechanism_off, fixed = result.arms[offset : offset + 4]
        assert cyclic.resources.environment_steps == 16
        assert cyclic.resources.policy_queries == 16
        assert sarsa.resources.policy_queries == 24
        assert sarsa.resources.persistent_agent_bytes > 0
        assert mechanism_off.resources.persistent_agent_bytes == 0
        assert mechanism_off.action_sha256 == fixed.action_sha256
        assert mechanism_off.reward_sha256 == fixed.reward_sha256
        assert mechanism_off.observation_sha256 == fixed.observation_sha256


def test_contract_feature_path_is_jittable() -> None:
    observation = np.arange(8 * 8 * 3, dtype=np.uint8).reshape((8, 8, 3))
    eager = _feature(observation, 2)
    compiled = jax.jit(_feature, static_argnums=(1,))(observation, 2)
    assert eager.shape == (200,)
    np.testing.assert_array_equal(np.asarray(eager), np.asarray(compiled))


def test_one_arm_replays_deterministically_without_external_runtime(
    result: COOMSmokeResult,
) -> None:
    first = result.arms[0]
    second = _run_arm(result.protocol, first.seed, first.arm_id)
    assert first == second


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"scientific_promotion_allowed": True}, "nonpromoting"),
        ({"benchmark_result_claimed": True}, "nonpromoting"),
        ({"synthetic_contract_trace": False}, "nonpromoting"),
        ({"qualification_blockers": ()}, "blockers"),
    ],
)
def test_result_rejects_promotion_benchmark_aliases_and_open_gates(
    result: COOMSmokeResult,
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(result, **changes)


def test_hostile_expanded_and_forged_payloads_fail_closed(result: COOMSmokeResult) -> None:
    with pytest.raises(ValueError, match="exact dict"):
        validate_coom_smoke_payload(object())
    expanded = result.to_payload()
    expanded["extra"] = 1
    with pytest.raises(ValueError, match="fields differ"):
        validate_coom_smoke_payload(expanded)
    forged = copy.deepcopy(result.to_payload())
    forged["dependencies"]["external_runtime_executed"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="may not import"):
        validate_coom_smoke_payload(forged)
    promoted = copy.deepcopy(result.to_payload())
    promoted["arms"][0]["performance_metrics_computed"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="may not compute"):
        validate_coom_smoke_payload(promoted)
    forged_bytes = copy.deepcopy(result.to_payload())
    forged_bytes["arms"][1]["resources"]["persistent_agent_bytes"] = 1  # type: ignore[index]
    with pytest.raises(ValueError, match="replay|resource"):
        validate_coom_smoke_payload(forged_bytes)
    forged_trace = copy.deepcopy(result.to_payload())
    forged_trace["arms"][0]["action_sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(ValueError, match="replay"):
        validate_coom_smoke_payload(forged_trace)
    forged_identity = copy.deepcopy(result.to_payload())
    forged_identity["identity"]["runtime_identity"][0][1] = "forged"  # type: ignore[index]
    with pytest.raises(ValueError, match="current tree/runtime"):
        validate_coom_smoke_payload(forged_identity)


@pytest.mark.parametrize(
    "changes",
    [
        {"steps_per_task": True},
        {"steps_per_task": 17},
        {"sequence": "CO16"},
        {"task_id_available": False},
        {"seeds": (1,)},
        {"observation_shape": (192,)},
    ],
)
def test_protocol_rejects_hostile_unbounded_or_changed_axes(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        COOMSmokeProtocol(**changes)  # type: ignore[arg-type]
