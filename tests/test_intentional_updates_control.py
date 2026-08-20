"""Prospective end-to-end Intentional Updates TD/control contract."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Never

import pytest

from alberta_framework.benchmarks import intentional_updates_control as lane

pytestmark = pytest.mark.integration


def test_plan_is_fresh_prospective_and_permanently_nonpromoting() -> None:
    plan = lane.frozen_plan()
    assert plan["seeds"] == [25610, 25611, 25612, 25613]
    assert plan["execution_authorized"] is False
    assert plan["scientific_promotion_allowed"] is False
    assert plan["negative_outcomes_retained"] is True
    assert plan["confidence_critical"] == 5.391949071934058
    assert plan["confidence_critical"].hex() == "0x1.5915b18f69e09p+2"
    assert plan["protocol_families"] == ["supervised_ipmnist", "td_control"]


@pytest.mark.parametrize(
    ("fixed", "off"),
    [
        ("fixed_td0", "intentional_td0_off"),
        ("fixed_trace", "intentional_trace_off"),
        ("fixed_q_lambda", "intentional_q_lambda_off"),
    ],
)
def test_mechanism_off_reduces_bit_exactly_to_fixed_consumer(
    fixed: str, off: str,
) -> None:
    expected = lane.run_control_shard(fixed, seed=25610, horizon=48, phase_length=12)
    actual = lane.run_control_shard(off, seed=25610, horizon=48, phase_length=12)
    assert actual["arm"] == off
    assert actual["execution_arm"] == fixed
    for key in ("trajectory", "final_state", "metrics", "resources"):
        assert actual[key] == expected[key]


@pytest.mark.parametrize("arm", lane.CONTROL_ARMS)
def test_each_control_arm_runs_end_to_end_with_exact_resources(arm: str) -> None:
    record = lane.run_control_shard(arm, seed=25611, horizon=48, phase_length=12)
    assert lane.validate_control_shard(record) == record
    assert len(record["trajectory"]["rewards"]) == 48
    assert record["resources"]["environment_steps"] == 48
    assert record["resources"]["observations"] == 48
    assert record["resources"]["updates"] == 48
    assert record["resources"]["model_queries"] == 96
    assert record["resources"]["timing_is_selection_metric"] is False
    assert type(record["resources"]["timing_telemetry_ns"]) is int
    assert record["policy"]["scientific_promotion_allowed"] is False


def test_prediction_and_control_information_and_rng_are_explicit() -> None:
    prediction = lane.run_control_shard(
        "intentional_trace", seed=25612, horizon=16, phase_length=4
    )
    control = lane.run_control_shard(
        "intentional_q_lambda", seed=25612, horizon=16, phase_length=4
    )
    assert prediction["resources"]["action_queries"] == 0
    assert prediction["resources"]["rng_fold_ins"] == 0
    assert control["resources"]["action_queries"] == 16
    assert control["resources"]["rng_fold_ins"] == 16
    assert control["identity"]["agent_rng_impl"] == "threefry2x32"
    assert prediction["information"]["boundary_information"] == []
    assert prediction["information"]["task_information"] == []


def test_validator_rejects_nested_subclasses_without_hooks() -> None:
    record = lane.run_control_shard(
        "intentional_trace", seed=25613, horizon=16, phase_length=4
    )

    class HostileDict(dict[object, object]):
        calls = 0

        def __iter__(self) -> Never:
            type(self).calls += 1
            raise AssertionError("must not iterate")

        def __eq__(self, other: object) -> Never:
            type(self).calls += 1
            raise AssertionError("must not compare")

    hostile = copy.deepcopy(record)
    hostile["trajectory"] = HostileDict(hostile["trajectory"])
    with pytest.raises(ValueError, match="exact JSON"):
        lane.validate_control_shard(hostile)
    assert HostileDict.calls == 0


def test_validator_rejects_resource_result_identity_and_policy_forgery() -> None:
    record = lane.run_control_shard("intentional_td0", seed=25610, horizon=16, phase_length=4)
    for path, replacement in (
        (("resources", "updates"), 15),
        (("trajectory", "rewards"), [99.0] * 16),
        (("identity", "source_sha256"), {"forged": "0" * 64}),
        (("policy", "scientific_promotion_allowed"), True),
    ):
        hostile = copy.deepcopy(record)
        hostile[path[0]][path[1]] = replacement
        with pytest.raises(ValueError):
            lane.validate_control_shard(hostile)


def test_campaign_execution_is_closed_before_independent_review() -> None:
    with pytest.raises(RuntimeError, match="not authorized"):
        lane.run_campaign(Path("unused.npz"), Path("unused.json"))


@pytest.mark.parametrize(
    ("horizon", "phase_length"),
    [(0, 1), (10_001, 1), (8, 0), (8, 9), (True, 1)],
)
def test_control_bounds_fail_before_execution(horizon: int, phase_length: int) -> None:
    with pytest.raises(ValueError):
        lane.run_control_shard(
            "fixed_td0", seed=25610, horizon=horizon, phase_length=phase_length
        )
