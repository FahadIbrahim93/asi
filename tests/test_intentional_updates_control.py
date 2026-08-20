"""Prospective end-to-end Intentional Updates TD/control contract."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Never

import numpy as np
import pytest

from alberta_framework.benchmarks import intentional_updates_control as lane
from alberta_framework.benchmarks.ipmnist_screening import (
    ScreeningRunResult,
    intentional_updates_development_record,
    screening_spec,
)

pytestmark = pytest.mark.integration


def _control(
    arm: str, *, seed: int, horizon: int = 512, phase_length: int = 64
) -> dict[str, object]:
    return lane._run_control_shard_authorized(
        arm,
        seed=seed,
        horizon=horizon,
        phase_length=phase_length,
        _capability=lane._EXECUTION_CAPABILITY,
    )


def test_plan_is_fresh_prospective_and_permanently_nonpromoting() -> None:
    plan = lane.frozen_plan()
    assert plan["seeds"] == [31_561_001, 31_561_002, 31_561_003, 31_561_004]
    assert plan["execution_authorized"] is False
    assert plan["scientific_promotion_allowed"] is False
    assert plan["negative_outcomes_retained"] is True
    assert plan["confidence_critical"] == 5.391949071934058
    assert plan["confidence_critical"].hex() == "0x1.5915b18f69e09p+2"
    assert plan["protocol_families"] == ["supervised_ipmnist", "td_control"]
    assert plan["dataset"]["x"]["sha256"] == (
        "b8078cd833f53d89828a5e28d728517be9add34076f13fe973399f1f16381313"
    )
    assert plan["dataset"]["y"]["sha256"] == (
        "4f1dd9551f104f8153409e0add59f0a71568f7bad5a5f8e2274480c186fe219a"
    )


def test_catalog_cli_is_read_only_and_execution_stays_closed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert lane.main(["--catalog"]) == 0
    assert json.loads(capsys.readouterr().out) == lane.frozen_plan()


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
    expected = _control(fixed, seed=lane.SEEDS[0], horizon=48, phase_length=12)
    actual = _control(off, seed=lane.SEEDS[0], horizon=48, phase_length=12)
    assert actual["arm"] == off
    assert actual["execution_arm"] == fixed
    for key in ("trajectory", "final_state", "metrics"):
        assert actual[key] == expected[key]
    expected_resources = dict(expected["resources"])
    actual_resources = dict(actual["resources"])
    expected_resources.pop("timing_telemetry_ns")
    actual_resources.pop("timing_telemetry_ns")
    assert actual_resources == expected_resources


@pytest.mark.parametrize("arm", lane.CONTROL_ARMS)
def test_each_control_arm_runs_end_to_end_with_exact_resources(arm: str) -> None:
    record = _control(arm, seed=lane.SEEDS[1], horizon=48, phase_length=12)
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
    prediction = _control(
        "intentional_trace", seed=lane.SEEDS[2], horizon=16, phase_length=4
    )
    control = _control(
        "intentional_q_lambda", seed=lane.SEEDS[2], horizon=16, phase_length=4
    )
    assert prediction["resources"]["action_queries"] == 0
    assert prediction["resources"]["rng_fold_ins"] == 0
    assert control["resources"]["action_queries"] == 16
    assert control["resources"]["rng_fold_ins"] == 16
    assert control["identity"]["agent_rng_impl"] == "threefry2x32"
    assert prediction["information"]["boundary_information"] == []
    assert prediction["information"]["task_information"] == []


def test_validator_rejects_nested_subclasses_without_hooks() -> None:
    record = _control(
        "intentional_trace", seed=lane.SEEDS[3], horizon=16, phase_length=4
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
    record = _control("intentional_td0", seed=lane.SEEDS[0], horizon=16, phase_length=4)
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
        lane.run_control_shard("fixed_td0", seed=lane.SEEDS[0])
    with pytest.raises(RuntimeError, match="not authorized"):
        lane.run_campaign(Path("unused.npz"), Path("unused.json"))


def test_execution_cli_fails_before_dataset_or_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"dataset": 0, "consumer": 0}

    def forbidden_dataset(*args: object, **kwargs: object) -> object:
        calls["dataset"] += 1
        raise AssertionError("dataset load occurred before authorization")

    def forbidden_consumer(*args: object, **kwargs: object) -> object:
        calls["consumer"] += 1
        raise AssertionError("consumer ran before authorization")

    monkeypatch.setattr(lane, "_load_dataset", forbidden_dataset)
    monkeypatch.setattr(lane, "_run", forbidden_consumer)
    with pytest.raises(RuntimeError, match="not authorized"):
        lane.main(["--dataset", "unused.npz"])
    assert calls == {"dataset": 0, "consumer": 0}


def test_validator_bounds_config_before_reexecution(monkeypatch: pytest.MonkeyPatch) -> None:
    record = _control("fixed_td0", seed=lane.SEEDS[0], horizon=16, phase_length=4)
    hostile = copy.deepcopy(record)
    hostile["config"]["horizon"] = 1 << 40
    calls = 0

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("validator reexecuted before exact bounds")

    monkeypatch.setattr(lane, "_run", forbidden)
    with pytest.raises(ValueError, match="horizon"):
        lane.validate_control_shard(hostile)
    assert calls == 0


def _synthetic_supervised_records() -> list[dict[str, object]]:
    records = []
    for seed in lane.SEEDS:
        for arm, offset in (
            ("intentional_updates_off", 0.0),
            ("intentional_updates_ipmnist", 0.01),
        ):
            spec = screening_spec(arm)
            result = ScreeningRunResult(
                config_name=arm,
                base_learner=spec.base_learner,
                hyperparameters=spec.hyperparameters,
                seed=seed,
                config=lane.SUPERVISED_CONFIG,
                per_task_accuracy=np.full(8, 0.5 + offset, dtype=np.float64),
                per_task_loss=np.full(8, 0.7 - offset, dtype=np.float64),
                per_task_plasticity=np.full(8, 0.4 + offset, dtype=np.float64),
                wall_clock_seconds=0.125,
            )
            records.append(intentional_updates_development_record(result))
    return records


@pytest.fixture(scope="module")
def complete_records() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    supervised = _synthetic_supervised_records()
    control = [
        _control(arm, seed=seed)
        for seed in lane.SEEDS
        for arm in lane.CONTROL_ARMS
    ]
    return supervised, control


def test_report_recomputes_all_four_bonferroni_paired_questions(
    complete_records: tuple[list[dict[str, object]], list[dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lane, "_current_source", lambda: {"git_commit": "c" * 40})
    monkeypatch.setattr(lane, "_current_runtime", lambda: {"backend": "cpu"})
    report = lane.build_report(
        *complete_records,
        dataset_provenance=lane.frozen_plan()["dataset"],
        execution_source_commit="c" * 40,
    )
    assert set(report["paired_comparisons"]) == {
        "supervised_ipmnist", "td0", "trace", "q_lambda"
    }
    assert all(
        item["outcome"] in {"supported", "rejected", "inconclusive"}
        for item in report["paired_comparisons"].values()
    )
    assert lane.validate_report(report, require_current_source=True) == report


def test_report_rejects_missing_shard_arithmetic_and_promotion(
    complete_records: tuple[list[dict[str, object]], list[dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lane, "_current_source", lambda: {"git_commit": "c" * 40})
    monkeypatch.setattr(lane, "_current_runtime", lambda: {"backend": "cpu"})
    report = lane.build_report(
        *complete_records,
        dataset_provenance=lane.frozen_plan()["dataset"],
        execution_source_commit="c" * 40,
    )
    missing = copy.deepcopy(report)
    missing["runs"][0]["control"].pop()
    with pytest.raises(ValueError, match="complete"):
        lane.validate_report(missing, require_current_source=True)
    forged = copy.deepcopy(report)
    forged["paired_comparisons"]["td0"]["mean_delta"] = 9.0
    with pytest.raises(ValueError, match="paired arithmetic"):
        lane.validate_report(forged, require_current_source=True)
    promoting = copy.deepcopy(report)
    promoting["policy"]["scientific_promotion_allowed"] = True
    with pytest.raises(ValueError, match="nonpromoting"):
        lane.validate_report(promoting, require_current_source=True)


def test_report_publication_is_no_replace_and_rejects_symlink_parent(
    tmp_path: Path,
    complete_records: tuple[list[dict[str, object]], list[dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lane, "_current_source", lambda: {"git_commit": "c" * 40})
    monkeypatch.setattr(lane, "_current_runtime", lambda: {"backend": "cpu"})
    report = lane.build_report(
        *complete_records,
        dataset_provenance=lane.frozen_plan()["dataset"],
        execution_source_commit="c" * 40,
    )
    destination = tmp_path / "new" / "report.json"
    monkeypatch.setattr(lane, "OUTPUT_PATH", destination)
    assert lane.publish_report(destination, report) == destination
    with pytest.raises(FileExistsError):
        lane.publish_report(destination, report)

    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)
    linked_destination = linked / "report.json"
    monkeypatch.setattr(lane, "OUTPUT_PATH", linked_destination)
    with pytest.raises(OSError):
        lane.publish_report(linked_destination, report)
    assert not (target / "report.json").exists()


def test_reservation_is_exclusive_and_parent_swap_stays_descriptor_pinned(
    tmp_path: Path,
    complete_records: tuple[list[dict[str, object]], list[dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lane, "_current_source", lambda: {"git_commit": "c" * 40})
    monkeypatch.setattr(lane, "_current_runtime", lambda: {"backend": "cpu"})
    report = lane.build_report(
        *complete_records,
        dataset_provenance=lane.frozen_plan()["dataset"],
        execution_source_commit="c" * 40,
    )
    requested = tmp_path / "requested"
    destination = requested / "report.json"
    monkeypatch.setattr(lane, "OUTPUT_PATH", destination)
    reservation = lane._reserve(destination)
    try:
        with pytest.raises(FileExistsError):
            lane._reserve(destination)
        moved = tmp_path / "moved"
        requested.rename(moved)
        requested.mkdir()
        lane._publish_reserved(reservation, report)
    finally:
        lane._release(reservation)
    assert (moved / "report.json").is_file()
    assert not destination.exists()


@pytest.mark.parametrize(
    ("horizon", "phase_length"),
    [(0, 1), (10_001, 1), (8, 0), (8, 9), (True, 1)],
)
def test_control_bounds_fail_before_execution(horizon: int, phase_length: int) -> None:
    with pytest.raises(ValueError):
        lane._run_control_shard_authorized(
            "fixed_td0",
            seed=lane.SEEDS[0],
            horizon=horizon,
            phase_length=phase_length,
            _capability=lane._EXECUTION_CAPABILITY,
        )
