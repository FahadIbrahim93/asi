from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, cast

import pytest

import alberta_framework.evaluation.adamo_matched_runner as runner
from alberta_framework.benchmarks.adamo_diagnostic import ARMS


def _receipt(seed: int) -> dict[str, object]:
    arms: list[dict[str, object]] = []
    offsets = {
        "adamw_control": 0.0,
        "adamo_inert": 0.0,
        "adamo_l1e3": 0.03,
        "adam_iso_joint_l1e3": -0.01,
    }
    for arm_index, arm in enumerate(ARMS):
        offset = offsets[arm]
        arms.append(
            {
                "arm": arm,
                "per_task_accuracy": [0.5 + offset] * 8,
                "per_task_loss": [0.7 - offset] * 8,
                "per_task_plasticity": [0.2 + offset] * 8,
                "post_task_diagnostics": [
                    {
                        "jacobian_rms_distance_from_one": 0.8 - offset,
                        "jacobian_condition_number_clipped_1e12": 4.0 - offset,
                        "weight_gram_penalty": 10.0 - offset,
                    }
                ],
                "resources": {
                    "observations": 512,
                    "updates": 512,
                    "data_steps": 512,
                    "environment_steps": 0,
                    "model_queries": 1040,
                    "jacobian_reverse_rows": 80,
                    "logical_compute_units": 1000 + arm_index,
                    "parameter_count": 282_160,
                    "persistent_numeric_bytes": 3_385_920,
                    "peak_gram_working_bytes": 360_000,
                    "timing_seconds": 1.25,
                },
            }
        )
    return {
        "profile": runner.PROFILE,
        "seed": seed,
        "frozen_development_seeds": list(runner.SEEDS),
        "dataset": {
            "rows": 60_000,
            "sha256": "a" * 64,
            "x_sha256": "b" * 64,
            "y_sha256": "c" * 64,
        },
        "runtime": {"identity": "same"},
        "arms": arms,
    }


def _install_validation_boundary(monkeypatch: pytest.MonkeyPatch) -> tuple[
    dict[str, object], dict[str, object], dict[str, object]
]:
    source = {"source": "bound"}
    dataset = {
        "x": {"shape": [60_000, 784], "sha256": "b" * 64},
        "y": {"shape": [60_000], "sha256": "c" * 64},
    }
    runtime = {"runtime": "bound"}

    def validate_receipt(value: object) -> dict[str, object]:
        if type(value) is not dict:
            raise ValueError("receipt must be an exact object")
        receipt = cast(dict[str, object], value)
        if receipt.get("profile") != runner.PROFILE:
            raise ValueError("profile drift")
        raw_arms = receipt.get("arms")
        if type(raw_arms) is not list or [item.get("arm") for item in raw_arms] != list(ARMS):
            raise ValueError("arm drift")
        if raw_arms[0]["per_task_accuracy"] != raw_arms[1]["per_task_accuracy"]:
            raise ValueError("inert reduction drift")
        return receipt

    monkeypatch.setattr(runner, "validate_adamo_diagnostic", validate_receipt)
    monkeypatch.setattr(runner, "_validated_source_provenance", lambda value, **_: value)
    monkeypatch.setattr(runner, "_validated_dataset_provenance", lambda value, **_: value)
    monkeypatch.setattr(runner, "_validated_runtime_environment", lambda value, **_: value)
    monkeypatch.setattr(runner, "_screening_source_provenance", lambda: source)
    monkeypatch.setattr(runner, "_screening_runtime_environment", lambda: runtime)
    return source, dataset, runtime


def _report(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    source, dataset, runtime = _install_validation_boundary(monkeypatch)
    return runner.build_report(
        [_receipt(seed) for seed in runner.SEEDS],
        source_provenance=source,
        dataset_provenance=dataset,
        environment=runtime,
    )


def test_plan_freezes_fresh_roster_data_statistics_and_nonpromotion() -> None:
    plan = runner.frozen_plan()
    assert runner.SEEDS == (15610, 15611, 15612, 15613)
    assert plan["profile"] == "bounded-development"
    assert plan["confidence_method"] == "two_sided_student_t"
    assert plan["confidence_degrees_of_freedom"] == 3
    assert "OpenML_mnist_784_v1" in plan["matched_axes"][0]
    assert plan["allowed_boundary_information"] == []
    assert len(plan["paper_protocol_differences"]) >= 5
    assert plan["development_only"] is True
    assert plan["scientific_promotion_allowed"] is False
    assert runner.OUTPUT_PATH.relative_to(runner._REPO_ROOT).as_posix() == (
        "outputs/adamo_matched_development/report.v1.json"
    )


def test_report_reconstructs_paired_outcomes_diagnostics_and_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report(monkeypatch)
    assert runner.validate_report(report) == report
    paired = cast(dict[str, dict[str, object]], report["paired_comparisons"])
    assert paired["adamo_l1e3"]["primary_outcome"] == "development_positive"
    assert paired["adam_iso_joint_l1e3"]["primary_outcome"] == "development_negative"
    assert paired["adamo_l1e3"]["final_jacobian_rms_improvements"] == pytest.approx(
        [0.03] * 4
    )
    resources = cast(dict[str, dict[str, object]], report["resource_totals"])
    assert resources["adamo_l1e3"]["observations"] == 2048
    assert resources["adamo_l1e3"]["model_queries"] == 4160
    assert resources["adamo_l1e3"]["timing_seconds"] == 5.0
    assert resources["adamo_l1e3"]["timing_is_telemetry_only"] is True


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value["receipts"].pop(), "every frozen seed"),
        (
            lambda value: value["records"][0].update(mean_online_accuracy=0.99),
            "do not reconstruct",
        ),
        (
            lambda value: value["resource_totals"]["adamo_l1e3"].update(model_queries=1),
            "do not reconstruct",
        ),
        (
            lambda value: value["policy"].update(scientific_promotion_allowed=True),
            "nonpromotion",
        ),
        (lambda value: value["plan"].update(seeds=[1, 2, 3, 4]), "literal frozen"),
        (
            lambda value: value["receipts"][0]["dataset"].update(x_sha256="d" * 64),
            "hashes do not match",
        ),
    ],
)
def test_validator_rejects_hostile_aggregate_changes(
    monkeypatch: pytest.MonkeyPatch,
    mutate: Any,
    message: str,
) -> None:
    hostile = copy.deepcopy(_report(monkeypatch))
    mutate(hostile)
    with pytest.raises(ValueError, match=message):
        runner.validate_report(hostile)


def test_report_rejects_duplicate_seed_and_nonexact_inert_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, dataset, runtime = _install_validation_boundary(monkeypatch)
    receipts = [_receipt(seed) for seed in runner.SEEDS]
    receipts[-1]["seed"] = runner.SEEDS[0]
    with pytest.raises(ValueError, match="unique frozen"):
        runner.build_report(
            receipts,
            source_provenance=source,
            dataset_provenance=dataset,
            environment=runtime,
        )
    receipts = [_receipt(seed) for seed in runner.SEEDS]
    cast(list[dict[str, object]], receipts[0]["arms"])[1]["per_task_accuracy"] = [0.6] * 8
    with pytest.raises(ValueError, match="inert reduction"):
        runner.build_report(
            receipts,
            source_provenance=source,
            dataset_provenance=dataset,
            environment=runtime,
        )


def test_output_reservation_rejects_symlink_and_existing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "outputs").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(runner, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        runner, "OUTPUT_PATH", tmp_path / "outputs/adamo_matched_development/report.v1.json"
    )
    with pytest.raises(OSError):
        runner._open_output_transaction()

    (tmp_path / "outputs").unlink()
    target = tmp_path / "outputs/adamo_matched_development/report.v1.json"
    target.parent.mkdir(parents=True)
    target.write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner._open_output_transaction()


def test_publication_is_pinned_create_only_and_reloaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _report(monkeypatch)
    monkeypatch.setattr(runner, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        runner, "OUTPUT_PATH", tmp_path / "outputs/adamo_matched_development/report.v1.json"
    )
    directory_fd, temporary_fd, temporary_name = runner._open_output_transaction()
    try:
        with pytest.raises(FileExistsError, match="already reserved"):
            runner._open_output_transaction()
        runner._publish_report(directory_fd, temporary_fd, temporary_name, report)
    finally:
        os.close(temporary_fd)
        os.unlink(temporary_name, dir_fd=directory_fd)
        os.close(directory_fd)
    retained = json.loads(runner.OUTPUT_PATH.read_bytes())
    assert runner.validate_report(retained) == retained
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner._open_output_transaction()


def test_cli_has_no_profile_dataset_or_output_override() -> None:
    with pytest.raises(SystemExit):
        runner.main(["--profile", "contract-smoke"])
    with pytest.raises(SystemExit):
        runner.main(["--dataset", "arbitrary.npz"])
    with pytest.raises(SystemExit):
        runner.main(["--output", "arbitrary.json"])
