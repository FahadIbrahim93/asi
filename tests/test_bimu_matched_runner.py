from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import alberta_framework.evaluation.bimu_matched_nonpromoting as bimu_plan
import alberta_framework.evaluation.bimu_matched_runner as runner
from alberta_framework.evaluation.bimu_matched_nonpromoting import (
    BiMUMatchedDevelopmentPlan,
    _test_plan,
)
from alberta_framework.evaluation.bimu_matched_runner import (
    _execute_bimu_matched_development,
    validate_bimu_matched_result,
    write_bimu_matched_result,
)


def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = np.arange(8 * 3, dtype=np.float32).reshape(8, 3) / 32.0
    labels = np.arange(8, dtype=np.int32) % 2
    return features[:4], labels[:4], features[4:], labels[4:]


def test_matched_runner_executes_complete_two_arm_three_seed_roster() -> None:
    plan = _test_plan(input_dim=3, n_classes=2, examples=4)
    result = cast(dict[str, Any], _execute_bimu_matched_development(*_data(), plan=plan))
    validate_bimu_matched_result(result, *_data(), plan=plan)

    assert result["policy"] == {
        "development_only": True,
        "scientific_promotion_allowed": False,
        "sota_claim_allowed": False,
        "negative_results_retained": True,
        "execution_authorized": False,
        "authorization_transition_approved": False,
    }
    rows = result["rows"]
    assert [(row["seed"], row["arm"]) for row in rows] == [
        (seed, arm) for seed in plan.seeds for arm in plan.arm_names
    ]
    for seed in plan.seeds:
        control, candidate = [row for row in rows if row["seed"] == seed]
        assert control["result"]["dataset_sha256"] == candidate["result"]["dataset_sha256"]
        assert control["result"]["schedule_sha256"] == candidate["result"]["schedule_sha256"]
        assert (
            control["result"]["initial_state_sha256"] == candidate["result"]["initial_state_sha256"]
        )
        assert control["result"]["counters"] == candidate["result"]["counters"]


def test_matched_validator_recomputes_aggregate_and_rejects_axis_forgery() -> None:
    plan = _test_plan(input_dim=3, n_classes=2, examples=4)
    result = cast(dict[str, Any], _execute_bimu_matched_development(*_data(), plan=plan))

    forged = copy.deepcopy(result)
    forged["aggregate"]["paired_late_five_delta_mean"] += 0.01
    with pytest.raises(ValueError, match="aggregate"):
        validate_bimu_matched_result(forged, *_data(), plan=plan)

    forged = copy.deepcopy(result)
    forged["rows"][1]["result"]["schedule_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="matched|digest"):
        validate_bimu_matched_result(forged, *_data(), plan=plan)


def test_matched_runner_rejects_dataset_before_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _test_plan(input_dim=3, n_classes=2, examples=4)
    bad = list(_data())
    bad[0] = np.asarray(bad[0]).copy()
    bad[0][0, 0] += 1.0

    called = False

    def fail(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("execution must not start")

    monkeypatch.setattr(
        "alberta_framework.evaluation.bimu_matched_runner.run_bimu_development", fail
    )
    with pytest.raises(ValueError, match="dataset"):
        _execute_bimu_matched_development(*bad, plan=plan)
    assert called is False


def test_public_runner_and_writer_fail_closed_before_input_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(RuntimeError, match="not independently authorized"):
        runner.run_bimu_matched_development(object(), object(), object(), object())
    with pytest.raises(RuntimeError, match="not independently authorized"):
        write_bimu_matched_result(object(), object(), object(), object(), object())
    monkeypatch.setattr(runner, "AUTHORIZATION_TRANSITION_APPROVED", True)
    with pytest.raises(RuntimeError, match="not independently authorized"):
        runner.run_bimu_matched_development(object(), object(), object(), object())
    monkeypatch.setattr(runner, "AUTHORIZATION_TRANSITION_APPROVED", False)
    monkeypatch.setattr(runner, "EXECUTION_AUTHORIZED", True)
    with pytest.raises(RuntimeError, match="not independently authorized"):
        runner.run_bimu_matched_development(object(), object(), object(), object())


def test_matched_result_writer_is_append_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "EXECUTION_AUTHORIZED", True)
    monkeypatch.setattr(runner, "AUTHORIZATION_TRANSITION_APPROVED", True)
    monkeypatch.setattr(runner, "_REGISTERED_REPOSITORY_ROOT", tmp_path)
    plan = _test_plan(input_dim=3, n_classes=2, examples=4)
    result = _execute_bimu_matched_development(*_data(), plan=plan)
    destination = write_bimu_matched_result(result, *_data(), plan=plan)
    retained = json.loads(destination.read_bytes())
    validate_bimu_matched_result(retained, *_data(), plan=plan)
    with pytest.raises(FileExistsError):
        write_bimu_matched_result(result, *_data(), plan=plan)


def test_writer_rejects_invalid_root_and_symlinked_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "EXECUTION_AUTHORIZED", True)
    monkeypatch.setattr(runner, "AUTHORIZATION_TRANSITION_APPROVED", True)
    plan = _test_plan(input_dim=3, n_classes=2, examples=4)
    result = _execute_bimu_matched_development(*_data(), plan=plan)
    monkeypatch.setattr(runner, "_REGISTERED_REPOSITORY_ROOT", Path("relative"))
    with pytest.raises(RuntimeError, match="absolute POSIX Path"):
        write_bimu_matched_result(result, *_data(), plan=plan)
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "outputs").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(runner, "_REGISTERED_REPOSITORY_ROOT", tmp_path)
    with pytest.raises(OSError):
        write_bimu_matched_result(result, *_data(), plan=plan)
    assert list(outside.iterdir()) == []


def _resign(result: dict[str, Any], plan: BiMUMatchedDevelopmentPlan) -> None:
    result["aggregate"] = runner._aggregate(result["rows"], plan)
    unsigned = dict(result)
    unsigned.pop("result_sha256")
    result["result_sha256"] = hashlib.sha256(runner._canonical(unsigned)).hexdigest()


def test_validator_derives_frozen_counters_schedule_and_initial_state() -> None:
    plan = _test_plan(input_dim=3, n_classes=2, examples=4)
    original = cast(dict[str, Any], _execute_bimu_matched_development(*_data(), plan=plan))

    forged = copy.deepcopy(original)
    for row in forged["rows"]:
        counters = row["result"]["counters"]
        counters["label_queries"] = 0
        counters["optimizer_updates"] = 0
        counters["model_forward_queries"] = 120
    _resign(forged, plan)
    with pytest.raises(ValueError, match="counter|does not reproduce"):
        validate_bimu_matched_result(forged, *_data(), plan=plan)

    for field in ("schedule_sha256", "initial_state_sha256"):
        forged = copy.deepcopy(original)
        for row in forged["rows"]:
            row["result"][field] = "0" * 64
        _resign(forged, plan)
        with pytest.raises(ValueError, match="schedule|initial|does not reproduce"):
            validate_bimu_matched_result(forged, *_data(), plan=plan)


def test_validator_reexecutes_and_rejects_rehashed_forged_metrics() -> None:
    plan = _test_plan(input_dim=3, n_classes=2, examples=4)
    forged = copy.deepcopy(_execute_bimu_matched_development(*_data(), plan=plan))
    metrics = forged["rows"][0]["result"]["metrics"]
    metrics["asi_whole_stream_online_accuracy"] = 0.0
    metrics["paper_late_five_test_accuracy"] = 0.0
    metrics["online_correct"] = 0
    metrics["final_five_test_accuracy"] = [0.0] * 5
    metrics["final_five_test_correct"] = [0] * 5
    _resign(forged, plan)

    with pytest.raises(ValueError, match="does not reproduce"):
        validate_bimu_matched_result(forged, *_data(), plan=plan)


def test_validator_rejects_rehashed_authorization_flag_mismatch() -> None:
    plan = _test_plan(input_dim=3, n_classes=2, examples=4)
    forged = copy.deepcopy(_execute_bimu_matched_development(*_data(), plan=plan))
    forged["identity"]["authorization"]["execution_authorized"] = True
    forged["policy"]["execution_authorized"] = True
    _resign(forged, plan)
    with pytest.raises(ValueError, match="identity|policy"):
        validate_bimu_matched_result(forged, *_data(), plan=plan)


def test_source_identity_preserves_audited_dependency_lock() -> None:
    identity = runner._source_identity()
    assert "uv.lock" not in identity
    assert set(identity) == {
        "alberta_framework/benchmarks/bimu.py",
        "alberta_framework/evaluation/bimu_matched_nonpromoting.py",
        "alberta_framework/evaluation/bimu_matched_runner.py",
        "alberta_framework/evaluation/prospective_publication.py",
    }
    assert "uv.lock" in bimu_plan._source_identity()
