import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import alberta_framework.evaluation.optimization_readiness_panel as panel_module
from alberta_framework.evaluation.optimization_readiness_panel import (
    PANEL_SCHEMA,
    OptimizationReadinessPanelCase,
)

run_optimization_readiness_panel = panel_module._run_optimization_readiness_panel
validate_optimization_readiness_panel = panel_module._validate_optimization_readiness_panel


def _cases() -> tuple[OptimizationReadinessPanelCase, ...]:
    x = np.linspace(-1.0, 1.0, 20_000, dtype=np.float64).reshape(10_000, 2)
    y = 0.75 * x[:, 0] - 0.25 * x[:, 1]
    return tuple(
        OptimizationReadinessPanelCase(
            task_id="bounded-linear-regression-v1",
            checkpoint_id=f"checkpoint-{index:02d}",
            seed=2_684_771_901 + index,
            validation_inputs=x,
            validation_labels=y,
            checkpoint_parameters=np.asarray(parameters, dtype=np.float64),
        )
        for index, parameters in enumerate(
            ([0.0, 0.0], [0.1, -0.05], [0.25, -0.1], [0.4, -0.15], [0.6, -0.2], [0.7, -0.23])
        )
    )


def test_panel_executes_unique_real_cases_and_derives_association() -> None:
    cases = _cases()
    result = run_optimization_readiness_panel(cases)

    assert result["schema"] == PANEL_SCHEMA
    assert result["policy"] == {
        "authorization_transition_approved": False,
        "execution_authorized": False,
        "development_only": True,
        "negative_outcomes_retained": True,
        "scientific_promotion_allowed": False,
    }
    assert result["case_count"] == 6
    assert [case["execution"]["checkpoint_id"] for case in result["cases"]] == [
        "checkpoint-00",
        "checkpoint-01",
        "checkpoint-02",
        "checkpoint-03",
        "checkpoint-04",
        "checkpoint-05",
    ]
    correlations = result["association"]["spearman_by_predictor_and_gain_horizon"]
    assert set(correlations) == {
        "optimization_readiness",
        "gradient_strength_mechanism_off",
        "gradient_norm",
        "representation_rank",
        "curvature_rank",
        "parameter_norm",
    }
    assert all(set(values) == {"1", "10", "100"} for values in correlations.values())
    assert result["association"]["status"] in {"supported", "rejected", "inconclusive"}
    assert result["resources"]["primary_case_executions"] == 6
    assert result["resources"]["strict_validation_case_executions"] == 6
    assert result["resources"]["aggregate_caller_bytes"] == sum(
        case.validation_inputs.nbytes
        + case.validation_labels.nbytes
        + case.checkpoint_parameters.nbytes
        for case in cases
    )
    assert result["resources"]["model_queries"] == sum(
        case["resources"]["model_queries"] for case in result["cases"]
    )
    assert result["resources"]["data_steps"] == sum(
        case["resources"]["data_steps"] for case in result["cases"]
    )
    assert result["resources"]["environment_steps"] == 0
    assert result["resources"]["retained_transaction_model_queries"] == 2 * result[
        "resources"
    ]["model_queries"]
    assert result["resources"]["timing_seconds"] == 0.0
    assert result["resources"]["timing_is_telemetry_only"] is True
    validate_optimization_readiness_panel(result, cases=cases)


def test_panel_validator_reexecutes_and_rejects_forged_outcome() -> None:
    cases = _cases()
    result = run_optimization_readiness_panel(cases)
    forged = copy.deepcopy(result)
    forged["association"]["status"] = "supported"
    if result["association"]["status"] == "supported":
        forged["association"]["status"] = "rejected"
    with pytest.raises(ValueError, match="recompute"):
        validate_optimization_readiness_panel(forged, cases=cases)


def test_panel_requires_exact_unique_case_roster_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = list(_cases())
    cases[1] = OptimizationReadinessPanelCase(
        task_id=cases[0].task_id,
        checkpoint_id=cases[0].checkpoint_id,
        seed=999,
        validation_inputs=cases[1].validation_inputs,
        validation_labels=cases[1].validation_labels,
        checkpoint_parameters=cases[1].checkpoint_parameters,
    )

    def forbidden(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("duplicate roster must reject before execution")

    monkeypatch.setattr(
        panel_module.readiness_executor,
        "_execute_optimization_readiness",
        forbidden,
    )
    with pytest.raises(ValueError, match="unique"):
        run_optimization_readiness_panel(tuple(cases))


def test_panel_rejects_renamed_duplicate_dataset_checkpoint_content() -> None:
    cases = list(_cases())
    original = cases[0]
    registered = cases[1]
    cases[1] = OptimizationReadinessPanelCase(
        task_id=registered.task_id,
        checkpoint_id=registered.checkpoint_id,
        seed=registered.seed,
        validation_inputs=original.validation_inputs.copy(),
        validation_labels=original.validation_labels.copy(),
        checkpoint_parameters=original.checkpoint_parameters.copy(),
    )

    with pytest.raises(ValueError, match="checkpoint content identities must be unique"):
        run_optimization_readiness_panel(tuple(cases))


def test_panel_requires_one_exact_matched_task_dataset() -> None:
    cases = list(_cases())
    changed = cases[1]
    labels = changed.validation_labels.copy()
    labels[0] += 1.0
    cases[1] = OptimizationReadinessPanelCase(
        task_id=changed.task_id,
        checkpoint_id=changed.checkpoint_id,
        seed=changed.seed,
        validation_inputs=changed.validation_inputs,
        validation_labels=labels,
        checkpoint_parameters=changed.checkpoint_parameters,
    )
    with pytest.raises(ValueError, match="matched task dataset"):
        run_optimization_readiness_panel(tuple(cases))


@pytest.mark.skipif(sys.platform != "linux", reason="publication requires Linux")
def test_panel_retention_is_create_only_and_round_trips(tmp_path: Path) -> None:
    cases = _cases()
    destination = panel_module._run_and_retain_optimization_readiness_panel(
        cases,
        repository_root=type(panel_module.REGISTERED_REPOSITORY_ROOT)(tmp_path),
    )
    loaded = json.loads(destination.read_bytes())
    validate_optimization_readiness_panel(loaded, cases=cases)
    assert destination.parent == tmp_path / "outputs/optimization_readiness/prospective.v1"
    with pytest.raises(FileExistsError):
        panel_module._run_and_retain_optimization_readiness_panel(
            cases,
            repository_root=type(panel_module.REGISTERED_REPOSITORY_ROOT)(tmp_path),
        )


@pytest.mark.skipif(sys.platform != "linux", reason="publication requires Linux")
def test_panel_retention_rejects_namespace_symlink(tmp_path: Path) -> None:
    cases = _cases()
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "outputs").symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        panel_module._run_and_retain_optimization_readiness_panel(
            cases,
            repository_root=type(panel_module.REGISTERED_REPOSITORY_ROOT)(tmp_path),
        )
    assert not (outside / "optimization_readiness").exists()


@pytest.mark.skipif(sys.platform != "linux", reason="publication requires Linux")
def test_panel_retention_removes_partial_bytes_after_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = _cases()
    calls = 0

    def failed_write(_descriptor: int, _payload: bytes) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            return len(_payload)
        raise OSError("injected write failure")

    monkeypatch.setattr(panel_module.os, "write", failed_write)
    with pytest.raises(OSError, match="injected"):
        panel_module._run_and_retain_optimization_readiness_panel(
            cases,
            repository_root=type(panel_module.REGISTERED_REPOSITORY_ROOT)(tmp_path),
        )
    destination = tmp_path / "outputs/optimization_readiness/prospective.v1"
    assert list(destination.iterdir()) == []


def test_panel_rejects_oversized_case_roster_before_array_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _cases()[0]

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("oversized roster must reject before array access")

    monkeypatch.setattr(np, "asarray", forbidden)
    with pytest.raises(ValueError, match="six-case"):
        run_optimization_readiness_panel((case,) * 17)


def test_public_runner_fails_before_roster_or_namespace_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Hostile:
        def __iter__(self):
            raise AssertionError("unauthorized runner must not inspect cases")

    monkeypatch.setattr(
        panel_module,
        "REGISTERED_REPOSITORY_ROOT",
        type(panel_module.REGISTERED_REPOSITORY_ROOT)(tmp_path),
    )
    with pytest.raises(PermissionError, match="separately reviewed"):
        panel_module.run_and_retain_optimization_readiness_panel(Hostile())
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize(
    ("transition", "execution"),
    [(True, False), (False, True)],
)
def test_public_runner_rejects_each_authorization_flag_mismatch_before_work(
    transition: bool,
    execution: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        panel_module,
        "REGISTERED_REPOSITORY_ROOT",
        type(panel_module.REGISTERED_REPOSITORY_ROOT)(tmp_path),
    )
    monkeypatch.setattr(
        panel_module.readiness_executor,
        "AUTHORIZATION_TRANSITION_APPROVED",
        transition,
    )
    monkeypatch.setattr(
        panel_module.readiness_executor,
        "EXECUTION_AUTHORIZED",
        execution,
    )
    with pytest.raises(PermissionError, match="separately reviewed"):
        panel_module.run_and_retain_optimization_readiness_panel(object())
    assert not (tmp_path / "outputs").exists()


def test_public_runner_rejects_unreviewed_plan_transition_before_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        panel_module,
        "REGISTERED_REPOSITORY_ROOT",
        type(panel_module.REGISTERED_REPOSITORY_ROOT)(tmp_path),
    )
    monkeypatch.setattr(
        panel_module.readiness_executor,
        "AUTHORIZATION_TRANSITION_APPROVED",
        True,
    )
    monkeypatch.setattr(panel_module.readiness_executor, "EXECUTION_AUTHORIZED", True)
    with pytest.raises(PermissionError, match="plan has not passed"):
        panel_module.run_and_retain_optimization_readiness_panel(object())
    assert not (tmp_path / "outputs").exists()


def test_public_validator_denies_before_payload_or_case_inspection() -> None:
    class Hostile(dict[str, object]):
        def __iter__(self):
            raise AssertionError("unauthorized validator must not inspect the payload")

    with pytest.raises(PermissionError, match="separately reviewed"):
        panel_module.validate_optimization_readiness_panel(Hostile(), cases=object())


@pytest.mark.skipif(sys.platform != "linux", reason="publication requires Linux")
def test_private_transaction_reserves_before_roster_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = (
        tmp_path
        / "outputs/optimization_readiness/prospective.v1/.result.json.reservation"
    )

    def stop_after_reservation(_cases: object) -> dict[str, object]:
        assert marker.is_file()
        raise RuntimeError("stop after reservation")

    monkeypatch.setattr(panel_module, "_run_optimization_readiness_panel", stop_after_reservation)
    with pytest.raises(RuntimeError, match="stop after reservation"):
        panel_module._run_and_retain_optimization_readiness_panel(
            object(),
            repository_root=type(panel_module.REGISTERED_REPOSITORY_ROOT)(tmp_path),
        )
    assert not marker.exists()


def test_frozen_plan_binds_fresh_roster_authorization_and_parity_gaps() -> None:
    plan = panel_module.frozen_optimization_readiness_plan()
    assert plan["authorization"] == {
        "authorization_transition_approved": False,
        "execution_authorized": False,
    }
    assert [case["seed"] for case in plan["roster"]] == list(range(2_684_771_901, 2_684_771_907))
    assert plan["seed_status"]["classification"] == (
        "frozen_exposed_unexecuted_consumed_for_promotion"
    )
    assert plan["seed_status"]["execution_status"] == "unexecuted"
    assert plan["seed_status"]["promotion_eligible"] is False
    assert plan["paper_parity_gaps"]
