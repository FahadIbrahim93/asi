import copy
import json
from pathlib import Path

import numpy as np
import pytest

import alberta_framework.evaluation.optimization_readiness_panel as panel_module
from alberta_framework.evaluation.optimization_readiness_panel import (
    PANEL_SCHEMA,
    OptimizationReadinessPanelCase,
    retain_optimization_readiness_panel,
    run_optimization_readiness_panel,
    validate_optimization_readiness_panel,
)


def _cases() -> tuple[OptimizationReadinessPanelCase, ...]:
    x = np.linspace(-1.0, 1.0, 20_000, dtype=np.float64).reshape(10_000, 2)
    y = 0.75 * x[:, 0] - 0.25 * x[:, 1]
    return tuple(
        OptimizationReadinessPanelCase(
            task_id="ipmnist-linear-readiness",
            checkpoint_id=f"checkpoint-{index}",
            seed=1_568_001 + index,
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
        "development_only": True,
        "negative_outcomes_retained": True,
        "scientific_promotion_allowed": False,
    }
    assert result["case_count"] == 6
    assert [case["execution"]["checkpoint_id"] for case in result["cases"]] == [
        "checkpoint-0",
        "checkpoint-1",
        "checkpoint-2",
        "checkpoint-3",
        "checkpoint-4",
        "checkpoint-5",
    ]
    correlations = result["association"]["spearman_by_predictor_and_gain_horizon"]
    assert set(correlations) == {
        "optimization_readiness",
        "gradient_norm",
        "representation_rank",
        "curvature_rank",
        "parameter_norm",
    }
    assert all(set(values) == {"1", "10", "100"} for values in correlations.values())
    assert result["association"]["status"] in {"supported", "rejected", "inconclusive"}
    assert result["resources"]["case_executions"] == 6
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
        "alberta_framework.evaluation.optimization_readiness_panel.execute_optimization_readiness",
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

    with pytest.raises(ValueError, match="content identities must be unique"):
        run_optimization_readiness_panel(tuple(cases))


def test_panel_retention_is_create_only_and_round_trips(tmp_path: Path) -> None:
    cases = _cases()
    result = run_optimization_readiness_panel(cases)
    destination = retain_optimization_readiness_panel(
        result,
        cases=cases,
        repository_root=tmp_path,
    )
    loaded = json.loads(destination.read_bytes())
    validate_optimization_readiness_panel(loaded, cases=cases)
    assert destination.parent == tmp_path / "outputs/optimization_readiness/development.v1"
    with pytest.raises(FileExistsError):
        retain_optimization_readiness_panel(
            result,
            cases=cases,
            repository_root=tmp_path,
        )


def test_panel_retention_rejects_namespace_symlink(tmp_path: Path) -> None:
    cases = _cases()
    result = run_optimization_readiness_panel(cases)
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "outputs").symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        retain_optimization_readiness_panel(result, cases=cases, repository_root=tmp_path)
    assert not (outside / "optimization_readiness").exists()


def test_panel_retention_removes_partial_bytes_after_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = _cases()
    result = run_optimization_readiness_panel(cases)

    def failed_write(_descriptor: int, _payload: bytes) -> int:
        raise OSError("injected write failure")

    monkeypatch.setattr(panel_module.os, "write", failed_write)
    with pytest.raises(OSError, match="injected"):
        retain_optimization_readiness_panel(result, cases=cases, repository_root=tmp_path)
    destination = tmp_path / "outputs/optimization_readiness/development.v1"
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
