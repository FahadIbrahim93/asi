from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import alberta_framework.evaluation.bounded_elastic_matched_runner as runner
from alberta_framework.benchmarks.ipmnist_screening import ScreeningRunResult, screening_spec
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig
from alberta_framework.evaluation.bounded_elastic_ipmnist_nonpromoting import (
    registered_bounded_elastic_hyperparameters,
)

SMALL = IPMNISTConfig(n_tasks=1, task_length=5000, input_dim=2, hidden1=4, hidden2=2, n_classes=2)


def _run_for_test(
    data_x: object, data_y: object, *, config: IPMNISTConfig
) -> dict[str, object]:
    return runner._run_bounded_elastic_matched_authorized(
        data_x,
        data_y,
        config=config,
        seeds=runner.TEST_ONLY_SEEDS,
        _capability=runner._TEST_EXECUTION_CAPABILITY,
    )


def test_plan_is_prospective_and_public_execution_is_hard_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = runner.frozen_plan()
    assert plan["seeds"] == [51_562_001, 51_562_002, 51_562_003, 51_562_004, 51_562_005]
    assert plan["reviewed_execution_transition"] is False
    assert plan["execution_authorized"] is False
    calls = 0

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("runner dispatched before authorization")

    monkeypatch.setattr(runner, "run_screening_config", forbidden)
    with pytest.raises(RuntimeError, match="not authorized"):
        runner.run_bounded_elastic_matched(*_data(), config=SMALL)
    assert calls == 0


def _data() -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(10_000, dtype=np.float32).reshape(5000, 2) / 10_000.0
    y = np.arange(5000, dtype=np.int32) % 2
    return x, y


def _resign(value: dict[str, Any]) -> None:
    unsigned = dict(value)
    unsigned.pop("result_sha256", None)
    value["result_sha256"] = hashlib.sha256(runner._canonical(unsigned)).hexdigest()


def _fake_run(
    data_x: np.ndarray,
    data_y: np.ndarray,
    spec: object,
    seed: int,
    config: IPMNISTConfig,
) -> ScreeningRunResult:
    del data_x, data_y
    resolved = cast(Any, spec)
    value = 0.5 + float(
        runner.TEST_ONLY_SEEDS.index(seed) + tuple(runner.ARMS).index(resolved.name)
    ) / 100.0
    return ScreeningRunResult(
        config_name=resolved.name,
        base_learner="upgd_w",
        hyperparameters=registered_bounded_elastic_hyperparameters(resolved.name),
        seed=seed,
        config=config,
        per_task_accuracy=np.asarray([value], dtype=np.float64),
        per_task_loss=np.asarray([1.0 - value], dtype=np.float64),
        per_task_plasticity=np.asarray([value], dtype=np.float64),
        wall_clock_seconds=1.0,
    )


def test_campaign_runs_all_four_arms_across_frozen_seeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "run_screening_config", _fake_run)
    result = cast(dict[str, Any], _run_for_test(*_data(), config=SMALL))
    runner._validate_bounded_elastic_matched_authorized(
        result, *_data(), config=SMALL, seeds=runner.TEST_ONLY_SEEDS,
        _capability=runner._TEST_EXECUTION_CAPABILITY,
    )
    assert [(row["seed"], row["arm"]) for row in result["rows"]] == [
        (seed, arm) for seed in runner.TEST_ONLY_SEEDS for arm in runner.ARMS
    ]
    assert all(row["result"]["outcome_retained"] is True for row in result["rows"])
    for seed in runner.TEST_ONLY_SEEDS:
        rows = [row for row in result["rows"] if row["seed"] == seed]
        assert len({row["execution_identity"]["schedule_sha256"] for row in rows}) == 1
        assert len({row["execution_identity"]["initial_parameters_sha256"] for row in rows}) == 1


def test_validator_rejects_identity_resource_and_roster_forgery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "run_screening_config", _fake_run)
    result = cast(dict[str, Any], _run_for_test(*_data(), config=SMALL))

    forged = copy.deepcopy(result)
    forged["rows"][0]["execution_identity"]["schedule_sha256"] = "0" * 64
    _resign(forged)
    with pytest.raises(ValueError, match="execution identity"):
        runner._validate_bounded_elastic_matched_authorized(
            forged, *_data(), config=SMALL, seeds=runner.TEST_ONLY_SEEDS,
            _capability=runner._TEST_EXECUTION_CAPABILITY,
        )

    forged = copy.deepcopy(result)
    forged["rows"][0]["result"]["resources"]["data_steps"] = 1
    _resign(forged)
    with pytest.raises(ValueError, match="step/query"):
        runner._validate_bounded_elastic_matched_authorized(
            forged, *_data(), config=SMALL, seeds=runner.TEST_ONLY_SEEDS,
            _capability=runner._TEST_EXECUTION_CAPABILITY,
        )

    missing = copy.deepcopy(result)
    missing["rows"].pop()
    _resign(missing)
    with pytest.raises(ValueError, match="roster"):
        runner._validate_bounded_elastic_matched_authorized(
            missing, *_data(), config=SMALL, seeds=runner.TEST_ONLY_SEEDS,
            _capability=runner._TEST_EXECUTION_CAPABILITY,
        )


def test_validator_reexecutes_and_rejects_self_consistent_metric_forgery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "run_screening_config", _fake_run)
    result = cast(dict[str, Any], _run_for_test(*_data(), config=SMALL))
    forged = copy.deepcopy(result)
    forged["rows"][0]["result"]["metrics"]["mean_online_accuracy"] = 0.99
    forged["aggregate"] = runner._aggregate(forged["rows"])
    _resign(forged)

    with pytest.raises(ValueError, match="reexecution"):
        runner._validate_bounded_elastic_matched_authorized(
            forged, *_data(), config=SMALL, seeds=runner.TEST_ONLY_SEEDS,
            _capability=runner._TEST_EXECUTION_CAPABILITY,
        )


def test_campaign_rejects_unregistered_outcome_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "run_screening_config", _fake_run)
    result = cast(dict[str, Any], _run_for_test(*_data(), config=SMALL))
    forged = copy.deepcopy(result)
    forged["rows"][0]["result"]["outcome"] = "supported"
    _resign(forged)

    with pytest.raises(ValueError, match="inconclusive"):
        runner._validate_bounded_elastic_matched_authorized(
            forged, *_data(), config=SMALL, seeds=runner.TEST_ONLY_SEEDS,
            _capability=runner._TEST_EXECUTION_CAPABILITY,
        )


def test_writer_is_create_only_and_retains_negative_outcomes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "run_screening_config", _fake_run)
    result = _run_for_test(*_data(), config=SMALL)
    destination = tmp_path / "bounded-elastic.json"
    monkeypatch.setattr(runner, "OUTPUT_PATH", destination)
    runner._write_bounded_elastic_matched_authorized(
        destination, result, *_data(), config=SMALL, seeds=runner.TEST_ONLY_SEEDS,
        _capability=runner._TEST_EXECUTION_CAPABILITY,
    )
    retained = json.loads(destination.read_bytes())
    runner._validate_bounded_elastic_matched_authorized(
        retained, *_data(), config=SMALL, seeds=runner.TEST_ONLY_SEEDS,
        _capability=runner._TEST_EXECUTION_CAPABILITY,
    )
    with pytest.raises(FileExistsError):
        runner._write_bounded_elastic_matched_authorized(
            destination, result, *_data(), config=SMALL, seeds=runner.TEST_ONLY_SEEDS,
            _capability=runner._TEST_EXECUTION_CAPABILITY,
        )


def test_public_writer_is_closed_before_creating_output(tmp_path: Path) -> None:
    destination = tmp_path / "never" / "report.json"
    with pytest.raises(RuntimeError, match="not authorized"):
        runner.write_bounded_elastic_matched(
            destination, {}, *_data(), config=SMALL
        )
    assert not destination.parent.exists()


def test_preflight_rejects_unbounded_or_wrong_dataset_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("screening must not execute")

    monkeypatch.setattr(runner, "run_screening_config", forbidden)
    x, y = _data()
    with pytest.raises(ValueError, match="float32"):
        runner._run_bounded_elastic_matched_authorized(
            x.astype(np.float64), y, config=SMALL, seeds=runner.TEST_ONLY_SEEDS,
            _capability=runner._TEST_EXECUTION_CAPABILITY,
        )
    huge = IPMNISTConfig(
        n_tasks=1,
        task_length=5000,
        input_dim=8_000,
        hidden1=10_000,
        hidden2=1,
        n_classes=2,
    )
    with pytest.raises(ValueError, match="persistent-memory"):
        runner._run_bounded_elastic_matched_authorized(
            x, y, config=huge, seeds=runner.TEST_ONLY_SEEDS,
            _capability=runner._TEST_EXECUTION_CAPABILITY,
        )
    assert calls == 0


def test_registered_controls_are_exact() -> None:
    for arm in runner.ARMS:
        assert screening_spec(arm).hyperparameters == registered_bounded_elastic_hyperparameters(
            arm
        )


def test_source_identity_uses_only_installed_package_files() -> None:
    assert set(runner._source_identity()) == {
        "alberta_framework/_seed_validation.py",
        "alberta_framework/benchmarks/ipmnist_screening.py",
        "alberta_framework/benchmarks/upgd_ipmnist.py",
        "alberta_framework/evaluation/bounded_elastic_ipmnist_nonpromoting.py",
        "alberta_framework/evaluation/bounded_elastic_matched_runner.py",
    }
