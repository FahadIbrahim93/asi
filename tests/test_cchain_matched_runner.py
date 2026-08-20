from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import alberta_framework.evaluation.cchain_matched_runner as runner
from alberta_framework.benchmarks.ipmnist_screening import ScreeningRunResult
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig
from alberta_framework.evaluation.cchain_ipmnist_nonpromoting import DEVELOPMENT_SEEDS

SMALL = IPMNISTConfig(
    n_tasks=1, task_length=4, input_dim=4, hidden1=3, hidden2=2, n_classes=2
)


def _data() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(
            [
                [-1.0, -0.5, 0.5, 1.0],
                [1.0, 0.5, -0.5, -1.0],
                [-0.5, 1.0, -1.0, 0.5],
                [0.5, -1.0, 1.0, -0.5],
            ],
            dtype=np.float32,
        ),
        np.asarray([0, 1, 0, 1], dtype=np.int32),
    )


def _fake_run(
    data_x: np.ndarray,
    data_y: np.ndarray,
    spec: object,
    seed: int,
    config: IPMNISTConfig,
) -> ScreeningRunResult:
    del data_x, data_y
    resolved = cast(Any, spec)
    value = float(seed + runner.ARMS.index(resolved.name)) / 100.0
    return ScreeningRunResult(
        config_name=resolved.name,
        base_learner="adamw",
        hyperparameters=dict(resolved.hyperparameters),
        seed=seed,
        config=config,
        per_task_accuracy=np.full(config.n_tasks, value, dtype=np.float64),
        per_task_loss=np.full(config.n_tasks, 1.0 - value, dtype=np.float64),
        per_task_plasticity=np.full(config.n_tasks, value, dtype=np.float64),
        wall_clock_seconds=1.0,
        mechanism_diagnostics={
            "mean_probability_kl": value,
            "mean_logit_mse": value,
            "final_coefficient": 1.0,
            "diagnostic_updates": float(max(config.n_steps - 2, 0)),
            "ntk_threshold_rank": 1.0,
            "ntk_off_diagonal_abs_mean": value,
            "ntk_diagonal_mean": 1.0,
            "ntk_examples": float(min(config.n_steps, 4)),
        },
    )


def _resign(value: dict[str, Any]) -> None:
    unsigned = dict(value)
    unsigned.pop("result_sha256", None)
    value["result_sha256"] = hashlib.sha256(runner._canonical(unsigned)).hexdigest()


def test_campaign_runs_exact_frozen_roster_and_binds_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "run_screening_config", _fake_run)
    result = cast(dict[str, Any], runner.run_cchain_matched(*_data(), config=SMALL))
    runner.validate_cchain_matched(result, *_data(), config=SMALL)
    assert [(row["seed"], row["arm"]) for row in result["rows"]] == [
        (seed, arm) for seed in DEVELOPMENT_SEEDS for arm in runner.ARMS
    ]
    assert all(row["result"]["outcome"] == "inconclusive" for row in result["rows"])
    assert result["policy"]["scientific_promotion_allowed"] is False
    first = result["rows"][0]["execution_identity"]
    assert first["prng_implementation"] == "threefry2x32"
    assert len(first["initial_parameters_sha256"]) == 64
    assert len(first["schedule_sha256"]) == 64


def test_validator_rejects_forgery_roster_and_unregistered_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "run_screening_config", _fake_run)
    result = cast(dict[str, Any], runner.run_cchain_matched(*_data(), config=SMALL))

    forged = copy.deepcopy(result)
    forged["rows"][0]["result"]["metrics"]["mean_online_accuracy"] = 0.99
    forged["aggregate"] = runner._aggregate(forged["rows"])
    _resign(forged)
    with pytest.raises(ValueError, match="reexecution"):
        runner.validate_cchain_matched(forged, *_data(), config=SMALL)

    missing = copy.deepcopy(result)
    missing["rows"].pop()
    _resign(missing)
    with pytest.raises(ValueError, match="roster"):
        runner.validate_cchain_matched(missing, *_data(), config=SMALL)

    decided = copy.deepcopy(result)
    decided["rows"][0]["result"]["outcome"] = "supported"
    _resign(decided)
    with pytest.raises(ValueError, match="inconclusive"):
        runner.validate_cchain_matched(decided, *_data(), config=SMALL)


def test_validator_rejects_execution_and_resource_forgery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "run_screening_config", _fake_run)
    result = cast(dict[str, Any], runner.run_cchain_matched(*_data(), config=SMALL))

    forged = copy.deepcopy(result)
    forged["rows"][0]["execution_identity"]["initial_parameters_sha256"] = "0" * 64
    _resign(forged)
    with pytest.raises(ValueError, match="execution identity"):
        runner.validate_cchain_matched(forged, *_data(), config=SMALL)

    forged = copy.deepcopy(result)
    forged["rows"][0]["result"]["resources"]["model_queries"] += 1
    _resign(forged)
    with pytest.raises(ValueError, match="model_queries"):
        runner.validate_cchain_matched(forged, *_data(), config=SMALL)


def test_writer_is_create_only_and_replays_before_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "run_screening_config", _fake_run)
    result = runner.run_cchain_matched(*_data(), config=SMALL)
    destination = tmp_path / "cchain-matched.json"
    runner.write_cchain_matched(destination, result, *_data(), config=SMALL)
    retained = json.loads(destination.read_bytes())
    runner.validate_cchain_matched(retained, *_data(), config=SMALL)
    with pytest.raises(FileExistsError):
        runner.write_cchain_matched(destination, result, *_data(), config=SMALL)


def test_preflight_rejects_hostile_dataset_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("C-CHAIN must not execute")

    monkeypatch.setattr(runner, "run_screening_config", forbidden)
    x, y = _data()
    with pytest.raises(ValueError, match="float32"):
        runner.run_cchain_matched(x.astype(np.float64), y, config=SMALL)
    assert calls == 0


def test_source_identity_covers_the_current_execution_closure() -> None:
    assert set(runner._source_identity()) == {
        "alberta_framework/_seed_validation.py",
        "alberta_framework/benchmarks/cchain_ipmnist.py",
        "alberta_framework/benchmarks/ipmnist_screening.py",
        "alberta_framework/benchmarks/upgd_ipmnist.py",
        "alberta_framework/core/_float32_scalars.py",
        "alberta_framework/core/baseline_optimizers.py",
        "alberta_framework/core/optimizers.py",
        "alberta_framework/core/update_safety.py",
        "alberta_framework/evaluation/cchain_ipmnist_nonpromoting.py",
        "alberta_framework/evaluation/cchain_matched_runner.py",
    }
