from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import jax
import jax.random as jr
import numpy as np
import pytest

import alberta_framework.benchmarks.activation_feature_ipmnist as activation
import alberta_framework.evaluation.activation_feature_matched_runner as runner
from alberta_framework.benchmarks.activation_feature_ipmnist import (
    ACTIVATION_FEATURE_SPECS,
    DEVELOPMENT_SEEDS,
    run_activation_feature_arm,
)
from alberta_framework.benchmarks.ipmnist_screening import ScreeningRunResult
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig, build_schedule

SMALL = IPMNISTConfig(n_tasks=1, task_length=2, input_dim=2, hidden1=2, hidden2=2, n_classes=2)


def _data() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([[-1.0, 1.0], [1.0, -1.0]], dtype=np.float32),
        np.asarray([0, 1], dtype=np.int32),
    )


def _resign(value: dict[str, Any]) -> None:
    unsigned = dict(value)
    unsigned.pop("result_sha256", None)
    value["result_sha256"] = hashlib.sha256(runner._canonical(unsigned)).hexdigest()


def _fake_arm(
    data_x: np.ndarray,
    data_y: np.ndarray,
    *,
    arm: str,
    seed: int,
    config: IPMNISTConfig,
) -> activation.ActivationFeatureRunResult:
    spec = ACTIVATION_FEATURE_SPECS[arm]
    value = float(seed + tuple(ACTIVATION_FEATURE_SPECS).index(arm)) / 100.0
    screening = ScreeningRunResult(
        config_name=arm,
        base_learner="upgd_w",
        hyperparameters=dict(spec.hyperparameters),
        seed=seed,
        config=config,
        per_task_accuracy=np.full(config.n_tasks, value, dtype=np.float64),
        per_task_loss=np.full(config.n_tasks, 1.0 - value, dtype=np.float64),
        per_task_plasticity=np.full(config.n_tasks, value, dtype=np.float64),
        wall_clock_seconds=1.0,
    )
    _, schedule_key, _ = jr.split(jr.key(np.uint32(seed), impl="threefry2x32"), 3)
    schedule = build_schedule(schedule_key, config, data_x.shape[0])
    return activation.ActivationFeatureRunResult(
        screening=screening,
        dataset_sha256=activation._array_bundle_sha256(data_x, data_y),
        schedule_sha256=activation._schedule_sha256(
            schedule.permutations, schedule.example_indices
        ),
        source_identity=activation._current_source_identity(),
        runtime_identity=activation._runtime_identity(),
        n_train=data_x.shape[0],
        peak_schedule_working_bytes=activation._preflight_activation_feature_resources(
            config, n_train=data_x.shape[0]
        ),
    )


def test_campaign_executes_complete_frozen_roster_and_validates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "run_activation_feature_arm", _fake_arm)
    result = cast(dict[str, Any], runner.run_activation_feature_matched(*_data(), config=SMALL))
    runner.validate_activation_feature_matched(result, *_data(), config=SMALL)
    assert result["policy"] == {
        "development_only": True,
        "scientific_promotion_allowed": False,
        "sota_claim_allowed": False,
        "negative_results_retained": True,
    }
    assert [(row["seed"], row["arm"]) for row in result["rows"]] == [
        (seed, arm) for seed in DEVELOPMENT_SEEDS for arm in ACTIVATION_FEATURE_SPECS
    ]
    assert all(row["result"]["outcome"] == "inconclusive" for row in result["rows"])


def test_validator_rejects_roster_identity_and_aggregate_forgery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "run_activation_feature_arm", _fake_arm)
    result = cast(dict[str, Any], runner.run_activation_feature_matched(*_data(), config=SMALL))
    missing = copy.deepcopy(result)
    missing["rows"].pop()
    _resign(missing)
    with pytest.raises(ValueError, match="roster"):
        runner.validate_activation_feature_matched(missing, *_data(), config=SMALL)

    forged = copy.deepcopy(result)
    forged["identity"]["dataset_sha256"] = "0" * 64
    _resign(forged)
    with pytest.raises(ValueError, match="identity"):
        runner.validate_activation_feature_matched(forged, *_data(), config=SMALL)

    forged = copy.deepcopy(result)
    forged["aggregate"]["arms"]["aid"]["mean_accuracy"] += 0.1
    _resign(forged)
    with pytest.raises(ValueError, match="aggregate"):
        runner.validate_activation_feature_matched(forged, *_data(), config=SMALL)


def test_validator_reexecutes_and_rejects_self_consistent_metric_forgery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "run_activation_feature_arm", _fake_arm)
    result = cast(dict[str, Any], runner.run_activation_feature_matched(*_data(), config=SMALL))
    forged = copy.deepcopy(result)
    forged["rows"][0]["result"]["metrics"]["asi_whole_stream_mean_accuracy"] = 0.99
    forged["aggregate"] = runner._aggregate(forged["rows"])
    _resign(forged)

    with pytest.raises(ValueError, match="reexecution"):
        runner.validate_activation_feature_matched(forged, *_data(), config=SMALL)


def test_campaign_rejects_unregistered_outcome_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "run_activation_feature_arm", _fake_arm)
    result = cast(dict[str, Any], runner.run_activation_feature_matched(*_data(), config=SMALL))
    forged = copy.deepcopy(result)
    forged["rows"][0]["result"]["outcome"] = "supported"
    _resign(forged)

    with pytest.raises(ValueError, match="inconclusive"):
        runner.validate_activation_feature_matched(forged, *_data(), config=SMALL)


def test_rows_bind_initial_parameters_and_schedule(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "run_activation_feature_arm", _fake_arm)
    result = cast(dict[str, Any], runner.run_activation_feature_matched(*_data(), config=SMALL))
    first = result["rows"][0]["execution_identity"]
    assert first["prng_implementation"] == "threefry2x32"
    assert len(first["initial_parameters_sha256"]) == 64
    assert len(first["schedule_sha256"]) == 64

    forged = copy.deepcopy(result)
    forged["rows"][0]["execution_identity"]["initial_parameters_sha256"] = "0" * 64
    _resign(forged)
    with pytest.raises(ValueError, match="execution identity"):
        runner.validate_activation_feature_matched(forged, *_data(), config=SMALL)


def test_writer_is_create_only_and_revalidates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "run_activation_feature_arm", _fake_arm)
    result = runner.run_activation_feature_matched(*_data(), config=SMALL)
    destination = tmp_path / "activation-feature.json"
    runner.write_activation_feature_matched(destination, result, *_data(), config=SMALL)
    retained = json.loads(destination.read_bytes())
    runner.validate_activation_feature_matched(retained, *_data(), config=SMALL)
    with pytest.raises(FileExistsError):
        runner.write_activation_feature_matched(destination, result, *_data(), config=SMALL)


def test_campaign_rejects_dataset_before_arm_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("arm must not execute")

    monkeypatch.setattr(runner, "run_activation_feature_arm", forbidden)
    x, y = _data()
    bad = x.astype(np.float64)
    with pytest.raises(ValueError, match="float32"):
        runner.run_activation_feature_matched(bad, y, config=SMALL)
    assert calls == 0


def test_source_identity_uses_only_installed_package_files() -> None:
    identity = runner._source_identity()
    assert "uv.lock" not in identity
    assert set(identity) == {
        "alberta_framework/_seed_validation.py",
        "alberta_framework/benchmarks/activation_feature_ipmnist.py",
        "alberta_framework/benchmarks/plasticity_comparators.py",
        "alberta_framework/benchmarks/ipmnist_screening.py",
        "alberta_framework/benchmarks/upgd_ipmnist.py",
        "alberta_framework/evaluation/activation_feature_matched_runner.py",
    }


def test_shard_execution_is_independent_of_ambient_default_prng() -> None:
    with jax.default_prng_impl("threefry2x32"):
        expected = run_activation_feature_arm(*_data(), arm="aid", seed=4, config=SMALL)
    with jax.default_prng_impl("rbg"):
        actual = run_activation_feature_arm(*_data(), arm="aid", seed=4, config=SMALL)
    np.testing.assert_array_equal(actual.per_task_accuracy, expected.per_task_accuracy)
    np.testing.assert_array_equal(actual.per_task_loss, expected.per_task_loss)
    np.testing.assert_array_equal(actual.per_task_plasticity, expected.per_task_plasticity)
