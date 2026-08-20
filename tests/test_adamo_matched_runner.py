from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import alberta_framework.evaluation.adamo_matched_runner as runner
from alberta_framework.benchmarks.adamo_diagnostic import (
    ARMS,
    FROZEN_DEVELOPMENT_SEEDS,
    run_adamo_diagnostic,
)


@pytest.fixture(scope="module")
def data() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(8, 4),
        np.arange(8, dtype=np.int32) % 2,
    )


@pytest.fixture(scope="module")
def base_receipt(data: tuple[np.ndarray, np.ndarray]) -> dict[str, object]:
    return run_adamo_diagnostic(
        *data,
        profile="contract-smoke",
        seed=FROZEN_DEVELOPMENT_SEEDS[0],
    )


def _fake_runner(base: dict[str, object]) -> Any:
    def fake(
        inputs: np.ndarray, labels: np.ndarray, *, profile: str, seed: int
    ) -> dict[str, object]:
        del inputs, labels
        value = copy.deepcopy(base)
        value["profile"] = profile
        value["seed"] = seed
        return value

    return fake


def _resign(value: dict[str, Any]) -> None:
    unsigned = dict(value)
    unsigned.pop("result_sha256", None)
    value["result_sha256"] = hashlib.sha256(runner._canonical(unsigned)).hexdigest()


def test_campaign_runs_five_seed_roster_and_binds_execution(
    monkeypatch: pytest.MonkeyPatch,
    data: tuple[np.ndarray, np.ndarray],
    base_receipt: dict[str, object],
) -> None:
    monkeypatch.setattr(runner, "run_adamo_diagnostic", _fake_runner(base_receipt))
    result = cast(
        dict[str, Any], runner.run_adamo_matched(*data, profile="contract-smoke")
    )
    runner.validate_adamo_matched(result, *data, profile="contract-smoke")
    assert len(FROZEN_DEVELOPMENT_SEEDS) == 5
    assert [shard["seed"] for shard in result["shards"]] == list(
        FROZEN_DEVELOPMENT_SEEDS
    )
    assert result["arms"] == list(ARMS)
    assert result["decision"] == "inconclusive"
    first = result["shards"][0]["execution_identity"]
    assert first["prng_implementation"] == "threefry2x32"
    assert len(first["initial_parameters_sha256"]) == 64
    assert len(first["schedule_sha256"]) == 64


def test_validator_rejects_self_consistent_metric_forgery_and_decision(
    monkeypatch: pytest.MonkeyPatch,
    data: tuple[np.ndarray, np.ndarray],
    base_receipt: dict[str, object],
) -> None:
    monkeypatch.setattr(runner, "run_adamo_diagnostic", _fake_runner(base_receipt))
    result = cast(
        dict[str, Any], runner.run_adamo_matched(*data, profile="contract-smoke")
    )
    forged = copy.deepcopy(result)
    forged["shards"][0]["result"]["arms"][2]["per_task_accuracy"][0] = 0.99
    forged["aggregate"] = runner._aggregate(forged["shards"])
    _resign(forged)
    with pytest.raises(ValueError, match="reexecution"):
        runner.validate_adamo_matched(forged, *data, profile="contract-smoke")

    decided = copy.deepcopy(result)
    decided["decision"] = "supported"
    _resign(decided)
    with pytest.raises(ValueError, match="inconclusive"):
        runner.validate_adamo_matched(decided, *data, profile="contract-smoke")


def test_validator_rejects_roster_identity_and_resource_forgery(
    monkeypatch: pytest.MonkeyPatch,
    data: tuple[np.ndarray, np.ndarray],
    base_receipt: dict[str, object],
) -> None:
    monkeypatch.setattr(runner, "run_adamo_diagnostic", _fake_runner(base_receipt))
    result = cast(
        dict[str, Any], runner.run_adamo_matched(*data, profile="contract-smoke")
    )
    missing = copy.deepcopy(result)
    missing["shards"].pop()
    _resign(missing)
    with pytest.raises(ValueError, match="roster"):
        runner.validate_adamo_matched(missing, *data, profile="contract-smoke")

    forged = copy.deepcopy(result)
    forged["shards"][0]["execution_identity"]["schedule_sha256"] = "0" * 64
    _resign(forged)
    with pytest.raises(ValueError, match="execution identity"):
        runner.validate_adamo_matched(forged, *data, profile="contract-smoke")

    forged = copy.deepcopy(result)
    forged["shards"][0]["result"]["arms"][0]["resources"]["model_queries"] += 1
    _resign(forged)
    with pytest.raises(ValueError, match="accounting mismatch"):
        runner.validate_adamo_matched(forged, *data, profile="contract-smoke")


def test_writer_is_create_only_and_replays(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    data: tuple[np.ndarray, np.ndarray],
    base_receipt: dict[str, object],
) -> None:
    monkeypatch.setattr(runner, "run_adamo_diagnostic", _fake_runner(base_receipt))
    result = runner.run_adamo_matched(*data, profile="contract-smoke")
    destination = tmp_path / "adamo-matched.json"
    runner.write_adamo_matched(
        destination, result, *data, profile="contract-smoke"
    )
    retained = json.loads(destination.read_bytes())
    runner.validate_adamo_matched(retained, *data, profile="contract-smoke")
    with pytest.raises(FileExistsError):
        runner.write_adamo_matched(
            destination, result, *data, profile="contract-smoke"
        )


def test_preflight_rejects_dataset_before_shard_execution(
    monkeypatch: pytest.MonkeyPatch,
    data: tuple[np.ndarray, np.ndarray],
) -> None:
    calls = 0

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("AdamO shard must not execute")

    monkeypatch.setattr(runner, "run_adamo_diagnostic", forbidden)
    with pytest.raises(ValueError, match="float32"):
        runner.run_adamo_matched(data[0].astype(np.float64), data[1], profile="contract-smoke")
    assert calls == 0


def test_source_identity_covers_current_execution_closure() -> None:
    assert set(runner._source_identity()) == {
        "alberta_framework/_seed_validation.py",
        "alberta_framework/benchmarks/adamo_diagnostic.py",
        "alberta_framework/benchmarks/ipmnist_screening.py",
        "alberta_framework/benchmarks/upgd_ipmnist.py",
        "alberta_framework/core/_float32_scalars.py",
        "alberta_framework/core/adamo.py",
        "alberta_framework/core/baseline_optimizers.py",
        "alberta_framework/core/optimizers.py",
        "alberta_framework/core/update_safety.py",
        "alberta_framework/evaluation/adamo_matched_runner.py",
    }
