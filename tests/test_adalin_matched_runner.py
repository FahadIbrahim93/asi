from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import alberta_framework.evaluation.adalin_matched_runner as runner
from alberta_framework.benchmarks.adalin import run_adalin_development


@pytest.fixture(scope="module")
def data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x = np.eye(4, dtype=np.float32)
    train_y = np.asarray([0, 1, 0, 1], dtype=np.int32)
    return train_x, train_y, train_x.copy(), train_y.copy()


@pytest.fixture(scope="module")
def base_receipts(
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> dict[bool, dict[str, object]]:
    config = runner.CAMPAIGN_PROFILES["contract-smoke"]
    return {
        enabled: run_adalin_development(
            *data,
            config=config,
            seed=runner.DEVELOPMENT_SEEDS[0],
            mechanism_enabled=enabled,
        )
        for enabled in (False, True)
    }


def _fake_runner(base: dict[bool, dict[str, object]]) -> Any:
    def fake(
        train_inputs: np.ndarray,
        train_labels: np.ndarray,
        test_inputs: np.ndarray,
        test_labels: np.ndarray,
        *,
        config: object,
        seed: int,
        mechanism_enabled: bool,
    ) -> dict[str, object]:
        del train_inputs, train_labels, test_inputs, test_labels, config
        value = copy.deepcopy(base[mechanism_enabled])
        identity = runner._execution_identity(
            seed, "contract-smoke", 4, mechanism_enabled=mechanism_enabled
        )
        value["seed"] = seed
        value["provenance"]["schedule_sha256"] = identity["schedule_sha256"]
        value["provenance"]["initial_state_sha256"] = identity["initial_state_sha256"]
        return value

    return fake


def _resign(value: dict[str, Any]) -> None:
    unsigned = dict(value)
    unsigned.pop("result_sha256", None)
    value["result_sha256"] = hashlib.sha256(runner._canonical(unsigned)).hexdigest()


def test_campaign_runs_exact_five_seed_two_arm_roster(
    monkeypatch: pytest.MonkeyPatch,
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    base_receipts: dict[bool, dict[str, object]],
) -> None:
    monkeypatch.setattr(runner, "run_adalin_development", _fake_runner(base_receipts))
    result = cast(
        dict[str, Any], runner.run_adalin_matched(*data, profile="contract-smoke")
    )
    runner.validate_adalin_matched(result, *data, profile="contract-smoke")
    assert len(runner.DEVELOPMENT_SEEDS) == 5
    assert [(row["seed"], row["arm"]) for row in result["rows"]] == [
        (seed, arm) for seed in runner.DEVELOPMENT_SEEDS for arm in runner.ARMS
    ]
    assert result["decision"] == "inconclusive"
    first = result["rows"][0]["execution_identity"]
    assert first["prng_implementation"] == "threefry2x32"
    assert len(first["initial_state_sha256"]) == 64
    assert len(first["schedule_sha256"]) == 64


def test_validator_rejects_self_consistent_metric_forgery_and_decision(
    monkeypatch: pytest.MonkeyPatch,
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    base_receipts: dict[bool, dict[str, object]],
) -> None:
    monkeypatch.setattr(runner, "run_adalin_development", _fake_runner(base_receipts))
    result = cast(
        dict[str, Any], runner.run_adalin_matched(*data, profile="contract-smoke")
    )
    forged = copy.deepcopy(result)
    metrics = forged["rows"][1]["result"]["metrics"]
    metrics["task_preupdate_online_accuracy"][0] = 0.99
    metrics["asi_whole_stream_preupdate_online_accuracy"] = float(
        np.mean(metrics["task_preupdate_online_accuracy"])
    )
    forged["aggregate"] = runner._aggregate(forged["rows"])
    _resign(forged)
    with pytest.raises(ValueError, match="reexecution"):
        runner.validate_adalin_matched(forged, *data, profile="contract-smoke")

    decided = copy.deepcopy(result)
    decided["decision"] = "supported"
    _resign(decided)
    with pytest.raises(ValueError, match="inconclusive"):
        runner.validate_adalin_matched(decided, *data, profile="contract-smoke")

    mistyped_policy = copy.deepcopy(result)
    mistyped_policy["policy"]["development_only"] = 1
    _resign(mistyped_policy)
    with pytest.raises(ValueError, match="nonpromoting"):
        runner.validate_adalin_matched(
            mistyped_policy, *data, profile="contract-smoke"
        )


def test_validator_rejects_roster_identity_and_resource_forgery(
    monkeypatch: pytest.MonkeyPatch,
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    base_receipts: dict[bool, dict[str, object]],
) -> None:
    monkeypatch.setattr(runner, "run_adalin_development", _fake_runner(base_receipts))
    result = cast(
        dict[str, Any], runner.run_adalin_matched(*data, profile="contract-smoke")
    )
    missing = copy.deepcopy(result)
    missing["rows"].pop()
    _resign(missing)
    with pytest.raises(ValueError, match="roster"):
        runner.validate_adalin_matched(missing, *data, profile="contract-smoke")

    forged = copy.deepcopy(result)
    forged["rows"][0]["execution_identity"]["initial_state_sha256"] = "0" * 64
    _resign(forged)
    with pytest.raises(ValueError, match="execution identity"):
        runner.validate_adalin_matched(forged, *data, profile="contract-smoke")

    forged = copy.deepcopy(result)
    forged["rows"][0]["result"]["resources"]["model_queries"] += 1
    _resign(forged)
    with pytest.raises(ValueError, match="canonical count"):
        runner.validate_adalin_matched(forged, *data, profile="contract-smoke")


def test_writer_is_create_only_and_replays(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    base_receipts: dict[bool, dict[str, object]],
) -> None:
    monkeypatch.setattr(runner, "run_adalin_development", _fake_runner(base_receipts))
    result = runner.run_adalin_matched(*data, profile="contract-smoke")
    destination = tmp_path / "adalin-matched.json"
    runner.write_adalin_matched(
        destination, result, *data, profile="contract-smoke"
    )
    retained = json.loads(destination.read_bytes())
    runner.validate_adalin_matched(retained, *data, profile="contract-smoke")
    with pytest.raises(FileExistsError):
        runner.write_adalin_matched(
            destination, result, *data, profile="contract-smoke"
        )


def test_preflight_rejects_dataset_before_execution(
    monkeypatch: pytest.MonkeyPatch,
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    calls = 0

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("AdaLin row must not execute")

    monkeypatch.setattr(runner, "run_adalin_development", forbidden)
    with pytest.raises(ValueError, match="float32"):
        runner.run_adalin_matched(
            data[0].astype(np.float64), *data[1:], profile="contract-smoke"
        )
    assert calls == 0


def test_source_identity_covers_current_execution_closure() -> None:
    assert set(runner._source_identity()) == {
        "alberta_framework/benchmarks/adalin.py",
        "alberta_framework/evaluation/adalin_matched_runner.py",
    }
