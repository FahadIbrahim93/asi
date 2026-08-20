from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, cast

import jax
import numpy as np
import pytest

import alberta_framework.evaluation.replay_frozen_campaign as campaign_module
from alberta_framework.benchmarks.ipmnist_screening import ScreeningRunResult
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig
from alberta_framework.evaluation.replay_frozen_campaign import (
    CAMPAIGN_SCHEMA,
    retain_replay_frozen_campaign,
    run_replay_frozen_campaign,
    validate_replay_frozen_campaign,
)
from alberta_framework.evaluation.replay_frozen_ipmnist_nonpromoting import (
    DEVELOPMENT_SEEDS,
    PROTOCOL_GAPS,
    registered_arms,
)


def _config() -> IPMNISTConfig:
    return IPMNISTConfig(
        n_tasks=1,
        task_length=4,
        input_dim=4,
        hidden1=3,
        hidden2=2,
        n_classes=2,
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


@pytest.fixture(autouse=True)
def _bounded_campaign_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    def dataset_provenance(_x: object, _y: object) -> dict[str, object]:
        return {
            "schema": "alberta.ipmnist_screening.dataset_provenance.v1",
            "source": {
                "provider": "openml",
                "name": "mnist_784",
                "version": 1,
                "row_start": 0,
                "row_stop_exclusive": 60_000,
            },
            "materialization": "alberta.ipmnist.float32-neg1-pos1-int32-labels.v1",
            "x": {
                "dtype": "<f4",
                "shape": [60_000, 784],
                "sha256": campaign_module._CANONICAL_X_SHA256,
            },
            "y": {
                "dtype": "<i4",
                "shape": [60_000],
                "sha256": campaign_module._CANONICAL_Y_SHA256,
            },
        }

    def execute(
        _data_x: object,
        _data_y: object,
        spec: Any,
        seed: int,
        config: IPMNISTConfig,
        **_kwargs: object,
    ) -> ScreeningRunResult:
        arm_index = registered_arms().index(spec.name)
        seed_index = DEVELOPMENT_SEEDS.index(seed)
        accuracy = 0.3 + 0.01 * arm_index + 0.001 * seed_index
        return ScreeningRunResult(
            config_name=spec.name,
            base_learner=spec.base_learner,
            hyperparameters=dict(spec.hyperparameters),
            seed=seed,
            config=config,
            per_task_accuracy=np.full(config.n_tasks, accuracy, dtype=np.float64),
            per_task_loss=np.full(config.n_tasks, 1.0 - accuracy, dtype=np.float64),
            per_task_plasticity=np.full(config.n_tasks, 0.1, dtype=np.float64),
            wall_clock_seconds=99.0,
            noise_mode="step",
            noise_pool_steps=None,
        )

    monkeypatch.setattr(campaign_module, "_screening_dataset_provenance", dataset_provenance)
    monkeypatch.setattr(campaign_module, "run_screening_config", execute)


@pytest.fixture
def completed_campaign() -> dict[str, Any]:
    x, y = _data()
    return cast(dict[str, Any], run_replay_frozen_campaign(x, y, config=_config()))


def test_campaign_has_exact_roster_inconclusive_policy_and_limitations(
    completed_campaign: dict[str, Any],
) -> None:
    assert completed_campaign["schema"] == CAMPAIGN_SCHEMA
    assert completed_campaign["status"] == "complete"
    assert completed_campaign["development_outcome"] == "inconclusive"
    assert completed_campaign["policy"] == {
        "development_only": True,
        "scientific_promotion_allowed": False,
        "sota_claim_allowed": False,
        "negative_outcomes_retained": True,
        "pretrained_ceiling_claim_allowed": False,
        "hillclimb_gate_evaluated": False,
    }
    assert completed_campaign["plan"]["decision_rule"] == "inconclusive_only_no_selection"
    assert [(row["seed"], row["arm"]) for row in completed_campaign["runs"]] == [
        (seed, arm) for seed in DEVELOPMENT_SEEDS for arm in registered_arms()
    ]
    assert completed_campaign["plan"]["protocol_gaps"] == list(PROTOCOL_GAPS)
    assert completed_campaign["resources"]["run_count"] == 40
    assert completed_campaign["resources"]["timing_measured"] is False
    assert completed_campaign["resources"]["peak_working_set_claimed"] is False
    assert completed_campaign["resources"][
        "persistent_numeric_bytes_across_seed_arm_identities"
    ] == completed_campaign["resources"]["receipt_integer_totals"]["persistent_bytes"]
    assert all(row["receipt"]["outcome"] == "inconclusive" for row in completed_campaign["runs"])


def test_source_identity_covers_every_package_source_and_lock(
    completed_campaign: dict[str, Any],
) -> None:
    source = completed_campaign["identity"]["source_sha256"]
    root = Path(campaign_module.__file__).resolve().parents[2]
    assert set(source) == {
        "pyproject.toml",
        "uv.lock",
        *(path.relative_to(root).as_posix() for path in (root / "alberta_framework").rglob("*.py")),
    }


def test_seed_identity_binds_explicit_threefry_schedule_parameters_and_all_states(
    completed_campaign: dict[str, Any],
) -> None:
    identities = completed_campaign["seed_identities"]
    assert [item["seed"] for item in identities] == list(DEVELOPMENT_SEEDS)
    assert all(item["rng_impl"] == "threefry2x32" for item in identities)
    assert all(
        [state["arm"] for state in item["initial_states"]] == list(registered_arms())
        for item in identities
    )
    by_seed = {item["seed"]: item["identity_sha256"] for item in identities}
    assert all(
        row["seed_identity_sha256"] == by_seed[row["seed"]]
        for row in completed_campaign["runs"]
    )


def test_seed_identity_is_ambient_prng_invariant() -> None:
    prior = str(jax.config.jax_default_prng_impl)
    try:
        jax.config.update("jax_default_prng_impl", "threefry2x32")
        seed = DEVELOPMENT_SEEDS[0]
        threefry = campaign_module._seed_identity(seed, config=_config(), n_train=4)
        jax.config.update("jax_default_prng_impl", "rbg")
        rbg = campaign_module._seed_identity(seed, config=_config(), n_train=4)
    finally:
        jax.config.update("jax_default_prng_impl", prior)
    assert threefry == rbg


def test_strict_validator_replays_every_shard(
    completed_campaign: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    execute = campaign_module.run_screening_config
    calls: list[tuple[int, str]] = []

    def counting_execute(
        data_x: object,
        data_y: object,
        spec: Any,
        seed: int,
        config: IPMNISTConfig,
        **kwargs: object,
    ) -> ScreeningRunResult:
        calls.append((seed, spec.name))
        return execute(data_x, data_y, spec, seed, config, **kwargs)

    monkeypatch.setattr(campaign_module, "run_screening_config", counting_execute)
    x, y = _data()
    validate_replay_frozen_campaign(completed_campaign, x, y, config=_config())
    assert calls == [(seed, arm) for seed in DEVELOPMENT_SEEDS for arm in registered_arms()]


def test_hostile_self_consistent_metric_forgery_requires_replay(
    completed_campaign: dict[str, Any],
) -> None:
    forged = copy.deepcopy(completed_campaign)
    forged["runs"][0]["receipt"]["metrics"]["mean_online_accuracy"] += 0.01
    campaign_module._recompute_derived_fields_for_test(forged)
    x, y = _data()
    with pytest.raises(ValueError, match="recompute exactly"):
        validate_replay_frozen_campaign(forged, x, y, config=_config())


def test_identity_forgery_rejects_before_shard_execution(
    completed_campaign: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    forged = copy.deepcopy(completed_campaign)
    forged["seed_identities"][0]["initial_states"][0]["sha256"] = "0" * 64
    campaign_module._recompute_derived_fields_for_test(forged)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("identity forgery must reject before execution")

    monkeypatch.setattr(campaign_module, "run_screening_config", forbidden)
    x, y = _data()
    with pytest.raises(ValueError, match="seed identities"):
        validate_replay_frozen_campaign(forged, x, y, config=_config())


def test_default_plan_is_bounded_without_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    config = campaign_module._preflight_config(campaign_module.FROZEN_CAMPAIGN_CONFIG)
    assert config.n_steps * len(DEVELOPMENT_SEEDS) * len(registered_arms()) == 4_000_000

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("preflight must not execute")

    monkeypatch.setattr(campaign_module, "run_screening_config", forbidden)
    assert campaign_module._plan(config)["selected_ipmnist_configuration"] is False
    assert campaign_module._plan(config)["execution_authorized"] is False


def test_input_preflight_rejects_before_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    x, y = _data()

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("execution must not start")

    monkeypatch.setattr(campaign_module, "run_screening_config", forbidden)
    with pytest.raises(ValueError, match="float32"):
        run_replay_frozen_campaign(x.astype(np.float64), y, config=_config())


@pytest.mark.parametrize("field", ("source", "materialization", "x", "y"))
def test_frozen_dataset_gate_rejects_self_consistent_identity_drift(field: str) -> None:
    identity = {
        "schema": "alberta.ipmnist_screening.dataset_provenance.v1",
        "source": dict(campaign_module._DATASET_SOURCE),
        "materialization": campaign_module._DATASET_MATERIALIZATION,
        "x": {
            "dtype": "<f4",
            "shape": [60_000, 784],
            "sha256": campaign_module._CANONICAL_X_SHA256,
        },
        "y": {
            "dtype": "<i4",
            "shape": [60_000],
            "sha256": campaign_module._CANONICAL_Y_SHA256,
        },
    }
    if field == "source":
        identity[field]["version"] = 2
    elif field == "materialization":
        identity[field] = "forged"
    else:
        identity[field]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="canonical identity"):
        campaign_module._require_frozen_dataset_identity(identity)


def test_source_drift_across_execution_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    x, y = _data()
    calls = 0

    def drifting_source() -> dict[str, str]:
        nonlocal calls
        calls += 1
        return {"source": ("a" if calls == 1 else "b") * 64}

    monkeypatch.setattr(campaign_module, "_source_identity", drifting_source)
    with pytest.raises(RuntimeError, match="source changed"):
        run_replay_frozen_campaign(x, y, config=_config())


def test_oversized_json_rejects_before_execution(
    completed_campaign: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    hostile = copy.deepcopy(completed_campaign)
    hostile["runs"] = [None] * 100_001

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("hostile payload must reject before execution")

    monkeypatch.setattr(campaign_module, "run_screening_config", forbidden)
    x, y = _data()
    with pytest.raises(ValueError, match="JSON|roster"):
        validate_replay_frozen_campaign(hostile, x, y, config=_config())


def test_create_only_retention_roundtrip(
    completed_campaign: dict[str, Any], tmp_path: Path
) -> None:
    x, y = _data()
    destination = retain_replay_frozen_campaign(
        completed_campaign,
        x,
        y,
        config=_config(),
        repository_root=tmp_path,
    )
    assert destination.name == f"result.{completed_campaign['result_sha256']}.json"
    validate_replay_frozen_campaign(json.loads(destination.read_bytes()), x, y, config=_config())
    with pytest.raises(FileExistsError):
        retain_replay_frozen_campaign(
            completed_campaign,
            x,
            y,
            config=_config(),
            repository_root=tmp_path,
        )


def test_retention_rejects_namespace_symlink(
    completed_campaign: dict[str, Any], tmp_path: Path
) -> None:
    x, y = _data()
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "outputs").symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        retain_replay_frozen_campaign(
            completed_campaign,
            x,
            y,
            config=_config(),
            repository_root=tmp_path,
        )
    assert list(outside.iterdir()) == []
