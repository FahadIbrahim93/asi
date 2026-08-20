from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, cast

import jax
import numpy as np
import pytest

import alberta_framework.evaluation.noise_curvature_campaign as campaign_module
from alberta_framework.benchmarks.ipmnist_screening import (
    ScreeningRunResult,
    _screening_dataset_provenance,
)
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig
from alberta_framework.evaluation.noise_curvature_campaign import (
    CAMPAIGN_SCHEMA,
    retain_noise_curvature_campaign,
    run_noise_curvature_campaign,
    validate_noise_curvature_campaign,
)
from alberta_framework.evaluation.noise_curvature_ipmnist_nonpromoting import (
    DEVELOPMENT_SEEDS,
    registered_arms,
)


def _data() -> tuple[np.ndarray, np.ndarray, IPMNISTConfig]:
    x = np.linspace(-1.0, 1.0, 160, dtype=np.float32).reshape(40, 4)
    y = np.asarray([index % 2 for index in range(40)], dtype=np.int32)
    config = IPMNISTConfig(
        n_tasks=1,
        task_length=40,
        input_dim=4,
        hidden1=3,
        hidden2=2,
        n_classes=2,
    )
    return x, y, config


@pytest.fixture(autouse=True)
def _fake_campaign_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    def dataset_provenance(data_x: object, data_y: object) -> dict[str, object]:
        del data_x, data_y
        return {
            "schema": "alberta.ipmnist_screening.dataset_provenance.v1",
            "source": dict(campaign_module._CANONICAL_DATASET_SOURCE),
            "materialization": campaign_module._CANONICAL_DATASET_MATERIALIZATION,
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
        mode = float(spec.hyperparameters["controller_mode"])
        accuracy = 0.4 + 0.01 * mode + 0.001 * seed
        return ScreeningRunResult(
            config_name=spec.name,
            base_learner=spec.base_learner,
            hyperparameters=dict(spec.hyperparameters),
            seed=seed,
            config=config,
            per_task_accuracy=np.full(config.n_tasks, accuracy, dtype=np.float64),
            per_task_loss=np.full(config.n_tasks, 1.0 - accuracy, dtype=np.float64),
            per_task_plasticity=np.full(config.n_tasks, 0.1, dtype=np.float64),
            wall_clock_seconds=123.0,
            noise_mode="step",
            noise_pool_steps=None,
        )

    monkeypatch.setattr(
        campaign_module, "_screening_dataset_provenance", dataset_provenance
    )
    monkeypatch.setattr(campaign_module, "run_screening_config", execute)


@pytest.fixture
def completed_campaign() -> dict[str, Any]:
    x, y, config = _data()
    return cast(dict[str, Any], run_noise_curvature_campaign(x, y, config=config))


def test_campaign_executes_exact_five_seed_four_arm_roster(
    completed_campaign: dict[str, Any],
) -> None:
    result = completed_campaign
    assert result["schema"] == CAMPAIGN_SCHEMA
    assert result["status"] == "complete"
    assert result["policy"] == {
        "development_only": True,
        "scientific_promotion_allowed": False,
        "sota_claim_allowed": False,
        "negative_outcomes_retained": True,
        "live_control_included": False,
        "hillclimb_gate_evaluated": False,
    }
    assert [(row["seed"], row["arm"]) for row in result["runs"]] == [
        (seed, arm) for seed in DEVELOPMENT_SEEDS for arm in registered_arms()
    ]
    assert [item["seed"] for item in result["seed_identities"]] == list(
        DEVELOPMENT_SEEDS
    )
    assert all(item["rng_impl"] == "threefry2x32" for item in result["seed_identities"])
    identities = {item["seed"]: item["identity_sha256"] for item in result["seed_identities"]}
    assert all(row["seed_identity_sha256"] == identities[row["seed"]] for row in result["runs"])
    assert result["plan"]["selected_ipmnist_configuration"] is False
    assert result["plan"]["multiple_comparison_correction"] == (
        "none_development_screen_only"
    )
    assert result["resources"]["run_count"] == 20
    assert result["resources"]["timing_measured"] is False
    assert result["resources"]["peak_working_set_claimed"] is False
    assert result["identity"]["consistency_not_attestation"] is True
    assert result["identity"]["dataset"]["source"] == (
        campaign_module._CANONICAL_DATASET_SOURCE
    )
    source_identity = result["identity"]["source_sha256"]
    repository_root = Path(campaign_module.__file__).resolve().parents[2]
    assert set(source_identity) == {
        "pyproject.toml",
        "uv.lock",
        *(
            path.relative_to(repository_root).as_posix()
            for path in (repository_root / "alberta_framework").rglob("*.py")
        ),
    }
    assert len(result["comparisons"]) == 3
    assert result["resources"]["counter_totals"]["model_queries"] == sum(
        row["receipt"]["resources"]["model_queries"] for row in result["runs"]
    )


def test_strict_validator_reexecutes_and_rejects_metric_forgery(
    completed_campaign: dict[str, Any],
) -> None:
    x, y, config = _data()
    validate_noise_curvature_campaign(copy.deepcopy(completed_campaign), x, y, config=config)
    forged = copy.deepcopy(completed_campaign)
    forged["runs"][0]["receipt"]["metrics"]["mean_online_accuracy"] += 0.01
    campaign_module._resign_for_test(forged)
    with pytest.raises(ValueError, match="recompute exactly"):
        validate_noise_curvature_campaign(forged, x, y, config=config)


def test_strict_validator_replays_every_seed_arm_shard(
    completed_campaign: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    x, y, config = _data()
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
    validate_noise_curvature_campaign(completed_campaign, x, y, config=config)
    assert calls == [
        (seed, arm) for seed in DEVELOPMENT_SEEDS for arm in registered_arms()
    ]


def test_seed_identity_is_explicit_threefry_and_ambient_invariant() -> None:
    _, _, config = _data()
    prior = str(jax.config.jax_default_prng_impl)
    try:
        jax.config.update("jax_default_prng_impl", "threefry2x32")
        threefry = campaign_module._seed_identity(0, config=config, n_train=40)
        jax.config.update("jax_default_prng_impl", "rbg")
        rbg = campaign_module._seed_identity(0, config=config, n_train=40)
    finally:
        jax.config.update("jax_default_prng_impl", prior)
    assert threefry == rbg


def test_selected_full_configuration_fits_the_frozen_preflight() -> None:
    config = campaign_module._preflight_config(IPMNISTConfig())
    assert config.n_steps * len(DEVELOPMENT_SEEDS) * len(registered_arms()) == 20_000_000
    assert campaign_module._plan(config)["selected_ipmnist_configuration"] is True


def test_input_preflight_rejects_before_any_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, y, config = _data()
    bad = x.astype(np.float64)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("execution must not start")

    monkeypatch.setattr(campaign_module, "run_screening_config", forbidden)
    with pytest.raises(ValueError, match="float32"):
        run_noise_curvature_campaign(bad, y, config=config)


def test_campaign_requires_the_current_frozen_dataset_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, y, config = _data()

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("execution must not start")

    monkeypatch.setattr(
        campaign_module,
        "_screening_dataset_provenance",
        _screening_dataset_provenance,
    )
    monkeypatch.setattr(campaign_module, "run_screening_config", forbidden)
    with pytest.raises(ValueError, match="OpenML mnist_784"):
        run_noise_curvature_campaign(x, y, config=config)


def test_frozen_dataset_gate_rejects_shape_valid_noncanonical_hashes() -> None:
    forged = {
        "schema": "alberta.ipmnist_screening.dataset_provenance.v1",
        "source": dict(campaign_module._CANONICAL_DATASET_SOURCE),
        "materialization": campaign_module._CANONICAL_DATASET_MATERIALIZATION,
        "x": {"dtype": "<f4", "shape": [60_000, 784], "sha256": "0" * 64},
        "y": {"dtype": "<i4", "shape": [60_000], "sha256": "1" * 64},
    }
    with pytest.raises(ValueError, match="canonical.*checksum|checksum.*canonical"):
        campaign_module._require_frozen_dataset_identity(forged)


def test_frozen_dataset_gate_rejects_source_or_materialization_drift() -> None:
    identity = {
        "schema": "alberta.ipmnist_screening.dataset_provenance.v1",
        "source": dict(campaign_module._CANONICAL_DATASET_SOURCE),
        "materialization": campaign_module._CANONICAL_DATASET_MATERIALIZATION,
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
    for field, forged_value in (
        ("source", {"provider": "not-openml"}),
        ("materialization", "forged"),
    ):
        forged = copy.deepcopy(identity)
        forged[field] = forged_value
        with pytest.raises(ValueError, match="source|materialization"):
            campaign_module._require_frozen_dataset_identity(forged)


def test_campaign_rejects_source_drift_across_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, y, config = _data()
    calls = 0

    def drifting_source() -> dict[str, str]:
        nonlocal calls
        calls += 1
        return {"source": "a" * 64 if calls == 1 else "b" * 64}

    monkeypatch.setattr(campaign_module, "_source_identity", drifting_source)
    with pytest.raises(RuntimeError, match="source changed"):
        run_noise_curvature_campaign(x, y, config=config)


def test_json_preflight_rejects_oversized_payload_before_reexecution(
    completed_campaign: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, y, config = _data()
    hostile = copy.deepcopy(completed_campaign)
    hostile["runs"] = [None] * 100_001

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("reexecution must not start")

    monkeypatch.setattr(campaign_module, "run_screening_config", forbidden)
    with pytest.raises(ValueError, match="JSON|roster"):
        validate_noise_curvature_campaign(hostile, x, y, config=config)


def test_validator_rejects_schedule_identity_before_reexecution(
    completed_campaign: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, y, config = _data()
    forged = copy.deepcopy(completed_campaign)
    forged["seed_identities"][0]["schedule_sha256"] = "0" * 64
    campaign_module._resign_for_test(forged)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("identity drift must reject before benchmark execution")

    monkeypatch.setattr(campaign_module, "run_screening_config", forbidden)
    with pytest.raises(ValueError, match="seed identities"):
        validate_noise_curvature_campaign(forged, x, y, config=config)


def test_retention_is_content_named_create_only_and_round_trips(
    completed_campaign: dict[str, Any], tmp_path: Path
) -> None:
    x, y, config = _data()
    destination = retain_noise_curvature_campaign(
        completed_campaign,
        x,
        y,
        config=config,
        repository_root=tmp_path,
    )
    assert destination.name == f"result.{completed_campaign['result_sha256']}.json"
    retained = json.loads(destination.read_bytes())
    validate_noise_curvature_campaign(retained, x, y, config=config)
    with pytest.raises(FileExistsError):
        retain_noise_curvature_campaign(
            completed_campaign,
            x,
            y,
            config=config,
            repository_root=tmp_path,
        )


def test_retention_rejects_namespace_symlink(
    completed_campaign: dict[str, Any], tmp_path: Path
) -> None:
    x, y, config = _data()
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "outputs").symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        retain_noise_curvature_campaign(
            completed_campaign,
            x,
            y,
            config=config,
            repository_root=tmp_path,
        )
    assert list(outside.iterdir()) == []
