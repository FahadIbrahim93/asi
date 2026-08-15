"""Integrity contracts for IPMNIST screening shard and summary v2."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import operator
from pathlib import Path
from types import FunctionType

import jax
import numpy as np
import pytest

import alberta_framework.benchmarks.ipmnist_screening as screening
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig

pytestmark = pytest.mark.unit

TINY = IPMNISTConfig(
    n_tasks=1,
    task_length=3,
    input_dim=4,
    hidden1=3,
    hidden2=2,
    n_classes=2,
)


@pytest.fixture(scope="module")
def v2_result() -> screening.ScreeningRunResult:
    x = np.linspace(-1.0, 1.0, 48, dtype=np.float32).reshape(12, 4)
    y = np.asarray([0, 1] * 6, dtype=np.int32)
    return screening.run_screening_config(
        x,
        y,
        screening.screening_spec("upgd_w_control"),
        seed=7,
        config=TINY,
    )


@pytest.fixture(scope="module")
def v2_payload(v2_result: screening.ScreeningRunResult) -> dict[str, object]:
    return screening.shard_payload(v2_result)


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _resign(payload: dict[str, object]) -> None:
    digest = payload["content_digest"]
    assert isinstance(digest, dict)
    digest["sha256"] = screening.screening_content_sha256(payload)


def test_v2_shard_binds_arm_data_source_rng_environment_and_content(
    v2_payload: dict[str, object],
) -> None:
    payload = v2_payload
    assert payload["schema"] == screening.SHARD_SCHEMA
    assert payload["schema"] == "alberta.ipmnist_screening.shard.v2"

    expected_arm = screening.screening_arm_definition(
        screening.screening_spec("upgd_w_control")
    )
    assert payload["arm_definition"] == expected_arm
    assert payload["arm_definition_sha256"] == screening.screening_arm_fingerprint(
        screening.screening_spec("upgd_w_control")
    )

    dataset = payload["input_dataset"]
    assert isinstance(dataset, dict)
    assert dataset["features"] == {"dtype": "float32", "shape": [12, 4]}
    assert dataset["labels"] == {"dtype": "int32", "shape": [12]}
    assert dataset["provenance"] == "caller-supplied-post-cast-arrays"
    assert dataset["semantic_dataset_identity_attested"] is False
    assert len(str(dataset["sha256"])) == 64

    source = payload["source_identity"]
    assert isinstance(source, dict)
    assert source["scope"] == "conservative-package-python-disk-snapshot-v1"
    assert source["capture_version"] == "stable-package-python-byte-snapshot.v1"
    assert source["whole_package_python_included"] is True
    assert source["loaded_code_bytes_attested"] is False
    assert source["execution_authenticated"] is False
    dependency = source["dependency_identity"]
    assert isinstance(dependency, dict)
    assert dependency["active_environment_matches_lock_attested"] is False
    assert dependency["dependency_lock_document_bound"] is (
        "uv.lock" in source["explicit_source_extras"]
    )
    assert source["files"]
    assert len(str(source["sha256"])) == 64

    assert payload["rng_contract"] == screening.screening_rng_contract()
    invocation = payload["execution_invocation"]
    assert isinstance(invocation, dict)
    assert invocation["authentication"] == "self-declared-unverified"
    assert invocation["execution_authenticated"] is False
    assert invocation["interface"] == "python-library-call"
    assert invocation["argv"] is None
    assert invocation["resolved"] == {
        "config_name": "upgd_w_control",
        "seed": 7,
        "config": TINY.to_config(),
        "noise_mode": "step",
        "noise_pool_steps": 64,
        "progress_every": None,
    }
    environment = payload["environment"]
    assert isinstance(environment, dict)
    assert {"python", "jax", "numpy", "platform", "jax_backend"} <= environment.keys()
    expected_jax_runtime = {
        "jax_default_prng_impl": str(jax.config.jax_default_prng_impl),
        "jax_random_seed_offset": int(jax.config.jax_random_seed_offset),
        "jax_threefry_partitionable": bool(jax.config.jax_threefry_partitionable),
        "jax_disable_jit": bool(jax.config.jax_disable_jit),
        "jax_disable_most_optimizations": bool(
            jax.config.values["jax_disable_most_optimizations"]
        ),
    }
    assert {field: environment[field] for field in expected_jax_runtime} == (
        expected_jax_runtime
    )
    rng_contract = payload["rng_contract"]
    assert isinstance(rng_contract, dict)
    assert rng_contract["jax_runtime_config"] == {
        field: value
        for field, value in expected_jax_runtime.items()
        if field != "jax_disable_most_optimizations"
    }
    assert environment["jax_config_values"] == screening._jax_config_values_identity()

    digest = payload["content_digest"]
    assert isinstance(digest, dict)
    assert digest["authentication"] == "none-unkeyed-self-hash"
    assert digest["sha256"] == screening.screening_content_sha256(payload)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("base_learner", "forged", "registered arm definition"),
        ("mechanism", "forged", "registered arm definition"),
        ("hyperparameters", {"step_size": 99.0}, "registered arm definition"),
        (
            "factory",
            {"module": "forged", "qualname": "forged"},
            "registered arm definition",
        ),
    ],
)
def test_strict_loader_rejects_resigned_arm_definition_tampering(
    tmp_path: Path,
    v2_payload: dict[str, object],
    field: str,
    replacement: object,
    message: str,
) -> None:
    payload = copy.deepcopy(v2_payload)
    arm = payload["arm_definition"]
    assert isinstance(arm, dict)
    arm[field] = replacement
    payload["arm_definition_sha256"] = screening.canonical_json_sha256(arm)
    _resign(payload)
    path = tmp_path / "tampered.json"
    _write(path, payload)

    with pytest.raises(ValueError, match=message):
        screening.load_shard(path)


def test_strict_loader_rejects_bool_numeric_alias_in_arm_identity(
    tmp_path: Path,
    v2_payload: dict[str, object],
) -> None:
    payload = copy.deepcopy(v2_payload)
    spec = screening.screening_spec("lin_rls")
    arm = screening.screening_arm_definition(spec)
    arm_hyperparameters = arm["hyperparameters"]
    assert isinstance(arm_hyperparameters, dict)
    arm_hyperparameters["rff_gamma"] = False
    payload.update(
        {
            "config_name": spec.name,
            "base_learner": spec.base_learner,
            "mechanism": spec.mechanism,
            "hyperparameters": {**spec.hyperparameters, "rff_gamma": False},
            "arm_definition": arm,
            "arm_definition_sha256": screening.screening_arm_fingerprint(spec),
        }
    )
    invocation = payload["execution_invocation"]
    assert isinstance(invocation, dict)
    resolved = invocation["resolved"]
    assert isinstance(resolved, dict)
    resolved["config_name"] = spec.name
    _resign(payload)
    path = tmp_path / "bool-numeric-alias.json"
    _write(path, payload)

    with pytest.raises(ValueError, match="registered arm definition"):
        screening.load_shard(path)


def test_registered_arm_hyperparameters_are_immutable() -> None:
    spec = screening.screening_spec("upgd_w_control")
    before = screening.screening_arm_fingerprint(spec)
    with pytest.raises(TypeError):
        operator.setitem(spec.hyperparameters, "step_size", 99.0)
    assert screening.screening_arm_fingerprint(spec) == before


def test_arm_fingerprint_binds_closure_captured_callable() -> None:
    upgd = screening.screening_spec("upgd_w_control")
    adamw = screening.screening_spec("adamw_control")
    assert upgd.factory.__module__ == adamw.factory.__module__
    assert upgd.factory.__qualname__ == adamw.factory.__qualname__

    upgd_factory = screening.screening_arm_definition(upgd)["factory"]
    adamw_factory = screening.screening_arm_definition(adamw)["factory"]
    assert upgd_factory != adamw_factory
    assert screening.screening_arm_fingerprint(upgd) != (
        screening.screening_arm_fingerprint(
            dataclasses.replace(upgd, factory=adamw.factory)
        )
    )


def test_arm_fingerprint_binds_callback_code_body() -> None:
    registered = screening.screening_spec("upgd_idbd")

    def replacement_factory(hyperparameters: object) -> object:
        del hyperparameters
        raise RuntimeError("different callback body")

    replacement = FunctionType(
        replacement_factory.__code__.replace(
            co_name=registered.factory.__code__.co_name,
            co_qualname=registered.factory.__code__.co_qualname,
        ),
        registered.factory.__globals__,
        registered.factory.__name__,
    )
    replacement.__module__ = registered.factory.__module__
    replacement.__qualname__ = registered.factory.__qualname__
    forged = dataclasses.replace(registered, factory=replacement)

    registered_factory = screening.screening_arm_definition(registered)["factory"]
    forged_factory = screening.screening_arm_definition(forged)["factory"]
    assert isinstance(registered_factory, dict)
    assert isinstance(forged_factory, dict)
    for field in ("module", "qualname", "defaults", "kwdefaults", "closure"):
        assert registered_factory[field] == forged_factory[field]
    assert registered_factory["code"] != forged_factory["code"]
    assert screening.screening_arm_fingerprint(registered) != (
        screening.screening_arm_fingerprint(forged)
    )

    relocated = FunctionType(
        registered.factory.__code__.replace(
            co_filename="/different/checkout/ipmnist_screening.py",
            co_firstlineno=registered.factory.__code__.co_firstlineno + 1000,
        ),
        registered.factory.__globals__,
        registered.factory.__name__,
        registered.factory.__defaults__,
        registered.factory.__closure__,
    )
    relocated.__module__ = registered.factory.__module__
    relocated.__qualname__ = registered.factory.__qualname__
    relocated.__kwdefaults__ = registered.factory.__kwdefaults__
    relocated_factory = screening.screening_arm_definition(
        dataclasses.replace(registered, factory=relocated)
    )["factory"]
    assert isinstance(relocated_factory, dict)
    assert registered_factory == relocated_factory


def test_arm_definition_rejects_callback_with_alternate_global_namespace() -> None:
    registered = screening.screening_spec("upgd_idbd")
    alternate_globals = dict(registered.factory.__globals__)
    clone = FunctionType(
        registered.factory.__code__,
        alternate_globals,
        registered.factory.__name__,
        registered.factory.__defaults__,
        registered.factory.__closure__,
    )
    clone.__module__ = registered.factory.__module__
    clone.__qualname__ = registered.factory.__qualname__
    clone.__kwdefaults__ = registered.factory.__kwdefaults__
    assert clone.__globals__ is not registered.factory.__globals__

    with pytest.raises(TypeError, match="canonical module namespace"):
        screening.screening_arm_definition(
            dataclasses.replace(registered, factory=clone)
        )


def test_cli_argv_must_parse_to_exact_resolved_inputs(
    tmp_path: Path,
    v2_payload: dict[str, object],
    v2_result: screening.ScreeningRunResult,
) -> None:
    contradictory_argv = [
        "run",
        "--config-name",
        "upgd_w_control",
        "--seed",
        "999",
        "--n-tasks",
        "1",
        "--task-length",
        "3",
        "--out",
        str(tmp_path / "unused.json"),
    ]
    with pytest.raises(ValueError, match="CLI argv contradicts resolved"):
        screening.shard_payload(v2_result, cli_argv=contradictory_argv)

    payload = copy.deepcopy(v2_payload)
    invocation = payload["execution_invocation"]
    assert isinstance(invocation, dict)
    invocation.update(
        {
            "interface": "cli",
            "entrypoint": "python -m alberta_framework.benchmarks.ipmnist_screening",
            "argv": contradictory_argv,
        }
    )
    _resign(payload)
    path = tmp_path / "contradictory-cli.json"
    _write(path, payload)
    with pytest.raises(ValueError, match="CLI argv contradicts resolved"):
        screening.load_shard(path)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("input_dataset", "content digest"),
        ("environment", "content digest"),
        ("rng_contract", "content digest"),
        ("source_identity", "content digest"),
    ],
)
def test_strict_loader_rejects_unsigned_bound_content_tampering(
    tmp_path: Path,
    v2_payload: dict[str, object],
    field: str,
    message: str,
) -> None:
    payload = copy.deepcopy(v2_payload)
    bound = payload[field]
    assert isinstance(bound, dict)
    bound["tampered"] = True
    path = tmp_path / f"tampered-{field}.json"
    _write(path, payload)

    with pytest.raises(ValueError, match=message):
        screening.load_shard(path)


def test_strict_loader_rejects_resigned_source_or_rng_contract_drift(
    tmp_path: Path,
    v2_payload: dict[str, object],
) -> None:
    for field, message in (
        ("source_identity", "source identity"),
        ("rng_contract", "RNG contract"),
    ):
        payload = copy.deepcopy(v2_payload)
        bound = payload[field]
        assert isinstance(bound, dict)
        bound["contract_version"] = "forged"
        _resign(payload)
        path = tmp_path / f"resigned-{field}.json"
        _write(path, payload)
        with pytest.raises(ValueError, match=message):
            screening.load_shard(path)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("jax_default_prng_impl", 1),
        ("jax_random_seed_offset", True),
        ("jax_threefry_partitionable", 0),
        ("jax_disable_jit", 0),
        ("jax_disable_most_optimizations", 0),
    ],
)
def test_strict_loader_rejects_invalid_jax_runtime_identity_types(
    tmp_path: Path,
    v2_payload: dict[str, object],
    field: str,
    invalid_value: object,
) -> None:
    payload = copy.deepcopy(v2_payload)
    environment = payload["environment"]
    assert isinstance(environment, dict)
    environment[field] = invalid_value
    _resign(payload)
    path = tmp_path / f"invalid-{field}.json"
    _write(path, payload)

    with pytest.raises(ValueError, match="environment identity types"):
        screening.load_shard(path)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        (
            "jax_default_prng_impl",
            "rbg" if str(jax.config.jax_default_prng_impl) != "rbg" else "threefry2x32",
        ),
        ("jax_random_seed_offset", int(jax.config.jax_random_seed_offset) + 1),
        (
            "jax_threefry_partitionable",
            not bool(jax.config.jax_threefry_partitionable),
        ),
        ("jax_disable_jit", not bool(jax.config.jax_disable_jit)),
    ],
)
def test_merge_rejects_shards_from_different_jax_runtime_rng_configs(
    tmp_path: Path,
    v2_payload: dict[str, object],
    field: str,
    replacement: object,
) -> None:
    paths: list[Path] = []
    for seed in (7, 11):
        payload = copy.deepcopy(v2_payload)
        payload["seed"] = seed
        invocation = payload["execution_invocation"]
        assert isinstance(invocation, dict)
        resolved = invocation["resolved"]
        assert isinstance(resolved, dict)
        resolved["seed"] = seed
        if seed == 11:
            environment = payload["environment"]
            assert isinstance(environment, dict)
            environment[field] = replacement
            config_values = environment["jax_config_values"]
            assert isinstance(config_values, dict)
            config_values[field] = replacement
            payload["rng_contract"] = (
                screening._screening_rng_contract_for_environment(environment)
            )
        _resign(payload)
        path = tmp_path / f"{field}-seed-{seed}.json"
        _write(path, payload)
        paths.append(path)

    with pytest.raises(ValueError, match="multiple RNG contracts"):
        screening.merge_shards(
            paths,
            control_name="upgd_w_control",
            slope_window=1,
            expected_seeds=(7, 11),
        )


def test_merge_rejects_different_non_rng_jax_optimization_configs(
    tmp_path: Path,
    v2_payload: dict[str, object],
) -> None:
    paths: list[Path] = []
    for seed in (7, 11):
        payload = copy.deepcopy(v2_payload)
        payload["seed"] = seed
        invocation = payload["execution_invocation"]
        assert isinstance(invocation, dict)
        resolved = invocation["resolved"]
        assert isinstance(resolved, dict)
        resolved["seed"] = seed
        if seed == 11:
            environment = payload["environment"]
            assert isinstance(environment, dict)
            replacement = not bool(environment["jax_disable_most_optimizations"])
            environment["jax_disable_most_optimizations"] = replacement
            config_values = environment["jax_config_values"]
            assert isinstance(config_values, dict)
            config_values["jax_disable_most_optimizations"] = replacement
        _resign(payload)
        path = tmp_path / f"optimization-seed-{seed}.json"
        _write(path, payload)
        paths.append(path)

    with pytest.raises(ValueError, match="multiple environment identities"):
        screening.merge_shards(
            paths,
            control_name="upgd_w_control",
            slope_window=1,
            expected_seeds=(7, 11),
        )


@pytest.mark.parametrize(
    ("labels", "error", "message"),
    [
        (np.asarray([0.0, 1.0] * 6), TypeError, "integer labels"),
        (np.asarray([-1, 1] * 6), ValueError, r"\[0, 2\)"),
        (np.asarray([0, 2] * 6), ValueError, r"\[0, 2\)"),
    ],
)
def test_runner_rejects_noninteger_or_out_of_range_labels(
    labels: np.ndarray,
    error: type[Exception],
    message: str,
) -> None:
    x = np.linspace(-1.0, 1.0, 48, dtype=np.float32).reshape(12, 4)
    with pytest.raises(error, match=message):
        screening.run_screening_config(
            x,
            labels,
            screening.screening_spec("upgd_w_control"),
            seed=7,
            config=TINY,
        )


@pytest.mark.parametrize(
    ("features", "labels", "message"),
    [
        (
            np.full((12, 4), np.nan, dtype=np.float32),
            np.asarray([0, 1] * 6),
            "finite float32",
        ),
        (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            "non-empty shape",
        ),
    ],
)
def test_runner_rejects_nonfinite_or_empty_features(
    features: np.ndarray,
    labels: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        screening.run_screening_config(
            features,
            labels,
            screening.screening_spec("upgd_w_control"),
            seed=7,
            config=TINY,
        )


def test_v1_requires_explicit_quarantine_parser(
    tmp_path: Path,
    v2_payload: dict[str, object],
) -> None:
    legacy = {
        "schema": screening.LEGACY_SHARD_SCHEMA,
        "config_name": "upgd_w_control",
        "base_learner": "upgd_w",
        "hyperparameters": dict(
            screening.screening_spec("upgd_w_control").hyperparameters
        ),
        "seed": 7,
        "noise_mode": "step",
        "config": TINY.to_config(),
        "per_task_accuracy": v2_payload["per_task_accuracy"],
        "per_task_loss": v2_payload["per_task_loss"],
        "per_task_plasticity": v2_payload["per_task_plasticity"],
    }
    path = tmp_path / "legacy.json"
    _write(path, legacy)

    with pytest.raises(ValueError, match="legacy.*quarantine"):
        screening.load_shard(path)
    loaded = screening.load_shard(path, legacy_v1_quarantine=True)
    assert loaded["schema"] == screening.LEGACY_SHARD_SCHEMA
    assert loaded["legacy_quarantine"] == {
        "integrity": "unbound-v1",
        "merge_allowed": False,
        "scientific_promotion_allowed": False,
    }


def test_summary_binds_exact_input_paths_digests_and_seed_contract(
    tmp_path: Path,
    v2_payload: dict[str, object],
) -> None:
    paths: list[Path] = []
    for seed in (7, 11):
        payload = copy.deepcopy(v2_payload)
        payload["seed"] = seed
        invocation = payload["execution_invocation"]
        assert isinstance(invocation, dict)
        resolved = invocation["resolved"]
        assert isinstance(resolved, dict)
        resolved["seed"] = seed
        _resign(payload)
        path = tmp_path / f"control-seed-{seed}.json"
        _write(path, payload)
        paths.append(path)

    summary = screening.merge_shards(
        paths,
        control_name="upgd_w_control",
        slope_window=1,
        expected_seeds=(7, 11),
    )

    assert summary["schema"] == screening.SUMMARY_SCHEMA
    assert summary["schema"] == "alberta.ipmnist_screening.summary.v2"
    assert summary["seed_set_contract"] == {
        "expected_seeds": [7, 11],
        "require_exact_per_arm": True,
    }
    assert summary["input_shards"] == [
        {
            "path": path.as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "config_name": "upgd_w_control",
            "seed": seed,
        }
        for path, seed in zip(paths, (7, 11), strict=True)
    ]
    digest = summary["content_digest"]
    assert isinstance(digest, dict)
    assert digest["sha256"] == screening.screening_content_sha256(summary)
    summary_path = tmp_path / "summary.json"
    _write(summary_path, summary)
    assert screening.load_summary(summary_path) == summary


@pytest.mark.parametrize("spelling", ["relative", "dotdot"])
def test_summary_loader_rejects_noncanonical_input_path_spellings(
    tmp_path: Path,
    v2_payload: dict[str, object],
    spelling: str,
) -> None:
    shard_path = tmp_path / "seed-7.json"
    _write(shard_path, v2_payload)
    summary = screening.merge_shards(
        (shard_path,), expected_seeds=(7,), slope_window=1
    )
    records = summary["input_shards"]
    assert isinstance(records, list) and isinstance(records[0], dict)
    if spelling == "relative":
        records[0]["path"] = shard_path.name
    else:
        redundant = tmp_path / "redundant"
        redundant.mkdir()
        records[0]["path"] = f"{redundant.as_posix()}/../{shard_path.name}"
    _resign(summary)
    summary_path = tmp_path / f"summary-{spelling}.json"
    _write(summary_path, summary)

    with pytest.raises(ValueError, match="input shard record is invalid"):
        screening.load_summary(summary_path)


def test_merge_rejects_any_arm_with_inexact_explicit_seed_set(
    tmp_path: Path,
    v2_payload: dict[str, object],
) -> None:
    path = tmp_path / "only-seed-7.json"
    _write(path, v2_payload)

    with pytest.raises(ValueError, match="seed-set contract"):
        screening.merge_shards(path for path in (path,))


def test_merge_requires_declared_control_arm_in_input_set(
    tmp_path: Path,
    v2_payload: dict[str, object],
) -> None:
    payload = copy.deepcopy(v2_payload)
    spec = screening.screening_spec("adamw_control")
    arm = screening.screening_arm_definition(spec)
    payload.update(
        {
            "config_name": spec.name,
            "base_learner": spec.base_learner,
            "mechanism": spec.mechanism,
            "hyperparameters": dict(spec.hyperparameters),
            "arm_definition": arm,
            "arm_definition_sha256": screening.screening_arm_fingerprint(spec),
        }
    )
    invocation = payload["execution_invocation"]
    assert isinstance(invocation, dict)
    resolved = invocation["resolved"]
    assert isinstance(resolved, dict)
    resolved["config_name"] = spec.name
    _resign(payload)
    path = tmp_path / "adamw-only.json"
    _write(path, payload)

    with pytest.raises(ValueError, match="control arm.*absent"):
        screening.merge_shards(
            (path,),
            control_name="upgd_w_control",
            expected_seeds=(7,),
            slope_window=1,
        )


def test_shard_payload_rejects_pool_mode_for_arm_without_noise_update(
    v2_result: screening.ScreeningRunResult,
) -> None:
    spec = screening.screening_spec("adamw_control")
    assert spec.noise_update is None
    impossible = dataclasses.replace(
        v2_result,
        config_name=spec.name,
        base_learner=spec.base_learner,
        hyperparameters=dict(spec.hyperparameters),
        arm_definition=screening.screening_arm_definition(spec),
        noise_mode="pool",
    )

    with pytest.raises(ValueError, match="pool mode is unsupported"):
        screening.shard_payload(impossible)


def test_strict_loader_rejects_pool_mode_for_arm_without_noise_update(
    tmp_path: Path,
    v2_payload: dict[str, object],
) -> None:
    payload = copy.deepcopy(v2_payload)
    spec = screening.screening_spec("adamw_control")
    arm = screening.screening_arm_definition(spec)
    payload.update(
        {
            "config_name": spec.name,
            "base_learner": spec.base_learner,
            "mechanism": spec.mechanism,
            "hyperparameters": dict(spec.hyperparameters),
            "arm_definition": arm,
            "arm_definition_sha256": screening.screening_arm_fingerprint(spec),
            "noise_mode": "pool",
        }
    )
    invocation = payload["execution_invocation"]
    assert isinstance(invocation, dict)
    resolved = invocation["resolved"]
    assert isinstance(resolved, dict)
    resolved.update({"config_name": spec.name, "noise_mode": "pool"})
    _resign(payload)
    path = tmp_path / "unsupported-pool.json"
    _write(path, payload)

    with pytest.raises(ValueError, match="pool mode is unsupported"):
        screening.load_shard(path)


@pytest.mark.parametrize("field", ["created_unix", "wall_clock_seconds"])
def test_content_digest_binds_operational_claims_too(
    tmp_path: Path,
    v2_payload: dict[str, object],
    field: str,
) -> None:
    payload = copy.deepcopy(v2_payload)
    value = payload[field]
    assert isinstance(value, float)
    payload[field] = value + 1.0
    path = tmp_path / f"changed-{field}.json"
    _write(path, payload)
    with pytest.raises(ValueError, match="content digest"):
        screening.load_shard(path)


def test_strict_json_rejects_duplicate_keys_nan_and_extra_v2_fields(
    tmp_path: Path,
    v2_payload: dict[str, object],
) -> None:
    duplicate = tmp_path / "duplicate.json"
    encoded = json.dumps(v2_payload)
    duplicate.write_text(
        '{"schema":"alberta.ipmnist_screening.shard.v2",' + encoded[1:],
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        screening.load_shard(duplicate)

    nonstandard = copy.deepcopy(v2_payload)
    nonstandard["created_unix"] = float("nan")
    nan_path = tmp_path / "nan.json"
    nan_path.write_text(json.dumps(nonstandard), encoding="utf-8")
    with pytest.raises(ValueError, match="non-standard JSON numeric constant"):
        screening.load_shard(nan_path)

    extra = copy.deepcopy(v2_payload)
    extra["unexpected"] = "resigned"
    _resign(extra)
    extra_path = tmp_path / "extra.json"
    _write(extra_path, extra)
    with pytest.raises(ValueError, match="field set mismatch"):
        screening.load_shard(extra_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("per_task_accuracy", ["0.5"], "JSON numbers"),
        ("per_task_plasticity", [True], "JSON numbers"),
        ("per_task_accuracy", [1.5], r"\[0, 1\]"),
        ("per_task_loss", [-0.1], "non-negative"),
        ("wall_clock_seconds", -1.0, "non-negative"),
    ],
)
def test_strict_v2_numeric_domains_reject_resigned_coercions(
    tmp_path: Path,
    v2_payload: dict[str, object],
    field: str,
    value: object,
    message: str,
) -> None:
    payload = copy.deepcopy(v2_payload)
    payload[field] = value
    _resign(payload)
    path = tmp_path / f"invalid-{field}.json"
    _write(path, payload)
    with pytest.raises(ValueError, match=message):
        screening.load_shard(path)


def test_shard_serializes_actual_cloned_spec_then_strict_loading_rejects_it(
    tmp_path: Path,
) -> None:
    x = np.linspace(-1.0, 1.0, 48, dtype=np.float32).reshape(12, 4)
    y = np.asarray([0, 1] * 6, dtype=np.int32)
    registered = screening.screening_spec("upgd_w_control")
    cloned = dataclasses.replace(registered, description="unregistered clone")
    result = screening.run_screening_config(x, y, cloned, seed=7, config=TINY)
    payload = screening.shard_payload(result)
    arm = payload["arm_definition"]
    assert isinstance(arm, dict)
    assert arm["description"] == "unregistered clone"
    assert payload["arm_definition_sha256"] == screening.canonical_json_sha256(arm)

    path = tmp_path / "clone.json"
    _write(path, payload)
    with pytest.raises(ValueError, match="registered arm definition"):
        screening.load_shard(path)


def test_whole_package_source_snapshot_includes_unimported_files_and_rejects_symlinks(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    package = repo / "alberta_framework"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "unimported.py").write_text("VALUE = 2\n", encoding="utf-8")

    snapshot = screening._capture_source_snapshot(repo_root=repo)
    assert snapshot == {
        Path("alberta_framework/__init__.py"): b"",
        Path("alberta_framework/a.py"): b"VALUE = 1\n",
        Path("alberta_framework/unimported.py"): b"VALUE = 2\n",
    }

    outside = tmp_path / "outside.py"
    outside.write_text("VALUE = 3\n", encoding="utf-8")
    (package / "escape.py").symlink_to(outside)
    with pytest.raises(ValueError, match="refuses symlinks"):
        screening._capture_source_snapshot(repo_root=repo)


def test_merge_materializes_one_shot_paths_once(
    tmp_path: Path,
    v2_payload: dict[str, object],
) -> None:
    path = tmp_path / "one.json"
    _write(path, v2_payload)

    class OneShotPaths:
        def __init__(self) -> None:
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            if self.iterations > 1:
                raise AssertionError("paths iterable was consumed twice")
            yield path

    paths = OneShotPaths()
    summary = screening.merge_shards(paths, expected_seeds=(7,), slope_window=1)
    assert summary["n_shards"] == 1
    assert paths.iterations == 1


def test_summary_loader_recomputes_results_and_rejects_duplicate_arm_seed(
    tmp_path: Path,
    v2_payload: dict[str, object],
) -> None:
    shard_path = tmp_path / "seed-7.json"
    _write(shard_path, v2_payload)
    summary = screening.merge_shards(
        (shard_path,), expected_seeds=(7,), slope_window=1
    )

    forged = copy.deepcopy(summary)
    results = forged["results"]
    assert isinstance(results, list) and isinstance(results[0], dict)
    results[0]["average_online_accuracy_mean"] = 0.123456
    _resign(forged)
    forged_path = tmp_path / "forged-summary.json"
    _write(forged_path, forged)
    with pytest.raises(ValueError, match="derived results"):
        screening.load_summary(forged_path)

    duplicate_shard_path = tmp_path / "same-seed-other-path.json"
    duplicate_shard_path.write_bytes(shard_path.read_bytes())
    duplicate = copy.deepcopy(summary)
    inputs = duplicate["input_shards"]
    assert isinstance(inputs, list)
    duplicate_record = copy.deepcopy(inputs[0])
    assert isinstance(duplicate_record, dict)
    duplicate_record["path"] = duplicate_shard_path.as_posix()
    duplicate_record["sha256"] = hashlib.sha256(
        duplicate_shard_path.read_bytes()
    ).hexdigest()
    inputs.append(duplicate_record)
    duplicate["n_shards"] = 2
    _resign(duplicate)
    duplicate_path = tmp_path / "duplicate-summary.json"
    _write(duplicate_path, duplicate)
    with pytest.raises(ValueError, match="input shard record is invalid"):
        screening.load_summary(duplicate_path)
