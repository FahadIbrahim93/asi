"""Run the complete, resource-matched issue #1562 development comparison."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import stat
import sys
from pathlib import Path
from typing import Final, cast

import jax
import jax.random as jr
import numpy as np

from alberta_framework.benchmarks.ipmnist_screening import (
    _screening_dataset_provenance,
    bounded_elastic_development_result_payload,
    run_screening_config,
    screening_spec,
)
from alberta_framework.benchmarks.upgd_ipmnist import (
    IPMNISTConfig,
    build_schedule,
    default_openml_data_home,
    init_mlp_params,
)
from alberta_framework.evaluation.bounded_elastic_ipmnist_nonpromoting import (
    bounded_elastic_resource_expectations,
    validate_bounded_elastic_development_result,
    validate_matched_bounded_elastic_results,
)

RESULT_SCHEMA: Final = "asi.bounded-elastic-ipmnist.matched-development.v1"
CAMPAIGN_SEEDS: Final = (51_562_001, 51_562_002, 51_562_003, 51_562_004, 51_562_005)
TEST_ONLY_SEEDS: Final = (201, 202, 203, 204, 205)
ARMS: Final = (
    "bounded_structure_off",
    "bounded_growth",
    "bounded_elastic",
    "bounded_fixed_cbp",
)
_POLICY: Final = {
    "development_only": True,
    "scientific_promotion_allowed": False,
    "sota_claim_allowed": False,
    "negative_results_retained": True,
}
_MAX_STEPS: Final = 2_000_000
_MAX_PERSISTENT_BYTES: Final = 256 * 1024 * 1024
_MAX_DATASET_BYTES: Final = 256 * 1024 * 1024
_MAX_SCHEDULE_BYTES: Final = 256 * 1024 * 1024
_MAX_COMBINED_NUMERIC_BYTES: Final = 256 * 1024 * 1024
_REVIEWED_EXECUTION_TRANSITION: Final = False
_EXECUTION_AUTHORIZED: Final = False
_EXECUTION_CAPABILITY: Final = object()
_TEST_EXECUTION_CAPABILITY: Final = object()
CAMPAIGN_CONFIG: Final = IPMNISTConfig(n_tasks=8, task_length=5000)
OUTPUT_PATH: Final = (
    Path(__file__).resolve().parents[2]
    / "outputs/bounded_elastic_matched_development/report.v1.json"
)


def frozen_plan() -> dict[str, object]:
    config = CAMPAIGN_CONFIG
    return {
        "schema": "asi.bounded-elastic-ipmnist.matched-plan.v1",
        "paper_revision": "arXiv:2608.01475v1",
        "official_code": None,
        "seeds": list(CAMPAIGN_SEEDS),
        "arms": list(ARMS),
        "config": _config_payload(config),
        "dataset": {
            "schema": "alberta.ipmnist_screening.dataset_provenance.v1",
            "source": {"provider": "openml", "name": "mnist_784", "version": 1,
                       "row_start": 0, "row_stop_exclusive": 60_000},
            "materialization": "alberta.ipmnist.float32-neg1-pos1-int32-labels.v1",
            "x": {"dtype": "<f4", "shape": [60_000, 784],
                  "sha256": "b8078cd833f53d89828a5e28d728517be9add34076f13fe973399f1f16381313"},
            "y": {"dtype": "<i4", "shape": [60_000],
                  "sha256": "4f1dd9551f104f8153409e0add59f0a71568f7bad5a5f8e2274480c186fe219a"},
        },
        "source_identity": _source_identity(),
        "runtime_identity": _runtime_identity(),
        "per_arm_resources": {
            arm: bounded_elastic_resource_expectations(
                arm=arm, n_tasks=config.n_tasks, input_dim=config.input_dim,
                hidden1=config.hidden1, hidden2=config.hidden2, n_classes=config.n_classes,
            )
            for arm in ARMS
        },
        "numeric_resource_envelope": _numeric_resource_envelope(
            config=config, dataset_rows=60_000
        ),
        "matched_axes": ["seed", "observations", "updates", "example_schedule",
                         "allowed_boundary_information", "allowed_task_information"],
        "decision_rule": {
            "metric": "mean_online_accuracy",
            "baseline": "bounded_fixed_cbp",
            "candidates": ["bounded_growth", "bounded_elastic"],
            "candidate_supported": "all five paired deltas are strictly positive",
            "candidate_rejected": "all five paired deltas are nonpositive",
            "campaign_supported": "at least one candidate is supported",
            "campaign_rejected": "both candidates are rejected",
            "otherwise": "inconclusive",
        },
        "reviewed_execution_transition": _REVIEWED_EXECUTION_TRANSITION,
        "execution_authorized": _EXECUTION_AUTHORIZED,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "negative_results_retained": True,
        "output_path": "outputs/bounded_elastic_matched_development/report.v1.json",
    }


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _json_preflight(value: object) -> None:
    pending: list[tuple[object, int]] = [(value, 0)]
    seen: set[int] = set()
    nodes = 0
    while pending:
        current, depth = pending.pop()
        actual_type = type(current)
        nodes += 1
        if nodes > 100_000 or depth > 16:
            raise ValueError("matched result exceeds its JSON structure bound")
        if current is None or actual_type is bool or actual_type is int:
            continue
        if actual_type is float:
            if not math.isfinite(cast(float, current)):
                raise ValueError("matched result contains a non-finite float")
            continue
        if actual_type is str:
            if len(cast(str, current).encode("utf-8")) > 16_384:
                raise ValueError("matched result contains an oversized string")
            continue
        if actual_type is not dict and actual_type is not list:
            raise ValueError("matched result must contain only exact JSON values")
        identity = id(current)
        if identity in seen:
            raise ValueError("matched result contains an aliased or cyclic container")
        seen.add(identity)
        if actual_type is dict:
            mapping = cast(dict[object, object], current)
            if len(mapping) > 4096 or any(type(key) is not str for key in mapping):
                raise ValueError("matched result object exceeds its field bound")
            pending.extend((item, depth + 1) for item in mapping.values())
        else:
            sequence = cast(list[object], current)
            if len(sequence) > 4096:
                raise ValueError("matched result list exceeds its item bound")
            pending.extend((item, depth + 1) for item in sequence)


def _exact_object(value: object, fields: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict or set(cast(dict[object, object], value)) != fields:
        raise ValueError(f"{label} fields drifted")
    return cast(dict[str, object], value)


def _source_identity() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    relative_paths = (
        "alberta_framework/_seed_validation.py",
        "alberta_framework/benchmarks/ipmnist_screening.py",
        "alberta_framework/benchmarks/upgd_ipmnist.py",
        "alberta_framework/evaluation/bounded_elastic_ipmnist_nonpromoting.py",
        "alberta_framework/evaluation/bounded_elastic_matched_runner.py",
        "pyproject.toml",
        "uv.lock",
    )
    return {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in relative_paths
    }


def _runtime_identity() -> dict[str, object]:
    environment_names = (
        "JAX_DEFAULT_MATMUL_PRECISION",
        "JAX_DEFAULT_PRNG_IMPL",
        "JAX_ENABLE_X64",
        "JAX_NUM_CPU_DEVICES",
        "JAX_PLATFORMS",
        "JAX_PLATFORM_NAME",
        "JAX_RANDOM_SEED_OFFSET",
        "XLA_FLAGS",
    )
    return {
        "schema": "asi.bounded-elastic-ipmnist.runtime.v1",
        "python": list(sys.version_info[:3]),
        "python_implementation": platform.python_implementation(),
        "byteorder": sys.byteorder,
        "platform": sys.platform,
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "chex",
                "jax",
                "jaxlib",
                "jaxtyping",
                "numpy",
                "orbax-checkpoint",
                "scikit-learn",
                "scipy",
            )
        },
        "backend": jax.default_backend(),
        "devices": [
            {
                "platform": device.platform,
                "device_kind": device.device_kind,
                "id": device.id,
                "process_index": device.process_index,
            }
            for device in jax.devices()
        ],
        "jax_config": {
            "jax_default_matmul_precision": str(jax.config.jax_default_matmul_precision),
            "jax_default_prng_impl": str(jax.config.jax_default_prng_impl),
            "jax_disable_jit": bool(jax.config.jax_disable_jit),
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "jax_numpy_dtype_promotion": str(jax.config.jax_numpy_dtype_promotion.value),
            "jax_numpy_rank_promotion": str(jax.config.jax_numpy_rank_promotion),
            "jax_random_seed_offset": int(jax.config.jax_random_seed_offset),
            "jax_threefry_partitionable": bool(jax.config.jax_threefry_partitionable),
        },
        "environment": {name: os.environ.get(name) for name in environment_names},
    }


def _checked_config(value: object) -> IPMNISTConfig:
    if type(value) is not IPMNISTConfig:
        raise ValueError("config must be an exact IPMNISTConfig")
    try:
        config = IPMNISTConfig(
            **{
                name: getattr(value, name)
                for name in (
                    "n_tasks",
                    "task_length",
                    "input_dim",
                    "hidden1",
                    "hidden2",
                    "n_classes",
                )
            }
        )
    except (TypeError, ValueError) as error:
        raise ValueError("config is invalid") from error
    if config.task_length != 5000:
        raise ValueError("bounded-elastic campaign requires task_length=5000")
    if config.n_tasks * config.task_length > _MAX_STEPS:
        raise ValueError("bounded-elastic campaign exceeds its 2000000-step bound")
    resources = bounded_elastic_resource_expectations(
        arm="bounded_fixed_cbp",
        n_tasks=config.n_tasks,
        input_dim=config.input_dim,
        hidden1=config.hidden1,
        hidden2=config.hidden2,
        n_classes=config.n_classes,
    )
    if resources["peak_persistent_bytes_budget"] > _MAX_PERSISTENT_BYTES:
        raise ValueError("bounded-elastic campaign exceeds its persistent-memory bound")
    return config


def _numeric_resource_envelope(
    *, config: IPMNISTConfig, dataset_rows: int
) -> dict[str, int]:
    dataset_bytes = dataset_rows * (config.input_dim * 4 + 4)
    schedule_bytes = config.n_tasks * (dataset_rows + config.input_dim) * 4
    peak_persistent_bytes = max(
        bounded_elastic_resource_expectations(
            arm=arm,
            n_tasks=config.n_tasks,
            input_dim=config.input_dim,
            hidden1=config.hidden1,
            hidden2=config.hidden2,
            n_classes=config.n_classes,
        )["peak_persistent_bytes_budget"]
        for arm in ARMS
    )
    return {
        "dataset_bytes": dataset_bytes,
        "schedule_bytes": schedule_bytes,
        "peak_persistent_bytes_budget": peak_persistent_bytes,
        "combined_numeric_bytes": dataset_bytes + schedule_bytes + peak_persistent_bytes,
        "combined_numeric_bytes_limit": _MAX_COMBINED_NUMERIC_BYTES,
    }


def _validated_arrays(
    data_x: object, data_y: object, config: IPMNISTConfig
) -> tuple[np.ndarray, np.ndarray]:
    if type(data_x) is not np.ndarray or data_x.dtype != np.dtype(np.float32):
        raise ValueError("data_x must be an exact float32 NumPy array")
    if type(data_y) is not np.ndarray or data_y.dtype != np.dtype(np.int32):
        raise ValueError("data_y must be an exact int32 NumPy array")
    if (
        data_x.ndim != 2
        or data_y.ndim != 1
        or data_x.shape[0] != data_y.shape[0]
        or data_x.shape[0] < config.task_length
        or data_x.shape[1] != config.input_dim
    ):
        raise ValueError("dataset shape does not match the campaign config")
    dataset_bytes = data_x.nbytes + data_y.nbytes
    if dataset_bytes > _MAX_DATASET_BYTES:
        raise ValueError("dataset exceeds the campaign's 256 MiB byte bound")
    schedule_bytes = config.n_tasks * (data_x.shape[0] + config.input_dim) * 4
    if schedule_bytes > _MAX_SCHEDULE_BYTES:
        raise ValueError("schedule exceeds the campaign's 256 MiB byte bound")
    envelope = _numeric_resource_envelope(config=config, dataset_rows=data_x.shape[0])
    if envelope["combined_numeric_bytes"] > _MAX_COMBINED_NUMERIC_BYTES:
        raise ValueError("campaign exceeds its combined 256 MiB numeric allocation bound")
    if not np.isfinite(data_x).all():
        raise ValueError("data_x must be finite")
    if np.any(data_y < 0) or np.any(data_y >= config.n_classes):
        raise ValueError("data_y is outside the configured class range")
    return np.array(data_x, copy=True, order="C"), np.array(data_y, copy=True, order="C")


def _dataset_sha256(data_x: np.ndarray, data_y: np.ndarray) -> str:
    digest = hashlib.sha256(b"asi-bounded-elastic-dataset-v1\0")
    for value in (data_x, data_y):
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _array_tree_sha256(value: object) -> str:
    digest = hashlib.sha256(b"asi-bounded-elastic-initial-parameters-v1\0")
    leaves, structure = jax.tree.flatten(value)
    digest.update(str(structure).encode("ascii"))
    for leaf in leaves:
        host = np.asarray(leaf)
        digest.update(np.asarray(host.shape, dtype="<i8").tobytes())
        digest.update(host.dtype.str.encode("ascii"))
        digest.update(host.tobytes(order="C"))
    return digest.hexdigest()


def _execution_identity(seed: int, config: IPMNISTConfig, n_train: int) -> dict[str, str]:
    root = jr.key(np.uint32(seed), impl="threefry2x32")
    key_init, key_schedule, _ = jr.split(root, 3)
    parameters = init_mlp_params(key_init, config)
    schedule = build_schedule(key_schedule, config, n_train)
    digest = hashlib.sha256(b"asi-bounded-elastic-schedule-v1\0")
    for value in (schedule.permutations, schedule.example_indices):
        host = np.asarray(value, dtype=np.int32)
        digest.update(np.asarray(host.shape, dtype="<i8").tobytes())
        digest.update(host.astype("<i4", copy=False).tobytes(order="C"))
    return {
        "schedule_sha256": digest.hexdigest(),
        "initial_parameters_sha256": _array_tree_sha256(parameters),
        "prng_implementation": "threefry2x32",
    }


def _config_payload(config: IPMNISTConfig) -> dict[str, int]:
    return {
        name: getattr(config, name)
        for name in ("n_tasks", "task_length", "input_dim", "hidden1", "hidden2", "n_classes")
    }


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    arms: dict[str, object] = {}
    for arm in ARMS:
        metrics = [
            cast(dict[str, float], cast(dict[str, object], row["result"])["metrics"])
            for row in rows
            if row["arm"] == arm
        ]
        arms[arm] = {
            "mean_accuracy": math.fsum(item["mean_online_accuracy"] for item in metrics)
            / len(metrics),
            "mean_loss": math.fsum(item["mean_loss"] for item in metrics) / len(metrics),
            "mean_plasticity": math.fsum(item["mean_plasticity"] for item in metrics)
            / len(metrics),
        }
    comparisons: list[dict[str, object]] = []
    for candidate in ("bounded_growth", "bounded_elastic"):
        deltas: list[float] = []
        for seed in sorted({cast(int, row["seed"]) for row in rows}):
            by_arm = {
                cast(str, row["arm"]): cast(
                    dict[str, float], cast(dict[str, object], row["result"])["metrics"]
                )
                for row in rows
                if row["seed"] == seed
            }
            deltas.append(
                by_arm[candidate]["mean_online_accuracy"]
                - by_arm["bounded_fixed_cbp"]["mean_online_accuracy"]
            )
        outcome = (
            "supported"
            if all(delta > 0.0 for delta in deltas)
            else "rejected"
            if all(delta <= 0.0 for delta in deltas)
            else "inconclusive"
        )
        comparisons.append(
            {
                "candidate": candidate,
                "baseline": "bounded_fixed_cbp",
                "paired_accuracy_deltas": deltas,
                "mean_accuracy_delta": math.fsum(deltas) / len(deltas),
                "outcome": outcome,
            }
        )
    outcomes = [cast(str, comparison["outcome"]) for comparison in comparisons]
    campaign_outcome = (
        "supported"
        if "supported" in outcomes
        else "rejected"
        if all(outcome == "rejected" for outcome in outcomes)
        else "inconclusive"
    )
    return {
        "arms": arms,
        "primary_comparisons": comparisons,
        "outcome": campaign_outcome,
        "row_count": len(rows),
    }


def run_bounded_elastic_matched(
    data_x: object, data_y: object, *, config: IPMNISTConfig
) -> dict[str, object]:
    """Fail closed pending a separately reviewed authorization transition."""
    if (_REVIEWED_EXECUTION_TRANSITION is not True or _EXECUTION_AUTHORIZED is not True):
        raise RuntimeError("bounded-elastic matched campaign execution is not authorized")
    return _run_bounded_elastic_matched_authorized(
        data_x, data_y, config=config, seeds=CAMPAIGN_SEEDS,
        _capability=_EXECUTION_CAPABILITY,
    )


def _run_bounded_elastic_matched_authorized(
    data_x: object, data_y: object, *, config: IPMNISTConfig,
    seeds: tuple[int, ...], _capability: object,
) -> dict[str, object]:
    """Capability-private full runner used by tests or an authorized campaign."""
    expected_seeds = (
        CAMPAIGN_SEEDS if _capability is _EXECUTION_CAPABILITY
        else TEST_ONLY_SEEDS if _capability is _TEST_EXECUTION_CAPABILITY
        else None
    )
    if expected_seeds is None or type(seeds) is not tuple or seeds != expected_seeds:
        raise RuntimeError("private bounded-elastic execution capability or seeds are invalid")
    if _capability is _EXECUTION_CAPABILITY and (
        _REVIEWED_EXECUTION_TRANSITION is not True or _EXECUTION_AUTHORIZED is not True
    ):
        raise RuntimeError("bounded-elastic matched campaign execution is not authorized")
    checked_config = _checked_config(config)
    x, y = _validated_arrays(data_x, data_y, checked_config)
    execution_source = _source_identity()
    execution_runtime = _runtime_identity()
    if _capability is _EXECUTION_CAPABILITY:
        if checked_config != CAMPAIGN_CONFIG:
            raise ValueError("campaign config differs from the frozen reviewed plan")
        if _screening_dataset_provenance(x, y) != frozen_plan()["dataset"]:
            raise ValueError("campaign dataset differs from the frozen reviewed identity")
    rows: list[dict[str, object]] = []
    for seed in seeds:
        execution_identity = _execution_identity(seed, checked_config, x.shape[0])
        for arm in ARMS:
            spec = screening_spec(arm)
            arm_result = run_screening_config(x, y, spec, seed, checked_config)
            if _source_identity() != execution_source or _runtime_identity() != execution_runtime:
                raise RuntimeError("source or runtime changed during matched execution")
            rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "execution_identity": dict(execution_identity),
                    "result": bounded_elastic_development_result_payload(
                        arm_result, outcome="inconclusive"
                    ),
                }
            )
    campaign: dict[str, object] = {
        "schema": RESULT_SCHEMA,
        "status": "complete",
        "config": _config_payload(checked_config),
        "development_seeds": list(seeds),
        "arms": list(ARMS),
        "identity": {
            "dataset_sha256": _dataset_sha256(x, y),
            "source_sha256": execution_source,
            "runtime": execution_runtime,
            "consistency_not_attestation": True,
        },
        "policy": dict(_POLICY),
        "rows": rows,
        "aggregate": _aggregate(rows),
    }
    campaign["result_sha256"] = hashlib.sha256(_canonical(campaign)).hexdigest()
    _validate_bounded_elastic_matched(
        campaign, x, y, config=checked_config, seeds=seeds, reexecute=False
    )
    return campaign


def validate_bounded_elastic_matched(
    value: object, data_x: object, data_y: object, *, config: IPMNISTConfig
) -> None:
    """Fail closed because validation reexecutes the reserved campaign roster."""
    if (_REVIEWED_EXECUTION_TRANSITION is not True or _EXECUTION_AUTHORIZED is not True):
        raise RuntimeError("bounded-elastic matched campaign execution is not authorized")
    _validate_bounded_elastic_matched_authorized(
        value, data_x, data_y, config=config, seeds=CAMPAIGN_SEEDS,
        _capability=_EXECUTION_CAPABILITY,
    )


def _validate_bounded_elastic_matched_authorized(
    value: object, data_x: object, data_y: object, *, config: IPMNISTConfig,
    seeds: tuple[int, ...], _capability: object,
) -> None:
    expected_seeds = (
        CAMPAIGN_SEEDS if _capability is _EXECUTION_CAPABILITY
        else TEST_ONLY_SEEDS if _capability is _TEST_EXECUTION_CAPABILITY
        else None
    )
    if expected_seeds is None or type(seeds) is not tuple or seeds != expected_seeds:
        raise RuntimeError("private bounded-elastic validation capability or seeds are invalid")
    if _capability is _EXECUTION_CAPABILITY and (
        _REVIEWED_EXECUTION_TRANSITION is not True or _EXECUTION_AUTHORIZED is not True
    ):
        raise RuntimeError("bounded-elastic matched campaign execution is not authorized")
    _validate_bounded_elastic_matched(
        value, data_x, data_y, config=config, seeds=seeds, reexecute=True
    )


def _validate_bounded_elastic_matched(
    value: object,
    data_x: object,
    data_y: object,
    *,
    config: IPMNISTConfig,
    seeds: tuple[int, ...],
    reexecute: bool,
) -> None:
    """Validate the full roster, identities, and optionally current execution."""
    _json_preflight(value)
    root = _exact_object(
        value,
        {
            "schema",
            "status",
            "config",
            "development_seeds",
            "arms",
            "identity",
            "policy",
            "rows",
            "aggregate",
            "result_sha256",
        },
        "matched result",
    )
    if root["schema"] != RESULT_SCHEMA or root["status"] != "complete":
        raise ValueError("matched result identity drifted")
    checked_config = _checked_config(config)
    x, y = _validated_arrays(data_x, data_y, checked_config)
    if root["config"] != _config_payload(checked_config):
        raise ValueError("matched result config drifted")
    if root["development_seeds"] != list(seeds) or root["arms"] != list(ARMS):
        raise ValueError("matched result protocol roster drifted")
    expected_identity = {
        "dataset_sha256": _dataset_sha256(x, y),
        "source_sha256": _source_identity(),
        "runtime": _runtime_identity(),
        "consistency_not_attestation": True,
    }
    if root["identity"] != expected_identity:
        raise ValueError("matched result identity drifted")
    if root["policy"] != _POLICY:
        raise ValueError("matched result must remain permanently nonpromoting")
    expected_roster = [(seed, arm) for seed in seeds for arm in ARMS]
    raw_rows = root["rows"]
    if type(raw_rows) is not list or len(raw_rows) != len(expected_roster):
        raise ValueError("matched result roster is incomplete")
    rows = cast(list[dict[str, object]], raw_rows)
    observed: list[tuple[object, object]] = []
    identities = {
        seed: _execution_identity(seed, checked_config, x.shape[0]) for seed in seeds
    }
    for row in rows:
        checked_row = _exact_object(
            row, {"seed", "arm", "execution_identity", "result"}, "matched row"
        )
        seed = checked_row["seed"]
        arm = checked_row["arm"]
        if (
            type(seed) is not int
            or seed not in identities
            or type(arm) is not str
            or arm not in ARMS
        ):
            raise ValueError("matched row roster identity drifted")
        if checked_row["execution_identity"] != identities[seed]:
            raise ValueError("matched row execution identity drifted")
        payload = validate_bounded_elastic_development_result(checked_row["result"])
        if payload["outcome"] != "inconclusive":
            raise ValueError("matched campaign outcomes must remain inconclusive")
        if payload["seed"] != seed or payload["arm"] != arm:
            raise ValueError("matched row result identity drifted")
        observed.append((seed, arm))
    if observed != expected_roster:
        raise ValueError("matched result roster order drifted")
    for offset in range(0, len(rows), len(ARMS)):
        validate_matched_bounded_elastic_results(
            [row["result"] for row in rows[offset : offset + len(ARMS)]]
        )
    if root["aggregate"] != _aggregate(rows):
        raise ValueError("matched result aggregate drifted")
    unsigned = dict(root)
    claimed = unsigned.pop("result_sha256")
    if type(claimed) is not str or claimed != hashlib.sha256(_canonical(unsigned)).hexdigest():
        raise ValueError("matched result digest drifted")
    if reexecute:
        replay_source = _source_identity()
        replay_runtime = _runtime_identity()
        for row in rows:
            claimed_payload = cast(dict[str, object], row["result"])
            replay = run_screening_config(
                x,
                y,
                screening_spec(cast(str, row["arm"])),
                cast(int, row["seed"]),
                checked_config,
            )
            if _source_identity() != replay_source or _runtime_identity() != replay_runtime:
                raise RuntimeError("source or runtime changed during strict reexecution")
            expected_payload = bounded_elastic_development_result_payload(
                replay, outcome="inconclusive"
            )
            expected_resources = cast(dict[str, object], expected_payload["resources"])
            claimed_resources = cast(dict[str, object], claimed_payload["resources"])
            expected_resources["timing_seconds"] = claimed_resources["timing_seconds"]
            if expected_payload != claimed_payload:
                raise ValueError("matched row disagrees with strict current-source reexecution")


def write_bounded_elastic_matched(
    destination: Path,
    value: object,
    data_x: object,
    data_y: object,
    *,
    config: IPMNISTConfig,
) -> None:
    """Fail closed until a reviewed authorization enables publication."""
    if (_REVIEWED_EXECUTION_TRANSITION is not True or _EXECUTION_AUTHORIZED is not True):
        raise RuntimeError("bounded-elastic matched campaign execution is not authorized")
    _write_bounded_elastic_matched_authorized(
        destination, value, data_x, data_y, config=config, seeds=CAMPAIGN_SEEDS,
        _capability=_EXECUTION_CAPABILITY,
    )


def _open_output_parent(path: Path) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    absolute = Path(os.path.abspath(os.fspath(path)))
    descriptor = os.open(os.path.sep, flags)
    try:
        for component in absolute.parent.parts[1:]:
            if component in {"", ".", ".."}:
                raise ValueError("output path contains an unsafe component")
            created = False
            try:
                os.mkdir(component, 0o755, dir_fd=descriptor)
                created = True
            except FileExistsError:
                pass
            if created:
                os.fsync(descriptor)
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


OutputReservation = tuple[int, str, str, int, int, int]


def _reserve_output(path: Path) -> OutputReservation:
    if type(path) is not type(Path()):
        raise TypeError("destination must be an exact Path")
    if not all(hasattr(os, name) for name in ("O_DIRECTORY", "O_NOFOLLOW", "O_TMPFILE")):
        raise OSError("immutable output publication requires Linux descriptor support")
    if path.absolute() != OUTPUT_PATH.absolute() or path.name in {"", ".", ".."}:
        raise ValueError(f"output must be the exact reserved NEW path {OUTPUT_PATH}")
    directory_fd = _open_output_parent(path)
    reservation_name = f".{path.name}.reservation"
    marker_fd = -1
    acquired = False
    marker_identity: tuple[int, int] | None = None
    try:
        try:
            os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("refusing to replace immutable bounded-elastic output")
        marker_fd = os.open(
            reservation_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o400,
            dir_fd=directory_fd,
        )
        acquired = True
        marker = b"asi-bounded-elastic-output-reservation-v1\n"
        view = memoryview(marker)
        while view:
            written = os.write(marker_fd, view)
            if written <= 0:
                raise OSError("short reservation write")
            view = view[written:]
        os.fsync(marker_fd)
        metadata = os.fstat(marker_fd)
        marker_identity = (metadata.st_dev, metadata.st_ino)
        os.fsync(directory_fd)
        return (
            directory_fd,
            path.name,
            reservation_name,
            marker_fd,
            metadata.st_dev,
            metadata.st_ino,
        )
    except BaseException:
        if marker_fd >= 0:
            os.close(marker_fd)
        if acquired:
            try:
                current = os.stat(
                    reservation_name, dir_fd=directory_fd, follow_symlinks=False
                )
                if (current.st_dev, current.st_ino) == marker_identity:
                    os.unlink(reservation_name, dir_fd=directory_fd)
                    os.fsync(directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)
        raise


def _require_owned_reservation(reservation: OutputReservation) -> None:
    directory_fd, _name, marker_name, marker_fd, device, inode = reservation
    held = os.fstat(marker_fd)
    visible = os.stat(marker_name, dir_fd=directory_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(held.st_mode)
        or held.st_nlink != 1
        or (held.st_dev, held.st_ino) != (device, inode)
        or (visible.st_dev, visible.st_ino) != (device, inode)
    ):
        raise ValueError("output reservation is not the owned visible regular marker")


def _release_output(reservation: OutputReservation) -> None:
    directory_fd, _name, marker_name, marker_fd, device, inode = reservation
    try:
        metadata = os.stat(marker_name, dir_fd=directory_fd, follow_symlinks=False)
        if (metadata.st_dev, metadata.st_ino) == (device, inode):
            os.unlink(marker_name, dir_fd=directory_fd)
            os.fsync(directory_fd)
    except FileNotFoundError:
        pass
    finally:
        os.close(marker_fd)
        os.close(directory_fd)


def _link_unnamed_file(file_fd: int, directory_fd: int, name: str) -> None:
    linkat = ctypes.CDLL(None, use_errno=True).linkat
    linkat.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    linkat.restype = ctypes.c_int
    if linkat(file_fd, b"", directory_fd, os.fsencode(name), 0x1000) != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(error_number, os.strerror(error_number), name)
        raise OSError(error_number, os.strerror(error_number), name)


def _write_bounded_elastic_matched_authorized(
    destination: Path, value: object, data_x: object, data_y: object, *,
    config: IPMNISTConfig, seeds: tuple[int, ...], _capability: object,
) -> None:
    expected_seeds = (
        CAMPAIGN_SEEDS if _capability is _EXECUTION_CAPABILITY
        else TEST_ONLY_SEEDS if _capability is _TEST_EXECUTION_CAPABILITY
        else None
    )
    if expected_seeds is None or seeds != expected_seeds:
        raise RuntimeError("private bounded-elastic publication capability is invalid")
    if _capability is _EXECUTION_CAPABILITY and (
        _REVIEWED_EXECUTION_TRANSITION is not True or _EXECUTION_AUTHORIZED is not True
    ):
        raise RuntimeError("bounded-elastic matched campaign execution is not authorized")
    reservation = _reserve_output(destination)
    directory_fd, destination_name, _marker, _marker_fd, _device, _inode = reservation
    descriptor = -1
    try:
        _require_owned_reservation(reservation)
        _validate_bounded_elastic_matched(
            value, data_x, data_y, config=config, seeds=seeds, reexecute=True
        )
        _require_owned_reservation(reservation)
        encoded = _canonical(value) + b"\n"
        descriptor = os.open(
            ".",
            os.O_WRONLY | os.O_TMPFILE | os.O_CLOEXEC,
            0o600,
            dir_fd=directory_fd,
        )
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("immutable output write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        _require_owned_reservation(reservation)
        try:
            _link_unnamed_file(descriptor, directory_fd, destination_name)
        except FileExistsError as error:
            raise FileExistsError(
                "refusing to replace immutable bounded-elastic output"
            ) from error
        os.fsync(directory_fd)
        read_fd = os.open(destination_name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                          dir_fd=directory_fd)
        try:
            metadata = os.fstat(read_fd)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ValueError("published output must be one unique regular file")
            actual = os.read(read_fd, len(encoded) + 1)
            final = os.fstat(read_fd)
            if actual != encoded or metadata.st_size != len(encoded) or (
                metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns
            ) != (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns):
                raise ValueError("published output changed during strict reread")
            def exact_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
                result: dict[str, object] = {}
                for key, item in pairs:
                    if key in result:
                        raise ValueError("published output contains a duplicate JSON key")
                    result[key] = item
                return result

            try:
                reread = json.loads(
                    actual.decode("utf-8"), object_pairs_hook=exact_object
                )
            except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
                raise ValueError("published output is not bounded strict JSON") from error
            _validate_bounded_elastic_matched(
                reread, data_x, data_y, config=config, seeds=seeds, reexecute=False
            )
        finally:
            os.close(read_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        _release_output(reservation)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", action="store_true")
    parser.add_argument("--data-home", type=Path, default=default_openml_data_home())
    args = parser.parse_args(argv)
    if args.catalog:
        print(json.dumps(frozen_plan(), sort_keys=True))
        return 0
    raise RuntimeError("bounded-elastic matched campaign execution is not authorized")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
