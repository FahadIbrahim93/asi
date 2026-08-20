"""Run the complete, permanently nonpromoting issue #1565 C-CHAIN comparison."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Final, cast

import jax
import jax.random as jr
import numpy as np

from alberta_framework.benchmarks.cchain_ipmnist import (
    COEFFICIENT_WINDOW,
    MAX_NTK_EXAMPLES,
    REFERENCE_CAPACITY,
)
from alberta_framework.benchmarks.ipmnist_screening import (
    cchain_development_result_payload,
    run_screening_config,
    screening_spec,
)
from alberta_framework.benchmarks.upgd_ipmnist import (
    IPMNISTConfig,
    build_schedule,
    default_openml_data_home,
    init_mlp_params,
    load_mnist_train,
)
from alberta_framework.evaluation.cchain_ipmnist_nonpromoting import (
    DEVELOPMENT_SEEDS,
    validate_cchain_development_result,
    validate_matched_cchain_development_results,
)

RESULT_SCHEMA: Final = "asi.cchain-ipmnist.matched-development.v1"
ARMS: Final = (
    "cchain_mechanism_off",
    "cchain_full",
    "cchain_orthogonal_only",
    "cchain_projective_only",
)
_POLICY: Final = {
    "development_only": True,
    "scientific_promotion_allowed": False,
    "sota_claim_allowed": False,
    "negative_results_retained": True,
}
_MAX_STEPS: Final = 2_000_000
_MAX_BYTES: Final = 256 * 1024 * 1024
_MAX_JSON_NODES: Final = 100_000
_MAX_JSON_STRING_BYTES: Final = 16_384
_PARAMETER_LEAVES: Final = 6


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
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > 16:
            raise ValueError("matched result exceeds its JSON structure bound")
        if current is None or type(current) in {bool, int}:
            continue
        if type(current) is float:
            if not math.isfinite(current):
                raise ValueError("matched result contains a non-finite float")
            continue
        if type(current) is str:
            if len(current.encode("utf-8")) > _MAX_JSON_STRING_BYTES:
                raise ValueError("matched result contains an oversized string")
            continue
        if type(current) not in {dict, list}:
            raise ValueError("matched result must contain only exact JSON values")
        identity = id(current)
        if identity in seen:
            raise ValueError("matched result contains an aliased or cyclic container")
        seen.add(identity)
        if type(current) is dict:
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
        "alberta_framework/benchmarks/cchain_ipmnist.py",
        "alberta_framework/benchmarks/ipmnist_screening.py",
        "alberta_framework/benchmarks/upgd_ipmnist.py",
        "alberta_framework/core/_float32_scalars.py",
        "alberta_framework/core/baseline_optimizers.py",
        "alberta_framework/core/optimizers.py",
        "alberta_framework/core/update_safety.py",
        "alberta_framework/evaluation/cchain_ipmnist_nonpromoting.py",
        "alberta_framework/evaluation/cchain_matched_runner.py",
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
        "schema": "asi.cchain-ipmnist.runtime.v1",
        "python": list(sys.version_info[:3]),
        "python_implementation": platform.python_implementation(),
        "byteorder": sys.byteorder,
        "platform": sys.platform,
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("chex", "jax", "jaxlib", "numpy", "scikit-learn")
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
    steps = config.n_tasks * config.task_length
    if steps > _MAX_STEPS:
        raise ValueError("C-CHAIN campaign exceeds its 2000000-step bound")
    parameter_count = (
        config.input_dim * config.hidden1
        + config.hidden1
        + config.hidden1 * config.hidden2
        + config.hidden2
        + config.hidden2 * config.n_classes
        + config.n_classes
    )
    persistent_scalars = (
        4 * parameter_count
        + 5 * _PARAMETER_LEAVES
        + REFERENCE_CAPACITY * config.input_dim
        + 2 * COEFFICIENT_WINDOW
        + 9
    )
    ntk_examples = min(steps, REFERENCE_CAPACITY, MAX_NTK_EXAMPLES)
    ntk_bytes = 2 * ntk_examples * config.n_classes * parameter_count * 4
    if 4 * persistent_scalars > _MAX_BYTES or ntk_bytes > _MAX_BYTES:
        raise ValueError("C-CHAIN campaign exceeds its 256 MiB numeric-memory bound")
    if REFERENCE_CAPACITY * config.input_dim > 1_000_000:
        raise ValueError("C-CHAIN campaign exceeds its reference-buffer element bound")
    return config


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
    if data_x.nbytes + data_y.nbytes > _MAX_BYTES:
        raise ValueError("dataset exceeds the campaign's 256 MiB byte bound")
    schedule_bytes = config.n_tasks * (data_x.shape[0] + config.input_dim) * 4
    if schedule_bytes > _MAX_BYTES:
        raise ValueError("schedule exceeds the campaign's 256 MiB byte bound")
    if not np.isfinite(data_x).all():
        raise ValueError("data_x must be finite")
    if np.any(data_y < 0) or np.any(data_y >= config.n_classes):
        raise ValueError("data_y is outside the configured class range")
    return np.array(data_x, copy=True, order="C"), np.array(data_y, copy=True, order="C")


def _dataset_sha256(data_x: np.ndarray, data_y: np.ndarray) -> str:
    digest = hashlib.sha256(b"asi-cchain-ipmnist-dataset-v1\0")
    for value in (data_x, data_y):
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _array_tree_sha256(value: object) -> str:
    digest = hashlib.sha256(b"asi-cchain-ipmnist-initial-parameters-v1\0")
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
    digest = hashlib.sha256(b"asi-cchain-ipmnist-schedule-v1\0")
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
        arm_rows = [row for row in rows if row["arm"] == arm]
        results = [cast(dict[str, object], row["result"]) for row in arm_rows]
        metrics = [cast(dict[str, float], result["metrics"]) for result in results]
        resources = [cast(dict[str, object], result["resources"]) for result in results]
        metric_names = sorted(metrics[0])
        resource_names = sorted(
            name
            for name in resources[0]
            if name not in {"timing_seconds", "timing_is_telemetry_only"}
        )
        arms[arm] = {
            "mean_metrics": {
                name: math.fsum(metric[name] for metric in metrics) / len(metrics)
                for name in metric_names
            },
            "total_resources": {
                name: sum(cast(int, resource[name]) for resource in resources)
                for name in resource_names
            },
        }
    return {"arms": arms, "row_count": len(rows)}


def run_cchain_matched(
    data_x: object, data_y: object, *, config: IPMNISTConfig
) -> dict[str, object]:
    """Execute every frozen seed/arm shard under the exact current runner."""
    checked_config = _checked_config(config)
    x, y = _validated_arrays(data_x, data_y, checked_config)
    rows: list[dict[str, object]] = []
    for seed in DEVELOPMENT_SEEDS:
        execution_identity = _execution_identity(seed, checked_config, x.shape[0])
        for arm in ARMS:
            result = run_screening_config(x, y, screening_spec(arm), seed, checked_config)
            rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "execution_identity": dict(execution_identity),
                    "result": cchain_development_result_payload(
                        result, outcome="inconclusive"
                    ),
                }
            )
    campaign: dict[str, object] = {
        "schema": RESULT_SCHEMA,
        "status": "complete",
        "config": _config_payload(checked_config),
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "arms": list(ARMS),
        "identity": {
            "dataset_sha256": _dataset_sha256(x, y),
            "source_sha256": _source_identity(),
            "runtime": _runtime_identity(),
            "consistency_not_attestation": True,
        },
        "policy": dict(_POLICY),
        "rows": rows,
        "aggregate": _aggregate(rows),
    }
    campaign["result_sha256"] = hashlib.sha256(_canonical(campaign)).hexdigest()
    _validate_cchain_matched(campaign, x, y, config=checked_config, reexecute=False)
    return campaign


def validate_cchain_matched(
    value: object, data_x: object, data_y: object, *, config: IPMNISTConfig
) -> None:
    """Strictly validate and replay every shard, excluding timing telemetry."""
    _validate_cchain_matched(value, data_x, data_y, config=config, reexecute=True)


def _validate_cchain_matched(
    value: object,
    data_x: object,
    data_y: object,
    *,
    config: IPMNISTConfig,
    reexecute: bool,
) -> None:
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
    if root["development_seeds"] != list(DEVELOPMENT_SEEDS) or root["arms"] != list(ARMS):
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
    expected_roster = [(seed, arm) for seed in DEVELOPMENT_SEEDS for arm in ARMS]
    raw_rows = root["rows"]
    if type(raw_rows) is not list or len(raw_rows) != len(expected_roster):
        raise ValueError("matched result roster is incomplete")
    rows = cast(list[dict[str, object]], raw_rows)
    identities = {
        seed: _execution_identity(seed, checked_config, x.shape[0])
        for seed in DEVELOPMENT_SEEDS
    }
    observed: list[tuple[object, object]] = []
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
        payload = validate_cchain_development_result(checked_row["result"])
        if payload["outcome"] != "inconclusive":
            raise ValueError("matched campaign outcomes must remain inconclusive")
        if payload["seed"] != seed or payload["arm"] != arm:
            raise ValueError("matched row result identity drifted")
        observed.append((seed, arm))
    if observed != expected_roster:
        raise ValueError("matched result roster order drifted")
    for offset in range(0, len(rows), len(ARMS)):
        validate_matched_cchain_development_results(
            [row["result"] for row in rows[offset : offset + len(ARMS)]]
        )
    if root["aggregate"] != _aggregate(rows):
        raise ValueError("matched result aggregate drifted")
    unsigned = dict(root)
    claimed = unsigned.pop("result_sha256")
    if type(claimed) is not str or claimed != hashlib.sha256(_canonical(unsigned)).hexdigest():
        raise ValueError("matched result digest drifted")
    if reexecute:
        for row in rows:
            claimed_payload = cast(dict[str, object], row["result"])
            replay = run_screening_config(
                x,
                y,
                screening_spec(cast(str, row["arm"])),
                cast(int, row["seed"]),
                checked_config,
            )
            expected_payload = cchain_development_result_payload(
                replay, outcome="inconclusive"
            )
            expected_resources = cast(dict[str, object], expected_payload["resources"])
            claimed_resources = cast(dict[str, object], claimed_payload["resources"])
            expected_resources["timing_seconds"] = claimed_resources["timing_seconds"]
            if expected_payload != claimed_payload:
                raise ValueError("matched row disagrees with strict current-source reexecution")


def write_cchain_matched(
    destination: Path,
    value: object,
    data_x: object,
    data_y: object,
    *,
    config: IPMNISTConfig,
) -> None:
    """Strictly replay and publish one result without replacing retained data."""
    if type(destination) is not type(Path()):
        raise TypeError("destination must be an exact Path")
    validate_cchain_matched(value, data_x, data_y, config=config)
    parent = destination.parent.resolve(strict=True)
    resolved = parent / destination.name
    if not parent.is_dir() or destination.name in {"", ".", ".."}:
        raise ValueError("destination parent must be an existing directory")
    if os.path.lexists(resolved):
        raise FileExistsError(f"refusing to replace existing result: {resolved}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, resolved, follow_symlinks=False)
        published = True
        directory = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        if published:
            resolved.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-home", type=Path, default=default_openml_data_home())
    parser.add_argument("--tasks", type=int, default=200)
    parser.add_argument("--task-length", type=int, default=5_000)
    args = parser.parse_args(argv)
    config = IPMNISTConfig(n_tasks=args.tasks, task_length=args.task_length)
    x, y = load_mnist_train(args.data_home)
    result = run_cchain_matched(x, y, config=config)
    write_cchain_matched(args.output, result, x, y, config=config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
