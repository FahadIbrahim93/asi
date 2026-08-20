"""Run the complete, permanently nonpromoting issue #1566 comparison."""

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
import numpy as np

from alberta_framework.benchmarks.activation_feature_ipmnist import (
    ACTIVATION_FEATURE_SPECS,
    DEVELOPMENT_SEEDS,
    _array_bundle_sha256,
    _preflight_activation_feature_resources,
    _require_activation_feature_config,
    activation_feature_result_payload,
    run_activation_feature_arm,
    validate_activation_feature_result,
    validate_matched_activation_feature_results,
)
from alberta_framework.benchmarks.upgd_ipmnist import (
    IPMNISTConfig,
    build_schedule,
    default_openml_data_home,
    init_mlp_params,
    load_mnist_train,
)

RESULT_SCHEMA: Final = "asi.activation-feature-ipmnist.matched-development.v1"
_POLICY: Final = {
    "development_only": True,
    "scientific_promotion_allowed": False,
    "sota_claim_allowed": False,
    "negative_results_retained": True,
}
_MAX_JSON_NODES: Final = 100_000
_MAX_JSON_STRING_BYTES: Final = 16_384


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
        "alberta_framework/benchmarks/activation_feature_ipmnist.py",
        "alberta_framework/benchmarks/plasticity_comparators.py",
        "alberta_framework/benchmarks/ipmnist_screening.py",
        "alberta_framework/benchmarks/upgd_ipmnist.py",
        "alberta_framework/evaluation/activation_feature_matched_runner.py",
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
        "schema": "asi.activation-feature-ipmnist.runtime.v1",
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


def _validated_arrays(
    data_x: object, data_y: object, config: IPMNISTConfig
) -> tuple[np.ndarray, np.ndarray]:
    checked_config = _require_activation_feature_config(config)
    if type(data_x) is not np.ndarray or data_x.dtype != np.dtype(np.float32):
        raise ValueError("data_x must be an exact float32 NumPy array")
    if type(data_y) is not np.ndarray or data_y.dtype != np.dtype(np.int32):
        raise ValueError("data_y must be an exact int32 NumPy array")
    if (
        data_x.ndim != 2
        or data_y.ndim != 1
        or data_x.shape[0] != data_y.shape[0]
        or data_x.shape[0] < checked_config.task_length
        or data_x.shape[1] != checked_config.input_dim
    ):
        raise ValueError("dataset shape does not match the campaign config")
    if data_x.nbytes + data_y.nbytes > 256 * 1024 * 1024:
        raise ValueError("dataset exceeds the campaign's 256 MiB byte bound")
    if not np.isfinite(data_x).all():
        raise ValueError("data_x must be finite")
    if np.any(data_y < 0) or np.any(data_y >= checked_config.n_classes):
        raise ValueError("data_y is outside the configured class range")
    return np.array(data_x, copy=True, order="C"), np.array(data_y, copy=True, order="C")


def _config_payload(config: IPMNISTConfig) -> dict[str, int]:
    return {
        name: getattr(config, name)
        for name in ("n_tasks", "task_length", "input_dim", "hidden1", "hidden2", "n_classes")
    }


def _array_tree_sha256(value: object) -> str:
    digest = hashlib.sha256(b"asi-activation-feature-initial-parameters-v1\0")
    leaves, structure = jax.tree.flatten(value)
    digest.update(str(structure).encode("ascii"))
    for leaf in leaves:
        host = np.asarray(leaf)
        digest.update(np.asarray(host.shape, dtype="<i8").tobytes())
        digest.update(host.dtype.str.encode("ascii"))
        digest.update(host.tobytes(order="C"))
    return digest.hexdigest()


def _execution_identity(seed: int, config: IPMNISTConfig, n_train: int) -> dict[str, str]:
    root = jax.random.key(np.uint32(seed), impl="threefry2x32")
    key_init, key_schedule, _ = jax.random.split(root, 3)
    parameters = init_mlp_params(key_init, config)
    schedule = build_schedule(key_schedule, config, n_train)
    digest = hashlib.sha256(b"asi-activation-feature-schedule-v1\0")
    for value in (schedule.permutations, schedule.example_indices):
        host = np.asarray(value, dtype=np.int32)
        digest.update(np.asarray(host.shape, dtype="<i8").tobytes())
        digest.update(host.astype("<i4", copy=False).tobytes(order="C"))
    return {
        "schedule_sha256": digest.hexdigest(),
        "initial_parameters_sha256": _array_tree_sha256(parameters),
        "prng_implementation": "threefry2x32",
    }


def _preflight_campaign(config: IPMNISTConfig, n_train: int) -> None:
    steps = config.n_tasks * config.task_length
    if steps > 2_000_000:
        raise ValueError("activation/feature campaign exceeds its 2000000-step bound")
    _preflight_activation_feature_resources(config, n_train=n_train)
    allocated = (
        config.input_dim * config.hidden1
        + config.hidden1
        + config.hidden1 * config.hidden2
        + config.hidden2
        + config.hidden2 * config.n_classes
        + config.n_classes
    )
    persistent_bytes = 4 * (allocated + 2 * config.input_dim + 1)
    if persistent_bytes > 256 * 1024 * 1024:
        raise ValueError("activation/feature campaign exceeds its persistent-memory bound")


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    arms: dict[str, object] = {}
    for arm in ACTIVATION_FEATURE_SPECS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        metrics = [
            cast(dict[str, float], cast(dict[str, object], row["result"])["metrics"])
            for row in arm_rows
        ]
        arms[arm] = {
            "mean_accuracy": math.fsum(item["asi_whole_stream_mean_accuracy"] for item in metrics)
            / len(metrics),
            "mean_loss": math.fsum(item["asi_whole_stream_mean_loss"] for item in metrics)
            / len(metrics),
            "mean_plasticity": math.fsum(
                item["asi_whole_stream_mean_plasticity"] for item in metrics
            )
            / len(metrics),
        }
    return {"arms": arms, "row_count": len(rows)}


def run_activation_feature_matched(
    data_x: object, data_y: object, *, config: IPMNISTConfig
) -> dict[str, object]:
    """Execute every frozen seed/arm shard under the exact current runner."""
    checked_config = _require_activation_feature_config(config)
    x, y = _validated_arrays(data_x, data_y, checked_config)
    _preflight_campaign(checked_config, x.shape[0])
    rows: list[dict[str, object]] = []
    for seed in DEVELOPMENT_SEEDS:
        execution_identity = _execution_identity(seed, checked_config, x.shape[0])
        for arm in ACTIVATION_FEATURE_SPECS:
            arm_result = run_activation_feature_arm(x, y, arm=arm, seed=seed, config=checked_config)
            rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "execution_identity": dict(execution_identity),
                    "result": activation_feature_result_payload(arm_result, outcome="inconclusive"),
                }
            )
    identity = {
        "dataset_sha256": _array_bundle_sha256(x, y),
        "source_sha256": _source_identity(),
        "runtime": _runtime_identity(),
        "consistency_not_attestation": True,
    }
    campaign: dict[str, object] = {
        "schema": RESULT_SCHEMA,
        "status": "complete",
        "config": _config_payload(checked_config),
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "arms": list(ACTIVATION_FEATURE_SPECS),
        "identity": identity,
        "policy": dict(_POLICY),
        "rows": rows,
        "aggregate": _aggregate(rows),
    }
    campaign["result_sha256"] = hashlib.sha256(_canonical(campaign)).hexdigest()
    _validate_activation_feature_matched(
        campaign, x, y, config=checked_config, reexecute=False
    )
    return campaign


def validate_activation_feature_matched(
    value: object, data_x: object, data_y: object, *, config: IPMNISTConfig
) -> None:
    """Strictly validate and replay every shard, excluding timing telemetry."""
    _validate_activation_feature_matched(value, data_x, data_y, config=config, reexecute=True)


def _validate_activation_feature_matched(
    value: object,
    data_x: object,
    data_y: object,
    *,
    config: IPMNISTConfig,
    reexecute: bool,
) -> None:
    """Validate roster, identities, aggregates, policy, and optionally execution."""
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
    checked_config = _require_activation_feature_config(config)
    x, y = _validated_arrays(data_x, data_y, checked_config)
    _preflight_campaign(checked_config, x.shape[0])
    if root["config"] != _config_payload(checked_config):
        raise ValueError("matched result config drifted")
    if root["development_seeds"] != list(DEVELOPMENT_SEEDS):
        raise ValueError("matched result seed protocol drifted")
    if root["arms"] != list(ACTIVATION_FEATURE_SPECS):
        raise ValueError("matched result arm protocol drifted")
    expected_identity = {
        "dataset_sha256": _array_bundle_sha256(x, y),
        "source_sha256": _source_identity(),
        "runtime": _runtime_identity(),
        "consistency_not_attestation": True,
    }
    if root["identity"] != expected_identity:
        raise ValueError("matched result identity drifted")
    if root["policy"] != _POLICY:
        raise ValueError("matched result must remain permanently nonpromoting")
    raw_rows = root["rows"]
    expected_roster = [
        (seed, arm) for seed in DEVELOPMENT_SEEDS for arm in ACTIVATION_FEATURE_SPECS
    ]
    if type(raw_rows) is not list or len(raw_rows) != len(expected_roster):
        raise ValueError("matched result roster is incomplete")
    rows = cast(list[dict[str, object]], raw_rows)
    observed: list[tuple[object, object]] = []
    identities = {
        seed: _execution_identity(seed, checked_config, x.shape[0])
        for seed in DEVELOPMENT_SEEDS
    }
    for row in rows:
        checked_row = _exact_object(
            row, {"seed", "arm", "execution_identity", "result"}, "matched row"
        )
        payload = validate_activation_feature_result(checked_row["result"])
        if payload["outcome"] != "inconclusive":
            raise ValueError("matched campaign outcomes must remain inconclusive")
        if checked_row["seed"] != payload["seed"] or checked_row["arm"] != payload["arm"]:
            raise ValueError("matched row identity drifted")
        seed = cast(int, checked_row["seed"])
        if checked_row["execution_identity"] != identities[seed]:
            raise ValueError("matched row execution identity drifted")
        execution = cast(dict[str, object], payload["execution_identity"])
        if (
            execution["dataset_sha256"] != expected_identity["dataset_sha256"]
            or execution["schedule_sha256"] != identities[seed]["schedule_sha256"]
        ):
            raise ValueError("matched row dataset or schedule identity drifted")
        observed.append((checked_row["seed"], checked_row["arm"]))
    if observed != expected_roster:
        raise ValueError("matched result roster order drifted")
    for offset in range(0, len(rows), len(ACTIVATION_FEATURE_SPECS)):
        validate_matched_activation_feature_results(
            [row["result"] for row in rows[offset : offset + len(ACTIVATION_FEATURE_SPECS)]]
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
            replay = run_activation_feature_arm(
                x,
                y,
                arm=cast(str, row["arm"]),
                seed=cast(int, row["seed"]),
                config=checked_config,
            )
            expected_payload = activation_feature_result_payload(replay, outcome="inconclusive")
            expected_resources = cast(dict[str, object], expected_payload["resources"])
            claimed_resources = cast(dict[str, object], claimed_payload["resources"])
            expected_resources["timing_seconds"] = claimed_resources["timing_seconds"]
            if expected_payload != claimed_payload:
                raise ValueError("matched row disagrees with strict current-source reexecution")


def write_activation_feature_matched(
    destination: Path,
    value: object,
    data_x: object,
    data_y: object,
    *,
    config: IPMNISTConfig,
) -> None:
    """Validate and publish one result without replacing any retained outcome."""
    if type(destination) is not type(Path()):
        raise TypeError("destination must be an exact Path")
    validate_activation_feature_matched(value, data_x, data_y, config=config)
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
    result = run_activation_feature_matched(x, y, config=config)
    write_activation_feature_matched(args.output, result, x, y, config=config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
