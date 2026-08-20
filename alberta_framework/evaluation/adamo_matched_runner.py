"""Run the complete, permanently nonpromoting issue #1560 AdamO campaign."""

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

from alberta_framework.benchmarks.adamo_diagnostic import (
    ARMS,
    FROZEN_DEVELOPMENT_SEEDS,
    PROFILES,
    _load_dataset,
    run_adamo_diagnostic,
    validate_adamo_diagnostic,
)
from alberta_framework.benchmarks.upgd_ipmnist import build_schedule, init_mlp_params

RESULT_SCHEMA: Final = "asi.adamo-diagnostic.matched-development.v1"
_POLICY: Final = {
    "development_only": True,
    "scientific_promotion_allowed": False,
    "sota_claim_allowed": False,
    "negative_results_retained": True,
}
_MAX_BYTES: Final = 256 * 1024 * 1024
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
        if nodes > _MAX_JSON_NODES or depth > 18:
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
        "alberta_framework/benchmarks/adamo_diagnostic.py",
        "alberta_framework/benchmarks/ipmnist_screening.py",
        "alberta_framework/benchmarks/upgd_ipmnist.py",
        "alberta_framework/core/_float32_scalars.py",
        "alberta_framework/core/adamo.py",
        "alberta_framework/core/baseline_optimizers.py",
        "alberta_framework/core/optimizers.py",
        "alberta_framework/core/update_safety.py",
        "alberta_framework/evaluation/adamo_matched_runner.py",
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
        "schema": "asi.adamo-diagnostic.runtime.v1",
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


def _checked_profile(value: object) -> str:
    if type(value) is not str or value not in PROFILES:
        raise ValueError("profile must name one registered AdamO diagnostic profile")
    return value


def _validated_arrays(
    inputs: object, labels: object, profile: str
) -> tuple[np.ndarray, np.ndarray]:
    config = PROFILES[profile].config
    if type(inputs) is not np.ndarray or inputs.dtype != np.dtype(np.float32):
        raise ValueError("inputs must be an exact float32 NumPy array")
    if type(labels) is not np.ndarray or labels.dtype != np.dtype(np.int32):
        raise ValueError("labels must be an exact int32 NumPy array")
    if (
        inputs.ndim != 2
        or labels.ndim != 1
        or inputs.shape[0] != labels.shape[0]
        or inputs.shape[0] < config.task_length
        or inputs.shape[1] != config.input_dim
    ):
        raise ValueError("dataset shape does not match the AdamO profile")
    if inputs.nbytes + labels.nbytes > _MAX_BYTES:
        raise ValueError("dataset exceeds the campaign's 256 MiB byte bound")
    schedule_bytes = config.n_tasks * (inputs.shape[0] + config.input_dim) * 4
    if schedule_bytes > _MAX_BYTES:
        raise ValueError("schedule exceeds the campaign's 256 MiB byte bound")
    if not np.isfinite(inputs).all():
        raise ValueError("inputs must be finite")
    if np.any(labels < 0) or np.any(labels >= config.n_classes):
        raise ValueError("labels are outside the configured class range")
    return np.array(inputs, copy=True, order="C"), np.array(labels, copy=True, order="C")


def _dataset_sha256(inputs: np.ndarray, labels: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in (inputs, labels):
        contiguous = np.ascontiguousarray(value)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _array_tree_sha256(value: object) -> str:
    digest = hashlib.sha256(b"asi-adamo-diagnostic-initial-parameters-v1\0")
    leaves, structure = jax.tree.flatten(value)
    digest.update(str(structure).encode("ascii"))
    for leaf in leaves:
        host = np.asarray(leaf)
        digest.update(np.asarray(host.shape, dtype="<i8").tobytes())
        digest.update(host.dtype.str.encode("ascii"))
        digest.update(host.tobytes(order="C"))
    return digest.hexdigest()


def _execution_identity(seed: int, profile: str, n_train: int) -> dict[str, str]:
    config = PROFILES[profile].config
    root = jr.key(np.uint32(seed), impl="threefry2x32")
    key_init, key_schedule, _ = jr.split(root, 3)
    parameters = init_mlp_params(key_init, config)
    schedule = build_schedule(key_schedule, config, n_train)
    digest = hashlib.sha256(b"asi-adamo-diagnostic-schedule-v1\0")
    for value in (schedule.permutations, schedule.example_indices):
        host = np.asarray(value, dtype=np.int32)
        digest.update(np.asarray(host.shape, dtype="<i8").tobytes())
        digest.update(host.astype("<i4", copy=False).tobytes(order="C"))
    return {
        "schedule_sha256": digest.hexdigest(),
        "initial_parameters_sha256": _array_tree_sha256(parameters),
        "prng_implementation": "threefry2x32",
    }


def _aggregate(shards: list[dict[str, object]]) -> dict[str, object]:
    arms: dict[str, object] = {}
    for arm_index, arm_name in enumerate(ARMS):
        arm_payloads = [
            cast(list[dict[str, object]], cast(dict[str, object], shard["result"])["arms"])[
                arm_index
            ]
            for shard in shards
        ]
        curves = {
            name: [
                cast(float, value)
                for arm in arm_payloads
                for value in cast(list[object], arm[name])
            ]
            for name in ("per_task_accuracy", "per_task_loss", "per_task_plasticity")
        }
        final_diagnostics = [
            cast(list[dict[str, object]], arm["post_task_diagnostics"])[-1]
            for arm in arm_payloads
        ]
        diagnostic_names = (
            "jacobian_min_singular_value",
            "jacobian_max_singular_value",
            "jacobian_mean_singular_value",
            "jacobian_condition_number_clipped_1e12",
            "jacobian_rms_distance_from_one",
            "weight_gram_penalty",
        )
        resources = [cast(dict[str, object], arm["resources"]) for arm in arm_payloads]
        additive_resource_names = (
            "observations",
            "updates",
            "data_steps",
            "environment_steps",
            "model_queries",
            "jacobian_reverse_rows",
            "logical_compute_units",
        )
        peak_resource_names = (
            "parameter_count",
            "persistent_numeric_bytes",
            "peak_gram_working_bytes",
        )
        arms[arm_name] = {
            "mean_curves": {
                name: math.fsum(values) / len(values) for name, values in curves.items()
            },
            "mean_final_diagnostics": {
                name: math.fsum(cast(float, item[name]) for item in final_diagnostics)
                / len(final_diagnostics)
                for name in diagnostic_names
            },
            "total_additive_resources": {
                name: sum(cast(int, resource[name]) for resource in resources)
                for name in additive_resource_names
            },
            "max_per_shard_resources": {
                name: max(cast(int, resource[name]) for resource in resources)
                for name in peak_resource_names
            },
        }
    return {"arms": arms, "shard_count": len(shards)}


def run_adamo_matched(
    inputs: object, labels: object, *, profile: str = "bounded-development"
) -> dict[str, object]:
    """Execute every frozen AdamO development seed and all nested arms."""
    checked_profile = _checked_profile(profile)
    x, y = _validated_arrays(inputs, labels, checked_profile)
    shards: list[dict[str, object]] = []
    for seed in FROZEN_DEVELOPMENT_SEEDS:
        shards.append(
            {
                "seed": seed,
                "execution_identity": _execution_identity(seed, checked_profile, x.shape[0]),
                "result": run_adamo_diagnostic(
                    x, y, profile=checked_profile, seed=seed
                ),
            }
        )
    campaign: dict[str, object] = {
        "schema": RESULT_SCHEMA,
        "status": "complete",
        "profile": checked_profile,
        "development_seeds": list(FROZEN_DEVELOPMENT_SEEDS),
        "arms": list(ARMS),
        "identity": {
            "dataset_sha256": _dataset_sha256(x, y),
            "source_sha256": _source_identity(),
            "runtime": _runtime_identity(),
            "consistency_not_attestation": True,
        },
        "policy": dict(_POLICY),
        "decision": "inconclusive",
        "shards": shards,
        "aggregate": _aggregate(shards),
    }
    campaign["result_sha256"] = hashlib.sha256(_canonical(campaign)).hexdigest()
    _validate_adamo_matched(
        campaign, x, y, profile=checked_profile, reexecute=False
    )
    return campaign


def validate_adamo_matched(
    value: object,
    inputs: object,
    labels: object,
    *,
    profile: str = "bounded-development",
) -> None:
    """Strictly validate and replay every seed shard, excluding timing."""
    _validate_adamo_matched(value, inputs, labels, profile=profile, reexecute=True)


def _validate_adamo_matched(
    value: object,
    inputs: object,
    labels: object,
    *,
    profile: str,
    reexecute: bool,
) -> None:
    _json_preflight(value)
    root = _exact_object(
        value,
        {
            "schema",
            "status",
            "profile",
            "development_seeds",
            "arms",
            "identity",
            "policy",
            "decision",
            "shards",
            "aggregate",
            "result_sha256",
        },
        "matched result",
    )
    if root["schema"] != RESULT_SCHEMA or root["status"] != "complete":
        raise ValueError("matched result identity drifted")
    checked_profile = _checked_profile(profile)
    x, y = _validated_arrays(inputs, labels, checked_profile)
    if root["profile"] != checked_profile:
        raise ValueError("matched result profile drifted")
    if root["development_seeds"] != list(FROZEN_DEVELOPMENT_SEEDS):
        raise ValueError("matched result seed roster drifted")
    if root["arms"] != list(ARMS):
        raise ValueError("matched result arm roster drifted")
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
    if root["decision"] != "inconclusive" or type(root["decision"]) is not str:
        raise ValueError("matched campaign decision must remain inconclusive")
    raw_shards = root["shards"]
    if type(raw_shards) is not list or len(raw_shards) != len(FROZEN_DEVELOPMENT_SEEDS):
        raise ValueError("matched result roster is incomplete")
    shards = cast(list[dict[str, object]], raw_shards)
    observed: list[object] = []
    identities = {
        seed: _execution_identity(seed, checked_profile, x.shape[0])
        for seed in FROZEN_DEVELOPMENT_SEEDS
    }
    for shard in shards:
        checked_shard = _exact_object(
            shard, {"seed", "execution_identity", "result"}, "matched shard"
        )
        seed = checked_shard["seed"]
        if type(seed) is not int or seed not in identities:
            raise ValueError("matched shard seed identity drifted")
        if checked_shard["execution_identity"] != identities[seed]:
            raise ValueError("matched shard execution identity drifted")
        result = validate_adamo_diagnostic(checked_shard["result"])
        if result["seed"] != seed or result["profile"] != checked_profile:
            raise ValueError("matched shard result identity drifted")
        dataset = cast(dict[str, object], result["dataset"])
        if dataset["sha256"] != expected_identity["dataset_sha256"]:
            raise ValueError("matched shard dataset identity drifted")
        observed.append(seed)
    if observed != list(FROZEN_DEVELOPMENT_SEEDS):
        raise ValueError("matched result roster order drifted")
    if root["aggregate"] != _aggregate(shards):
        raise ValueError("matched result aggregate drifted")
    unsigned = dict(root)
    claimed = unsigned.pop("result_sha256")
    if type(claimed) is not str or claimed != hashlib.sha256(_canonical(unsigned)).hexdigest():
        raise ValueError("matched result digest drifted")
    if reexecute:
        for shard in shards:
            seed = cast(int, shard["seed"])
            claimed_result = cast(dict[str, object], shard["result"])
            expected_result = run_adamo_diagnostic(
                x, y, profile=checked_profile, seed=seed
            )
            expected_arms = cast(list[dict[str, object]], expected_result["arms"])
            claimed_arms = cast(list[dict[str, object]], claimed_result["arms"])
            for expected_arm, claimed_arm in zip(expected_arms, claimed_arms, strict=True):
                expected_resources = cast(dict[str, object], expected_arm["resources"])
                claimed_resources = cast(dict[str, object], claimed_arm["resources"])
                expected_resources["timing_seconds"] = claimed_resources["timing_seconds"]
            if expected_result != claimed_result:
                raise ValueError("matched shard disagrees with strict current-source reexecution")


def _preflight_destination(destination: Path) -> Path:
    if type(destination) is not type(Path()):
        raise TypeError("destination must be an exact Path")
    parent = destination.parent.resolve(strict=True)
    resolved = parent / destination.name
    if not parent.is_dir() or destination.name in {"", ".", ".."}:
        raise ValueError("destination parent must be an existing directory")
    if os.path.lexists(resolved):
        raise FileExistsError(f"refusing to replace existing result: {resolved}")
    return resolved


def write_adamo_matched(
    destination: Path,
    value: object,
    inputs: object,
    labels: object,
    *,
    profile: str = "bounded-development",
) -> None:
    """Strictly replay and publish one result without replacing retained data."""
    resolved = _preflight_destination(destination)
    validate_adamo_matched(value, inputs, labels, profile=profile)
    parent = resolved.parent
    descriptor, temporary_name = tempfile.mkstemp(
        dir=parent, prefix=f".{resolved.name}.", suffix=".tmp"
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
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--profile", choices=tuple(PROFILES), default="bounded-development"
    )
    args = parser.parse_args(argv)
    _preflight_destination(args.output)
    inputs, labels = _load_dataset(args.dataset)
    result = run_adamo_matched(inputs, labels, profile=args.profile)
    write_adamo_matched(
        args.output, result, inputs, labels, profile=args.profile
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
