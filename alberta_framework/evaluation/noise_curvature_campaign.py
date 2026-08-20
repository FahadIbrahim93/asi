"""Matched, five-seed, permanently nonpromoting noise-curvature campaign.

This module composes the four registered scheduler arms through the current
IPMNIST runner.  It binds the supplied dataset, exact Threefry schedules and
initializations, package sources, runtime, counters, and deterministic numeric
resources.  Validation repeats all twenty runs.  Timing is deliberately
discarded because no timing protocol has been qualified.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import stat
import sys
from importlib.metadata import version
from pathlib import Path, PosixPath
from typing import Any, Final, cast

import jax
import jax.random as jr
import numpy as np

from alberta_framework.benchmarks.ipmnist_screening import (
    _screening_dataset_provenance,
    _screening_root_key,
    noise_curvature_development_result_payload,
    run_screening_config,
    screening_spec,
)
from alberta_framework.benchmarks.noise_curvature_ipmnist import (
    PAPER_REVISION,
    noise_curvature_persistent_bytes,
)
from alberta_framework.benchmarks.upgd_ipmnist import (
    IPMNISTConfig,
    build_schedule,
    init_mlp_params,
)
from alberta_framework.evaluation.noise_curvature_ipmnist_nonpromoting import (
    COMPARISON_ID,
    DEVELOPMENT_SEEDS,
    OFFICIAL_CODE_STATUS,
    PROTOCOL_DIFFERENCES,
    registered_arms,
    validate_matched_noise_curvature_results,
    validate_noise_curvature_development_result,
)

CAMPAIGN_SCHEMA: Final[str] = "asi.noise-curvature-ipmnist.matched-campaign.v1"
PLAN_SCHEMA: Final[str] = "asi.noise-curvature-ipmnist.matched-plan.v1"
_RNG_IMPL: Final[str] = "threefry2x32"
_MAX_DATASET_BYTES: Final[int] = 256 * 1024 * 1024
_MAX_RESULT_BYTES: Final[int] = 8 * 1024 * 1024
_MAX_JSON_NODES: Final[int] = 100_000
_MAX_JSON_STRING_BYTES: Final[int] = 2 * 1024 * 1024
_MAX_CAMPAIGN_OBSERVATIONS: Final[int] = 20_000_000
_MAX_CAMPAIGN_MODEL_QUERIES: Final[int] = 100_000_000
_MAX_SCHEDULE_BYTES: Final[int] = 128 * 1024 * 1024
_T_CRITICAL_DF4_95: Final[float] = 2.7764451051977987
_CANONICAL_X_SHA256: Final[str] = (
    "b8078cd833f53d89828a5e28d728517be9add34076f13fe973399f1f16381313"
)
_CANONICAL_Y_SHA256: Final[str] = (
    "4f1dd9551f104f8153409e0add59f0a71568f7bad5a5f8e2274480c186fe219a"
)
_CANONICAL_DATASET_SOURCE: Final[dict[str, object]] = {
    "provider": "openml",
    "name": "mnist_784",
    "version": 1,
    "row_start": 0,
    "row_stop_exclusive": 60_000,
}
_CANONICAL_DATASET_MATERIALIZATION: Final[str] = (
    "alberta.ipmnist.float32-neg1-pos1-int32-labels.v1"
)
_SOURCE_ROOT_FILES: Final[tuple[str, ...]] = ("pyproject.toml", "uv.lock")
_MAX_SOURCE_FILES: Final[int] = 1_024
_POLICY: Final[dict[str, bool]] = {
    "development_only": True,
    "scientific_promotion_allowed": False,
    "sota_claim_allowed": False,
    "negative_outcomes_retained": True,
    "live_control_included": False,
    "hillclimb_gate_evaluated": False,
}


def _canonical(value: object) -> bytes:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    if len(encoded) > _MAX_RESULT_BYTES:
        raise ValueError("campaign artifact exceeds its encoded byte ceiling")
    return encoded


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _assert_plain_json(value: object) -> None:
    seen: set[int] = set()
    nodes = 0
    string_bytes = 0
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > 14:
            raise ValueError("campaign artifact exceeds its JSON work bound")
        if item is None or type(item) is bool:
            continue
        if type(item) is int:
            if not -(1 << 63) <= item <= (1 << 63) - 1:
                raise ValueError("campaign JSON integer exceeds signed int64")
            continue
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("campaign JSON float must be finite")
            continue
        if type(item) is str:
            try:
                encoded = item.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError("campaign JSON string must be valid UTF-8") from exc
            if len(encoded) > 4_096 or b"\x00" in encoded:
                raise ValueError("campaign JSON string must be bounded UTF-8")
            string_bytes += len(encoded)
            if string_bytes > _MAX_JSON_STRING_BYTES:
                raise ValueError("campaign JSON string budget exceeded")
            continue
        if type(item) is list:
            identity = id(item)
            if identity in seen or list.__len__(item) > 1_024:
                raise ValueError("campaign JSON contains an alias, cycle, or oversized list")
            seen.add(identity)
            stack.extend((child, depth + 1) for child in list.__iter__(item))
            continue
        if type(item) is dict:
            identity = id(item)
            if identity in seen or dict.__len__(item) > 256:
                raise ValueError("campaign JSON contains an alias, cycle, or oversized object")
            seen.add(identity)
            for key, child in dict.items(item):
                if type(key) is not str:
                    raise ValueError("campaign JSON keys must be exact strings")
                stack.append((key, depth + 1))
                stack.append((child, depth + 1))
            continue
        raise ValueError("campaign artifact must be exact plain JSON")


def _preflight_config(config: object) -> IPMNISTConfig:
    if type(config) is not IPMNISTConfig:
        raise ValueError("config must be an exact IPMNISTConfig")
    checked = IPMNISTConfig(**config.to_config())
    if checked.task_length % 40:
        raise ValueError("task_length must be divisible by the frozen control interval")
    run_count = len(DEVELOPMENT_SEEDS) * len(registered_arms())
    observations = checked.n_steps * run_count
    if observations > _MAX_CAMPAIGN_OBSERVATIONS:
        raise ValueError("campaign observation plan exceeds its bound")
    events = checked.n_steps // 40
    queries_per_run = checked.n_steps * 2 + events * (40 + 3)
    if queries_per_run * run_count > _MAX_CAMPAIGN_MODEL_QUERIES:
        raise ValueError("campaign model-query plan exceeds its bound")
    schedule_bytes = (
        len(DEVELOPMENT_SEEDS)
        * checked.n_tasks
        * (checked.input_dim + checked.task_length)
        * np.dtype(np.int32).itemsize
    )
    if schedule_bytes > _MAX_SCHEDULE_BYTES:
        raise ValueError("campaign schedule plan exceeds its byte bound")
    noise_curvature_persistent_bytes(
        parameter_count=checked.parameter_count,
        input_dim=checked.input_dim,
        control_interval=40,
    )
    return checked


def _validated_arrays(
    data_x: object, data_y: object, *, config: IPMNISTConfig
) -> tuple[np.ndarray, np.ndarray]:
    if type(data_x) is not np.ndarray or data_x.dtype != np.dtype(np.float32):
        raise ValueError("data_x must be an exact float32 numpy.ndarray")
    if type(data_y) is not np.ndarray or data_y.dtype != np.dtype(np.int32):
        raise ValueError("data_y must be an exact int32 numpy.ndarray")
    if (
        data_x.ndim != 2
        or data_x.shape[1] != config.input_dim
        or data_y.shape != (data_x.shape[0],)
        or data_x.shape[0] < config.task_length
    ):
        raise ValueError("dataset shapes do not match the campaign config")
    caller_bytes = int(data_x.nbytes + data_y.nbytes)
    if caller_bytes > _MAX_DATASET_BYTES:
        raise ValueError("dataset exceeds the campaign byte ceiling")
    checked_x = np.array(data_x, dtype=np.float32, order="C", copy=True)
    checked_y = np.array(data_y, dtype=np.int32, order="C", copy=True)
    if not np.all(np.isfinite(checked_x)):
        raise ValueError("data_x must contain only finite values")
    if np.any(checked_y < 0) or np.any(checked_y >= config.n_classes):
        raise ValueError("data_y contains a label outside the configured classes")
    checked_x.flags.writeable = False
    checked_y.flags.writeable = False
    return checked_x, checked_y


def _array_identity(value: np.ndarray) -> dict[str, object]:
    digest = hashlib.sha256()
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return {
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "sha256": digest.hexdigest(),
        "numeric_bytes": int(value.nbytes),
    }


def _require_frozen_dataset_identity(value: object) -> dict[str, object]:
    """Reject shape-valid substitutes for the retained canonical MNIST bytes."""
    if type(value) is not dict or set(value) != {
        "schema",
        "source",
        "materialization",
        "x",
        "y",
    }:
        raise ValueError("frozen dataset identity fields drifted")
    identity = cast(dict[str, object], value)
    if identity["schema"] != "alberta.ipmnist_screening.dataset_provenance.v1":
        raise ValueError("frozen dataset identity schema drifted")
    if identity["source"] != _CANONICAL_DATASET_SOURCE:
        raise ValueError("frozen dataset OpenML source identity drifted")
    if identity["materialization"] != _CANONICAL_DATASET_MATERIALIZATION:
        raise ValueError("frozen dataset materialization identity drifted")
    x = identity["x"]
    y = identity["y"]
    if type(x) is not dict or type(y) is not dict:
        raise ValueError("frozen dataset array identities must be exact objects")
    expected_x = {
        "dtype": "<f4",
        "shape": [60_000, 784],
        "sha256": _CANONICAL_X_SHA256,
    }
    expected_y = {
        "dtype": "<i4",
        "shape": [60_000],
        "sha256": _CANONICAL_Y_SHA256,
    }
    if x != expected_x or y != expected_y:
        raise ValueError("dataset does not match the retained canonical checksums")
    return identity


def _parameter_identity(parameters: dict[str, jax.Array]) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    for name in sorted(parameters):
        value = np.asarray(jax.device_get(parameters[name]))
        digest.update(name.encode("utf-8"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(value.tobytes(order="C"))
        total += int(value.nbytes)
    return digest.hexdigest(), total


def _key_sha256(key: jax.Array) -> str:
    data = np.asarray(jr.key_data(key), dtype=np.uint32)
    return hashlib.sha256(data.tobytes(order="C")).hexdigest()


def _seed_identity(seed: int, *, config: IPMNISTConfig, n_train: int) -> dict[str, object]:
    if type(seed) is not int or seed not in DEVELOPMENT_SEEDS:
        raise ValueError("seed must belong to the frozen development roster")
    root = _screening_root_key(seed)
    key_init, key_schedule, key_noise = jr.split(root, 3)
    parameters = init_mlp_params(key_init, config)
    parameter_sha256, parameter_bytes = _parameter_identity(parameters)
    schedule = build_schedule(key_schedule, config, n_train)
    permutations = np.asarray(jax.device_get(schedule.permutations), dtype=np.int32)
    examples = np.asarray(jax.device_get(schedule.example_indices), dtype=np.int32)
    schedule_digest = hashlib.sha256()
    for value in (permutations, examples):
        schedule_digest.update(
            json.dumps(list(value.shape), separators=(",", ":")).encode("ascii")
        )
        schedule_digest.update(value.dtype.str.encode("ascii"))
        schedule_digest.update(value.tobytes(order="C"))
    result: dict[str, object] = {
        "seed": seed,
        "rng_impl": _RNG_IMPL,
        "root_key_data_sha256": _key_sha256(root),
        "initial_parameters_sha256": parameter_sha256,
        "initial_parameter_numeric_bytes": parameter_bytes,
        "schedule_sha256": schedule_digest.hexdigest(),
        "schedule_numeric_bytes": int(permutations.nbytes + examples.nbytes),
        "noise_root_key_data_sha256": _key_sha256(key_noise),
    }
    result["identity_sha256"] = _sha256(result)
    return result


def _read_source(path: Path) -> bytes:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 2 * 1024 * 1024:
            raise ValueError("campaign source must be a bounded regular file")
        payload = bytearray()
        while len(payload) <= before.st_size:
            chunk = os.read(descriptor, min(64 * 1024, before.st_size + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        after = os.fstat(descriptor)
        if len(payload) != before.st_size or (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError("campaign source changed during its bounded read")
        return bytes(payload)
    finally:
        os.close(descriptor)


def _source_inventory(root: Path) -> tuple[str, ...]:
    package_root = root / "alberta_framework"
    package_files = tuple(
        path.relative_to(root).as_posix()
        for path in sorted(package_root.rglob("*.py"))
    )
    inventory = (*_SOURCE_ROOT_FILES, *package_files)
    if not package_files or len(inventory) > _MAX_SOURCE_FILES:
        raise ValueError("campaign source inventory is empty or exceeds its file bound")
    if len(set(inventory)) != len(inventory):
        raise ValueError("campaign source inventory contains duplicate paths")
    return inventory


def _source_identity() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    inventory = _source_inventory(root)
    result = {
        relative: hashlib.sha256(_read_source(root / relative)).hexdigest()
        for relative in inventory
    }
    if _source_inventory(root) != inventory:
        raise ValueError("campaign source inventory changed during capture")
    return result


def _runtime_identity() -> dict[str, object]:
    numpy_build = json.dumps(
        np.__config__.CONFIG,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    environment_names = (
        "JAX_PLATFORMS",
        "JAX_PLATFORM_NAME",
        "JAX_ENABLE_X64",
        "JAX_DEFAULT_PRNG_IMPL",
        "JAX_DEFAULT_MATMUL_PRECISION",
        "JAX_RANDOM_SEED_OFFSET",
        "JAX_NUM_CPU_DEVICES",
        "XLA_FLAGS",
    )
    environment: dict[str, str | None] = {}
    for name in environment_names:
        value = os.environ.get(name)
        if value is not None:
            try:
                encoded = value.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError("runtime environment must be valid UTF-8") from exc
            if len(encoded) > 4_096 or b"\x00" in encoded:
                raise ValueError("runtime environment exceeds its text bound")
        environment[name] = value
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "numpy": np.__version__,
        "numpy_build_sha256": hashlib.sha256(numpy_build).hexdigest(),
        "jax": jax.__version__,
        "jaxlib": version("jaxlib"),
        "chex": version("chex"),
        "jaxtyping": version("jaxtyping"),
        "jax_backend": jax.default_backend(),
        "jax_device_count": jax.device_count(),
        "jax_local_device_count": jax.local_device_count(),
        "jax_devices": [
            {
                "platform": device.platform,
                "device_kind": device.device_kind,
                "process_index": int(device.process_index),
                "id": int(device.id),
            }
            for device in jax.devices()
        ],
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_default_prng_impl": str(jax.config.jax_default_prng_impl),
        "jax_default_matmul_precision": str(jax.config.jax_default_matmul_precision),
        "jax_random_seed_offset": int(jax.config.jax_random_seed_offset),
        "jax_threefry_partitionable": bool(jax.config.jax_threefry_partitionable),
        "jax_numpy_dtype_promotion": str(jax.config.jax_numpy_dtype_promotion),
        "jax_numpy_rank_promotion": str(jax.config.jax_numpy_rank_promotion),
        "environment": environment,
        "agent_rng_impl": "jax.random.key(seed, impl='threefry2x32')",
    }


def _plan(config: IPMNISTConfig) -> dict[str, object]:
    return {
        "schema": PLAN_SCHEMA,
        "paper_revision": PAPER_REVISION,
        "official_code_status": OFFICIAL_CODE_STATUS,
        "comparison_id": COMPARISON_ID,
        "seeds": list(DEVELOPMENT_SEEDS),
        "arms": list(registered_arms()),
        "config": config.to_config(),
        "run_order": "seed_major_arm_minor",
        "rng_impl": _RNG_IMPL,
        "noise_mode": "step",
        "timing_measured": False,
        "runner_timing_telemetry_discarded": True,
        "live_control": "excluded_requires_separate_current-source_receipt",
        "selected_ipmnist_configuration": config.matches_selected_publication_configuration,
        "multiple_comparison_correction": "none_development_screen_only",
        "protocol_differences": list(PROTOCOL_DIFFERENCES),
    }


def _normalized_receipt(result: object) -> dict[str, object]:
    receipt = noise_curvature_development_result_payload(
        cast(Any, result), outcome="inconclusive"
    )
    resources = cast(dict[str, object], receipt["resources"])
    resources["timing_seconds"] = 0.0
    return validate_noise_curvature_development_result(receipt)


def _paired_summary(deltas: list[float]) -> dict[str, object]:
    if len(deltas) != len(DEVELOPMENT_SEEDS) or not all(
        type(value) is float and math.isfinite(value) for value in deltas
    ):
        raise RuntimeError("paired campaign deltas are malformed")
    mean = math.fsum(deltas) / len(deltas)
    variance = math.fsum((value - mean) ** 2 for value in deltas) / (len(deltas) - 1)
    standard_error = math.sqrt(variance / len(deltas))
    margin = _T_CRITICAL_DF4_95 * standard_error
    return {
        "paired_deltas": deltas,
        "mean_delta": mean,
        "standard_error": standard_error,
        "confidence_interval_95": [mean - margin, mean + margin],
        "interval_method": "two_sided_student_t_df4",
    }


def _comparisons(runs: list[dict[str, object]]) -> tuple[list[dict[str, object]], str]:
    combined = "noise_curvature_combined"
    controls = (
        "noise_curvature_fixed_adam_l2",
        "noise_curvature_gradient_only",
        "noise_curvature_volatility_only",
    )
    comparisons: list[dict[str, object]] = []
    for control in controls:
        deltas: list[float] = []
        for seed in DEVELOPMENT_SEEDS:
            by_arm = {
                cast(str, row["arm"]): cast(dict[str, object], row["receipt"])
                for row in runs
                if row["seed"] == seed
            }
            candidate_metrics = cast(dict[str, object], by_arm[combined]["metrics"])
            control_metrics = cast(dict[str, object], by_arm[control]["metrics"])
            deltas.append(
                cast(float, candidate_metrics["mean_online_accuracy"])
                - cast(float, control_metrics["mean_online_accuracy"])
            )
        summary = _paired_summary(deltas)
        interval = cast(list[float], summary["confidence_interval_95"])
        comparisons.append(
            {
                "candidate": combined,
                "control": control,
                "metric": "mean_online_accuracy",
                **summary,
                "positive_interval": interval[0] > 0.0,
            }
        )
    if all(cast(bool, item["positive_interval"]) for item in comparisons):
        status = "supported"
    elif any(
        cast(list[float], item["confidence_interval_95"])[1] <= 0.0
        for item in comparisons
    ):
        status = "rejected"
    else:
        status = "inconclusive"
    return comparisons, status


def _resources(
    runs: list[dict[str, object]],
    seeds: list[dict[str, object]],
    data_x: np.ndarray,
    data_y: np.ndarray,
) -> dict[str, object]:
    receipts = [cast(dict[str, object], row["receipt"]) for row in runs]
    child = [cast(dict[str, object], receipt["resources"]) for receipt in receipts]
    integer_fields = (
        "environment_steps",
        "data_steps",
        "model_queries",
        "first_order_gradient_queries",
        "loss_only_queries",
        "hessian_vector_product_queries",
        "controller_events",
    )
    totals = {
        field: sum(cast(int, resource[field]) for resource in child)
        for field in integer_fields
    }
    persistent = [cast(int, resource["persistent_bytes"]) for resource in child]
    return {
        "run_count": len(runs),
        "dataset_snapshot_numeric_bytes": int(data_x.nbytes + data_y.nbytes),
        "identity_schedule_derivations": len(seeds),
        "execution_schedule_derivations": len(runs),
        "schedule_numeric_bytes_across_seed_identities": sum(
            cast(int, seed["schedule_numeric_bytes"]) for seed in seeds
        ),
        "initial_parameter_numeric_bytes_across_seed_identities": sum(
            cast(int, seed["initial_parameter_numeric_bytes"]) for seed in seeds
        ),
        "summed_arm_persistent_bytes": sum(persistent),
        "max_arm_persistent_bytes": max(persistent),
        "counter_totals": totals,
        "timing_seconds": 0.0,
        "timing_measured": False,
        "timing_is_telemetry_only": True,
        "peak_working_set_claimed": False,
        "preflight_scope": (
            "bounded observations model queries dataset schedules and persistent numeric state; "
            "not scalar FLOPs compiler temporaries or a timing comparison"
        ),
    }


def _build_campaign(
    data_x: np.ndarray,
    data_y: np.ndarray,
    *,
    config: IPMNISTConfig,
    dataset_identity: dict[str, object],
) -> dict[str, object]:
    source_identity = _source_identity()
    runtime_identity = _runtime_identity()
    seed_identities = [
        _seed_identity(seed, config=config, n_train=int(data_x.shape[0]))
        for seed in DEVELOPMENT_SEEDS
    ]
    identity_by_seed = {
        cast(int, item["seed"]): cast(str, item["identity_sha256"])
        for item in seed_identities
    }
    runs: list[dict[str, object]] = []
    for seed in DEVELOPMENT_SEEDS:
        for arm in registered_arms():
            result = run_screening_config(
                data_x,
                data_y,
                screening_spec(arm),
                seed=seed,
                config=config,
                noise_mode="step",
            )
            runs.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "seed_identity_sha256": identity_by_seed[seed],
                    "receipt": _normalized_receipt(result),
                }
            )
    comparisons, outcome = _comparisons(runs)
    if _source_identity() != source_identity:
        raise RuntimeError("campaign source changed during execution")
    if _runtime_identity() != runtime_identity:
        raise RuntimeError("campaign runtime changed during execution")
    plan = _plan(config)
    payload: dict[str, object] = {
        "schema": CAMPAIGN_SCHEMA,
        "status": "complete",
        "plan": plan,
        "identity": {
            "dataset": dataset_identity,
            "plan_sha256": _sha256(plan),
            "source_sha256": source_identity,
            "runtime": runtime_identity,
            "consistency_not_attestation": True,
        },
        "policy": dict(_POLICY),
        "seed_identities": seed_identities,
        "runs": runs,
        "comparisons": comparisons,
        "development_outcome": outcome,
        "resources": _resources(runs, seed_identities, data_x, data_y),
    }
    payload["result_sha256"] = _sha256(payload)
    return payload


def run_noise_curvature_campaign(
    data_x: object,
    data_y: object,
    *,
    config: IPMNISTConfig = IPMNISTConfig(),
) -> dict[str, object]:
    """Execute all twenty matched development runs without promoting them."""
    checked_config = _preflight_config(config)
    checked_x, checked_y = _validated_arrays(data_x, data_y, config=checked_config)
    dataset_identity = _require_frozen_dataset_identity(
        _screening_dataset_provenance(checked_x, checked_y)
    )
    result = _build_campaign(
        checked_x,
        checked_y,
        config=checked_config,
        dataset_identity=dataset_identity,
    )
    _assert_plain_json(result)
    return result


def _static_preflight(value: object) -> dict[str, object]:
    _assert_plain_json(value)
    if type(value) is not dict:
        raise ValueError("campaign must be an exact object")
    root = cast(dict[str, object], value)
    expected = {
        "schema",
        "status",
        "plan",
        "identity",
        "policy",
        "seed_identities",
        "runs",
        "comparisons",
        "development_outcome",
        "resources",
        "result_sha256",
    }
    if set(root) != expected:
        raise ValueError("campaign fields drifted")
    if root["schema"] != CAMPAIGN_SCHEMA or root["status"] != "complete":
        raise ValueError("campaign schema or completion status drifted")
    if root["policy"] != _POLICY:
        raise ValueError("campaign policy must remain permanently nonpromoting")
    if type(root["plan"]) is not dict or type(root["identity"]) is not dict:
        raise ValueError("campaign plan and identity must be exact objects")
    if type(root["resources"]) is not dict:
        raise ValueError("campaign resources must be an exact object")
    seed_identities = root["seed_identities"]
    if type(seed_identities) is not list or len(seed_identities) != len(
        DEVELOPMENT_SEEDS
    ):
        raise ValueError("campaign seed-identity roster is incomplete")
    runs = root["runs"]
    if type(runs) is not list or len(runs) != len(DEVELOPMENT_SEEDS) * len(
        registered_arms()
    ):
        raise ValueError("campaign roster is incomplete")
    expected_roster = [
        (seed, arm) for seed in DEVELOPMENT_SEEDS for arm in registered_arms()
    ]
    observed_roster: list[tuple[object, object]] = []
    by_seed: dict[int, list[dict[str, object]]] = {
        seed: [] for seed in DEVELOPMENT_SEEDS
    }
    for row in runs:
        if type(row) is not dict or set(row) != {
            "seed",
            "arm",
            "seed_identity_sha256",
            "receipt",
        }:
            raise ValueError("campaign run row fields drifted")
        seed, arm = row["seed"], row["arm"]
        observed_roster.append((seed, arm))
        if type(seed) is not int or seed not in DEVELOPMENT_SEEDS:
            raise ValueError("campaign run seed drifted")
        if type(arm) is not str or arm not in registered_arms():
            raise ValueError("campaign run arm drifted")
        if type(row["seed_identity_sha256"]) is not str or len(
            row["seed_identity_sha256"]
        ) != 64:
            raise ValueError("campaign run seed identity drifted")
        receipt = validate_noise_curvature_development_result(row["receipt"])
        if receipt != row["receipt"] or receipt["seed"] != seed or receipt["arm"] != arm:
            raise ValueError("campaign run receipt drifted")
        receipt_resources = cast(dict[str, object], receipt["resources"])
        if receipt_resources["timing_seconds"] != 0.0:
            raise ValueError("campaign must discard unqualified timing telemetry")
        by_seed[seed].append(receipt)
    if observed_roster != expected_roster:
        raise ValueError("campaign run roster order drifted")
    for seed in DEVELOPMENT_SEEDS:
        validate_matched_noise_curvature_results(by_seed[seed])
    comparisons = root["comparisons"]
    if type(comparisons) is not list or len(comparisons) != 3:
        raise ValueError("campaign comparison roster is incomplete")
    if root["development_outcome"] not in {"supported", "rejected", "inconclusive"}:
        raise ValueError("campaign development outcome drifted")
    claimed = root["result_sha256"]
    unsigned = dict(root)
    del unsigned["result_sha256"]
    if type(claimed) is not str or claimed != _sha256(unsigned):
        raise ValueError("campaign result digest drifted")
    return root


def validate_noise_curvature_campaign(
    value: object,
    data_x: object,
    data_y: object,
    *,
    config: IPMNISTConfig = IPMNISTConfig(),
) -> dict[str, object]:
    """Fail closed unless all identities and twenty runs recompute exactly."""
    root = _static_preflight(value)
    checked_config = _preflight_config(config)
    checked_x, checked_y = _validated_arrays(data_x, data_y, config=checked_config)
    dataset_identity = _require_frozen_dataset_identity(
        _screening_dataset_provenance(checked_x, checked_y)
    )
    expected_plan = _plan(checked_config)
    if root["plan"] != expected_plan:
        raise ValueError("campaign plan does not match the supplied config")
    expected_seed_identities = [
        _seed_identity(seed, config=checked_config, n_train=int(checked_x.shape[0]))
        for seed in DEVELOPMENT_SEEDS
    ]
    if root["seed_identities"] != expected_seed_identities:
        raise ValueError("campaign seed identities do not match the supplied inputs")
    expected_seed_digests = {
        cast(int, item["seed"]): cast(str, item["identity_sha256"])
        for item in expected_seed_identities
    }
    for row in cast(list[dict[str, object]], root["runs"]):
        if row["seed_identity_sha256"] != expected_seed_digests[cast(int, row["seed"])]:
            raise ValueError("campaign run does not bind its exact seed identity")
    expected_identity = {
        "dataset": dataset_identity,
        "plan_sha256": _sha256(expected_plan),
        "source_sha256": _source_identity(),
        "runtime": _runtime_identity(),
        "consistency_not_attestation": True,
    }
    if root["identity"] != expected_identity:
        raise ValueError("campaign identity does not match current inputs/source/runtime")
    expected = _build_campaign(
        checked_x,
        checked_y,
        config=checked_config,
        dataset_identity=dataset_identity,
    )
    if root != expected:
        raise ValueError("campaign does not recompute exactly from the bound inputs")
    return expected


def _resign_for_test(value: dict[str, object]) -> None:
    """Re-sign a mutated test fixture so tests reach strict reexecution."""
    unsigned = dict(value)
    unsigned.pop("result_sha256", None)
    value["result_sha256"] = _sha256(unsigned)


def retain_noise_curvature_campaign(
    value: object,
    data_x: object,
    data_y: object,
    *,
    config: IPMNISTConfig = IPMNISTConfig(),
    repository_root: Path,
) -> Path:
    """Validate and publish one content-named result without replacement."""
    if type(repository_root) is not PosixPath or not repository_root.is_absolute():
        raise ValueError("repository_root must be an exact absolute POSIX Path")
    validated = validate_noise_curvature_campaign(
        value, data_x, data_y, config=config
    )
    encoded = _canonical(validated)
    digest = cast(str, validated["result_sha256"])
    segments = ("outputs", "noise_curvature", "development.v1")
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    directory_descriptor = os.open(repository_root, flags)
    published = False
    destination_name = f"result.{digest}.json"
    temporary_name = f".result.{digest}.tmp"
    try:
        for segment in segments:
            try:
                os.mkdir(segment, mode=0o755, dir_fd=directory_descriptor)
            except FileExistsError:
                pass
            next_descriptor = os.open(segment, flags, dir_fd=directory_descriptor)
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o444,
            dir_fd=directory_descriptor,
        )
        try:
            offset = 0
            while offset < len(encoded):
                written = os.write(descriptor, encoded[offset:])
                if written <= 0:
                    raise OSError("campaign retention write made no progress")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(
                temporary_name,
                destination_name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            published = True
        finally:
            os.unlink(temporary_name, dir_fd=directory_descriptor)
        read_descriptor = os.open(
            destination_name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_descriptor,
        )
        try:
            loaded = bytearray()
            while len(loaded) <= _MAX_RESULT_BYTES:
                chunk = os.read(
                    read_descriptor,
                    min(64 * 1024, _MAX_RESULT_BYTES + 1 - len(loaded)),
                )
                if not chunk:
                    break
                loaded.extend(chunk)
        finally:
            os.close(read_descriptor)
        if bytes(loaded) != encoded:
            raise RuntimeError("retained campaign bytes changed during publication")
        os.fsync(directory_descriptor)
    except BaseException:
        if published:
            os.unlink(destination_name, dir_fd=directory_descriptor)
        raise
    finally:
        os.close(directory_descriptor)
    return repository_root.joinpath(*segments, destination_name)


__all__ = [
    "CAMPAIGN_SCHEMA",
    "retain_noise_curvature_campaign",
    "run_noise_curvature_campaign",
    "validate_noise_curvature_campaign",
]
