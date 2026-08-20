"""Strict five-seed development campaign for gradual micro-phase arms.

The campaign is permanently nonpromoting and has no registered selection rule.
Validation reexecutes all five three-arm runs; consistency hashes are not
authenticated execution proof.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import stat
from importlib import metadata
from pathlib import Path
from typing import Final, cast

import jax
import jax.random as jr
import numpy as np

from alberta_framework.benchmarks.ipmnist_gradual_family import (
    _ADAMW_IDENTITY,
    GradualMicroPhaseConfig,
    GradualMicroPhaseResult,
    _realized_schedule,
    run_gradual_micro_phase_family,
)
from alberta_framework.benchmarks.upgd_ipmnist import (
    IPMNISTConfig,
    _make_adamw_learner,
    atomic_write_new,
    init_mlp_params,
    validated_ipmnist_data,
)

SCHEMA: Final[str] = "asi.gradual-micro-phase.matched-campaign.v1"
SEEDS: Final[tuple[int, ...]] = (156901, 156902, 156903, 156904, 156905)
ARM_IDS: Final[tuple[str, ...]] = ("abrupt", "output_interpolation", "task_sampling")
DEFAULT_FROZEN_CONFIG: Final[GradualMicroPhaseConfig] = GradualMicroPhaseConfig(
    transition_intervals=10,
    phase_examples=5000,
    input_dim=784,
    hidden1=300,
    hidden2=150,
    n_classes=10,
)
FROZEN_CONFIG = DEFAULT_FROZEN_CONFIG
POLICY: Final[dict[str, object]] = {
    "development_only": True,
    "scientific_promotion_allowed": False,
    "publication_equivalent": False,
    "sota_claim_allowed": False,
    "negative_outcomes_retained": True,
}
_DECISION: Final[dict[str, object]] = {
    "status": "inconclusive",
    "reason": "no_registered_selection_rule",
    "candidate_selected": None,
}
_MAX_JSON_BYTES = 32 * 1024 * 1024
_MAX_JSON_NODES = 200_000
_MAX_DATASET_BYTES = 256 * 1024 * 1024
_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_FILES = (
    "alberta_framework/evaluation/gradual_micro_phase_campaign.py",
    "alberta_framework/_seed_validation.py",
    "alberta_framework/benchmarks/ipmnist_gradual_family.py",
    "alberta_framework/benchmarks/ipmnist_gradual.py",
    "alberta_framework/benchmarks/upgd_ipmnist.py",
    "alberta_framework/core/baseline_optimizers.py",
    "alberta_framework/core/_float32_scalars.py",
    "alberta_framework/core/update_safety.py",
    "pyproject.toml",
    "uv.lock",
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _array_hash(domain: str, *values: np.ndarray) -> str:
    digest = hashlib.sha256(domain.encode("ascii") + b"\0")
    for value in values:
        array = np.ascontiguousarray(value.astype(value.dtype.newbyteorder("<"), copy=False))
        header = _canonical({"dtype": array.dtype.str, "shape": list(array.shape)})
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(array.nbytes.to_bytes(8, "little"))
        digest.update(memoryview(array).cast("B"))
    return "sha256:" + digest.hexdigest()


def _tree_hash(domain: str, value: object) -> str:
    leaves, treedef = jax.tree_util.tree_flatten(value)
    digest = hashlib.sha256(domain.encode("ascii") + b"\0" + str(treedef).encode())
    for leaf in leaves:
        array = np.asarray(jax.device_get(leaf))
        array = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
        digest.update(_canonical({"dtype": array.dtype.str, "shape": list(array.shape)}))
        digest.update(memoryview(array).cast("B"))
    return "sha256:" + digest.hexdigest()


def _sources() -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for relative in _SOURCE_FILES:
        path = _ROOT / relative
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
                chunk = os.read(
                    descriptor,
                    min(64 * 1024, before.st_size + 1 - len(payload)),
                )
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
                raise ValueError("campaign source changed during bounded capture")
        finally:
            os.close(descriptor)
        result.append(
            {
                "path": relative,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return result


def _runtime() -> dict[str, object]:
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jaxlib": metadata.version("jaxlib"),
        "numpy": np.__version__,
        "backend": jax.default_backend(),
        "devices": [
            {
                "id": int(device.id),
                "process_index": int(device.process_index),
                "platform": str(device.platform),
                "kind": str(device.device_kind),
            }
            for device in jax.devices()
        ],
        "jax_config": {
            "enable_x64": bool(jax.config.jax_enable_x64),
            "default_prng_impl": str(jax.config.jax_default_prng_impl),
            "default_matmul_precision": str(jax.config.jax_default_matmul_precision),
            "random_seed_offset": int(jax.config.jax_random_seed_offset),
            "threefry_partitionable": bool(jax.config.jax_threefry_partitionable),
            "numpy_dtype_promotion": str(jax.config.jax_numpy_dtype_promotion),
            "numpy_rank_promotion": str(jax.config.jax_numpy_rank_promotion),
            "disable_jit": bool(jax.config.jax_disable_jit),
        },
        "environment": {
            name: os.environ.get(name)
            for name in (
                "JAX_PLATFORMS",
                "JAX_PLATFORM_NAME",
                "JAX_ENABLE_X64",
                "JAX_DEFAULT_PRNG_IMPL",
                "JAX_DEFAULT_MATMUL_PRECISION",
                "JAX_RANDOM_SEED_OFFSET",
                "XLA_FLAGS",
            )
        },
        "agent_rng": "jax.random.key(seed, impl='threefry2x32')",
    }


def _config_payload(config: GradualMicroPhaseConfig) -> dict[str, int]:
    return {
        name: cast(int, getattr(config, name))
        for name in (
            "transition_intervals",
            "phase_examples",
            "input_dim",
            "hidden1",
            "hidden2",
            "n_classes",
        )
    }


def _datasets(
    old_x: object, old_y: object, new_x: object, new_y: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    config = FROZEN_CONFIG
    for name, value, dtype in (
        ("old_x", old_x, np.dtype(np.float32)),
        ("old_y", old_y, np.dtype(np.int32)),
        ("new_x", new_x, np.dtype(np.float32)),
        ("new_y", new_y, np.dtype(np.int32)),
    ):
        if type(value) is not np.ndarray or value.dtype != dtype:
            raise ValueError(f"{name} must be an exact {dtype} numpy array")
    old_x_array = cast(np.ndarray, old_x)
    old_y_array = cast(np.ndarray, old_y)
    new_x_array = cast(np.ndarray, new_x)
    new_y_array = cast(np.ndarray, new_y)
    if (
        old_x_array.shape != (config.phase_examples, config.input_dim)
        or new_x_array.shape != old_x_array.shape
        or old_y_array.shape != (config.phase_examples,)
        or new_y_array.shape != old_y_array.shape
    ):
        raise ValueError("campaign datasets must contain exactly one row-aligned phase")
    if (
        sum(value.nbytes for value in (old_x_array, old_y_array, new_x_array, new_y_array))
        > _MAX_DATASET_BYTES
    ):
        raise ValueError("campaign dataset byte bound exceeded before materialization")
    ox, oy = validated_ipmnist_data(
        old_x_array,
        old_y_array,
        input_dim=config.input_dim,
        n_classes=config.n_classes,
        min_length=config.phase_examples,
    )
    nx, ny = validated_ipmnist_data(
        new_x_array,
        new_y_array,
        input_dim=config.input_dim,
        n_classes=config.n_classes,
        min_length=config.phase_examples,
    )
    ox = np.array(ox, np.float32, order="C", copy=True)
    oy = np.array(oy, np.int32, order="C", copy=True)
    nx = np.array(nx, np.float32, order="C", copy=True)
    ny = np.array(ny, np.int32, order="C", copy=True)
    if np.any(ox < -1.0) or np.any(ox > 1.0) or np.any(nx < -1.0) or np.any(nx > 1.0):
        raise ValueError("campaign features must use the frozen [-1, 1] domain")
    if ox.shape != nx.shape or not np.array_equal(ox, nx):
        raise ValueError("campaign requires exact row-aligned old/new inputs")
    identity = {
        "protocol": "caller-fed-paired-micro-phase-data-not-official-dataset-acquisition",
        "old": _array_hash("asi.gradual-campaign.old.v1", ox, oy),
        "new": _array_hash("asi.gradual-campaign.new.v1", nx, ny),
        "rows": int(ox.shape[0]),
        "input_dim": config.input_dim,
        "numeric_bytes": int(ox.nbytes + oy.nbytes + nx.nbytes + ny.nbytes),
        "row_aligned_inputs": True,
    }
    return ox, oy, nx, ny, identity


def _seed_identity(seed: int, config: GradualMicroPhaseConfig, rows: int) -> dict[str, object]:
    root = jr.key(seed, impl="threefry2x32")
    init_key, _, learner_key = jr.split(root, 3)
    model_config = IPMNISTConfig(
        n_tasks=config.phase_count,
        task_length=config.phase_examples,
        input_dim=config.input_dim,
        hidden1=config.hidden1,
        hidden2=config.hidden2,
        n_classes=config.n_classes,
    )
    params = init_mlp_params(init_key, model_config)
    init_fn, _ = _make_adamw_learner(dict(_ADAMW_IDENTITY))
    state = init_fn(params)
    _, _, _, counts, schedule = _realized_schedule(seed, config, rows, rows)
    payload: dict[str, object] = {
        "seed": seed,
        "rng_impl": "threefry2x32",
        "root_sha256": _tree_hash("asi.gradual-campaign.root.v1", jr.key_data(root)),
        "learner_root_sha256": _tree_hash(
            "asi.gradual-campaign.learner-root.v1", jr.key_data(learner_key)
        ),
        "initial_parameters_sha256": _tree_hash("asi.gradual-campaign.initial-params.v1", params),
        "initial_learner_state_sha256": _tree_hash("asi.gradual-campaign.initial-state.v1", state),
        "schedule_sha256": schedule,
        "task_sampling_new_counts": counts,
    }
    payload["identity_sha256"] = _sha(payload)
    return payload


def _records(result: GradualMicroPhaseResult, seed_identity: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index, arm in enumerate(ARM_IDS):
        records.append(
            {
                "seed": result.seed,
                "arm": arm,
                "seed_identity_sha256": seed_identity,
                "training_loss_sums": result.training_loss_sums[index].tolist(),
                "new_task_eval_correct_counts": result.new_task_eval_correct_counts[index].tolist(),
                "new_task_eval_loss_sums": result.new_task_eval_loss_sums[index].tolist(),
                "persistent_numeric_bytes": int(result.persistent_numeric_bytes[index]),
                "observations": result.observations_per_arm,
                "updates": result.updates_per_arm,
                "data_steps": result.data_steps_per_arm,
                "environment_steps": result.environment_steps_per_arm,
                "model_queries": result.model_queries_per_arm,
                "soft_target_updates": result.soft_target_updates_per_arm[index],
                "timing_retained": False,
                "execution_attestation": False,
            }
        )
    return records


def _comparisons(
    records: list[dict[str, object]], config: GradualMicroPhaseConfig
) -> dict[str, object]:
    result: dict[str, object] = {}
    for arm in ARM_IDS[1:]:
        deltas: list[float] = []
        for seed in SEEDS:
            by_arm = {r["arm"]: r for r in records if r["seed"] == seed}
            candidate = cast(list[int], by_arm[arm]["new_task_eval_correct_counts"])[-1]
            control = cast(list[int], by_arm[ARM_IDS[0]]["new_task_eval_correct_counts"])[-1]
            deltas.append(float((candidate - control) / config.phase_examples))
        mean = math.fsum(deltas) / len(deltas)
        variance = math.fsum((value - mean) ** 2 for value in deltas) / 4
        half = 2.7764451051977987 * math.sqrt(variance / 5)
        result[arm] = {
            "control": "abrupt",
            "metric": "final_phase_new_task_accuracy",
            "paired_deltas": deltas,
            "mean_delta": mean,
            "confidence_interval_95": [mean - half, mean + half],
            "interval_method": "two_sided_student_t_df4_descriptive_only",
        }
    return result


def _resources(records: list[dict[str, object]], dataset: dict[str, object]) -> dict[str, object]:
    return {
        "runs": len(SEEDS),
        "arm_cells": len(records),
        "dataset_numeric_bytes": dataset["numeric_bytes"],
        "total_observations": sum(cast(int, row["observations"]) for row in records),
        "total_updates": sum(cast(int, row["updates"]) for row in records),
        "total_data_steps": sum(cast(int, row["data_steps"]) for row in records),
        "total_environment_steps": sum(cast(int, row["environment_steps"]) for row in records),
        "total_model_queries": sum(cast(int, row["model_queries"]) for row in records),
        "summed_persistent_numeric_bytes": sum(
            cast(int, row["persistent_numeric_bytes"]) for row in records
        ),
        "max_cell_persistent_numeric_bytes": max(
            cast(int, row["persistent_numeric_bytes"]) for row in records
        ),
        "physical_peak_rss_claimed": False,
        "timing_measured_but_discarded": True,
        "timing_is_selection_metric": False,
    }


def _execute(
    old_x: np.ndarray, old_y: np.ndarray, new_x: np.ndarray, new_y: np.ndarray
) -> dict[str, object]:
    config = FROZEN_CONFIG
    sources = _sources()
    runtime = _runtime()
    ox, oy, nx, ny, dataset = _datasets(old_x, old_y, new_x, new_y)
    identities = [_seed_identity(seed, config, int(ox.shape[0])) for seed in SEEDS]
    records: list[dict[str, object]] = []
    for identity in identities:
        seed = cast(int, identity["seed"])
        result = run_gradual_micro_phase_family(
            ox,
            oy,
            nx,
            ny,
            learner_name="adamw_control",
            seed=seed,
            config=config,
        )
        records.extend(_records(result, cast(str, identity["identity_sha256"])))
    if sources != _sources() or runtime != _runtime():
        raise RuntimeError("campaign source or runtime changed during execution")
    plan = {
        "seeds": list(SEEDS),
        "arms": list(ARM_IDS),
        "run_order": "seed_major_with_arm_major_records_per_seed",
        "config": _config_payload(config),
        "learner": "adamw_control",
        "rng_impl": "threefry2x32",
        "row_alignment": "old_x_equals_new_x_exactly",
        "feature_domain": "finite_float32_closed_interval_-1_1",
        "paper_scale": False,
        "execution_authorized": False,
        "seed_history_audit": (
            "156901--156905 had no exact current-main use when prospectively reserved on "
            "2026-08-20; seeds are immutable after merge"
        ),
    }
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "status": "complete",
        "plan": plan,
        "identity": {"dataset": dataset, "sources": sources, "runtime": runtime},
        "policy": copy.deepcopy(POLICY),
        "decision": copy.deepcopy(_DECISION),
        "seed_identities": identities,
        "records": records,
        "comparisons": _comparisons(records, config),
        "resources": _resources(records, dataset),
        "validation_scope": "strict_five_run_learner_reexecution_and_exact_report_replay",
    }
    payload["result_sha256"] = _sha(payload)
    return payload


def run_gradual_micro_phase_campaign(
    old_x: np.ndarray, old_y: np.ndarray, new_x: np.ndarray, new_y: np.ndarray
) -> dict[str, object]:
    result = _execute(old_x, old_y, new_x, new_y)
    _json_preflight(result)
    return result


def _json_preflight(value: object) -> None:
    seen: set[int] = set()
    nodes = 0
    text_bytes = 0
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > 24:
            raise ValueError("exact JSON work bound exceeded")
        if type(item) is dict:
            if id(item) in seen:
                raise ValueError("exact JSON alias or cycle detected")
            seen.add(id(item))
            mapping = cast(dict[object, object], item)
            if any(type(key) is not str for key in mapping):
                raise ValueError("exact JSON object keys required")
            stack.extend((child, depth + 1) for child in mapping.values())
        elif type(item) is list:
            if id(item) in seen:
                raise ValueError("exact JSON alias or cycle detected")
            seen.add(id(item))
            stack.extend((child, depth + 1) for child in cast(list[object], item))
        elif type(item) is str:
            text_bytes += len(item.encode("utf-8"))
            if text_bytes > _MAX_JSON_BYTES:
                raise ValueError("exact JSON text bound exceeded")
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError("exact JSON floats must be finite")
        elif type(item) not in (int, bool, type(None)):
            raise ValueError("exact JSON primitives required")


def _static_preflight(value: object) -> dict[str, object]:
    _json_preflight(value)
    if type(value) is not dict:
        raise ValueError("campaign must be an exact object")
    root = cast(dict[str, object], value)
    expected = {
        "schema",
        "status",
        "plan",
        "identity",
        "policy",
        "decision",
        "seed_identities",
        "records",
        "comparisons",
        "resources",
        "validation_scope",
        "result_sha256",
    }
    if set(root) != expected or root["schema"] != SCHEMA or root["status"] != "complete":
        raise ValueError("campaign exact JSON fields/schema drifted")
    if root["policy"] != POLICY:
        raise ValueError("campaign policy must remain permanently nonpromoting")
    if root["decision"] != _DECISION:
        raise ValueError("campaign decision must remain inconclusive")
    unsigned = dict(root)
    claimed = unsigned.pop("result_sha256")
    if claimed != _sha(unsigned):
        raise ValueError("campaign result digest drifted")
    records = root["records"]
    if type(records) is not list or len(records) != 15:
        raise ValueError("campaign record roster drifted")
    roster = [(row.get("seed"), row.get("arm")) for row in records if type(row) is dict]
    if roster != [(seed, arm) for seed in SEEDS for arm in ARM_IDS]:
        raise ValueError("campaign record roster drifted")
    return root


def validate_gradual_micro_phase_campaign(
    value: object,
    old_x: np.ndarray,
    old_y: np.ndarray,
    new_x: np.ndarray,
    new_y: np.ndarray,
) -> dict[str, object]:
    root = _static_preflight(value)
    identity = root["identity"]
    if type(identity) is not dict:
        raise ValueError("campaign identity must be an exact object")
    checked = _datasets(old_x, old_y, new_x, new_y)
    current_data = checked[-1]
    if identity.get("dataset") != current_data:
        raise ValueError("campaign dataset identity drifted")
    if identity.get("sources") != _sources():
        raise ValueError("campaign source identity drifted")
    if identity.get("runtime") != _runtime():
        raise ValueError("campaign runtime identity drifted")
    expected_seed_identities = [
        _seed_identity(seed, FROZEN_CONFIG, int(checked[0].shape[0])) for seed in SEEDS
    ]
    if root["seed_identities"] != expected_seed_identities:
        raise ValueError("campaign schedule/initial-state identity drifted")
    expected = _execute(old_x, old_y, new_x, new_y)
    if root != expected:
        raise ValueError("campaign record/resource replay mismatch")
    return expected


def _resign_for_test(value: dict[str, object]) -> None:
    unsigned = dict(value)
    unsigned.pop("result_sha256", None)
    value["result_sha256"] = _sha(unsigned)


def write_gradual_micro_phase_campaign_new(
    path: Path,
    value: object,
    old_x: np.ndarray,
    old_y: np.ndarray,
    new_x: np.ndarray,
    new_y: np.ndarray,
) -> Path:
    validated = validate_gradual_micro_phase_campaign(value, old_x, old_y, new_x, new_y)
    encoded = (
        json.dumps(validated, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True).encode(
            "ascii"
        )
        + b"\n"
    )
    if len(encoded) > _MAX_JSON_BYTES:
        raise ValueError("campaign JSON byte bound exceeded")
    return atomic_write_new(Path(path), encoded)


def load_gradual_micro_phase_campaign(
    path: Path,
    old_x: np.ndarray,
    old_y: np.ndarray,
    new_x: np.ndarray,
    new_y: np.ndarray,
) -> dict[str, object]:
    with Path(path).open("rb") as handle:
        encoded = handle.read(_MAX_JSON_BYTES + 1)
    if len(encoded) > _MAX_JSON_BYTES:
        raise ValueError("campaign JSON byte bound exceeded")
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("campaign must contain exact JSON") from exc
    return validate_gradual_micro_phase_campaign(value, old_x, old_y, new_x, new_y)
