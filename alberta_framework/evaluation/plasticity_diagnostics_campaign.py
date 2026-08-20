"""Strict nonpromoting five-seed campaign for bounded plasticity diagnostics.

The caller supplies the MNIST-shaped arrays and their provenance label.  This
module snapshots those arrays, executes the complete matched roster, and can
strictly replay a report.  It never downloads data or promotes a result.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import platform
import stat
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from typing import Final, cast

import jax
import jax.random as jr
import numpy as np

from alberta_framework.benchmarks import plasticity_diagnostics as diagnostic
from alberta_framework.benchmarks.plasticity_diagnostics import (
    ARM_IDS,
    FROZEN_SEEDS,
    PRNG_IMPLEMENTATION,
    ArmResult,
    DiagnosticResult,
)
from alberta_framework.benchmarks.upgd_ipmnist import atomic_write_new

SCHEMA: Final[str] = "asi.plasticity-diagnostics.matched-campaign.v1"
SEEDS: Final[tuple[int, ...]] = FROZEN_SEEDS
PROFILE_ID: Final[str] = "bounded-development"
POLICY: Final = MappingProxyType({
    "development_only": True,
    "scientific_promotion_allowed": False,
    "publication_equivalent": False,
    "sota_claim_allowed": False,
    "negative_outcomes_retained": True,
})
_DECISION: Final = MappingProxyType({
    "status": "inconclusive",
    "reason": "no_registered_selection_rule",
    "candidate_selected": None,
})
_GAPS: Final = (
    "no_800_task_mnist_protocol",
    "no_three_hidden_layer_width_2000_network",
    "no_official_continual_backprop_trace_or_code_parity",
    "no_continual_imagenet_protocol_or_accelerator_budget",
    "no_continual_rl_mujoco_protocol_or_environment_budget",
    "no_scientific_retention_or_fresh_seed_evaluation",
)
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_JSON_NODES = 100_000
_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_FILES = (
    "alberta_framework/evaluation/plasticity_diagnostics_campaign.py",
    "alberta_framework/benchmarks/plasticity_diagnostics.py",
    "alberta_framework/benchmarks/upgd_ipmnist.py",
    "alberta_framework/_seed_validation.py",
    "alberta_framework/core/_float32_scalars.py",
    "alberta_framework/core/baseline_optimizers.py",
    "alberta_framework/core/canonical_upgd.py",
    "alberta_framework/core/update_safety.py",
    "pyproject.toml",
    "uv.lock",
)
_SOURCE_KEYS = {
    "kind",
    "dataset_name",
    "dataset_version",
    "split",
    "acquisition",
    "artifact_sha256",
}


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


def _tree_sha(domain: str, value: object) -> str:
    leaves, treedef = jax.tree_util.tree_flatten(value)
    digest = hashlib.sha256(domain.encode("ascii") + b"\0" + str(treedef).encode("ascii"))
    for leaf in leaves:
        array = np.asarray(jax.device_get(leaf))
        array = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
        digest.update(_canonical({"dtype": array.dtype.str, "shape": list(array.shape)}))
        digest.update(memoryview(array).cast("B"))
    return "sha256:" + digest.hexdigest()


def _tree_bytes(value: object) -> int:
    return sum(
        int(np.asarray(jax.device_get(leaf)).nbytes)
        for leaf in jax.tree_util.tree_leaves(value)
    )


def _read_source(path: Path) -> bytes:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 4 * 1024 * 1024:
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
            raise ValueError("campaign source changed during bounded capture")
        return bytes(payload)
    finally:
        os.close(descriptor)


def _sources() -> list[dict[str, object]]:
    result = []
    for relative in _SOURCE_FILES:
        payload = _read_source(_ROOT / relative)
        result.append(
            {"path": relative, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
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


def _exact_source(source: object) -> dict[str, str]:
    if type(source) is not dict or set(cast(dict[object, object], source)) != _SOURCE_KEYS:
        raise ValueError("dataset source must be an exact object with exact fields")
    copied: dict[str, str] = {}
    for key, value in cast(dict[object, object], source).items():
        if type(key) is not str or type(value) is not str or not 0 < len(value) <= 512:
            raise ValueError("dataset source values must be bounded exact strings")
        copied[key] = value
    claimed = copied["artifact_sha256"]
    if len(claimed) != 64 or any(char not in "0123456789abcdef" for char in claimed):
        raise ValueError("dataset source artifact_sha256 must be a lowercase SHA-256")
    return copied


def _freeze_dataset(images: object, labels: object) -> tuple[np.ndarray, np.ndarray]:
    checked_images, checked_labels = diagnostic._arrays(images, labels)
    return np.ascontiguousarray(checked_images.copy()), np.ascontiguousarray(checked_labels.copy())


def _dataset_identity(
    images: np.ndarray, labels: np.ndarray, source: dict[str, str]
) -> dict[str, object]:
    identity: dict[str, object] = {
        "materialization": "mnist-flat-float32-unit-int32-labels.v1",
        "images_dtype": images.dtype.str,
        "labels_dtype": labels.dtype.str,
        "images_shape": list(images.shape),
        "labels_shape": list(labels.shape),
        "numeric_bytes": int(images.nbytes + labels.nbytes),
        "dataset_sha256": diagnostic._dataset_sha(images, labels),
        "source": source,
        "source_is_caller_attested_not_authenticated": True,
    }
    identity["identity_sha256"] = _sha(identity)
    return identity


def _profile_identity() -> dict[str, object]:
    payload: dict[str, object] = dataclasses.asdict(diagnostic.PROFILES[PROFILE_ID])
    payload["identity_sha256"] = _sha(payload)
    return payload


def _seed_identity(images: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, object]:
    profile = diagnostic.PROFILES[PROFILE_ID]
    root = jr.key(seed, impl=PRNG_IMPLEMENTATION)
    _, init_key = jr.split(root)
    initial = diagnostic._init_state(init_key, profile.hidden_width)
    schedule = diagnostic._schedule(images, labels, profile, seed)
    payload: dict[str, object] = {
        "seed": seed,
        "rng_impl": PRNG_IMPLEMENTATION,
        "root_sha256": _tree_sha("asi.plasticity-campaign.root.v1", jr.key_data(root)),
        "initial_state_sha256": _tree_sha("asi.plasticity-campaign.initial-state.v1", initial),
        "initial_state_numeric_bytes": _tree_bytes(initial),
        "schedule_sha256": _tree_sha("asi.plasticity-campaign.schedule.v1", schedule),
        "schedule_tasks": profile.n_tasks,
        "schedule_rows": profile.n_tasks * profile.examples_per_task,
        "schedule_semantics": "cumulative_input_permutation_then_seeded_without_replacement_rows",
    }
    payload["identity_sha256"] = _sha(payload)
    return payload


def _record(seed: int, arm: ArmResult, seed_identity: str) -> dict[str, object]:
    receipt = dataclasses.asdict(arm.receipt)
    receipt["elapsed_ns"] = 0
    return {
        "seed": seed,
        "arm": arm.arm_id,
        "seed_identity_sha256": seed_identity,
        "task_accuracy": list(arm.task_accuracy),
        "task_loss": list(arm.task_loss),
        "dead_unit_fraction": list(arm.dead_unit_fraction),
        "effective_rank": list(arm.effective_rank),
        "final_state_sha256": arm.final_state_sha256,
        "receipt": receipt,
        "timing_retained": False,
        "execution_attestation": False,
    }


def _summary(values: list[float]) -> dict[str, object]:
    mean = math.fsum(values) / len(SEEDS)
    variance = math.fsum((value - mean) ** 2 for value in values) / (len(SEEDS) - 1)
    half = 2.7764451051977987 * math.sqrt(variance / len(SEEDS))
    return {
        "paired_deltas": values,
        "mean_delta": mean,
        "confidence_interval_95": [mean - half, mean + half],
        "interval_method": "two_sided_student_t_df4_descriptive_only",
    }


def _comparisons(records: list[dict[str, object]]) -> dict[str, object]:
    bounded_accuracy: list[float] = []
    bounded_loss: list[float] = []
    mechanism_off_accuracy: list[float] = []
    for seed in SEEDS:
        rows = {row["arm"]: row for row in records if row["seed"] == seed}
        control = rows["sgd_control"]
        off = rows["cbp_mechanism_off"]
        candidate = rows["cbp_bounded"]
        control_accuracy = cast(list[float], control["task_accuracy"])
        off_accuracy = cast(list[float], off["task_accuracy"])
        candidate_accuracy = cast(list[float], candidate["task_accuracy"])
        control_loss = cast(list[float], control["task_loss"])
        candidate_loss = cast(list[float], candidate["task_loss"])
        bounded_accuracy.append(
            math.fsum(candidate_accuracy) / len(candidate_accuracy)
            - math.fsum(control_accuracy) / len(control_accuracy)
        )
        mechanism_off_accuracy.append(
            math.fsum(off_accuracy) / len(off_accuracy)
            - math.fsum(control_accuracy) / len(control_accuracy)
        )
        bounded_loss.append(
            math.fsum(candidate_loss) / len(candidate_loss)
            - math.fsum(control_loss) / len(control_loss)
        )
    return {
        "bounded_cbp_vs_sgd_mean_task_accuracy": {
            "metric": "mean_online_preupdate_task_accuracy",
            **_summary(bounded_accuracy),
        },
        "bounded_cbp_vs_sgd_mean_task_loss": {
            "metric": "mean_online_preupdate_cross_entropy_loss",
            **_summary(bounded_loss),
        },
        "mechanism_off_vs_sgd_mean_task_accuracy": {
            "metric": "mean_online_preupdate_task_accuracy",
            "exact_reduction_expected": True,
            **_summary(mechanism_off_accuracy),
        },
    }


def _resources(records: list[dict[str, object]]) -> dict[str, object]:
    receipts = [cast(dict[str, object], row["receipt"]) for row in records]
    total_fields = (
        "data_steps",
        "data_bytes_read",
        "training_model_queries",
        "diagnostic_model_queries",
        "model_queries",
        "parameter_updates",
        "replacements",
        "logical_forward_macs",
        "logical_gradient_macs",
    )
    result: dict[str, object] = {
        "runs": len(SEEDS),
        "arm_cells": len(records),
        "summed_persistent_bytes": sum(
            cast(int, receipt["persistent_bytes"]) for receipt in receipts
        ),
        "max_cell_persistent_bytes": max(
            cast(int, receipt["persistent_bytes"]) for receipt in receipts
        ),
        "physical_peak_rss_claimed": False,
        "timing_measured_but_discarded": True,
        "timing_is_selection_metric": False,
    }
    for field in total_fields:
        result[f"total_{field}"] = sum(cast(int, receipt[field]) for receipt in receipts)
    return result


def _plan() -> dict[str, object]:
    return {
        "seeds": list(SEEDS),
        "arms": list(ARM_IDS),
        "profile_id": PROFILE_ID,
        "run_order": "seed_major_arm_minor",
        "rng_impl": PRNG_IMPLEMENTATION,
    }


def _execute(images: np.ndarray, labels: np.ndarray, source: dict[str, str]) -> dict[str, object]:
    sources = _sources()
    runtime = _runtime()
    dataset = _dataset_identity(images, labels, source)
    seed_identities = [_seed_identity(images, labels, seed) for seed in SEEDS]
    records: list[dict[str, object]] = []
    for seed_identity in seed_identities:
        seed = cast(int, seed_identity["seed"])
        result: DiagnosticResult = diagnostic.run_diagnostic(
            images, labels, seed=seed, profile_id=PROFILE_ID
        )
        if result.dataset_sha256 != dataset["dataset_sha256"]:
            raise RuntimeError("runner dataset identity differs from campaign snapshot")
        for arm in result.arms:
            records.append(_record(seed, arm, cast(str, seed_identity["identity_sha256"])))
    if sources != _sources() or runtime != _runtime():
        raise RuntimeError("campaign source or runtime changed during execution")
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "status": "complete",
        "plan": _plan(),
        "identity": {
            "sources": sources,
            "runtime": runtime,
            "dataset": dataset,
            "profile": _profile_identity(),
        },
        "policy": dict(POLICY),
        "decision": dict(_DECISION),
        "scope_gaps": list(_GAPS),
        "seed_identities": seed_identities,
        "records": records,
        "comparisons": _comparisons(records),
        "resources": _resources(records),
        "validation_scope": "strict_five_seed_reexecution_and_exact_normalized_report_replay",
    }
    payload["result_sha256"] = _sha(payload)
    return payload


def run_plasticity_diagnostics_campaign(
    images: object, labels: object, *, source: object
) -> dict[str, object]:
    data, targets = _freeze_dataset(images, labels)
    provenance = _exact_source(source)
    result = _execute(data, targets, provenance)
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
            text_bytes += sum(len(cast(str, key).encode("utf-8")) for key in mapping)
            if text_bytes > _MAX_JSON_BYTES:
                raise ValueError("exact JSON text bound exceeded")
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
        elif type(item) is int:
            if type(item) is not int or not -(2**63) <= item < 2**63:
                raise ValueError("exact JSON integers must be bounded")
        elif type(item) not in (bool, type(None)):
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
        "scope_gaps",
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
    if root["scope_gaps"] != list(_GAPS):
        raise ValueError("campaign plasticity scope gaps drifted")
    if root["plan"] != _plan():
        raise ValueError("campaign plan/profile schedule drifted")
    unsigned = dict(root)
    claimed = unsigned.pop("result_sha256")
    if claimed != _sha(unsigned):
        raise ValueError("campaign result digest drifted")
    records = root["records"]
    if type(records) is not list or len(records) != len(SEEDS) * len(ARM_IDS):
        raise ValueError("campaign record roster drifted")
    roster = [(row.get("seed"), row.get("arm")) for row in records if type(row) is dict]
    if roster != [(seed, arm) for seed in SEEDS for arm in ARM_IDS]:
        raise ValueError("campaign record roster drifted")
    return root


def validate_plasticity_diagnostics_campaign(
    value: object, images: object, labels: object, *, source: object
) -> dict[str, object]:
    root = _static_preflight(value)
    data, targets = _freeze_dataset(images, labels)
    provenance = _exact_source(source)
    identity = root["identity"]
    if type(identity) is not dict:
        raise ValueError("campaign identity must be an exact object")
    if identity.get("sources") != _sources():
        raise ValueError("campaign source identity drifted")
    if identity.get("runtime") != _runtime():
        raise ValueError("campaign runtime identity drifted")
    if identity.get("dataset") != _dataset_identity(data, targets, provenance):
        raise ValueError("campaign supplied dataset/source identity drifted")
    if identity.get("profile") != _profile_identity():
        raise ValueError("campaign profile identity drifted")
    expected_seed_identities = [_seed_identity(data, targets, seed) for seed in SEEDS]
    if root["seed_identities"] != expected_seed_identities:
        raise ValueError("campaign schedule/initial-state identity drifted")
    expected = _execute(data, targets, provenance)
    if root != expected:
        raise ValueError("campaign record/resource replay mismatch")
    return expected


def _resign_for_test(value: dict[str, object]) -> None:
    unsigned = dict(value)
    unsigned.pop("result_sha256", None)
    value["result_sha256"] = _sha(unsigned)


def write_plasticity_diagnostics_campaign_new(
    path: Path, value: object, images: object, labels: object, *, source: object
) -> Path:
    validated = validate_plasticity_diagnostics_campaign(
        value, images, labels, source=source
    )
    encoded = (
        json.dumps(validated, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True).encode(
            "ascii"
        )
        + b"\n"
    )
    if len(encoded) > _MAX_JSON_BYTES:
        raise ValueError("campaign JSON byte bound exceeded")
    return atomic_write_new(Path(path), encoded)


def load_plasticity_diagnostics_campaign(
    path: Path, images: object, labels: object, *, source: object
) -> dict[str, object]:
    with Path(path).open("rb") as handle:
        encoded = handle.read(_MAX_JSON_BYTES + 1)
    if len(encoded) > _MAX_JSON_BYTES:
        raise ValueError("campaign JSON byte bound exceeded")
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("campaign must contain exact JSON") from exc
    return validate_plasticity_diagnostics_campaign(value, images, labels, source=source)
