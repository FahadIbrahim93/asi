"""Strict nonpromoting campaign for the bounded native Dreamer-family lane."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import stat
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, cast

import jax
import jax.random as jr
import numpy as np

from alberta_framework.benchmarks.dreamer_continual_development import (
    ARM_IDS,
    FROZEN_SEEDS,
    FROZEN_TASK_TARGETS,
    PRNG_IMPLEMENTATION,
    ArmResult,
    DevelopmentResult,
    run_development_lane,
)
from alberta_framework.benchmarks.upgd_ipmnist import atomic_write_new
from alberta_framework.core.world_model import (
    ActionConditionedWorldModel,
    ActionConditionedWorldModelConfig,
)

SCHEMA: Final[str] = "asi.dreamer-native.matched-campaign.v1"
SEEDS: Final[tuple[int, ...]] = FROZEN_SEEDS
FROZEN_PLAN: Final = MappingProxyType(
    {"steps_per_task": 4, "replay_capacity": 8, "imaginations_per_step": 2}
)
POLICY: Final = MappingProxyType(
    {
        "development_only": True,
        "scientific_promotion_allowed": False,
        "publication_equivalent": False,
        "sota_claim_allowed": False,
        "negative_outcomes_retained": True,
        "dreamer_parity_claimed": False,
    }
)
_DECISION: Final = MappingProxyType(
    {
        "status": "inconclusive",
        "reason": "no_registered_selection_rule",
        "candidate_selected": None,
    }
)
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_JSON_NODES = 100_000
_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_FILES = (
    "alberta_framework/evaluation/dreamer_continual_campaign.py",
    "alberta_framework/benchmarks/dreamer_continual_development.py",
    "alberta_framework/benchmarks/development_provenance.py",
    "alberta_framework/benchmarks/upgd_ipmnist.py",
    "alberta_framework/core/_float32_scalars.py",
    "alberta_framework/core/behavior_model.py",
    "alberta_framework/core/dreaming.py",
    "alberta_framework/core/initializers.py",
    "alberta_framework/core/learners.py",
    "alberta_framework/core/multi_head_learner.py",
    "alberta_framework/core/normalizers.py",
    "alberta_framework/core/optimizers.py",
    "alberta_framework/core/types.py",
    "alberta_framework/core/world_model.py",
    "alberta_framework/core/update_safety.py",
    "alberta_framework/_seed_validation.py",
    "pyproject.toml",
    "uv.lock",
)
_GAPS = (
    "no_rssm_or_stochastic_categorical_latent_state",
    "no_pixel_encoder_decoder_or_reconstruction_objective",
    "no_learned_actor_critic_lambda_returns_or_multistep_imagination",
    "no_official_continual_dreamer_source_or_checkpoint_parity",
    "no_minigrid_minihack_or_paper_task_schedule",
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


def _tree_sha(domain: str, value: object) -> str:
    leaves, treedef = jax.tree_util.tree_flatten(value)
    digest = hashlib.sha256(domain.encode("ascii") + b"\0" + str(treedef).encode())
    for leaf in leaves:
        array = np.asarray(jax.device_get(leaf))
        array = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
        digest.update(_canonical({"dtype": array.dtype.str, "shape": list(array.shape)}))
        digest.update(memoryview(array).cast("B"))
    return "sha256:" + digest.hexdigest()


def _tree_numeric_bytes(value: object) -> int:
    return sum(
        int(np.asarray(jax.device_get(leaf)).nbytes) for leaf in jax.tree_util.tree_leaves(value)
    )


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
            raise ValueError("campaign source changed during bounded capture")
        return bytes(payload)
    finally:
        os.close(descriptor)


def _sources() -> list[dict[str, object]]:
    result = []
    for relative in _SOURCE_FILES:
        payload = _read_source(_ROOT / relative)
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
        "chex": metadata.version("chex"),
        "jaxtyping": metadata.version("jaxtyping"),
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


def _replay_schedule(seed: int) -> list[dict[str, int]]:
    schedule: list[dict[str, int]] = []
    inserts = 0
    for task_index, _target in enumerate(FROZEN_TASK_TARGETS):
        for step_index in range(FROZEN_PLAN["steps_per_task"]):
            inserts += 1
            replay_length = min(inserts, FROZEN_PLAN["replay_capacity"])
            for imagination_index in range(FROZEN_PLAN["imaginations_per_step"]):
                schedule.append(
                    {
                        "task_index": task_index,
                        "step_index": step_index,
                        "replay_length": replay_length,
                        "imagination_index": imagination_index,
                        "anchor_index": (seed + imagination_index + inserts) % replay_length,
                        "imagined_action": imagination_index % 2,
                    }
                )
    return schedule


def _seed_identity(seed: int) -> dict[str, object]:
    root = jr.key(seed, impl=PRNG_IMPLEMENTATION)
    model = ActionConditionedWorldModel(
        ActionConditionedWorldModelConfig(
            observation_dim=2,
            n_actions=2,
            hidden_sizes=(),
            step_size=0.05,
            sparsity=0.0,
            error_decay=0.0,
        )
    )
    state = model.init(root)
    # MultiHeadMLPState retains host wall-clock lifecycle telemetry. It is not
    # consumed by prediction or learning, so bind the complete execution-relevant
    # numeric state after canonicalizing only those two telemetry scalars.
    learner_state = cast(Any, state.learner_state).replace(
        birth_timestamp=0.0,
        uptime_s=0.0,
    )
    canonical_state = cast(Any, state).replace(learner_state=learner_state)
    replay = _replay_schedule(seed)
    payload: dict[str, object] = {
        "seed": seed,
        "rng_impl": PRNG_IMPLEMENTATION,
        "root_sha256": _tree_sha("asi.dreamer-campaign.root.v1", jr.key_data(root)),
        "initial_world_model_state_sha256": _tree_sha(
            "asi.dreamer-campaign.initial-world-model.v1", canonical_state
        ),
        "initial_world_model_numeric_bytes": _tree_numeric_bytes(canonical_state),
        "initial_state_scope": (
            "all numeric state with host-only birth_timestamp and uptime_s canonicalized to zero"
        ),
        "initial_action_values_sha256": _tree_sha(
            "asi.dreamer-campaign.initial-values.v1", np.zeros((2,), np.float32)
        ),
        "initial_replay_items": 0,
        "replay_schedule_sha256": "sha256:" + hashlib.sha256(_canonical(replay)).hexdigest(),
        "replay_schedule_events": len(replay),
    }
    payload["identity_sha256"] = _sha(payload)
    return payload


def _record(seed: int, arm: ArmResult, seed_identity: str) -> dict[str, object]:
    receipt = asdict(arm.receipt)
    receipt["elapsed_ns"] = 0
    return {
        "seed": seed,
        "arm": arm.arm_id,
        "seed_identity_sha256": seed_identity,
        "task_returns": list(arm.task_returns),
        "final_action_values": list(arm.final_action_values),
        "receipt": receipt,
        "candidate_eligible": arm.candidate_eligible,
        "timing_retained": False,
        "execution_attestation": False,
    }


def _comparisons(records: list[dict[str, object]]) -> dict[str, object]:
    guarded_deltas: list[float] = []
    privileged_gaps: list[float] = []
    for seed in SEEDS:
        by_arm = {row["arm"]: row for row in records if row["seed"] == seed}
        totals = {
            arm: math.fsum(cast(list[float], row["task_returns"])) for arm, row in by_arm.items()
        }
        guarded_deltas.append(totals["guarded_imagination"] - totals["imagination_off"])
        privileged_gaps.append(totals["guarded_imagination"] - totals["privileged_task_control"])

    def summary(values: list[float]) -> dict[str, object]:
        mean = math.fsum(values) / 5
        variance = math.fsum((value - mean) ** 2 for value in values) / 4
        half = 2.7764451051977987 * math.sqrt(variance / 5)
        return {
            "paired_deltas": values,
            "mean_delta": mean,
            "confidence_interval_95": [mean - half, mean + half],
            "interval_method": "two_sided_student_t_df4_descriptive_only",
        }

    return {
        "guarded_vs_imagination_off": {
            "metric": "total_return",
            "candidate": "guarded_imagination",
            "control": "imagination_off",
            **summary(guarded_deltas),
        },
        "guarded_vs_privileged_normalization": {
            "metric": "total_return",
            "candidate": "guarded_imagination",
            "control": "privileged_task_control",
            "candidate_selection_allowed": False,
            **summary(privileged_gaps),
        },
    }


def _resources(records: list[dict[str, object]]) -> dict[str, object]:
    receipts = [cast(dict[str, object], row["receipt"]) for row in records]
    summed_fields = (
        "environment_steps",
        "world_model_updates",
        "world_model_queries",
        "replay_inserts",
        "replay_samples",
        "imagination_proposals",
        "imagination_accepts",
        "imagined_value_updates",
        "logical_compute_units",
    )
    result: dict[str, object] = {
        "runs": len(SEEDS),
        "arm_cells": len(records),
        "summed_persistent_bytes": sum(cast(int, item["persistent_bytes"]) for item in receipts),
        "max_cell_persistent_bytes": max(cast(int, item["persistent_bytes"]) for item in receipts),
        "max_cell_peak_replay_bytes": max(
            cast(int, item["peak_replay_bytes"]) for item in receipts
        ),
        "physical_peak_rss_claimed": False,
        "timing_measured_but_discarded": True,
        "timing_is_selection_metric": False,
    }
    for field in summed_fields:
        result[f"total_{field}"] = sum(cast(int, item[field]) for item in receipts)
    return result


def _plan() -> dict[str, object]:
    return {
        "seeds": list(SEEDS),
        "arms": list(ARM_IDS),
        "run_order": "seed_major_arm_minor",
        "rng_impl": PRNG_IMPLEMENTATION,
        **FROZEN_PLAN,
    }


def _workload() -> dict[str, object]:
    workload: dict[str, object] = {
        "task_targets": list(FROZEN_TASK_TARGETS),
        "transition": {
            "next_observation": "float32(0.8*observation[0]+direction,observation[0])",
            "direction": "action_0:+1,action_1:-1",
            "reward": "+1 iff action equals current target else -1",
            "task_boundary_observed_by_candidates": False,
            "privileged_control_reads_target": True,
        },
        "plan": dict(FROZEN_PLAN),
        "environment_steps_per_arm": len(FROZEN_TASK_TARGETS) * FROZEN_PLAN["steps_per_task"],
    }
    workload["workload_sha256"] = _sha(workload)
    return workload


def _execute() -> dict[str, object]:
    sources = _sources()
    runtime = _runtime()
    seed_identities = [_seed_identity(seed) for seed in SEEDS]
    records: list[dict[str, object]] = []
    for identity in seed_identities:
        seed = cast(int, identity["seed"])
        result: DevelopmentResult = run_development_lane(seed=seed, **FROZEN_PLAN)
        for arm in result.arms:
            records.append(_record(seed, arm, cast(str, identity["identity_sha256"])))
    if sources != _sources() or runtime != _runtime():
        raise RuntimeError("campaign source or runtime changed during execution")
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "status": "complete",
        "plan": _plan(),
        "identity": {"sources": sources, "runtime": runtime, "workload": _workload()},
        "policy": dict(POLICY),
        "decision": dict(_DECISION),
        "scope_gaps": list(_GAPS),
        "seed_identities": seed_identities,
        "records": records,
        "comparisons": _comparisons(records),
        "resources": _resources(records),
        "validation_scope": "strict_five_run_reexecution_and_exact_normalized_report_replay",
    }
    payload["result_sha256"] = _sha(payload)
    return payload


def run_dreamer_continual_campaign() -> dict[str, object]:
    result = _execute()
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
        raise ValueError("campaign Dreamer scope gaps drifted")
    if root["plan"] != _plan():
        raise ValueError("campaign plan/workload schedule drifted")
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


def validate_dreamer_continual_campaign(value: object) -> dict[str, object]:
    root = _static_preflight(value)
    identity = root["identity"]
    if type(identity) is not dict:
        raise ValueError("campaign identity must be an exact object")
    if identity.get("sources") != _sources():
        raise ValueError("campaign source identity drifted")
    if identity.get("runtime") != _runtime():
        raise ValueError("campaign runtime identity drifted")
    if identity.get("workload") != _workload():
        raise ValueError("campaign workload schedule identity drifted")
    expected_seed_identities = [_seed_identity(seed) for seed in SEEDS]
    if root["seed_identities"] != expected_seed_identities:
        raise ValueError("campaign replay/schedule/initial-state identity drifted")
    expected = _execute()
    if root != expected:
        raise ValueError("campaign record/resource replay mismatch")
    return expected


def _resign_for_test(value: dict[str, object]) -> None:
    unsigned = dict(value)
    unsigned.pop("result_sha256", None)
    value["result_sha256"] = _sha(unsigned)


def write_dreamer_continual_campaign_new(path: Path, value: object) -> Path:
    validated = validate_dreamer_continual_campaign(value)
    encoded = (
        json.dumps(validated, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True).encode(
            "ascii"
        )
        + b"\n"
    )
    if len(encoded) > _MAX_JSON_BYTES:
        raise ValueError("campaign JSON byte bound exceeded")
    return atomic_write_new(Path(path), encoded)


def load_dreamer_continual_campaign(path: Path) -> dict[str, object]:
    with Path(path).open("rb") as handle:
        encoded = handle.read(_MAX_JSON_BYTES + 1)
    if len(encoded) > _MAX_JSON_BYTES:
        raise ValueError("campaign JSON byte bound exceeded")
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("campaign must contain exact JSON") from exc
    return validate_dreamer_continual_campaign(value)
