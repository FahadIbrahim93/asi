"""Strict, permanently nonpromoting matched campaign for partial-reset arms.

This module qualifies execution and receipt machinery.  It contains no campaign
outcome and makes no scientific or publication-equivalence claim.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.benchmarks.ipmnist_screening import (
    _materialized_dataset_provenance,
    _partial_reset_peak_numeric_bytes,
    _screening_runtime_environment,
    _validated_ipmnist_data,
    partial_reset_development_record,
    run_screening_config,
    screening_spec,
    validate_partial_reset_development_record,
)
from alberta_framework.benchmarks.upgd_ipmnist import (
    IPMNISTConfig,
    atomic_write_new,
    build_schedule,
    init_mlp_params,
)

SCHEMA = "asi.partial-reset-matched-development.v1"
RNG_CONTRACT = "jax.threefry2x32:root=uint32(seed):split(init,schedule,noise)"
SEEDS = (156301, 156302, 156303, 156304, 156305)
ARM_IDS = (
    "cpr_ipmnist",
    "cpr_hard_reset",
    "cpr_l2_init",
    "cpr_utility_free",
    "cpr_off",
)
MAX_RECORDS = 25
_MAX_JSON_NODES = 100_000
_MAX_JSON_DEPTH = 32
_MAX_JSON_BYTES = 32 * 1024 * 1024
_MAX_DATASET_INPUT_BYTES = 256 * 1024 * 1024
_VALIDATION_SCOPE = (
    "strict_receipt_schedule_initial_state_and_arithmetic_replay_without_learner_reexecution"
)
_POLICY = {
    "development_only": True,
    "scientific_promotion_allowed": False,
    "publication_equivalent": False,
    "outcome_retention_required": True,
}
_CANONICAL_X_SHA256 = "b8078cd833f53d89828a5e28d728517be9add34076f13fe973399f1f16381313"
_CANONICAL_Y_SHA256 = "4f1dd9551f104f8153409e0add59f0a71568f7bad5a5f8e2274480c186fe219a"
_SOURCE_FILES = (
    "alberta_framework/benchmarks/partial_reset_matched_campaign.py",
    "alberta_framework/benchmarks/ipmnist_screening.py",
    "alberta_framework/benchmarks/upgd_ipmnist.py",
    "uv.lock",
)
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _lower_sha(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


@dataclass(frozen=True)
class PartialResetCampaignPlan:
    seeds: tuple[int, ...]
    arm_ids: tuple[str, ...]
    config: IPMNISTConfig
    dataset_x_sha256: str
    dataset_y_sha256: str
    canonical_dataset_required: bool

    def __post_init__(self) -> None:
        if (
            type(self.seeds) is not tuple
            or self.seeds != SEEDS
            or any(type(v) is not int for v in self.seeds)
        ):
            raise ValueError("seeds must be the exact frozen five-seed tuple")
        if (
            type(self.arm_ids) is not tuple
            or self.arm_ids != ARM_IDS
            or any(type(v) is not str for v in self.arm_ids)
        ):
            raise ValueError("arm IDs must be the exact frozen tuple")
        if type(self.config) is not IPMNISTConfig:
            raise ValueError("config must be an exact IPMNISTConfig")
        if not _lower_sha(self.dataset_x_sha256) or not _lower_sha(self.dataset_y_sha256):
            raise ValueError("dataset hashes must be lowercase SHA-256 strings")
        if type(self.canonical_dataset_required) is not bool:
            raise ValueError("canonical_dataset_required must be an exact bool")


FROZEN_PLAN = PartialResetCampaignPlan(
    seeds=SEEDS,
    arm_ids=ARM_IDS,
    config=IPMNISTConfig(),
    dataset_x_sha256=_CANONICAL_X_SHA256,
    dataset_y_sha256=_CANONICAL_Y_SHA256,
    canonical_dataset_required=True,
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")


def _sha_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _tree_sha256(domain: str, tree: object) -> str:
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    digest = hashlib.sha256(domain.encode("ascii") + b"\0" + str(treedef).encode("utf-8") + b"\0")
    for leaf in leaves:
        array = np.asarray(jax.device_get(leaf))
        if array.dtype.hasobject:
            raise ValueError("object learner state cannot be hashed")
        array = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
        header = _canonical_json({"dtype": array.dtype.str, "shape": list(array.shape)})
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(array.nbytes.to_bytes(8, "little"))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _snapshot_data(
    data_x: object, data_y: object, plan: PartialResetCampaignPlan
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if type(data_x) is not np.ndarray or type(data_y) is not np.ndarray:
        raise ValueError("dataset inputs must be exact numpy arrays")
    if (
        data_x.ndim != 2
        or data_y.ndim != 1
        or data_x.shape[0] != data_y.shape[0]
        or data_x.shape[1:] != (plan.config.input_dim,)
    ):
        raise ValueError("dataset shape does not match the campaign plan")
    if plan.canonical_dataset_required and (
        data_x.shape != (60_000, 784) or data_y.shape != (60_000,)
    ):
        raise ValueError("dataset is not the canonical 60,000-row IPMNIST materialization")
    if data_x.nbytes + data_y.nbytes > _MAX_DATASET_INPUT_BYTES:
        raise ValueError("dataset input byte bound exceeded before materialization")
    x, y = _validated_ipmnist_data(
        data_x,
        data_y,
        input_dim=plan.config.input_dim,
        n_classes=plan.config.n_classes,
        min_length=plan.config.task_length,
    )
    x = np.array(x, dtype=np.float32, order="C", copy=True)
    y = np.array(y, dtype=np.int32, order="C", copy=True)
    if np.any(x < -1.0) or np.any(x > 1.0):
        raise ValueError("dataset features must lie in [-1, 1]")
    provenance = _materialized_dataset_provenance(x, y)
    x_hash = cast(dict[str, object], provenance["x"])["sha256"]
    y_hash = cast(dict[str, object], provenance["y"])["sha256"]
    if x_hash != plan.dataset_x_sha256 or y_hash != plan.dataset_y_sha256:
        raise ValueError("dataset does not match the campaign plan")
    return x, y, provenance


def _test_plan(
    *, config: IPMNISTConfig, data_x: np.ndarray, data_y: np.ndarray
) -> PartialResetCampaignPlan:
    provenance = _materialized_dataset_provenance(data_x, data_y)
    return PartialResetCampaignPlan(
        seeds=SEEDS,
        arm_ids=ARM_IDS,
        config=config,
        dataset_x_sha256=cast(str, cast(dict[str, object], provenance["x"])["sha256"]),
        dataset_y_sha256=cast(str, cast(dict[str, object], provenance["y"])["sha256"]),
        canonical_dataset_required=False,
    )


def plan_payload(plan: PartialResetCampaignPlan) -> dict[str, object]:
    cells = len(plan.seeds) * len(plan.arm_ids)
    return {
        "seeds": list(plan.seeds),
        "arm_ids": list(plan.arm_ids),
        "config": plan.config.to_config(),
        "dataset_x_sha256": plan.dataset_x_sha256,
        "dataset_y_sha256": plan.dataset_y_sha256,
        "canonical_dataset_required": plan.canonical_dataset_required,
        "cells": cells,
        "observations_per_cell": plan.config.n_steps,
        "total_observations": cells * plan.config.n_steps,
        "rng_contract": RNG_CONTRACT,
        "policy": copy.deepcopy(_POLICY),
    }


def _source_identity() -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for relative in _SOURCE_FILES:
        data = (_REPO_ROOT / relative).read_bytes()
        result.append(
            {"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        )
    return result


def _schedule_sha256(schedule: object) -> str:
    return _tree_sha256("asi.partial-reset.schedule.v1", schedule)


def _seed_execution_identity(
    seed: int, plan: PartialResetCampaignPlan, *, n_train: int
) -> dict[str, object]:
    if (
        type(seed) is not int
        or seed not in plan.seeds
        or type(n_train) is not int
        or n_train < plan.config.task_length
    ):
        raise ValueError("seed execution identity inputs violate the plan")
    root = jr.key(jnp.uint32(seed), impl="threefry2x32")
    key_init, key_schedule, _ = jr.split(root, 3)
    params = init_mlp_params(key_init, plan.config)
    schedule = build_schedule(key_schedule, plan.config, n_train)
    initial_states = [
        screening_spec(arm).factory(screening_spec(arm).hyperparameters)[0](params)
        for arm in plan.arm_ids
    ]
    state_hashes = [
        _tree_sha256("asi.partial-reset.initial-state.v1", state) for state in initial_states
    ]
    if len(set(state_hashes)) != 1:
        raise RuntimeError("partial-reset arms do not share an exact initial learner state")
    resources = {
        "observations": plan.config.n_steps,
        "updates": plan.config.n_steps,
        "model_queries": 2 * plan.config.n_steps,
        "peak_numeric_bytes": _partial_reset_peak_numeric_bytes(plan.config),
        "schedule_bytes": int(
            np.asarray(schedule.permutations).nbytes + np.asarray(schedule.example_indices).nbytes
        ),
    }
    return {
        "rng_root_sha256": _tree_sha256("asi.partial-reset.rng-root.v1", jr.key_data(root)),
        "schedule_sha256": _schedule_sha256(schedule),
        "initial_parameters_sha256": _tree_sha256(
            "asi.partial-reset.initial-parameters.v1", params
        ),
        "initial_learner_state_sha256": state_hashes[0],
        "resources": resources,
        "resources_sha256": _sha_json(resources),
    }


def _paired(records: list[dict[str, object]]) -> dict[str, object]:
    control = {
        cast(int, r["seed"]): float(
            np.mean(cast(dict[str, Any], r["development_record"])["metrics"]["per_task_accuracy"])
        )
        for r in records
        if r["arm"] == ARM_IDS[-1]
    }
    result: dict[str, object] = {}
    for arm in ARM_IDS[:-1]:
        deltas = [
            float(
                np.mean(
                    cast(dict[str, Any], r["development_record"])["metrics"]["per_task_accuracy"]
                )
            )
            - control[cast(int, r["seed"])]
            for r in records
            if r["arm"] == arm
        ]
        mean = float(np.mean(np.asarray(deltas, dtype=np.float64)))
        sd = float(np.std(np.asarray(deltas, dtype=np.float64), ddof=1))
        half = float(2.7764451051977987 * sd / math.sqrt(5.0))
        result[arm] = {
            "control": ARM_IDS[-1],
            "metric": "mean_online_accuracy",
            "paired_deltas": deltas,
            "mean_delta": mean,
            "sample_sd": sd,
            "ci95": [mean - half, mean + half],
            "n": 5,
        }
    return result


def _root_resources(
    plan: PartialResetCampaignPlan, dataset: dict[str, object]
) -> dict[str, object]:
    cells = len(plan.seeds) * len(plan.arm_ids)
    x = cast(dict[str, object], dataset["x"])
    y = cast(dict[str, object], dataset["y"])
    dataset_bytes = (
        int(np.prod(cast(list[int], x["shape"]))) * np.dtype(np.float32).itemsize
        + int(np.prod(cast(list[int], y["shape"]))) * np.dtype(np.int32).itemsize
    )
    schedule_bytes = (
        plan.config.n_tasks * plan.config.input_dim * np.dtype(np.int32).itemsize
        + plan.config.n_steps * np.dtype(np.int32).itemsize
    )
    return {
        "cells": cells,
        "total_observations": cells * plan.config.n_steps,
        "total_updates": cells * plan.config.n_steps,
        "total_model_queries": 2 * cells * plan.config.n_steps,
        "dataset_numeric_bytes": dataset_bytes,
        "schedule_numeric_bytes_per_cell": schedule_bytes,
        "total_schedule_numeric_bytes_processed": cells * schedule_bytes,
        "per_cell_peak_numeric_bytes": _partial_reset_peak_numeric_bytes(plan.config),
        "retained_metric_values": cells * plan.config.n_tasks * 3,
        "physical_peak_rss_claimed": False,
        "timing_is_selection_metric": False,
        "numeric_scope": (
            "canonical dataset, per-cell schedule/state envelope, processed counters, "
            "and retained metric values; not allocator RSS"
        ),
    }


def build_partial_reset_campaign(data_x: np.ndarray, data_y: np.ndarray) -> dict[str, object]:
    plan = FROZEN_PLAN
    x, y, dataset = _snapshot_data(data_x, data_y, plan)
    sources = _source_identity()
    runtime = _screening_runtime_environment()
    records: list[dict[str, object]] = []
    for seed in plan.seeds:
        identity = _seed_execution_identity(seed, plan, n_train=x.shape[0])
        for arm in plan.arm_ids:
            final: dict[str, str] = {}

            def observer(task: int, params: object, state: object) -> None:
                if task == plan.config.n_tasks - 1:
                    if type(params) is MappingProxyType:
                        params = dict(params)
                    final["parameters"] = _tree_sha256(
                        "asi.partial-reset.final-parameters.v1", params
                    )
                    final["learner_state"] = _tree_sha256("asi.partial-reset.final-state.v1", state)

            result = run_screening_config(
                x, y, screening_spec(arm), seed, plan.config, _task_observer=observer
            )
            record = {
                "seed": seed,
                "arm": arm,
                **copy.deepcopy(identity),
                "final_parameters_sha256": final["parameters"],
                "final_learner_state_sha256": final["learner_state"],
                "development_record": partial_reset_development_record(result),
            }
            records.append(record)
    if sources != _source_identity() or runtime != _screening_runtime_environment():
        raise RuntimeError("source or runtime changed during campaign execution")
    report: dict[str, object] = {
        "schema": SCHEMA,
        "plan": plan_payload(plan),
        "plan_sha256": _sha_json(plan_payload(plan)),
        "source_identity": sources,
        "runtime_identity": runtime,
        "dataset_identity": dataset,
        "records": records,
        "paired_comparisons": _paired(records),
        "resources": _root_resources(plan, dataset),
        "policy": copy.deepcopy(_POLICY),
        "decision": {
            "status": "inconclusive",
            "reason": "no_registered_selection_rule",
            "candidate_selected": None,
        },
        "validation_scope": _VALIDATION_SCOPE,
    }
    return validate_partial_reset_campaign(report, x, y)


def _json_preflight(root: object) -> None:
    seen: set[int] = set()
    nodes = 0
    records_object = root.get("records") if type(root) is dict else None
    text_bytes = 0
    stack: list[tuple[object, int]] = [(root, 0)]
    while stack:
        value, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            raise ValueError("exact JSON node/depth bound exceeded")
        if type(value) in (dict, list):
            ident = id(value)
            if ident in seen:
                raise ValueError("exact JSON aliases and cycles are forbidden")
            seen.add(ident)
            if type(value) is dict:
                mapping = cast(dict[object, object], value)
                if any(type(k) is not str for k in mapping):
                    raise ValueError("exact JSON object keys required")
                stack.extend((v, depth + 1) for v in mapping.values())
            else:
                sequence = cast(list[object], value)
                if len(sequence) > MAX_RECORDS and sequence is records_object:
                    raise ValueError("records bound exceeded")
                stack.extend((v, depth + 1) for v in sequence)
        elif type(value) not in (str, int, float, bool, type(None)):
            raise ValueError("exact JSON primitive required")
        elif type(value) is str:
            text_bytes += len(value.encode("utf-8"))
            if text_bytes > _MAX_JSON_BYTES:
                raise ValueError("exact JSON text bound exceeded")
        elif type(value) is float and not math.isfinite(value):
            raise ValueError("exact JSON numbers must be finite")


def _exact_keys(value: object, keys: set[str], context: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{context} must be an exact object")
    return cast(dict[str, Any], value)


def validate_partial_reset_campaign(
    report: object, data_x: np.ndarray, data_y: np.ndarray
) -> dict[str, object]:
    _json_preflight(report)
    payload = _exact_keys(
        report,
        {
            "schema",
            "plan",
            "plan_sha256",
            "source_identity",
            "runtime_identity",
            "dataset_identity",
            "records",
            "paired_comparisons",
            "resources",
            "policy",
            "decision",
            "validation_scope",
        },
        "campaign",
    )
    plan = FROZEN_PLAN
    expected_plan = plan_payload(plan)
    if (
        payload["schema"] != SCHEMA
        or payload["plan"] != expected_plan
        or payload["plan_sha256"] != _sha_json(expected_plan)
    ):
        raise ValueError("campaign plan does not match the frozen plan")
    if payload["policy"] != _POLICY:
        raise ValueError("campaign nonpromoting policy was changed")
    if payload["decision"] != {
        "status": "inconclusive",
        "reason": "no_registered_selection_rule",
        "candidate_selected": None,
    }:
        raise ValueError("campaign decision must remain inconclusive without a selection rule")
    if payload["validation_scope"] != _VALIDATION_SCOPE:
        raise ValueError("validation scope was changed")
    if payload["source_identity"] != _source_identity():
        raise ValueError("source identity does not match current source bytes")
    if payload["runtime_identity"] != _screening_runtime_environment():
        raise ValueError("runtime identity does not match the current runtime")
    x, _, dataset = _snapshot_data(data_x, data_y, plan)
    if payload["dataset_identity"] != dataset:
        raise ValueError("dataset identity does not match supplied dataset")
    records = payload["records"]
    if type(records) is not list or len(records) != MAX_RECORDS:
        raise ValueError("records roster must contain exactly 25 cells")
    expected_roster = [(seed, arm) for seed in plan.seeds for arm in plan.arm_ids]
    actual_roster: list[tuple[object, object]] = []
    for raw in records:
        record = _exact_keys(
            raw,
            {
                "seed",
                "arm",
                "rng_root_sha256",
                "schedule_sha256",
                "initial_parameters_sha256",
                "initial_learner_state_sha256",
                "resources",
                "resources_sha256",
                "final_parameters_sha256",
                "final_learner_state_sha256",
                "development_record",
            },
            "record",
        )
        actual_roster.append((record["seed"], record["arm"]))
    if actual_roster != expected_roster:
        raise ValueError("records roster is incomplete or out of order")
    identities = {
        seed: _seed_execution_identity(seed, plan, n_train=x.shape[0]) for seed in plan.seeds
    }
    for record in cast(list[dict[str, Any]], records):
        identity = identities[record["seed"]]
        for key in (
            "rng_root_sha256",
            "schedule_sha256",
            "initial_parameters_sha256",
            "initial_learner_state_sha256",
            "resources",
            "resources_sha256",
        ):
            if record[key] != identity[key]:
                raise ValueError(f"record {key} identity/resource mismatch")
        if not _lower_sha(record["final_parameters_sha256"]) or not _lower_sha(
            record["final_learner_state_sha256"]
        ):
            raise ValueError("record final state identity is invalid")
        validated = validate_partial_reset_development_record(record["development_record"])
        if (
            validated["seed"] != record["seed"]
            or validated["arm"] != record["arm"]
            or validated["config"] != plan.config.to_config()
        ):
            raise ValueError("record development receipt does not match its roster cell")
    expected_paired = _paired(cast(list[dict[str, object]], records))
    if payload["paired_comparisons"] != expected_paired:
        raise ValueError("paired comparison mean/arithmetic mismatch")
    if payload["resources"] != _root_resources(plan, dataset):
        raise ValueError("campaign resource arithmetic mismatch")
    return cast(dict[str, object], report)


def write_partial_reset_campaign_new(
    path: Path, report: object, data_x: np.ndarray, data_y: np.ndarray
) -> Path:
    validated = validate_partial_reset_campaign(report, data_x, data_y)
    encoded = (
        json.dumps(validated, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True).encode(
            "ascii"
        )
        + b"\n"
    )
    if len(encoded) > _MAX_JSON_BYTES:
        raise ValueError("campaign JSON byte bound exceeded")
    return atomic_write_new(Path(path), encoded)


def load_partial_reset_campaign(
    path: Path, data_x: np.ndarray, data_y: np.ndarray
) -> dict[str, object]:
    with Path(path).open("rb") as handle:
        encoded = handle.read(_MAX_JSON_BYTES + 1)
    if len(encoded) > _MAX_JSON_BYTES:
        raise ValueError("campaign JSON byte bound exceeded")
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("campaign must contain exact JSON") from exc
    return validate_partial_reset_campaign(value, data_x, data_y)
