"""Execute and validate the frozen, permanently nonpromoting BiMU comparison."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Final, cast

import jax.random as jr
import numpy as np

from alberta_framework.benchmarks.bimu import (
    _EXAMPLE_ORDER_DOMAIN,
    _INIT_DOMAIN,
    _PRNG_IMPLEMENTATION,
    _dataset_sha256,
    _initialize_state,
    _state_sha256,
    _stream_key,
    build_task_schedule,
    run_bimu_development,
    validate_bimu_result,
)
from alberta_framework.evaluation.bimu_matched_nonpromoting import (
    FROZEN_BIMU_MATCHED_PLAN,
    BiMUMatchedDevelopmentPlan,
    _json_preflight,
    _plan_payload,
    _runtime_identity,
)
from alberta_framework.evaluation.prospective_publication import publish_prepared_json_at

RESULT_SCHEMA: Final = "asi.bimu.matched-development-result.v1"
_POLICY: Final = {
    "development_only": True,
    "scientific_promotion_allowed": False,
    "sota_claim_allowed": False,
    "negative_results_retained": True,
}
_MATCHED_COUNTERS: Final = (
    "environment_steps",
    "observations",
    "label_queries",
    "optimizer_seen",
    "model_forward_queries",
)
_RESOURCE_FIELDS: Final = (
    "trainable_scalar_count",
    "parameter_numeric_bytes",
    "optimizer_state_numeric_bytes",
    "initial_persistent_numeric_bytes",
    "final_persistent_numeric_bytes",
)
_MAX_DATASET_BYTES: Final = 16 * 1024 * 1024
_MAX_RESULT_BYTES: Final = 16 * 1024 * 1024
EXECUTION_AUTHORIZED: Final = False
AUTHORIZATION_TRANSITION_APPROVED: Final = False


def _require_execution_authorized() -> None:
    if AUTHORIZATION_TRANSITION_APPROVED is not True:
        raise RuntimeError("BiMU matched execution is not independently authorized")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _exact_object(value: object, fields: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{label} must be an exact object")
    resolved = cast(dict[str, object], value)
    if set(resolved) != fields:
        raise ValueError(f"{label} fields drifted")
    return resolved


def _source_identity() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    relative_paths = (
        "alberta_framework/benchmarks/bimu.py",
        "alberta_framework/evaluation/bimu_matched_nonpromoting.py",
        "alberta_framework/evaluation/bimu_matched_runner.py",
        "alberta_framework/evaluation/prospective_publication.py",
    )
    return {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in relative_paths
    }


def _validated_arrays(
    train_x: object,
    train_y: object,
    test_x: object,
    test_y: object,
    *,
    plan: BiMUMatchedDevelopmentPlan,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    config = plan.candidate_config
    expected = (
        (train_x, np.dtype(np.float32), (config.train_examples_per_task, config.input_dim)),
        (train_y, np.dtype(np.int32), (config.train_examples_per_task,)),
        (test_x, np.dtype(np.float32), (config.test_examples_per_task, config.input_dim)),
        (test_y, np.dtype(np.int32), (config.test_examples_per_task,)),
    )
    arrays: list[np.ndarray] = []
    total_bytes = 0
    for index, (value, dtype, shape) in enumerate(expected):
        if type(value) is not np.ndarray or value.dtype != dtype or value.shape != shape:
            raise ValueError(f"dataset array {index} does not match the plan")
        total_bytes += value.size * value.dtype.itemsize
        if total_bytes > _MAX_DATASET_BYTES:
            raise ValueError("dataset arrays exceed the matched runner byte ceiling")
        arrays.append(np.array(value, dtype=dtype, order="C", copy=True))
    if not np.all(np.isfinite(arrays[0])) or not np.all(np.isfinite(arrays[2])):
        raise ValueError("dataset features must be finite")
    if any(
        np.any(labels < 0) or np.any(labels >= config.n_classes)
        for labels in (arrays[1], arrays[3])
    ):
        raise ValueError("dataset labels are outside the plan class range")
    if _dataset_sha256(*arrays) != plan.dataset_sha256:
        raise ValueError("dataset does not match the plan digest")
    return arrays[0], arrays[1], arrays[2], arrays[3]


def _expected_fixed_identities(plan: BiMUMatchedDevelopmentPlan, seed: int) -> tuple[str, str]:
    config = plan.control_config
    if config.query_threshold != 0.0:
        raise ValueError("matched runner requires the frozen zero query threshold")
    root = jr.key(seed, impl=_PRNG_IMPLEMENTATION)
    state = _initialize_state(config, _stream_key(root, _INIT_DOMAIN))
    initial_state_sha256 = _state_sha256(state, optimizer_step=0, optimizer_seen=0)
    schedule_digest = hashlib.sha256()
    for task, permutation in enumerate(build_task_schedule(config, seed=seed)):
        order = np.asarray(
            jr.permutation(
                _stream_key(root, _EXAMPLE_ORDER_DOMAIN, task),
                config.train_examples_per_task,
            ),
            dtype=np.int32,
        )
        schedule_digest.update(
            json.dumps(
                {
                    "task": task,
                    "permutation": list(permutation),
                    "example_order": [int(value) for value in order],
                    "query_decisions": [True] * config.train_examples_per_task,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    return initial_state_sha256, schedule_digest.hexdigest()


def _sum_result_field(rows: list[dict[str, object]], section: str, field: str) -> int:
    total = 0
    for row in rows:
        result = cast(dict[str, object], row["result"])
        values = cast(dict[str, object], result[section])
        total += cast(int, values[field])
    return total


def _aggregate(
    rows: list[dict[str, object]], plan: BiMUMatchedDevelopmentPlan
) -> dict[str, object]:
    late_deltas: list[float] = []
    online_deltas: list[float] = []
    for seed in plan.seeds:
        pair = [row for row in rows if row["seed"] == seed]
        if len(pair) != 2:
            raise ValueError("aggregate does not have one complete pair per seed")
        by_arm = {cast(str, row["arm"]): cast(dict[str, object], row["result"]) for row in pair}
        control_metrics = cast(dict[str, object], by_arm["memory_off"]["metrics"])
        candidate_metrics = cast(dict[str, object], by_arm["bimu"]["metrics"])
        late_deltas.append(
            cast(float, candidate_metrics["paper_late_five_test_accuracy"])
            - cast(float, control_metrics["paper_late_five_test_accuracy"])
        )
        online_deltas.append(
            cast(float, candidate_metrics["asi_whole_stream_online_accuracy"])
            - cast(float, control_metrics["asi_whole_stream_online_accuracy"])
        )
    totals = {field: _sum_result_field(rows, "resources", field) for field in _RESOURCE_FIELDS}
    counter_totals = {
        field: _sum_result_field(rows, "counters", field)
        for field in (*_MATCHED_COUNTERS, "optimizer_updates")
    }
    return {
        "paired_late_five_deltas": late_deltas,
        "paired_late_five_delta_mean": math.fsum(late_deltas) / len(late_deltas),
        "paired_online_deltas": online_deltas,
        "paired_online_delta_mean": math.fsum(online_deltas) / len(online_deltas),
        "resource_totals": totals,
        "counter_totals": counter_totals,
    }


def _execute_bimu_matched_development(
    train_x: object,
    train_y: object,
    test_x: object,
    test_y: object,
    *,
    plan: BiMUMatchedDevelopmentPlan = FROZEN_BIMU_MATCHED_PLAN,
) -> dict[str, object]:
    """Run every matched arm and seed; this never promotes the observed result."""
    if type(plan) is not BiMUMatchedDevelopmentPlan:
        raise ValueError("plan must be an exact BiMUMatchedDevelopmentPlan")
    checked_plan = BiMUMatchedDevelopmentPlan(**plan.__dict__)
    arrays = _validated_arrays(train_x, train_y, test_x, test_y, plan=checked_plan)
    rows: list[dict[str, object]] = []
    configs = (checked_plan.control_config, checked_plan.candidate_config)
    for seed in checked_plan.seeds:
        for arm, config in zip(checked_plan.arm_names, configs, strict=True):
            arm_result = run_bimu_development(*arrays, config=config, seed=seed)
            validate_bimu_result(arm_result)
            rows.append({"seed": seed, "arm": arm, "result": arm_result})
    plan_payload = _plan_payload(checked_plan)
    result: dict[str, object] = {
        "schema": RESULT_SCHEMA,
        "status": "complete",
        "plan": plan_payload,
        "identity": {
            "dataset_sha256": checked_plan.dataset_sha256,
            "plan_sha256": hashlib.sha256(_canonical(plan_payload)).hexdigest(),
            "source_sha256": _source_identity(),
            "runtime": _runtime_identity(),
            "consistency_not_attestation": True,
        },
        "policy": dict(_POLICY),
        "rows": rows,
        "aggregate": _aggregate(rows, checked_plan),
    }
    result["result_sha256"] = hashlib.sha256(_canonical(result)).hexdigest()
    validate_bimu_matched_result(result, *arrays, plan=checked_plan)
    return result


def run_bimu_matched_development(
    train_x: object,
    train_y: object,
    test_x: object,
    test_y: object,
    *,
    plan: BiMUMatchedDevelopmentPlan = FROZEN_BIMU_MATCHED_PLAN,
) -> dict[str, object]:
    """Fail closed until a separately reviewed change authorizes this plan."""
    _require_execution_authorized()
    return _execute_bimu_matched_development(train_x, train_y, test_x, test_y, plan=plan)


def validate_bimu_matched_result(
    value: object,
    train_x: object,
    train_y: object,
    test_x: object,
    test_y: object,
    *,
    plan: BiMUMatchedDevelopmentPlan = FROZEN_BIMU_MATCHED_PLAN,
) -> None:
    """Validate roster, matched axes, derived aggregates, identities, and policy."""
    _json_preflight(value)
    root = _exact_object(
        value,
        {"schema", "status", "plan", "identity", "policy", "rows", "aggregate", "result_sha256"},
        "matched result",
    )
    if root["schema"] != RESULT_SCHEMA or root["status"] != "complete":
        raise ValueError("matched result identity drifted")
    if type(plan) is not BiMUMatchedDevelopmentPlan:
        raise ValueError("plan must be an exact BiMUMatchedDevelopmentPlan")
    checked_plan = BiMUMatchedDevelopmentPlan(**plan.__dict__)
    arrays = _validated_arrays(train_x, train_y, test_x, test_y, plan=checked_plan)
    expected_plan = _plan_payload(checked_plan)
    if root["plan"] != expected_plan:
        raise ValueError("matched result plan drifted")
    expected_identity = {
        "dataset_sha256": checked_plan.dataset_sha256,
        "plan_sha256": hashlib.sha256(_canonical(expected_plan)).hexdigest(),
        "source_sha256": _source_identity(),
        "runtime": _runtime_identity(),
        "consistency_not_attestation": True,
    }
    if root["identity"] != expected_identity:
        raise ValueError("matched result identity drifted")
    if root["policy"] != _POLICY:
        raise ValueError("matched result must remain permanently nonpromoting")
    raw_rows = root["rows"]
    if type(raw_rows) is not list or len(raw_rows) != len(checked_plan.seeds) * 2:
        raise ValueError("matched result roster is incomplete")
    rows = cast(list[dict[str, object]], raw_rows)
    expected_roster = [(seed, arm) for seed in checked_plan.seeds for arm in checked_plan.arm_names]
    observed_roster: list[tuple[object, object]] = []
    for row in rows:
        checked_row = _exact_object(row, {"seed", "arm", "result"}, "matched row")
        observed_roster.append((checked_row["seed"], checked_row["arm"]))
        arm_result = checked_row["result"]
        validate_bimu_result(arm_result)
        resolved_result = cast(dict[str, object], arm_result)
        if resolved_result["seed"] != checked_row["seed"]:
            raise ValueError("matched row seed drifted")
        config = (
            checked_plan.control_config
            if checked_row["arm"] == "memory_off"
            else checked_plan.candidate_config
        )
        if resolved_result["protocol"] != config.to_protocol_payload():
            raise ValueError("matched row protocol drifted")
        if resolved_result["dataset_sha256"] != checked_plan.dataset_sha256:
            raise ValueError("matched row dataset digest drifted")
        # The benchmark validator deliberately treats producer-reported metrics and
        # consistency digests as unauthenticated. A retained matched result has a
        # stronger contract: reproduce the exact seed/arm transaction from the
        # supplied, digest-bound arrays and compare every non-timing field.
        reproduced = run_bimu_development(
            *arrays,
            config=config,
            seed=cast(int, checked_row["seed"]),
        )
        reproduced["timing"] = resolved_result["timing"]
        if resolved_result != reproduced:
            raise ValueError("matched row does not reproduce from the frozen inputs")
    if observed_roster != expected_roster:
        raise ValueError("matched result roster order drifted")
    for offset in range(0, len(rows), 2):
        seed = checked_plan.seeds[offset // 2]
        control = cast(dict[str, object], rows[offset]["result"])
        candidate = cast(dict[str, object], rows[offset + 1]["result"])
        expected_initial, expected_schedule = _expected_fixed_identities(checked_plan, seed)
        if (
            control["initial_state_sha256"] != expected_initial
            or candidate["initial_state_sha256"] != expected_initial
        ):
            raise ValueError("matched pair initial-state identity drifted")
        if (
            control["schedule_sha256"] != expected_schedule
            or candidate["schedule_sha256"] != expected_schedule
        ):
            raise ValueError("matched pair schedule identity drifted")
        for field in ("dataset_sha256", "schedule_sha256", "initial_state_sha256"):
            if control[field] != candidate[field]:
                raise ValueError(f"matched pair {field} drifted")
        control_counters = cast(dict[str, object], control["counters"])
        candidate_counters = cast(dict[str, object], candidate["counters"])
        if any(control_counters[field] != candidate_counters[field] for field in _MATCHED_COUNTERS):
            raise ValueError("matched pair counters drifted")
        config = checked_plan.control_config
        observations = config.n_tasks * config.train_examples_per_task
        expected_counters = {
            "environment_steps": observations,
            "observations": observations,
            "label_queries": observations,
            "optimizer_seen": observations,
            "model_forward_queries": (
                observations * (config.query_samples + config.train_samples)
                + 5 * config.test_examples_per_task * config.test_samples
            ),
        }
        if any(control_counters[field] != value for field, value in expected_counters.items()):
            raise ValueError("matched pair counters do not match the frozen plan")
        control_resources = cast(dict[str, object], control["resources"])
        candidate_resources = cast(dict[str, object], candidate["resources"])
        if any(
            control_resources[field] != candidate_resources[field] for field in _RESOURCE_FIELDS
        ):
            raise ValueError("matched pair resources drifted")
    if root["aggregate"] != _aggregate(rows, checked_plan):
        raise ValueError("matched aggregate drifted")
    unsigned = dict(root)
    claimed = unsigned.pop("result_sha256")
    if type(claimed) is not str or claimed != hashlib.sha256(_canonical(unsigned)).hexdigest():
        raise ValueError("matched result digest drifted")


def write_bimu_matched_result(
    destination: Path,
    value: object,
    train_x: object,
    train_y: object,
    test_x: object,
    test_y: object,
    *,
    plan: BiMUMatchedDevelopmentPlan = FROZEN_BIMU_MATCHED_PLAN,
) -> None:
    """Durably publish one validated result without replacing existing evidence."""
    _require_execution_authorized()
    if type(destination) is not type(Path()):
        raise TypeError("destination must be an exact Path")
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    directory = os.open(destination.parent, flags)
    try:

        def prepare() -> bytes:
            validate_bimu_matched_result(value, train_x, train_y, test_x, test_y, plan=plan)
            return _canonical(value) + b"\n"

        def validate_loaded(loaded: object) -> None:
            validate_bimu_matched_result(loaded, train_x, train_y, test_x, test_y, plan=plan)

        publish_prepared_json_at(
            directory,
            destination.name,
            prepare=prepare,
            validate_loaded=validate_loaded,
            max_bytes=_MAX_RESULT_BYTES,
        )
    finally:
        os.close(directory)
