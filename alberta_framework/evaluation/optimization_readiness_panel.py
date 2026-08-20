"""Matched, permanently nonpromoting Optimization Readiness checkpoint panel.

The panel composes the model-bound executor over a bounded roster of real
caller-supplied task/checkpoint arrays.  It derives the association between the
prospective readiness score and subsequent measured gains; it does not promote
the association or turn a development panel into scientific evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path, PosixPath
from typing import Final, cast

import numpy as np

import alberta_framework.evaluation.optimization_readiness_executor as readiness_executor

PANEL_SCHEMA: Final[str] = "asi.optimization-readiness.checkpoint-panel.v1"
PLAN_SCHEMA: Final[str] = "asi.optimization-readiness.checkpoint-panel-plan.v1"
_MIN_CASES: Final[int] = 6
_MAX_CASES: Final[int] = 6
_MAX_CALLER_BYTES: Final[int] = 512 * 1024 * 1024
_MAX_JSON_NODES: Final[int] = 50_000
_MAX_JSON_STRING_BYTES: Final[int] = 1024 * 1024
_MAX_RESULT_BYTES: Final[int] = 4 * 1024 * 1024
REGISTERED_REPOSITORY_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_OUTPUT_SEGMENTS: Final[tuple[str, ...]] = (
    "outputs",
    "optimization_readiness",
    "prospective.v1",
)
_HORIZONS: Final[tuple[str, ...]] = ("1", "10", "100")
_PREDICTORS: Final[tuple[tuple[str, str], ...]] = (
    ("optimization_readiness", "optimization_readiness"),
    ("gradient_strength_mechanism_off", "gradient_strength"),
    ("gradient_norm", "gradient_norm"),
    ("representation_rank", "representation_energy_rank_0_99"),
    ("curvature_rank", "curvature_energy_rank_0_99"),
    ("parameter_norm", "parameter_norm"),
)
FROZEN_PANEL_ROSTER: Final[tuple[tuple[str, str, int], ...]] = tuple(
    ("bounded-linear-regression-v1", f"checkpoint-{index:02d}", 2_684_771_901 + index)
    for index in range(6)
)

def _frozen_panel_plan_payload() -> dict[str, object]:
    return {
        "schema": PLAN_SCHEMA,
        "authorization": {
            "authorization_transition_approved": False,
            "execution_authorized": False,
        },
        "roster": [
            {"task_id": task_id, "checkpoint_id": checkpoint_id, "seed": seed}
            for task_id, checkpoint_id, seed in FROZEN_PANEL_ROSTER
        ],
        "seed_status": {
            "classification": "frozen_exposed_unexecuted_consumed_for_promotion",
            "execution_status": "unexecuted",
            "history_search": (
                "2684771901--2684771906 had no exact match in fetched branch history or "
                "workspace content on 2026-08-20 before this plan was written"
            ),
            "promotion_eligible": False,
        },
        "paper_revision": "arXiv:2605.09044v1",
        "official_code_revision": "none-cited-in-arxiv-v1-as-of-2026-08-20",
        "execution_plan": dict(readiness_executor.FROZEN_EXECUTION_PLAN),
        "source_files": [
            "pyproject.toml",
            "uv.lock",
            "alberta_framework/evaluation/optimization_readiness.py",
            "alberta_framework/evaluation/optimization_readiness_executor.py",
            "alberta_framework/evaluation/optimization_readiness_panel.py",
        ],
        "runtime_contract": {
            "python": ">=3.12",
            "numpy": ">=1.26",
            "rng": "jax_threefry2x32_explicit_root",
            "exact_runtime_and_dependency_identity_bound_at_execution": True,
        },
        "dataset_contract": {
            "inputs": "exact_float64_matrix_10000_by_1_to_64",
            "labels": "exact_float64_vector_10000",
            "same_content_required_across_all_checkpoints": True,
            "checkpoint": "exact_float64_vector_matching_input_axis",
            "caller_labels_are_not_content_identity": True,
        },
        "information_contract": {
            "allowed_boundary_information": ["task_start"],
            "allowed_task_information": [
                "current_validation_inputs",
                "current_validation_labels",
            ],
            "future_boundaries_or_held_out_answers_allowed": False,
        },
        "resource_contract": {
            "aggregate_caller_bytes_ceiling": _MAX_CALLER_BYTES,
            "per_case_live_bytes_ceiling": 256 * 1024 * 1024,
            "per_case_preflight_work_units_ceiling": 500_000_000,
            "encoded_result_bytes_ceiling": _MAX_RESULT_BYTES,
            "timing": "unmeasured_telemetry_only",
        },
        "transaction": {
            "registered_namespace": "outputs/optimization_readiness/prospective.v1",
            "destination": "result.json",
            "reservation_before_case_inspection": True,
            "publication": "create_only_no_replace",
            "strict_reread_and_complete_reexecution": True,
        },
        "scope": "bounded_linear_adapter_not_scr_or_permuted_mnist",
        "paper_parity_gaps": [
            "linear model rather than the paper SCR and P-MNIST architectures",
            "input matrix is the linear representation and is checkpoint-invariant",
            "exact linear squared-loss Hessian is checkpoint-invariant",
            "no eNTK, effective-rank, active-neuron, or P-MNIST gradient-Gram baseline",
            "six checkpoints on one caller-supplied task rather than the paper task panels",
        ],
        "outcome_rule": (
            "supported iff optimization readiness has positive finite Spearman correlation and "
            "strictly exceeds each registered baseline at all 1/10/100-step horizons; ties or "
            "undefined correlations are inconclusive; otherwise rejected"
        ),
        "development_only": True,
        "negative_outcomes_retained": True,
        "scientific_promotion_allowed": False,
    }


@dataclass(frozen=True, slots=True)
class OptimizationReadinessPanelCase:
    """One supplied model checkpoint and its aligned validation task."""

    task_id: str
    checkpoint_id: str
    seed: int
    validation_inputs: np.ndarray
    validation_labels: np.ndarray
    checkpoint_parameters: np.ndarray


def _bounded_text(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be an exact string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must be valid UTF-8") from exc
    if not encoded or len(encoded) > 4_096 or b"\x00" in encoded:
        raise ValueError(f"{name} must be bounded non-empty UTF-8")
    return value


def _preflight_cases(
    cases: object,
) -> tuple[OptimizationReadinessPanelCase, ...]:
    if type(cases) is not tuple or not _MIN_CASES <= tuple.__len__(cases) <= _MAX_CASES:
        raise ValueError("panel must contain the exact six-case prospective roster")
    resolved = cast(tuple[object, ...], cases)
    identities: set[tuple[str, str]] = set()
    seeds: set[int] = set()
    dataset_identity: bytes | None = None
    checkpoint_identities: set[bytes] = set()
    caller_bytes = 0
    for index, value in enumerate(resolved):
        if type(value) is not OptimizationReadinessPanelCase:
            raise ValueError(f"cases[{index}] must be an exact panel case")
        case = value
        task_id = _bounded_text(case.task_id, name=f"cases[{index}].task_id")
        checkpoint_id = _bounded_text(case.checkpoint_id, name=f"cases[{index}].checkpoint_id")
        if type(case.seed) is not int or not 0 <= case.seed <= (1 << 32) - 1:
            raise ValueError(f"cases[{index}].seed must be a bounded exact int")
        identity = (task_id, checkpoint_id)
        if identity in identities:
            raise ValueError("panel task/checkpoint identities must be unique")
        identities.add(identity)
        if case.seed in seeds:
            raise ValueError("panel sampling seeds must be unique")
        seeds.add(case.seed)
        arrays = (
            case.validation_inputs,
            case.validation_labels,
            case.checkpoint_parameters,
        )
        if any(type(array) is not np.ndarray for array in arrays):
            raise ValueError(f"cases[{index}] arrays must be exact numpy.ndarray values")
        caller_bytes += sum(int(array.nbytes) for array in arrays)
        if caller_bytes > _MAX_CALLER_BYTES:
            raise ValueError("panel caller arrays exceed the aggregate byte ceiling")
        observed_dataset = json.dumps(
            {
                "inputs": readiness_executor._array_identity(case.validation_inputs),
                "labels": readiness_executor._array_identity(case.validation_labels),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if dataset_identity is None:
            dataset_identity = observed_dataset
        elif observed_dataset != dataset_identity:
            raise ValueError("panel cases must share one exact matched task dataset")
        checkpoint_identity = json.dumps(
            readiness_executor._array_identity(case.checkpoint_parameters),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if checkpoint_identity in checkpoint_identities:
            raise ValueError("panel checkpoint content identities must be unique")
        checkpoint_identities.add(checkpoint_identity)
    observed_roster = tuple(
        (case.task_id, case.checkpoint_id, case.seed)
        for case in cast(tuple[OptimizationReadinessPanelCase, ...], resolved)
    )
    if observed_roster != FROZEN_PANEL_ROSTER:
        raise ValueError("panel cases do not match the frozen prospective roster")
    return cast(tuple[OptimizationReadinessPanelCase, ...], resolved)


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + 1 + end) / 2.0
        for position in range(start, end):
            result[order[position]] = average
        start = end
    return result


def _spearman(left: list[float], right: list[float]) -> float | None:
    left_rank = _rank(left)
    right_rank = _rank(right)
    left_mean = sum(left_rank) / len(left_rank)
    right_mean = sum(right_rank) / len(right_rank)
    left_delta = [value - left_mean for value in left_rank]
    right_delta = [value - right_mean for value in right_rank]
    left_norm = math.sqrt(sum(value * value for value in left_delta))
    right_norm = math.sqrt(sum(value * value for value in right_delta))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    value = sum(a * b for a, b in zip(left_delta, right_delta, strict=True)) / (
        left_norm * right_norm
    )
    if not math.isfinite(value):
        raise ValueError("panel association must be finite")
    return float(max(-1.0, min(1.0, value)))


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def frozen_optimization_readiness_plan() -> dict[str, object]:
    """Return a detached audit copy without touching data or output state."""
    return cast(dict[str, object], json.loads(_canonical_bytes(_frozen_panel_plan_payload())))


def _plan_sha256() -> str:
    return hashlib.sha256(_canonical_bytes(_frozen_panel_plan_payload())).hexdigest()


def _internal_float(value: object) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise RuntimeError("model-bound executor returned a malformed float")
    return value


def _predictor_float(value: object) -> float:
    if type(value) is int and value >= 0:
        return float(value)
    return _internal_float(value)


def _internal_int(value: object) -> int:
    if type(value) is not int or value < 0:
        raise RuntimeError("model-bound executor returned a malformed counter")
    return value


def _panel_source_sha256() -> str:
    path = Path(__file__).resolve()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 2 * 1024 * 1024:
            raise ValueError("panel source must be a bounded regular file")
        payload = bytearray()
        while len(payload) <= before.st_size:
            chunk = os.read(descriptor, min(64 * 1024, before.st_size + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        after = os.fstat(descriptor)
        if len(payload) != before.st_size or (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns):
            raise ValueError("panel source changed during its bounded read")
    finally:
        os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


def _assert_plain_json(value: object) -> None:
    seen: set[int] = set()
    nodes = 0
    string_bytes = 0
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > 12:
            raise ValueError("panel artifact exceeds its JSON work bound")
        if item is None or type(item) is bool:
            continue
        if type(item) is int:
            if not -(1 << 63) <= item <= (1 << 63) - 1:
                raise ValueError("panel artifact contains an unbounded integer")
            continue
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("panel artifact contains a non-finite float")
            continue
        if type(item) is str:
            string_bytes += len(_bounded_text(item, name="panel string").encode("utf-8"))
            if string_bytes > _MAX_JSON_STRING_BYTES:
                raise ValueError("panel artifact exceeds its string-byte bound")
            continue
        if type(item) is list:
            identity = id(item)
            if identity in seen or list.__len__(item) > 512:
                raise ValueError("panel artifact contains an alias, cycle, or oversized list")
            seen.add(identity)
            stack.extend((child, depth + 1) for child in list.__iter__(item))
            continue
        if type(item) is dict:
            identity = id(item)
            if identity in seen or dict.__len__(item) > 128:
                raise ValueError("panel artifact contains an alias, cycle, or oversized mapping")
            seen.add(identity)
            for key, child in dict.items(item):
                if type(key) is not str:
                    raise ValueError("panel artifact keys must be exact strings")
                stack.append((key, depth + 1))
                stack.append((child, depth + 1))
            continue
        raise ValueError("panel artifact must be exact plain JSON")


def _run_optimization_readiness_panel(
    cases: object,
) -> dict[str, object]:
    """Execute and aggregate a bounded matched checkpoint panel."""
    panel_source_before = _panel_source_sha256()
    roster = _preflight_cases(cases)
    executions = [
        readiness_executor._execute_optimization_readiness(
            validation_inputs=case.validation_inputs,
            validation_labels=case.validation_labels,
            checkpoint_parameters=case.checkpoint_parameters,
            seed=case.seed,
            task_id=case.task_id,
            checkpoint_id=case.checkpoint_id,
        )
        for case in roster
    ]
    dataset_identity: bytes | None = None
    checkpoint_identities: set[bytes] = set()
    execution_context_identity: bytes | None = None
    for execution in executions:
        raw_identity = execution.get("identity")
        if type(raw_identity) is not dict:
            raise RuntimeError("model-bound executor returned a malformed identity")
        identity = cast(dict[str, object], raw_identity)
        if set(identity) != {
            "authorization",
            "source_sha256",
            "runtime",
            "dataset",
            "checkpoint",
        }:
            raise RuntimeError("model-bound executor returned a malformed identity")
        observed_dataset = _canonical_bytes(identity["dataset"])
        if dataset_identity is None:
            dataset_identity = observed_dataset
        elif observed_dataset != dataset_identity:
            raise ValueError("panel cases must share one exact matched task dataset")
        observed_checkpoint = _canonical_bytes(identity["checkpoint"])
        if observed_checkpoint in checkpoint_identities:
            raise ValueError("panel checkpoint content identities must be unique")
        checkpoint_identities.add(observed_checkpoint)
        observed_context = _canonical_bytes(
            {
                "authorization": identity["authorization"],
                "source_sha256": identity["source_sha256"],
                "runtime": identity["runtime"],
            }
        )
        if execution_context_identity is None:
            execution_context_identity = observed_context
        elif observed_context != execution_context_identity:
            raise RuntimeError("panel source, runtime, or authorization changed between cases")
    panel_source_after = _panel_source_sha256()
    if panel_source_after != panel_source_before:
        raise RuntimeError("panel source changed during execution")
    metrics: list[dict[str, object]] = []
    child_resources: list[dict[str, object]] = []
    for execution in executions:
        raw_metrics = execution["metrics"]
        raw_resources = execution["resources"]
        if type(raw_metrics) is not dict or type(raw_resources) is not dict:
            raise RuntimeError("model-bound executor returned malformed internal records")
        metrics.append(cast(dict[str, object], raw_metrics))
        child_resources.append(cast(dict[str, object], raw_resources))
    correlations: dict[str, dict[str, float | None]] = {name: {} for name, _field in _PREDICTORS}
    for horizon in _HORIZONS:
        gains: list[float] = []
        for metric in metrics:
            raw_gains = metric["future_relative_loss_reduction"]
            if type(raw_gains) is not dict:
                raise RuntimeError("model-bound executor returned malformed future gains")
            gains.append(_internal_float(cast(dict[str, object], raw_gains)[horizon]))
        for name, field in _PREDICTORS:
            predictor = [_predictor_float(metric[field]) for metric in metrics]
            correlations[name][horizon] = _spearman(predictor, gains)
    readiness = correlations["optimization_readiness"]
    baselines = {
        name: values for name, values in correlations.items() if name != "optimization_readiness"
    }
    paired_deltas: dict[str, dict[str, float | None]] = {}
    for name, values in baselines.items():
        paired_deltas[name] = {}
        for horizon in _HORIZONS:
            left, right = readiness[horizon], values[horizon]
            paired_deltas[name][horizon] = None if left is None or right is None else left - right
    all_deltas = [delta for values in paired_deltas.values() for delta in values.values()]
    if any(value is None for value in readiness.values()) or any(
        delta is None or delta == 0.0 for delta in all_deltas
    ):
        outcome = "inconclusive"
    elif all(cast(float, value) > 0.0 for value in readiness.values()) and all(
        cast(float, delta) > 0.0 for delta in all_deltas
    ):
        outcome = "supported"
    else:
        outcome = "rejected"
    resources = {
        "primary_case_executions": len(executions),
        "strict_validation_case_executions": len(executions),
        "aggregate_caller_bytes": sum(
            int(array.nbytes)
            for case in roster
            for array in (
                case.validation_inputs,
                case.validation_labels,
                case.checkpoint_parameters,
            )
        ),
        "model_queries": sum(
            _internal_int(resource["model_queries"]) for resource in child_resources
        ),
        "data_steps": sum(_internal_int(resource["data_steps"]) for resource in child_resources),
        "environment_steps": sum(
            _internal_int(resource["environment_steps"]) for resource in child_resources
        ),
        "parameter_updates": sum(
            _internal_int(resource["parameter_updates"]) for resource in child_resources
        ),
        "summed_child_preflight_work_units": sum(
            _internal_int(resource["preflight_work_units"]) for resource in child_resources
        ),
        "summed_child_persistent_bytes": sum(
            _internal_int(resource["persistent_bytes"]) for resource in child_resources
        ),
        "max_child_peak_working_set_bytes": max(
            _internal_int(resource["peak_working_set_bytes"]) for resource in child_resources
        ),
        "retained_transaction_model_queries": 2
        * sum(_internal_int(resource["model_queries"]) for resource in child_resources),
        "retained_transaction_data_steps": 2
        * sum(_internal_int(resource["data_steps"]) for resource in child_resources),
        "retained_transaction_parameter_updates": 2
        * sum(_internal_int(resource["parameter_updates"]) for resource in child_resources),
        "timing_seconds": 0.0,
        "timing_measured": False,
        "timing_is_telemetry_only": True,
    }
    payload: dict[str, object] = {
        "schema": PANEL_SCHEMA,
        "policy": {
            **readiness_executor._authorization_identity(),
            "development_only": True,
            "negative_outcomes_retained": True,
            "scientific_promotion_allowed": False,
        },
        "panel_source_sha256": panel_source_after,
        "plan": frozen_optimization_readiness_plan(),
        "plan_sha256": _plan_sha256(),
        "case_count": len(executions),
        "cases": executions,
        "association": {
            "metric": "spearman_rank_correlation",
            "predictors": [name for name, _field in _PREDICTORS],
            "target": "future_relative_loss_reduction",
            "spearman_by_predictor_and_gain_horizon": correlations,
            "optimization_readiness_minus_baseline": paired_deltas,
            "decision_rule": (
                "supported iff optimization readiness has positive finite correlation and "
                "strictly exceeds every registered baseline at all three horizons; ties or "
                "undefined correlations are inconclusive; otherwise rejected"
            ),
            "status": outcome,
            "scope": "development_association_not_scientific_evidence",
        },
        "resources": resources,
    }
    payload["result_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return payload


def _validate_optimization_readiness_panel(
    payload: object,
    *,
    cases: object,
) -> dict[str, object]:
    """Reexecute every supplied case and reject any panel drift or forgery."""
    if type(payload) is not dict:
        raise ValueError("panel artifact must be an exact dict")
    expected_keys = {
        "schema",
        "policy",
        "panel_source_sha256",
        "plan",
        "plan_sha256",
        "case_count",
        "cases",
        "association",
        "resources",
        "result_sha256",
    }
    if dict.__len__(payload) != len(expected_keys):
        raise ValueError("panel artifact has unexpected keys")
    if any(type(key) is not str or key not in expected_keys for key in dict.keys(payload)):
        raise ValueError("panel artifact has unexpected keys")
    _assert_plain_json(payload)
    recomputed = _run_optimization_readiness_panel(cases)
    if payload != recomputed:
        raise ValueError("panel artifact does not recompute exactly")
    return recomputed


def validate_optimization_readiness_panel(
    payload: object,
    *,
    cases: object,
) -> dict[str, object]:
    """Authorized strict replay of one retained prospective panel."""
    readiness_executor._require_execution_authorized()
    return _validate_optimization_readiness_panel(payload, cases=cases)


def _open_directory_chain(root: Path, segments: tuple[str, ...]) -> int:
    if sys.platform != "linux":
        raise RuntimeError("immutable panel publication requires Linux")
    if type(root) is not PosixPath or not root.is_absolute():
        raise ValueError("repository root must be an exact absolute POSIX Path")
    if type(segments) is not tuple or any(
        type(segment) is not str or not segment or segment in {".", ".."} or "/" in segment
        for segment in segments
    ):
        raise ValueError("output segments must be one exact safe tuple")
    descriptor = os.open(
        root,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW,
    )
    try:
        for segment in segments:
            try:
                os.mkdir(segment, mode=0o755, dir_fd=descriptor)
            except FileExistsError:
                pass
            next_descriptor = os.open(
                segment,
                os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _run_and_retain_optimization_readiness_panel(
    cases: object,
    *,
    repository_root: Path,
) -> Path:
    """Private testable transaction; caller must enforce the public gate and root."""
    directory_descriptor = _open_directory_chain(repository_root, _OUTPUT_SEGMENTS)
    reservation_name = ".result.json.reservation"
    result_name = "result.json"
    temporary_name = ".result.json.tmp"
    reservation_descriptor = os.open(
        reservation_name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o400,
        dir_fd=directory_descriptor,
    )
    marker = b"asi-optimization-readiness-prospective-v1\n"
    published_identity: tuple[int, int] | None = None
    temporary_identity: tuple[int, int] | None = None
    try:
        written = os.write(reservation_descriptor, marker)
        if written != len(marker):
            raise OSError("panel reservation write made no progress")
        os.fsync(reservation_descriptor)
        os.fsync(directory_descriptor)
        try:
            os.stat(result_name, dir_fd=directory_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("the prospective panel result already exists")

        result = _run_optimization_readiness_panel(cases)
        encoded = _canonical_bytes(result)
        if len(encoded) > _MAX_RESULT_BYTES:
            raise ValueError("panel artifact exceeds its encoded byte ceiling")
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o400,
            dir_fd=directory_descriptor,
        )
        try:
            temporary = os.fstat(descriptor)
            temporary_identity = (temporary.st_dev, temporary.st_ino)
            offset = 0
            while offset < len(encoded):
                count = os.write(descriptor, encoded[offset:])
                if count <= 0:
                    raise OSError("panel publication write made no progress")
                offset += count
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o444)
            source = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(
                temporary_name,
                result_name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            published_identity = (source.st_dev, source.st_ino)
        finally:
            try:
                visible_temporary = os.stat(
                    temporary_name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if (visible_temporary.st_dev, visible_temporary.st_ino) == temporary_identity:
                    os.unlink(temporary_name, dir_fd=directory_descriptor)
            except FileNotFoundError:
                pass

        read_descriptor = os.open(
            result_name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW,
            dir_fd=directory_descriptor,
        )
        try:
            before = os.fstat(read_descriptor)
            loaded = bytearray()
            while len(loaded) <= _MAX_RESULT_BYTES:
                chunk = os.read(
                    read_descriptor,
                    min(64 * 1024, _MAX_RESULT_BYTES + 1 - len(loaded)),
                )
                if not chunk:
                    break
                loaded.extend(chunk)
            after = os.fstat(read_descriptor)
        finally:
            os.close(read_descriptor)
        if (
            bytes(loaded) != encoded
            or published_identity != (before.st_dev, before.st_ino)
            or (before.st_size, before.st_mtime_ns)
            != (after.st_size, after.st_mtime_ns)
        ):
            raise RuntimeError("retained panel bytes or inode changed during publication")
        decoded = json.loads(loaded)
        _validate_optimization_readiness_panel(decoded, cases=cases)
        os.fsync(directory_descriptor)
    except BaseException:
        if published_identity is not None:
            try:
                visible = os.stat(result_name, dir_fd=directory_descriptor, follow_symlinks=False)
                if (visible.st_dev, visible.st_ino) == published_identity:
                    os.unlink(result_name, dir_fd=directory_descriptor)
                    os.fsync(directory_descriptor)
            except FileNotFoundError:
                pass
        raise
    finally:
        if temporary_identity is not None:
            try:
                visible_temporary = os.stat(
                    temporary_name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if (visible_temporary.st_dev, visible_temporary.st_ino) == temporary_identity:
                    os.unlink(temporary_name, dir_fd=directory_descriptor)
                    os.fsync(directory_descriptor)
            except FileNotFoundError:
                pass
        owned = os.fstat(reservation_descriptor)
        try:
            visible = os.stat(
                reservation_name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if (visible.st_dev, visible.st_ino) == (owned.st_dev, owned.st_ino):
                os.unlink(reservation_name, dir_fd=directory_descriptor)
                os.fsync(directory_descriptor)
        except FileNotFoundError:
            pass
        os.close(reservation_descriptor)
        os.close(directory_descriptor)
    return repository_root.joinpath(*_OUTPUT_SEGMENTS, result_name)


def run_and_retain_optimization_readiness_panel(cases: object) -> Path:
    """Authorized one-shot runner for the exact registered prospective namespace."""
    readiness_executor._require_execution_authorized()
    if (
        _frozen_panel_plan_payload()["authorization"]
        != readiness_executor._authorization_identity()
    ):
        raise PermissionError("the frozen panel plan has not passed the authorization transition")
    return _run_and_retain_optimization_readiness_panel(
        cases,
        repository_root=REGISTERED_REPOSITORY_ROOT,
    )


__all__ = [
    "PANEL_SCHEMA",
    "FROZEN_PANEL_ROSTER",
    "OptimizationReadinessPanelCase",
    "frozen_optimization_readiness_plan",
    "run_and_retain_optimization_readiness_panel",
    "validate_optimization_readiness_panel",
]
