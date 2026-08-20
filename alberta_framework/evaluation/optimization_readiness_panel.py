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
from dataclasses import dataclass
from pathlib import Path, PosixPath
from typing import Final, cast

import numpy as np

from alberta_framework.evaluation.optimization_readiness_executor import (
    execute_optimization_readiness,
)

PANEL_SCHEMA: Final[str] = "asi.optimization-readiness.checkpoint-panel.v1"
_MIN_CASES: Final[int] = 6
_MAX_CASES: Final[int] = 6
_MAX_CALLER_BYTES: Final[int] = 512 * 1024 * 1024
_MAX_JSON_NODES: Final[int] = 50_000
_MAX_JSON_STRING_BYTES: Final[int] = 1024 * 1024
_MAX_RESULT_BYTES: Final[int] = 4 * 1024 * 1024
_HORIZONS: Final[tuple[str, ...]] = ("1", "10", "100")
_PREDICTORS: Final[tuple[tuple[str, str], ...]] = (
    ("optimization_readiness", "optimization_readiness"),
    ("gradient_norm", "gradient_norm"),
    ("representation_rank", "representation_energy_rank_0_99"),
    ("curvature_rank", "curvature_energy_rank_0_99"),
    ("parameter_norm", "parameter_norm"),
)
FROZEN_PANEL_ROSTER: Final[tuple[tuple[str, str, int], ...]] = tuple(
    ("ipmnist-linear-readiness", f"checkpoint-{index}", 1_568_001 + index) for index in range(6)
)


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


def run_optimization_readiness_panel(
    cases: object,
) -> dict[str, object]:
    """Execute and aggregate a bounded matched checkpoint panel."""
    roster = _preflight_cases(cases)
    executions = [
        execute_optimization_readiness(
            validation_inputs=case.validation_inputs,
            validation_labels=case.validation_labels,
            checkpoint_parameters=case.checkpoint_parameters,
            seed=case.seed,
            task_id=case.task_id,
            checkpoint_id=case.checkpoint_id,
        )
        for case in roster
    ]
    content_identities: set[str] = set()
    for execution in executions:
        raw_identity = execution.get("identity")
        if type(raw_identity) is not dict:
            raise RuntimeError("model-bound executor returned a malformed identity")
        identity = cast(dict[str, object], raw_identity)
        if set(identity) != {"source_sha256", "runtime", "dataset", "checkpoint"}:
            raise RuntimeError("model-bound executor returned a malformed identity")
        content_identity = hashlib.sha256(
            _canonical_bytes(
                {
                    "dataset": identity["dataset"],
                    "checkpoint": identity["checkpoint"],
                }
            )
        ).hexdigest()
        if content_identity in content_identities:
            raise ValueError("panel dataset/checkpoint content identities must be unique")
        content_identities.add(content_identity)
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
        "case_executions": len(executions),
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
        "timing_measured": False,
    }
    payload: dict[str, object] = {
        "schema": PANEL_SCHEMA,
        "policy": {
            "development_only": True,
            "negative_outcomes_retained": True,
            "scientific_promotion_allowed": False,
        },
        "panel_source_sha256": _panel_source_sha256(),
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
            "execution_authorized": False,
            "seed_history_audit": (
                "1568001--1568006 had zero exact matches on current main when prospectively "
                "frozen on 2026-08-20"
            ),
            "status": outcome,
            "scope": "development_association_not_scientific_evidence",
        },
        "resources": resources,
    }
    payload["result_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return payload


def validate_optimization_readiness_panel(
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
    recomputed = run_optimization_readiness_panel(cases)
    if payload != recomputed:
        raise ValueError("panel artifact does not recompute exactly")
    return recomputed


def retain_optimization_readiness_panel(
    payload: object,
    *,
    cases: object,
    repository_root: Path,
) -> Path:
    """Publish one validated panel atomically without replacing prior bytes."""
    if type(repository_root) is not PosixPath or not repository_root.is_absolute():
        raise ValueError("repository_root must be an exact absolute POSIX Path")
    validated = validate_optimization_readiness_panel(payload, cases=cases)
    encoded = _canonical_bytes(validated)
    if len(encoded) > _MAX_RESULT_BYTES:
        raise ValueError("panel artifact exceeds its encoded byte ceiling")
    digest = cast(str, validated["result_sha256"])
    segments = ("outputs", "optimization_readiness", "development.v1")
    directory_descriptor = os.open(
        repository_root,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        for segment in segments:
            try:
                os.mkdir(segment, mode=0o755, dir_fd=directory_descriptor)
            except FileExistsError:
                pass
            next_descriptor = os.open(
                segment,
                os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_descriptor,
            )
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        name = f"result.{digest}.json"
        temporary_name = f".result.{digest}.tmp"
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
                    raise OSError("panel publication write made no progress")
                offset += written
            os.fsync(descriptor)
        except BaseException:
            os.close(descriptor)
            os.unlink(temporary_name, dir_fd=directory_descriptor)
            raise
        else:
            os.close(descriptor)
        try:
            os.link(
                temporary_name,
                name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        finally:
            os.unlink(temporary_name, dir_fd=directory_descriptor)
        read_descriptor = os.open(
            name,
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
            raise RuntimeError("retained panel bytes changed during publication")
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return repository_root.joinpath(*segments, f"result.{digest}.json")


__all__ = [
    "PANEL_SCHEMA",
    "FROZEN_PANEL_ROSTER",
    "OptimizationReadinessPanelCase",
    "retain_optimization_readiness_panel",
    "run_optimization_readiness_panel",
    "validate_optimization_readiness_panel",
]
