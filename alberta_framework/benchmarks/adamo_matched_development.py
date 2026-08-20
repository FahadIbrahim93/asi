"""Prospectively frozen, permanently nonpromoting AdamO matched screen.

The initial merge keeps execution closed. A separate reviewed authorization
change must flip ``_EXECUTION_AUTHORIZED`` before the reserved matrix can run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

from alberta_framework.benchmarks.adamo_diagnostic import (
    ARMS,
    FROZEN_DEVELOPMENT_SEEDS,
    _load_dataset,
    run_adamo_diagnostic,
    validate_adamo_diagnostic,
)
from alberta_framework.benchmarks.ipmnist_screening import (
    _screening_runtime_environment,
    _screening_source_provenance,
)
from alberta_framework.benchmarks.upgd_ipmnist import atomic_write_new_json

SCHEMA: Final[str] = "asi.adamo-matched-development.report.v1"
PLAN_ID: Final[str] = "issue-1560-adamo-bounded-development-v1"
PROFILE: Final[str] = "bounded-development"
SEEDS: Final[tuple[int, ...]] = FROZEN_DEVELOPMENT_SEEDS
CONSUMED_QUALIFICATION_SEEDS: Final[tuple[int, ...]] = (15600, 15601, 15602, 15603)
CONTROL_ARM: Final[str] = "adamw_control"
CANDIDATE_ARMS: Final[tuple[str, ...]] = ("adamo_l1e3", "adam_iso_joint_l1e3")
T95_DF3: Final[float] = 3.1824463052837078
DATASET_FILE_SHA256: Final[str] = (
    "58320c334531afce90c4899ea0c05976c9b9d1c10b7b37e8eb4289cabd0a00ba"
)
DATASET_SEMANTIC_SHA256: Final[str] = (
    "d25060db8f3f3f6ae7b0bb972e848733e15a1158f02021645e86a2923a5ee8a3"
)
DATASET_NUMERIC_BYTES: Final[int] = 219_800_000
_EXECUTION_AUTHORIZED: Final[bool] = False
_HEX: Final[frozenset[str]] = frozenset("0123456789abcdef")
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
OUTPUT_PATH: Final[Path] = _REPO_ROOT / "outputs/adamo_matched_development/report.v1.json"
_MAX_JSON_NODES: Final[int] = 100_000
_MAX_JSON_STRING_BYTES: Final[int] = 2 * 1024 * 1024


def frozen_plan() -> dict[str, object]:
    """Return the literal plan that must be merged before any execution."""
    return {
        "plan_id": PLAN_ID,
        "profile": PROFILE,
        "arms": list(ARMS),
        "control_arm": CONTROL_ARM,
        "candidate_arms": list(CANDIDATE_ARMS),
        "seeds": list(SEEDS),
        "consumed_qualification_seeds": list(CONSUMED_QUALIFICATION_SEEDS),
        "qualification_seed_note": (
            "15600-15603 were exercised by contract qualification and are excluded from "
            "the retained matched matrix"
        ),
        "primary_metric": "mean_online_accuracy",
        "selection_arm": "adamo_l1e3",
        "causal_ablation_arm": "adam_iso_joint_l1e3",
        "hypothesis": (
            "the decoupled AdamO isometry step reduces Jacobian/Gram degradation and yields "
            "a positive paired mean-online-accuracy delta against AdamW"
        ),
        "failure_condition": (
            "the AdamO accuracy interval is not wholly positive, the inert reduction fails, "
            "or any frozen identity/resource invariant fails"
        ),
        "paired_direction": "higher_is_better",
        "confidence_method": "two_sided_student_t",
        "confidence_level": 0.95,
        "confidence_degrees_of_freedom": 3,
        "confidence_critical": T95_DF3,
        "multiple_comparison_policy": (
            "only adamo_l1e3 informs the development disposition; the joint-gradient arm is "
            "a descriptive causal ablation"
        ),
        "matched_axes": [
            "dataset",
            "initialization_root",
            "task_permutations",
            "example_schedule",
            "observations",
            "updates",
            "allowed_learner_information",
        ],
        "allowed_boundary_information": [],
        "allowed_task_information": ["current_example_label"],
        "diagnostic_information": ["post_task_boundary_index", "fixed_input_row_0"],
        "dataset": {
            "file_sha256": DATASET_FILE_SHA256,
            "semantic_sha256": DATASET_SEMANTIC_SHA256,
            "numeric_bytes": DATASET_NUMERIC_BYTES,
            "keys": ["inputs", "labels"],
            "dtypes": ["float32", "int32"],
            "shapes": [[70000, 784], [70000]],
            "materialization": (
                "caller-frozen MNIST cache x/y copied without numeric conversion into "
                "compressed NPZ inputs/labels before plan review"
            ),
        },
        "mechanism_off_reduction": "adamo_inert == adamw_control bit-exact",
        "resource_policy": (
            "observations, updates, data/environment steps, model queries, Jacobian rows, "
            "persistent numeric bytes, peak Gram workspace, logical compute, and telemetry "
            "timing are retained per arm and seed"
        ),
        "output_path": "outputs/adamo_matched_development/report.v1.json",
        "execution_authorized": _EXECUTION_AUTHORIZED,
        "execution_status": (
            "authorized_after_separate_review"
            if _EXECUTION_AUTHORIZED
            else "blocked_pending_independent_plan_audit"
        ),
        "development_only": True,
        "scientific_promotion_allowed": False,
        "outcome_retention_required": True,
    }


def _digest(value: object, *, context: str, length: int = 64) -> str:
    if (
        type(value) is not str
        or len(value) != length
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase hexadecimal digest")
    return value


def _exact_object(
    value: object, expected_keys: frozenset[str], *, context: str
) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{context} must be an exact object")
    mapping = cast(dict[object, object], value)
    raw_keys = tuple(mapping)
    if (
        len(raw_keys) != len(expected_keys)
        or any(type(key) is not str for key in raw_keys)
        or frozenset(cast(tuple[str, ...], raw_keys)) != expected_keys
    ):
        raise ValueError(f"{context} keys do not match the frozen schema")
    return cast(dict[str, Any], mapping)


def _finite_float(value: object, *, context: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{context} must be an exact finite float")
    return value


def _bounded_json(value: object, *, context: str) -> object:
    """Copy exact JSON without invoking subclass hooks or unbounded traversal."""
    budget = [0, 0]

    def visit(item: object, *, depth: int, label: str) -> object:
        budget[0] += 1
        if budget[0] > _MAX_JSON_NODES or depth > 18:
            raise ValueError(f"{context} exceeds the JSON structure bound")
        if item is None or type(item) is bool:
            return item
        if type(item) is int:
            if not -(1 << 63) <= item <= (1 << 63) - 1:
                raise ValueError(f"{label} exceeds signed-int64")
            return item
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError(f"{label} must be finite")
            return item
        if type(item) is str:
            try:
                encoded = item.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ValueError(f"{label} must be valid UTF-8") from error
            if len(encoded) > 16_384 or b"\0" in encoded:
                raise ValueError(f"{label} must be a bounded string")
            budget[1] += len(encoded)
            if budget[1] > _MAX_JSON_STRING_BYTES:
                raise ValueError(f"{context} exceeds the JSON string-byte bound")
            return item
        if type(item) is list:
            if list.__len__(item) > 4096:
                raise ValueError(f"{label} exceeds the list bound")
            return [
                visit(child, depth=depth + 1, label=f"{label}[{index}]")
                for index, child in enumerate(list.__iter__(item))
            ]
        if type(item) is dict:
            if dict.__len__(item) > 4096:
                raise ValueError(f"{label} exceeds the object bound")
            result: dict[str, object] = {}
            for key, child in dict.items(item):
                if type(key) is not str:
                    raise ValueError(f"{label} keys must be exact strings")
                result[key] = visit(child, depth=depth + 1, label=f"{label}.{key}")
            return result
        raise ValueError(f"{label} must contain only exact JSON values")

    return visit(value, depth=0, label=context)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _current_source_provenance() -> dict[str, object]:
    return _screening_source_provenance(_REPO_ROOT)


def _current_runtime_environment() -> dict[str, object]:
    return _screening_runtime_environment()


def _validate_plan(value: object) -> dict[str, object]:
    expected = frozen_plan()
    plan = _exact_object(value, frozenset(expected), context="plan")
    normalized = _bounded_json(plan, context="plan")
    if normalized != expected:
        raise ValueError("report plan does not equal the literal frozen plan")
    return cast(dict[str, object], normalized)


def _arm_by_name(receipt: Mapping[str, object], name: str) -> dict[str, object]:
    arms = receipt["arms"]
    if type(arms) is not list:
        raise ValueError("receipt arms must be an exact list")
    for raw_arm in arms:
        if type(raw_arm) is dict and raw_arm.get("arm") == name:
            return cast(dict[str, object], raw_arm)
    raise ValueError(f"receipt is missing arm {name}")


def _mean_accuracy(receipt: Mapping[str, object], arm: str) -> float:
    values = _arm_by_name(receipt, arm)["per_task_accuracy"]
    if type(values) is not list or not values:
        raise ValueError("per_task_accuracy must be a nonempty exact list")
    converted = tuple(_finite_float(value, context="per_task_accuracy") for value in values)
    return math.fsum(converted) / len(converted)


def _final_diagnostic(receipt: Mapping[str, object], arm: str, key: str) -> float:
    diagnostics = _arm_by_name(receipt, arm)["post_task_diagnostics"]
    if type(diagnostics) is not list or not diagnostics or type(diagnostics[-1]) is not dict:
        raise ValueError("post_task_diagnostics must be a nonempty exact object list")
    return _finite_float(
        cast(dict[str, object], diagnostics[-1])[key], context=f"final {key}"
    )


def _resource_deltas(receipt: Mapping[str, object], candidate: str) -> dict[str, int]:
    control_resources = _arm_by_name(receipt, CONTROL_ARM)["resources"]
    candidate_resources = _arm_by_name(receipt, candidate)["resources"]
    if type(control_resources) is not dict or type(candidate_resources) is not dict:
        raise ValueError("arm resources must be exact objects")
    keys = (
        "observations",
        "updates",
        "data_steps",
        "environment_steps",
        "model_queries",
        "jacobian_reverse_rows",
        "persistent_numeric_bytes",
        "peak_gram_working_bytes",
        "logical_compute_units",
    )
    deltas: dict[str, int] = {}
    for key in keys:
        control = control_resources.get(key)
        value = candidate_resources.get(key)
        if type(control) is not int or type(value) is not int:
            raise ValueError(f"resource {key} must be an exact integer")
        deltas[key] = value - control
    return deltas


def _paired_comparison(
    receipts: Mapping[int, Mapping[str, object]], candidate: str
) -> dict[str, object]:
    accuracy_deltas = tuple(
        _mean_accuracy(receipts[seed], candidate)
        - _mean_accuracy(receipts[seed], CONTROL_ARM)
        for seed in SEEDS
    )
    mean = math.fsum(accuracy_deltas) / len(accuracy_deltas)
    centered = math.fsum((value - mean) ** 2 for value in accuracy_deltas)
    standard_error = math.sqrt(centered / (len(accuracy_deltas) - 1)) / math.sqrt(
        len(accuracy_deltas)
    )
    lower = mean - T95_DF3 * standard_error
    upper = mean + T95_DF3 * standard_error
    outcome = "supported" if lower > 0.0 else "rejected" if upper <= 0.0 else "inconclusive"
    rms_deltas = [
        _final_diagnostic(receipts[seed], candidate, "jacobian_rms_distance_from_one")
        - _final_diagnostic(receipts[seed], CONTROL_ARM, "jacobian_rms_distance_from_one")
        for seed in SEEDS
    ]
    gram_deltas = [
        _final_diagnostic(receipts[seed], candidate, "weight_gram_penalty")
        - _final_diagnostic(receipts[seed], CONTROL_ARM, "weight_gram_penalty")
        for seed in SEEDS
    ]
    resource_deltas = _resource_deltas(receipts[SEEDS[0]], candidate)
    if any(_resource_deltas(receipts[seed], candidate) != resource_deltas for seed in SEEDS[1:]):
        raise ValueError("candidate resource deltas must be seed-invariant")
    return {
        "accuracy_deltas": list(accuracy_deltas),
        "mean_accuracy_delta": mean,
        "ci95_lower": lower,
        "ci95_upper": upper,
        "outcome": outcome,
        "final_jacobian_rms_distance_deltas": rms_deltas,
        "final_weight_gram_penalty_deltas": gram_deltas,
        "resource_deltas_vs_control": resource_deltas,
    }


def build_report(
    receipts: Sequence[dict[str, object]],
    *,
    dataset_file_sha256: str,
    execution_source_commit: str,
) -> dict[str, object]:
    """Build and strictly validate the complete seed-by-arm campaign report."""
    if type(receipts) not in {list, tuple} or len(receipts) != len(SEEDS):
        raise ValueError("receipts must contain the complete frozen seed schedule")
    by_seed: dict[int, dict[str, object]] = {}
    ordered: list[dict[str, object]] = []
    for index, raw_receipt in enumerate(receipts):
        receipt = validate_adamo_diagnostic(raw_receipt)
        seed = receipt["seed"]
        if type(seed) is not int or seed != SEEDS[index] or seed in by_seed:
            raise ValueError("receipts must use deterministic frozen seed ordering")
        if receipt["profile"] != PROFILE:
            raise ValueError("receipt profile does not match the frozen plan")
        by_seed[seed] = receipt
        ordered.append(receipt)
    semantic_digests = {cast(dict[str, object], item["dataset"])["sha256"] for item in ordered}
    if len(semantic_digests) != 1:
        raise ValueError("all receipts must bind one identical semantic dataset")
    if semantic_digests != {DATASET_SEMANTIC_SHA256}:
        raise ValueError("receipt semantic dataset does not match the prospectively frozen input")
    source_receipts = [item["source"] for item in ordered]
    runtime_receipts = [item["runtime"] for item in ordered]
    if any(value != source_receipts[0] for value in source_receipts[1:]):
        raise ValueError("all receipts must bind one source identity")
    if any(value != runtime_receipts[0] for value in runtime_receipts[1:]):
        raise ValueError("all receipts must bind one runtime identity")
    report: dict[str, object] = {
        "schema": SCHEMA,
        "plan": frozen_plan(),
        "dataset_file_sha256": _digest(dataset_file_sha256, context="dataset file sha256"),
        "dataset_semantic_sha256": _digest(
            next(iter(semantic_digests)), context="dataset semantic sha256"
        ),
        "execution_source_commit": _digest(
            execution_source_commit, context="execution source commit", length=40
        ),
        "source_provenance": _current_source_provenance(),
        "runtime_environment": _current_runtime_environment(),
        "runs": ordered,
        "paired_comparisons": {
            candidate: _paired_comparison(by_seed, candidate) for candidate in CANDIDATE_ARMS
        },
        "development_disposition": "inconclusive",
        "policy": {
            "development_only": True,
            "scientific_promotion_allowed": False,
            "outcome_retained": True,
            "timing_is_telemetry_only": True,
        },
    }
    if report["dataset_file_sha256"] != DATASET_FILE_SHA256:
        raise ValueError("dataset file does not match the prospectively frozen input")
    comparisons = cast(dict[str, object], report["paired_comparisons"])
    primary_comparison = cast(dict[str, object], comparisons["adamo_l1e3"])
    report["development_disposition"] = primary_comparison["outcome"]
    return validate_report(report, require_current_source=True)


def validate_report(
    payload: object, *, require_current_source: bool = True
) -> dict[str, object]:
    """Fail closed over completeness, identities, resources, and paired arithmetic."""
    if type(require_current_source) is not bool:
        raise ValueError("require_current_source must be an exact bool")
    report = _exact_object(
        payload,
        frozenset(
            {
                "schema",
                "plan",
                "dataset_file_sha256",
                "dataset_semantic_sha256",
                "execution_source_commit",
                "source_provenance",
                "runtime_environment",
                "runs",
                "paired_comparisons",
                "development_disposition",
                "policy",
            }
        ),
        context="report",
    )
    if report["schema"] != SCHEMA or type(report["schema"]) is not str:
        raise ValueError("report schema does not match the frozen protocol")
    _validate_plan(report["plan"])
    file_digest = _digest(report["dataset_file_sha256"], context="dataset file sha256")
    semantic = _digest(report["dataset_semantic_sha256"], context="dataset semantic sha256")
    if file_digest != DATASET_FILE_SHA256 or semantic != DATASET_SEMANTIC_SHA256:
        raise ValueError("report dataset does not match the prospectively frozen input")
    _digest(report["execution_source_commit"], context="execution source commit", length=40)
    source = _bounded_json(report["source_provenance"], context="source provenance")
    runtime = _bounded_json(report["runtime_environment"], context="runtime environment")
    if type(source) is not dict or type(runtime) is not dict:
        raise ValueError("source and runtime identities must be exact objects")
    if source.get("git_commit") != report["execution_source_commit"]:
        raise ValueError("execution commit does not match source provenance")
    if require_current_source and source != _current_source_provenance():
        raise ValueError("report source provenance does not match current source")
    if require_current_source and runtime != _current_runtime_environment():
        raise ValueError("report runtime environment does not match the current runtime")
    policy = _exact_object(
        report["policy"],
        frozenset(
            {
                "development_only",
                "scientific_promotion_allowed",
                "outcome_retained",
                "timing_is_telemetry_only",
            }
        ),
        context="policy",
    )
    if any(type(value) is not bool for value in policy.values()) or policy != {
        "development_only": True,
        "scientific_promotion_allowed": False,
        "outcome_retained": True,
        "timing_is_telemetry_only": True,
    }:
        raise ValueError("report policy must remain permanently nonpromoting")
    runs = report["runs"]
    if type(runs) is not list or len(runs) != len(SEEDS):
        raise ValueError("runs must contain the complete frozen seed schedule")
    by_seed: dict[int, dict[str, object]] = {}
    for index, raw_run in enumerate(runs):
        run = validate_adamo_diagnostic(raw_run)
        seed = run["seed"]
        if type(seed) is not int or seed != SEEDS[index] or seed in by_seed:
            raise ValueError("runs must use deterministic frozen seed ordering")
        if run["profile"] != PROFILE:
            raise ValueError("run profile does not match the frozen plan")
        dataset = cast(dict[str, object], run["dataset"])
        if dataset["sha256"] != semantic:
            raise ValueError("run semantic dataset identity does not match the aggregate")
        by_seed[seed] = run
    paired = _exact_object(
        report["paired_comparisons"], frozenset(CANDIDATE_ARMS), context="paired comparisons"
    )
    normalized_paired = _bounded_json(paired, context="paired comparisons")
    expected_paired = {
        candidate: _paired_comparison(by_seed, candidate) for candidate in CANDIDATE_ARMS
    }
    if normalized_paired != expected_paired:
        raise ValueError("paired arithmetic does not match the retained runs")
    disposition = report["development_disposition"]
    if (
        type(disposition) is not str
        or disposition != expected_paired["adamo_l1e3"]["outcome"]
    ):
        raise ValueError("development disposition must equal the primary paired outcome")
    return cast(dict[str, object], report)


def publish_report(path: Path, report: object) -> Path:
    """Validate then atomically publish one immutable report generation."""
    if type(path) is not type(Path()):
        raise ValueError("output path must be an exact pathlib.Path")
    if path.absolute() != OUTPUT_PATH.absolute():
        raise ValueError(f"output path must be the reserved NEW path {OUTPUT_PATH}")
    validated = validate_report(report, require_current_source=True)
    return atomic_write_new_json(path, validated)


def _execution_commit() -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return _digest(completed.stdout.strip(), context="execution source commit", length=40)


def run_campaign(dataset_path: Path, output_path: Path = OUTPUT_PATH) -> dict[str, object]:
    """Execute exactly once after the separately reviewed authorization change."""
    if not _EXECUTION_AUTHORIZED:
        raise RuntimeError("AdamO matched-development execution is not authorized")
    if type(dataset_path) is not type(Path()) or type(output_path) is not type(Path()):
        raise ValueError("dataset and output paths must be exact pathlib.Path values")
    if output_path.absolute() != OUTPUT_PATH.absolute():
        raise ValueError(f"output path must be the reserved NEW path {OUTPUT_PATH}")
    if _sha256_file(dataset_path) != DATASET_FILE_SHA256:
        raise ValueError("dataset file does not match the prospectively frozen input")
    source_before = _current_source_provenance()
    runtime_before = _current_runtime_environment()
    inputs, labels = _load_dataset(dataset_path)
    receipts = [
        run_adamo_diagnostic(inputs, labels, profile=PROFILE, seed=seed) for seed in SEEDS
    ]
    if source_before != _current_source_provenance():
        raise RuntimeError("source identity changed during matched execution")
    if runtime_before != _current_runtime_environment():
        raise RuntimeError("runtime identity changed during matched execution")
    report = build_report(
        receipts,
        dataset_file_sha256=_sha256_file(dataset_path),
        execution_source_commit=_execution_commit(),
    )
    publish_report(output_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", action="store_true")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)
    if args.catalog:
        if args.dataset is not None:
            parser.error("--catalog and --dataset are mutually exclusive")
        print(json.dumps(frozen_plan(), sort_keys=True))
        return 0
    if args.dataset is None:
        parser.error("--dataset is required unless --catalog is used")
    run_campaign(args.dataset, args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
