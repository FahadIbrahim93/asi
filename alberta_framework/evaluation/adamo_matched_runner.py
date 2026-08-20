"""Frozen, append-only AdamO IPMNIST matched development campaign."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from alberta_framework.benchmarks.adamo_diagnostic import (
    ARMS,
    COMPARISON_ID,
    FROZEN_MATCHED_DEVELOPMENT_SEEDS,
    OFFICIAL_CODE,
    OFFICIAL_CODE_SEARCH_DATE,
    PAPER_URL,
    PROFILES,
    run_adamo_diagnostic,
    validate_adamo_diagnostic,
)
from alberta_framework.benchmarks.ipmnist_screening import (
    _screening_dataset_provenance,
    _screening_runtime_environment,
    _screening_source_provenance,
    _validated_dataset_provenance,
    _validated_runtime_environment,
    _validated_source_provenance,
)
from alberta_framework.benchmarks.upgd_ipmnist import (
    default_openml_data_home,
    load_mnist_train,
)
from alberta_framework.core.adamo import ADAMO_PAPER_REVISION

SCHEMA = "asi.adamo-ipmnist.matched-development-report.v1"
PLAN_ID = "asi.adamo-ipmnist.bounded-screen.v1"
PROFILE = "bounded-development"
SEEDS = FROZEN_MATCHED_DEVELOPMENT_SEEDS
ACTIVE_ARMS = ("adamo_l1e3", "adam_iso_joint_l1e3")
_T95_DF3 = 3.182446305284263
_MAX_REPORT_BYTES = 32 * 1024 * 1024
_PATH_TYPE = type(Path())
_REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = _REPO_ROOT / "outputs/adamo_matched_development/report.v1.json"


def frozen_plan() -> dict[str, object]:
    """Return the literal plan committed before any campaign execution."""
    config = PROFILES[PROFILE].config
    return {
        "plan_id": PLAN_ID,
        "paper_revision": ADAMO_PAPER_REVISION,
        "paper_url": PAPER_URL,
        "official_code": OFFICIAL_CODE,
        "official_code_search_date": OFFICIAL_CODE_SEARCH_DATE,
        "official_parity_status": "blocked_no_author_maintained_code_located",
        "comparison_id": COMPARISON_ID,
        "profile": PROFILE,
        "config": config.to_config(),
        "arms": list(ARMS),
        "active_comparison_arms": list(ACTIVE_ARMS),
        "seeds": list(SEEDS),
        "primary_metric": "mean_online_accuracy",
        "secondary_metrics": [
            "mean_online_plasticity",
            "final_jacobian_rms_distance_from_one",
            "final_weight_gram_penalty",
        ],
        "paired_direction": "positive_favors_candidate",
        "confidence_method": "two_sided_student_t",
        "confidence_level": 0.95,
        "confidence_degrees_of_freedom": 3,
        "confidence_critical": _T95_DF3,
        "multiplicity_policy": (
            "no multiplicity adjustment; this bounded screen is exploratory and nonpromoting"
        ),
        "matched_axes": [
            "OpenML_mnist_784_v1_rows_0_60000",
            "materialized_dataset",
            "initialization_root",
            "task_permutations",
            "example_schedule",
            "observations",
            "updates",
            "seed",
        ],
        "allowed_boundary_information": [],
        "allowed_task_information": ["current_example_label"],
        "diagnostic_information": ["post_task_boundary_index", "fixed_input_row_0"],
        "mechanism_off_reduction": "adamo_inert == adamw_control bit-exact",
        "paper_protocol_differences": [
            "IPMNIST adaptation, not reproduction of a paper task",
            "784-300-150-10 ReLU MLP instead of the paper depth-4 width-512 MLP",
            "existing protocol initialization instead of paper orthogonal initialization",
            "eight tasks of 64 updates instead of a paper training horizon",
            "no convolutional, RL, transformer, GroupSort, Newton-Schulz, NTK, or rank protocol",
        ],
        "execution_authorized": True,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "outcome_retention_required": True,
    }


def _object(value: object, keys: frozenset[str], *, context: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{context} must be an exact object")
    result = cast(dict[object, object], value)
    if (
        len(result) != len(keys)
        or any(type(key) is not str for key in result)
        or frozenset(result) != keys
    ):
        raise ValueError(f"{context} fields do not match the frozen schema")
    return cast(dict[str, Any], result)


def _finite(value: object, *, context: str, nonnegative: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{context} must be an exact finite float")
    if nonnegative and value < 0.0:
        raise ValueError(f"{context} must be nonnegative")
    return value


def _validated_plan(value: object) -> dict[str, object]:
    plan = _object(value, frozenset(frozen_plan()), context="plan")
    if plan != frozen_plan():
        raise ValueError("report plan does not match the literal frozen plan")
    return cast(dict[str, object], plan)


def _mean(values: object, *, context: str) -> float:
    if type(values) is not list or len(values) != PROFILES[PROFILE].config.n_tasks:
        raise ValueError(f"{context} must cover every task")
    checked = [_finite(value, context=context, nonnegative=True) for value in values]
    return float(np.mean(np.asarray(checked, dtype=np.float64)))


def _record(receipt: Mapping[str, object], arm: Mapping[str, object]) -> dict[str, object]:
    snapshots = arm["post_task_diagnostics"]
    if type(snapshots) is not list or not snapshots or type(snapshots[-1]) is not dict:
        raise ValueError("validated receipt has no final diagnostic")
    final = cast(dict[str, object], snapshots[-1])
    return {
        "seed": receipt["seed"],
        "arm": arm["arm"],
        "mean_online_accuracy": _mean(
            arm["per_task_accuracy"], context="per_task_accuracy"
        ),
        "mean_online_loss": _mean(arm["per_task_loss"], context="per_task_loss"),
        "mean_online_plasticity": _mean(
            arm["per_task_plasticity"], context="per_task_plasticity"
        ),
        "final_jacobian_rms_distance_from_one": _finite(
            final["jacobian_rms_distance_from_one"],
            context="final_jacobian_rms_distance_from_one",
            nonnegative=True,
        ),
        "final_jacobian_condition_number_clipped_1e12": _finite(
            final["jacobian_condition_number_clipped_1e12"],
            context="final_jacobian_condition_number_clipped_1e12",
            nonnegative=True,
        ),
        "final_weight_gram_penalty": _finite(
            final["weight_gram_penalty"],
            context="final_weight_gram_penalty",
            nonnegative=True,
        ),
    }


def _paired_interval(deltas: tuple[float, ...]) -> tuple[float, float, float, str]:
    values = np.asarray(deltas, dtype=np.float64)
    mean = float(values.mean())
    stderr = float(values.std(ddof=1) / math.sqrt(len(values)))
    lower = mean - _T95_DF3 * stderr
    upper = mean + _T95_DF3 * stderr
    outcome = (
        "development_positive"
        if lower > 0.0
        else "development_negative"
        if upper <= 0.0
        else "inconclusive"
    )
    return mean, lower, upper, outcome


def _derive(receipts: Sequence[dict[str, object]]) -> tuple[
    list[dict[str, object]], dict[str, dict[str, object]], dict[str, dict[str, object]]
]:
    by_seed: dict[int, dict[str, dict[str, object]]] = {}
    records: list[dict[str, object]] = []
    resource_rows: dict[str, list[dict[str, object]]] = {arm: [] for arm in ARMS}
    for raw_receipt in receipts:
        receipt = validate_adamo_diagnostic(raw_receipt)
        seed = receipt["seed"]
        if type(seed) is not int or seed not in SEEDS or seed in by_seed:
            raise ValueError("receipts must have unique frozen campaign seeds")
        if receipt["profile"] != PROFILE or receipt["frozen_development_seeds"] != list(SEEDS):
            raise ValueError("receipt profile or seed schedule differs from the campaign")
        raw_arms = receipt["arms"]
        if type(raw_arms) is not list:
            raise ValueError("receipt arms must be an exact list")
        indexed: dict[str, dict[str, object]] = {}
        for expected_arm, raw_arm in zip(ARMS, raw_arms, strict=True):
            if type(raw_arm) is not dict or raw_arm.get("arm") != expected_arm:
                raise ValueError("receipt arm identity differs from the campaign")
            arm_payload = cast(dict[str, object], raw_arm)
            record = _record(receipt, arm_payload)
            records.append(record)
            indexed[expected_arm] = record
            resources = arm_payload["resources"]
            if type(resources) is not dict:
                raise ValueError("validated receipt resources must be an exact object")
            resource_rows[expected_arm].append(cast(dict[str, object], resources))
        by_seed[seed] = indexed
    if set(by_seed) != set(SEEDS):
        raise ValueError("receipts must contain the complete frozen seed schedule")
    records.sort(key=lambda item: (cast(int, item["seed"]), cast(str, item["arm"])))

    paired: dict[str, dict[str, object]] = {}
    for arm_name in ACTIVE_ARMS:
        accuracy_deltas = tuple(
            cast(float, by_seed[seed][arm_name]["mean_online_accuracy"])
            - cast(float, by_seed[seed]["adamw_control"]["mean_online_accuracy"])
            for seed in SEEDS
        )
        mean, lower, upper, outcome = _paired_interval(accuracy_deltas)
        paired[arm_name] = {
            "mean_online_accuracy_deltas": list(accuracy_deltas),
            "mean_online_accuracy_delta": mean,
            "ci95_lower": lower,
            "ci95_upper": upper,
            "primary_outcome": outcome,
            "mean_online_plasticity_deltas": [
                cast(float, by_seed[seed][arm_name]["mean_online_plasticity"])
                - cast(float, by_seed[seed]["adamw_control"]["mean_online_plasticity"])
                for seed in SEEDS
            ],
            "final_jacobian_rms_improvements": [
                cast(float, by_seed[seed]["adamw_control"][
                    "final_jacobian_rms_distance_from_one"
                ])
                - cast(float, by_seed[seed][arm_name]["final_jacobian_rms_distance_from_one"])
                for seed in SEEDS
            ],
            "final_weight_gram_penalty_improvements": [
                cast(float, by_seed[seed]["adamw_control"]["final_weight_gram_penalty"])
                - cast(float, by_seed[seed][arm_name]["final_weight_gram_penalty"])
                for seed in SEEDS
            ],
        }

    totals: dict[str, dict[str, object]] = {}
    summed = (
        "observations",
        "updates",
        "data_steps",
        "environment_steps",
        "model_queries",
        "jacobian_reverse_rows",
        "logical_compute_units",
        "timing_seconds",
    )
    for arm_name in ARMS:
        rows = resource_rows[arm_name]
        totals[arm_name] = {
            "runs": len(rows),
            **{name: sum(cast(int | float, row[name]) for row in rows) for name in summed},
            "parameter_count": rows[0]["parameter_count"],
            "max_persistent_numeric_bytes": max(
                cast(int, row["persistent_numeric_bytes"]) for row in rows
            ),
            "max_peak_gram_working_bytes": max(
                cast(int, row["peak_gram_working_bytes"]) for row in rows
            ),
            "timing_is_telemetry_only": True,
        }
    return records, paired, totals


def build_report(
    receipts: Sequence[dict[str, object]],
    *,
    source_provenance: dict[str, object],
    dataset_provenance: dict[str, object],
    environment: dict[str, object],
) -> dict[str, object]:
    """Build one report from the complete validated campaign matrix."""
    if type(receipts) not in (list, tuple) or len(receipts) != len(SEEDS):
        raise ValueError("receipts must contain the complete frozen campaign")
    normalized_receipts = [validate_adamo_diagnostic(receipt) for receipt in receipts]
    records, paired, totals = _derive(normalized_receipts)
    report: dict[str, object] = {
        "schema": SCHEMA,
        "plan": frozen_plan(),
        "source_provenance": source_provenance,
        "dataset_provenance": dataset_provenance,
        "environment": environment,
        "receipts": normalized_receipts,
        "records": records,
        "paired_comparisons": paired,
        "resource_totals": totals,
        "policy": {
            "outcome_retained": True,
            "negative_outcomes_retained": True,
            "development_only": True,
            "scientific_promotion_allowed": False,
            "official_parity_claimed": False,
            "timing_used_for_selection": False,
        },
    }
    return validate_report(report)


def validate_report(
    payload: object, *, require_current_source: bool = True
) -> dict[str, object]:
    """Fail closed on forged, drifted, incomplete, or promoting reports."""
    report = _object(
        payload,
        frozenset(
            {
                "schema",
                "plan",
                "source_provenance",
                "dataset_provenance",
                "environment",
                "receipts",
                "records",
                "paired_comparisons",
                "resource_totals",
                "policy",
            }
        ),
        context="report",
    )
    if type(report["schema"]) is not str or report["schema"] != SCHEMA:
        raise ValueError("report schema does not match the frozen campaign")
    plan = _validated_plan(report["plan"])
    source = _validated_source_provenance(report["source_provenance"], context="AdamO")
    dataset = _validated_dataset_provenance(report["dataset_provenance"], context="AdamO")
    runtime = _validated_runtime_environment(report["environment"], context="AdamO")
    if require_current_source:
        if source != _screening_source_provenance():
            raise ValueError("report source provenance differs from the current source")
        if runtime != _screening_runtime_environment():
            raise ValueError("report runtime differs from the current runtime")
    receipts = report["receipts"]
    if type(receipts) is not list or len(receipts) != len(SEEDS):
        raise ValueError("report receipts must contain every frozen seed")
    normalized_receipts = [validate_adamo_diagnostic(receipt) for receipt in receipts]
    first_dataset: object | None = None
    first_runtime: object | None = None
    for receipt in normalized_receipts:
        receipt_dataset = receipt["dataset"]
        receipt_runtime = receipt["runtime"]
        if first_dataset is None:
            first_dataset = receipt_dataset
            first_runtime = receipt_runtime
        elif receipt_dataset != first_dataset or receipt_runtime != first_runtime:
            raise ValueError("diagnostic receipts do not share one dataset and runtime")
        receipt_rows = cast(dict[str, object], receipt_dataset)["rows"]
        dataset_shape = cast(list[int], cast(dict[str, object], dataset["x"])["shape"])
        if receipt_rows != dataset_shape[0]:
            raise ValueError("diagnostic receipt rows do not match dataset provenance")
        if cast(dict[str, object], receipt_dataset)["x_sha256"] != cast(
            dict[str, object], dataset["x"]
        )["sha256"] or cast(dict[str, object], receipt_dataset)["y_sha256"] != cast(
            dict[str, object], dataset["y"]
        )["sha256"]:
            raise ValueError("diagnostic receipt hashes do not match dataset provenance")
    expected_records, expected_paired, expected_totals = _derive(normalized_receipts)
    if report["records"] != expected_records:
        raise ValueError("report records do not reconstruct from the diagnostic receipts")
    if report["paired_comparisons"] != expected_paired:
        raise ValueError("paired comparisons do not reconstruct from the records")
    if report["resource_totals"] != expected_totals:
        raise ValueError("resource totals do not reconstruct from the receipts")
    expected_policy = {
        "outcome_retained": True,
        "negative_outcomes_retained": True,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "official_parity_claimed": False,
        "timing_used_for_selection": False,
    }
    if type(report["policy"]) is not dict or report["policy"] != expected_policy:
        raise ValueError("report policy violates the permanent nonpromotion contract")
    return {
        **dict(report),
        "plan": plan,
        "source_provenance": source,
        "dataset_provenance": dataset,
        "environment": runtime,
        "receipts": normalized_receipts,
    }


def run(*, data_home: Path) -> dict[str, object]:
    """Execute the frozen campaign against canonical OpenML MNIST."""
    if type(data_home) is not _PATH_TYPE:
        raise ValueError("data_home must be an exact Path")
    source = _screening_source_provenance()
    runtime = _screening_runtime_environment()
    data_x, data_y = load_mnist_train(data_home)
    dataset = _screening_dataset_provenance(data_x, data_y)
    receipts = [
        run_adamo_diagnostic(data_x, data_y, profile=PROFILE, seed=seed) for seed in SEEDS
    ]
    if source != _screening_source_provenance():
        raise RuntimeError("screening source changed during the AdamO campaign")
    if runtime != _screening_runtime_environment():
        raise RuntimeError("runtime changed during the AdamO campaign")
    if dataset != _screening_dataset_provenance(data_x, data_y):
        raise RuntimeError("dataset identity changed during the AdamO campaign")
    return build_report(
        receipts,
        source_provenance=source,
        dataset_provenance=dataset,
        environment=runtime,
    )


def _open_output_transaction() -> tuple[int, int, str]:
    """Reserve the immutable output through pinned, non-symlink directory handles."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    current = os.open(_REPO_ROOT, flags)
    try:
        for segment in ("outputs", "adamo_matched_development"):
            try:
                os.mkdir(segment, mode=0o755, dir_fd=current)
            except FileExistsError:
                pass
            child = os.open(segment, flags, dir_fd=current)
            os.close(current)
            current = child
        try:
            os.stat(OUTPUT_PATH.name, dir_fd=current, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"refusing to overwrite immutable output: {OUTPUT_PATH}")
        temporary_name = f".{OUTPUT_PATH.name}.pending"
        try:
            temporary_fd = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=current,
            )
        except FileExistsError as error:
            raise FileExistsError(
                f"another execution already reserved immutable output: {OUTPUT_PATH}"
            ) from error
        os.fsync(current)
        return current, temporary_fd, temporary_name
    except BaseException:
        os.close(current)
        raise


def _publish_report(
    directory_fd: int,
    temporary_fd: int,
    temporary_name: str,
    report: dict[str, object],
) -> None:
    """Validate, publish without replacement, and strictly reload one report."""
    normalized = validate_report(report)
    encoded = (
        json.dumps(normalized, allow_nan=False, indent=1, sort_keys=True) + "\n"
    ).encode("utf-8")
    if len(encoded) > _MAX_REPORT_BYTES:
        raise ValueError("encoded report exceeds the output byte limit")
    offset = 0
    while offset < len(encoded):
        written = os.write(temporary_fd, encoded[offset:])
        if written <= 0:
            raise OSError("short write while staging AdamO report")
        offset += written
    os.fchmod(temporary_fd, 0o444)
    os.fsync(temporary_fd)
    try:
        os.link(
            temporary_name,
            OUTPUT_PATH.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite immutable output: {OUTPUT_PATH}") from error
    os.fsync(directory_fd)
    published_fd = os.open(OUTPUT_PATH.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
    try:
        reloaded = bytearray()
        while True:
            remaining = _MAX_REPORT_BYTES + 1 - len(reloaded)
            chunk = os.read(published_fd, min(65_536, remaining))
            if not chunk:
                break
            reloaded.extend(chunk)
            if len(reloaded) > _MAX_REPORT_BYTES:
                raise ValueError("published report exceeds the output byte limit")
        if bytes(reloaded) != encoded:
            raise RuntimeError("published report differs from the staged report")
        if validate_report(json.loads(reloaded)) != normalized:
            raise RuntimeError("published report failed strict canonical reload")
    finally:
        os.close(published_fd)


def main(argv: Sequence[str] | None = None) -> int:
    """Run once and publish only to the frozen append-only namespace."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-home", type=Path, default=default_openml_data_home())
    args = parser.parse_args(argv)
    directory_fd, temporary_fd, temporary_name = _open_output_transaction()
    try:
        report = run(data_home=args.data_home)
        _publish_report(directory_fd, temporary_fd, temporary_name, report)
    finally:
        os.close(temporary_fd)
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
            os.fsync(directory_fd)
        except FileNotFoundError:
            pass
        os.close(directory_fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
