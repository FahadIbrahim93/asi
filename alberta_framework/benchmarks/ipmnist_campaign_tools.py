"""Maintained, nonpromoting IPMNIST campaign analysis tools.

Historical programs stored below ``outputs/`` are immutable run records.  This
module owns the live analysis contracts and takes every input path explicitly,
so reproduction never depends on a contributor's home directory.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from alberta_framework.benchmarks.ipmnist_provenance import analysis_provenance

DEFAULT_BASE = "upgd_ema_norm_sigma0"
DEFAULT_ARMS = (
    "sigma0_hidden_norm",
    "sigma0_localgate",
    "sigma0_ndecay099",
    "sigma0_ndecay09999",
    "sigma0_eps1e6",
    "sigma0_eps1e4",
    "sigma0_gate_beta05",
    "sigma0_gate_beta2",
)
DEFAULT_THRESHOLD = 0.002
CONFIRM_ALIGNMENT_ATOL = 1e-7
LATE_LO, LATE_HI = 4000, 5000
BUCKETS = (
    (0, 50),
    (50, 100),
    (100, 250),
    (250, 500),
    (500, 1000),
    (1000, 2000),
    (2000, 3500),
    (3500, 5000),
)


def across_seed_spread(values: Sequence[float] | np.ndarray[Any, Any]) -> float:
    """Return sample standard deviation, or zero below two observations."""
    array = np.asarray(values, dtype=np.float64).ravel()
    if array.size < 2:
        return 0.0
    return float(array.std(ddof=1))


def _json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if type(payload) is not dict:
        raise ValueError(f"{path} must contain a JSON object")
    return cast(dict[str, Any], payload)


def seed_means(root: Path, config_name: str) -> dict[int, float]:
    """Read mean per-task accuracy by seed for one campaign arm."""
    result: dict[int, float] = {}
    for path in sorted(root.glob(f"{config_name}_seed*.json")):
        payload = _json_object(path)
        seed = payload.get("seed")
        accuracies = payload.get("per_task_accuracy")
        if type(seed) is not int or type(accuracies) is not list or not accuracies:
            raise ValueError(f"{path} lacks a valid seed or per_task_accuracy")
        values = np.asarray(accuracies, dtype=np.float64)
        if values.ndim != 1 or not bool(np.all(np.isfinite(values))):
            raise ValueError(f"{path} has invalid per_task_accuracy")
        if seed in result:
            raise ValueError(f"duplicate seed {seed} for {config_name}")
        result[seed] = statistics.mean(float(value) for value in values)
    return result


def build_frontier(
    screen_dir: Path,
    confirm_dir: Path,
    *,
    base: str = DEFAULT_BASE,
    arms: Sequence[str] = DEFAULT_ARMS,
    threshold: float = DEFAULT_THRESHOLD,
    created_unix: float | None = None,
) -> dict[str, Any]:
    """Build a paired-seed development frontier without mixing seed sets."""
    screen_base = seed_means(screen_dir, base)
    confirm_base = seed_means(confirm_dir, base)
    if not screen_base:
        raise ValueError(f"screen base {base!r} has no seeds")
    rows: list[dict[str, Any]] = []
    for arm in arms:
        screen = seed_means(screen_dir, arm)
        confirm = seed_means(confirm_dir, arm)
        if screen.keys() != screen_base.keys():
            raise ValueError(
                f"screen seed sets differ for {arm!r} and base {base!r}: "
                f"{sorted(screen)} != {sorted(screen_base)}"
            )
        paired_seeds = sorted(screen)
        row: dict[str, Any] = {
            "config_name": arm,
            "n_screen_seeds": len(paired_seeds),
            "screen_mean": statistics.mean(screen[seed] for seed in paired_seeds),
            "screen_paired_delta_vs_base": statistics.mean(
                screen[seed] - screen_base[seed] for seed in paired_seeds
            ),
            "screen_per_seed_delta": [
                round(screen[seed] - screen_base[seed], 6) for seed in paired_seeds
            ],
        }
        row["confirmation_candidate"] = row["screen_paired_delta_vs_base"] > threshold
        if confirm:
            if confirm.keys() != confirm_base.keys():
                raise ValueError(
                    f"confirm seed sets differ for {arm!r} and base {base!r}: "
                    f"{sorted(confirm)} != {sorted(confirm_base)}"
                )
            confirm_seeds = sorted(confirm)
            row["confirm_mean"] = statistics.mean(
                confirm[seed] for seed in confirm_seeds
            )
            row["n_confirm_seeds"] = len(confirm_seeds)
            row["confirm_paired_delta_vs_base"] = statistics.mean(
                confirm[seed] - confirm_base[seed] for seed in confirm_seeds
            )
        rows.append(row)
    rows.sort(key=lambda row: -float(row.get("screen_mean", 0.0)))
    input_paths = [
        path
        for directory in (screen_dir, confirm_dir)
        for name in (base, *arms)
        for path in directory.glob(f"{name}_seed*.json")
    ]
    return {
        "schema": "asi.ipmnist.frontier.v2",
        "base": base,
        "base_screen_mean": statistics.mean(screen_base.values()) if screen_base else None,
        "base_confirm_mean": (
            statistics.mean(confirm_base.values()) if confirm_base else None
        ),
        "confirmation_threshold": threshold,
        "created_unix": time.time() if created_unix is None else created_unix,
        "evidence_policy": {
            "development_only": True,
            "evidence_class": "development_screening_diagnostic",
            "scientific_promotion_allowed": False,
        },
        "provenance": analysis_provenance(
            command="frontier",
            input_paths=input_paths,
            sources={"campaign_tools": Path(__file__)},
            repository_root=Path(__file__).resolve().parents[2],
        ),
        "results": rows,
    }


def _ceiling_runs(
    ceiling_dir: Path,
    prefix: str,
    *,
    input_paths: list[Path] | None = None,
) -> dict[int, dict[str, Any]]:
    runs: dict[int, dict[str, Any]] = {}
    for path in sorted(ceiling_dir.glob(f"{prefix}_seed*.json")):
        payload = _json_object(path)
        seed = payload.get("seed")
        tag = payload.get("tag")
        if type(seed) is not int or type(tag) is not str:
            raise ValueError(f"{path} lacks a valid seed or tag")
        per_step_path = ceiling_dir / f"{tag}_seed{seed}_per_step.npy"
        payload["per_step"] = np.load(per_step_path, allow_pickle=False)
        if input_paths is not None:
            input_paths.extend((path, per_step_path))
        runs[seed] = payload
    for path in sorted(ceiling_dir.glob(f"{prefix}_seed*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            payload = json.loads(str(archive["metadata"].item()))
            per_step = np.asarray(archive["per_step"])
        if type(payload) is not dict:
            raise ValueError(f"{path} metadata must be a JSON object")
        if payload.get("schema") != "asi.ipmnist_ceiling.run.v2":
            raise ValueError(f"{path} has an unsupported maintained run schema")
        if type(payload.get("provenance")) is not dict:
            raise ValueError(f"{path} lacks maintained run provenance")
        seed = payload.get("seed")
        if type(seed) is not int:
            raise ValueError(f"{path} lacks a valid seed")
        if seed in runs:
            raise ValueError(f"duplicate ceiling seed {seed} for {prefix}")
        if per_step.ndim != 2 or list(per_step.shape) != [
            payload.get("n_tasks"),
            payload.get("task_length"),
        ]:
            raise ValueError(f"{path} per_step shape disagrees with metadata")
        payload["per_step"] = per_step
        if input_paths is not None:
            input_paths.append(path)
        runs[seed] = payload
    return runs


def _finite_vector(value: object, *, name: str) -> np.ndarray[Any, np.dtype[np.float64]]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{name} must be a finite vector")
    return array


def validate_confirm_alignment(
    runs: Sequence[dict[str, Any]],
    confirm_dir: Path,
    *,
    input_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Validate rounded ceiling JSON against confirmation shards.

    Both files store decimal JSON derived from independently rounded floating
    computations.  The fixed absolute tolerance admits the observed encoding
    delta while rejecting a score-relevant disagreement; relative tolerance is
    deliberately zero so the gate does not widen with accuracy magnitude.
    """
    per_seed: dict[str, Any] = {}
    maximum = 0.0
    for run in runs:
        seed = cast(int, run["seed"])
        reference_path = confirm_dir / f"sigma0_ndecay099_seed{seed}.json"
        reference = _json_object(reference_path)
        if input_paths is not None:
            input_paths.append(reference_path)
        observed = _finite_vector(run["per_task_accuracy"], name="per_task_accuracy")
        expected = _finite_vector(reference["per_task_accuracy"], name="reference")
        if observed.shape != expected.shape:
            raise ValueError(f"ceiling and confirm shapes differ for seed {seed}")
        max_delta = float(np.max(np.abs(observed - expected), initial=0.0))
        if not bool(
            np.allclose(
                observed,
                expected,
                rtol=0.0,
                atol=CONFIRM_ALIGNMENT_ATOL,
                equal_nan=False,
            )
        ):
            raise ValueError(
                f"ceiling and confirm per-task values differ for seed {seed}: "
                f"max_abs_delta={max_delta:.17g} exceeds atol={CONFIRM_ALIGNMENT_ATOL}"
            )
        maximum = max(maximum, max_delta)
        per_seed[str(seed)] = {
            "n_values": int(observed.size),
            "max_abs_delta": max_delta,
        }
    return {
        "absolute_tolerance": CONFIRM_ALIGNMENT_ATOL,
        "relative_tolerance": 0.0,
        "max_abs_delta": maximum,
        "per_seed": per_seed,
    }


def build_ceiling_summary(ceiling_dir: Path, confirm_dir: Path) -> dict[str, Any]:
    """Recompute the maintained ceiling/error-budget summary from raw runs."""
    input_paths: list[Path] = []
    report: dict[str, Any] = {
        "schema": "asi.ipmnist.ceiling_summary.v2",
        "evidence_policy": {
            "development_only": True,
            "scientific_promotion_allowed": False,
        }
    }
    for spec in ("sigma0_ndecay099", "adamw_control", "sgd_ema_norm"):
        runs = _ceiling_runs(
            ceiling_dir, f"stationary_{spec}", input_paths=input_paths
        )
        if not runs:
            continue
        means = [float(run["mean_accuracy"]) for run in runs.values()]
        curves = np.stack(
            [np.asarray(run["per_step"])[0].astype(np.float64) for run in runs.values()]
        )
        mean_curve = curves.mean(axis=0)
        report[f"stationary_{spec}"] = {
            "avg_online_mean": float(np.mean(means)),
            "avg_online_per_seed": means,
            "sample_standard_deviation": across_seed_spread(means),
            "late_window": float(mean_curve[LATE_LO:LATE_HI].mean()),
            "buckets": {
                f"{start}-{end}": round(float(mean_curve[start:end].mean()), 5)
                for start, end in BUCKETS
            },
        }

    for spec in ("sigma0_ndecay099", "adamw_control"):
        runs = _ceiling_runs(ceiling_dir, f"carried_{spec}", input_paths=input_paths)
        if not runs:
            continue
        per_task = np.stack(
            [
                _finite_vector(run["per_task_accuracy"], name="per_task_accuracy")
                for run in runs.values()
            ]
        )
        late_tasks = per_task[:, -10:].mean(axis=1)
        last_task_late = np.stack(
            [
                np.asarray(run["per_step"], dtype=np.float64)[-10:, LATE_LO:LATE_HI].mean()
                for run in runs.values()
            ]
        )
        report[f"carried_{spec}"] = {
            "per_task_mean_curve": [round(float(value), 5) for value in per_task.mean(axis=0)],
            "late_task_avg_online": float(late_tasks.mean()),
            "late_task_sample_standard_deviation": across_seed_spread(late_tasks),
            "late_task_late_window": float(last_task_late.mean()),
        }

    batch = [
        _json_object(path)
        for path in sorted(ceiling_dir.glob("batch_reference_seed*.json"))
    ]
    input_paths.extend(sorted(ceiling_dir.glob("batch_reference_seed*.json")))
    if batch:
        best = [float(run["test_accuracy_best"]) for run in batch]
        report["batch_reference"] = {
            "test_best_mean": float(np.mean(best)),
            "test_best_sample_standard_deviation": across_seed_spread(best),
            "per_seed": best,
        }

    full = _ceiling_runs(
        ceiling_dir, "full_sigma0_ndecay099", input_paths=input_paths
    )
    if full:
        ordered = [full[seed] for seed in sorted(full)]
        alignment = validate_confirm_alignment(
            ordered, confirm_dir, input_paths=input_paths
        )
        array = np.stack(
            [np.asarray(run["per_step"], dtype=np.float64) for run in ordered]
        )
        overall = float(array.mean())
        plateau_by_task = array[:, :, LATE_LO:LATE_HI].mean(axis=2)
        asymptotic_error = float((1.0 - plateau_by_task).mean())
        total_error = 1.0 - overall
        curve = array.mean(axis=(0, 1))
        bucket_rows = []
        for start, end in BUCKETS:
            accuracy = float(curve[start:end].mean())
            excess = float(
                (plateau_by_task[:, :, None] - array[:, :, start:end]).mean()
            )
            contribution = excess * (end - start) / array.shape[2]
            bucket_rows.append(
                [
                    f"{start}-{end}",
                    round(accuracy, 5),
                    round(excess, 5),
                    round(contribution, 5),
                ]
            )
        task_plateau = plateau_by_task.mean(axis=0)
        task_index = np.arange(task_plateau.size)
        late_mask = task_index >= 20
        slope = float(np.polyfit(task_index[late_mask], task_plateau[late_mask], 1)[0])
        report["error_budget"] = {
            "overall": overall,
            "total_error": total_error,
            "asymptotic_error": asymptotic_error,
            "transient_excess": total_error - asymptotic_error,
            "plateau_mean": float(plateau_by_task.mean()),
            "first500": float(array[:, :, :500].mean()),
            "buckets": bucket_rows,
            "plateau_tasks_20_60": float(task_plateau[20:60].mean()),
            "plateau_tasks_160_200": float(task_plateau[160:200].mean()),
            "drift_slope_per_task": slope,
            "confirm_alignment": alignment,
        }
    report["provenance"] = analysis_provenance(
        command="ceiling",
        input_paths=input_paths,
        sources={"campaign_tools": Path(__file__)},
        repository_root=Path(__file__).resolve().parents[2],
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    frontier = subparsers.add_parser("frontier", help="build a paired-seed frontier")
    frontier.add_argument("--screen-dir", type=Path, required=True)
    frontier.add_argument("--confirm-dir", type=Path, required=True)
    frontier.add_argument("--base", default=DEFAULT_BASE)
    frontier.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    frontier.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ceiling = subparsers.add_parser("ceiling", help="recompute the ceiling summary")
    ceiling.add_argument("--ceiling-dir", type=Path, required=True)
    ceiling.add_argument("--confirm-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a maintained analyzer and emit JSON to stdout only."""
    args = _parser().parse_args(argv)
    if args.command == "frontier":
        result = build_frontier(
            args.screen_dir,
            args.confirm_dir,
            base=args.base,
            arms=args.arms,
            threshold=args.threshold,
        )
    else:
        result = build_ceiling_summary(args.ceiling_dir, args.confirm_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
