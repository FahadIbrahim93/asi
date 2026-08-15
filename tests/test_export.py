"""Machine-readable export must equal the measured values.

CSV ``.6f`` rounding and Python ``json.dump``'s NaN extension are not
presentation choices on this path: ``save_experiment_report`` treats the CSV
and JSON as the record of the run. LaTeX/markdown tables stay display-only.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from alberta_framework.utils.experiments import AggregatedResults, MetricSummary
from alberta_framework.utils.export import export_to_csv, export_to_json

pytestmark = pytest.mark.unit


def _agg(
    name: str,
    seeds: list[int],
    finals: list[float],
    series: np.ndarray | None = None,
) -> AggregatedResults:
    values = np.asarray(finals, dtype=np.float64)
    if series is None:
        series = values.reshape(len(seeds), 1)
    return AggregatedResults(
        config_name=name,
        seeds=list(seeds),
        metric_arrays={"squared_error": np.asarray(series, dtype=np.float64)},
        summary={
            "squared_error": MetricSummary(
                mean=float(np.mean(values)),
                std=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                min=float(np.min(values)),
                max=float(np.max(values)),
                n_seeds=len(values),
                values=values,
            )
        },
    )


def test_summary_csv_preserves_tiny_measured_means(tmp_path: Path) -> None:
    result = _agg("tiny", [0, 1], [1.23e-8, 2.34e-8])
    path = tmp_path / "tiny.csv"
    export_to_csv({"tiny": result}, path)

    row = next(csv.DictReader(path.open(encoding="utf-8")))
    assert float(row["mean"]) == result.summary["squared_error"].mean
    assert float(row["min"]) == result.summary["squared_error"].min
    assert float(row["max"]) == result.summary["squared_error"].max


def test_summary_csv_keeps_means_that_collapse_at_six_decimals(tmp_path: Path) -> None:
    left = _agg("a", [0], [0.1234564])
    right = _agg("b", [0], [0.1234565])
    assert left.summary["squared_error"].mean != right.summary["squared_error"].mean
    path = tmp_path / "collapse.csv"
    export_to_csv({"a": left, "b": right}, path)

    rows = {row["config"]: row for row in csv.DictReader(path.open(encoding="utf-8"))}
    assert float(rows["a"]["mean"]) == left.summary["squared_error"].mean
    assert float(rows["b"]["mean"]) == right.summary["squared_error"].mean
    assert rows["a"]["mean"] != rows["b"]["mean"]


def test_timeseries_csv_round_trips_step_values_and_seed_ids(tmp_path: Path) -> None:
    series = np.array([[1.23e-8, 2.0], [3.0, 4.0]], dtype=np.float64)
    result = _agg("arm", [10, 20], [2.0, 4.0], series=series)
    path = tmp_path / "ts.csv"
    export_to_csv({"arm": result}, path, include_timeseries=True)

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert list(rows[0].keys()) == ["step", "arm_seed10", "arm_seed20"]
    assert float(rows[0]["arm_seed10"]) == 1.23e-8
    assert float(rows[1]["arm_seed20"]) == 4.0


def test_json_export_round_trips_finite_summary(tmp_path: Path) -> None:
    result = _agg("arm", [3, 7], [0.25, 0.75])
    path = tmp_path / "arm.json"
    export_to_json({"arm": result}, path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    summary = result.summary["squared_error"]
    payload = loaded["arm"]["summary"]["squared_error"]
    assert loaded["arm"]["seeds"] == [3, 7]
    assert payload["mean"] == summary.mean
    assert payload["values"] == summary.values.tolist()


def test_json_export_refuses_nonfinite_measurements(tmp_path: Path) -> None:
    path = tmp_path / "nan.json"
    with pytest.raises(ValueError):
        export_to_json({"nf": _agg("nf", [0], [math.nan])}, path)
    assert not path.exists() or path.read_text(encoding="utf-8") == ""


def test_csv_export_refuses_nonfinite_measurements(tmp_path: Path) -> None:
    path = tmp_path / "inf.csv"
    with pytest.raises(ValueError, match="non-finite"):
        export_to_csv({"inf": _agg("inf", [0], [math.inf])}, path)
