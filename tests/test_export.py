"""Machine-readable exports preserve finite measurements exactly and fail closed."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

import alberta_framework.utils.export as export_module
from alberta_framework.utils.experiments import AggregatedResults, MetricSummary
from alberta_framework.utils.export import (
    export_to_csv,
    export_to_json,
    generate_latex_table,
    generate_markdown_table,
)

pytestmark = pytest.mark.unit

_METRIC = "squared_error"
_ROUND_TRIP_VALUES = (
    1.0000000000000002,
    0.12345678901234566,
    1.785e-8,
    float.fromhex("0x0.0000000000001p-1022"),
)


def _constant_result(name: str, value: float = 1.0) -> AggregatedResults:
    values = np.asarray([value], dtype=np.float64)
    return AggregatedResults(
        config_name=name,
        seeds=[17],
        metric_arrays={_METRIC: values.reshape(1, 1)},
        summary={
            _METRIC: MetricSummary(
                mean=value,
                std=0.0,
                min=value,
                max=value,
                n_seeds=1,
                values=values,
            )
        },
    )


def _timeseries_result() -> AggregatedResults:
    series = np.asarray(
        [
            [_ROUND_TRIP_VALUES[0], _ROUND_TRIP_VALUES[2]],
            [_ROUND_TRIP_VALUES[1], _ROUND_TRIP_VALUES[3]],
        ],
        dtype=np.float64,
    )
    final_values = series[:, -1]
    return AggregatedResults(
        config_name="trace",
        seeds=[17, 29],
        metric_arrays={_METRIC: series},
        summary={
            _METRIC: MetricSummary(
                mean=float(np.mean(final_values)),
                std=float(np.std(final_values, ddof=1)),
                min=float(np.min(final_values)),
                max=float(np.max(final_values)),
                n_seeds=2,
                values=final_values,
            )
        },
    )


def _result_with_nonfinite(surface: str, number: float) -> AggregatedResults:
    result = _constant_result("invalid")
    summary = result.summary[_METRIC]
    if surface == "summary":
        summary = summary._replace(mean=number)
        return result._replace(summary={_METRIC: summary})
    if surface == "values":
        summary = summary._replace(values=np.asarray([number], dtype=np.float64))
        return result._replace(summary={_METRIC: summary})
    if surface == "timeseries":
        return result._replace(
            metric_arrays={_METRIC: np.asarray([[number]], dtype=np.float64)}
        )
    raise AssertionError(f"unknown test surface: {surface}")


def _preflight_invalid_results(case: str) -> dict[str, AggregatedResults]:
    result = _constant_result("invalid")
    summary = result.summary[_METRIC]

    if case == "empty_results":
        return {}
    if case == "duplicate_seeds":
        duplicate = _timeseries_result()._replace(seeds=[17, 17])
        return {"invalid": duplicate}
    if case == "empty_seeds":
        empty_summary = summary._replace(
            mean=0.0,
            min=0.0,
            max=0.0,
            n_seeds=0,
            values=np.empty((0,), dtype=np.float64),
        )
        result = result._replace(
            seeds=[],
            metric_arrays={_METRIC: np.empty((0, 1), dtype=np.float64)},
            summary={_METRIC: empty_summary},
        )
    elif case == "bool_seed":
        result = result._replace(seeds=[True])
    elif case == "numpy_integer_seed":
        result = result._replace(seeds=[np.int64(17)])
    elif case == "negative_seed":
        result = result._replace(seeds=[-1])
    elif case == "seed_above_uint32":
        result = result._replace(seeds=[1 << 32])
    elif case == "zero_seed_axis":
        result = result._replace(
            metric_arrays={_METRIC: np.empty((0, 1), dtype=np.float64)}
        )
    elif case == "zero_step_axis":
        result = result._replace(
            metric_arrays={_METRIC: np.empty((1, 0), dtype=np.float64)}
        )
    elif case == "metric_ndim":
        result = result._replace(
            metric_arrays={_METRIC: np.ones((1, 1, 1), dtype=np.float64)}
        )
    elif case == "metric_row_count":
        result = result._replace(
            metric_arrays={_METRIC: np.ones((2, 1), dtype=np.float64)}
        )
    elif case == "summary_values_ndim":
        bad_summary = summary._replace(values=np.ones((1, 1), dtype=np.float64))
        result = result._replace(summary={_METRIC: bad_summary})
    elif case == "summary_count":
        bad_summary = summary._replace(n_seeds=2)
        result = result._replace(summary={_METRIC: bad_summary})
    elif case == "summary_value_count":
        bad_summary = summary._replace(values=np.ones((2,), dtype=np.float64))
        result = result._replace(summary={_METRIC: bad_summary})
    elif case == "nonfinite_unselected_array":
        result = result._replace(
            metric_arrays={
                _METRIC: result.metric_arrays[_METRIC],
                "unselected": np.asarray([[math.nan]], dtype=np.float64),
            }
        )
    elif case == "nonfinite_unselected_summary":
        bad_summary = summary._replace(mean=math.inf)
        result = result._replace(
            summary={_METRIC: summary, "unselected": bad_summary}
        )
    else:
        raise AssertionError(f"unknown preflight test case: {case}")

    return {"invalid": result}


def _export_mode(
    mode: str,
    results: dict[str, AggregatedResults],
    path: Path,
) -> None:
    if mode == "summary_csv":
        export_to_csv(results, path)
    elif mode == "timeseries_csv":
        export_to_csv(results, path, include_timeseries=True)
    elif mode == "json":
        export_to_json(results, path, include_timeseries=True)
    else:
        raise AssertionError(f"unknown export mode: {mode}")


@pytest.mark.parametrize("value", _ROUND_TRIP_VALUES)
def test_summary_csv_uses_shortest_binary64_round_trip(value: float, tmp_path: Path) -> None:
    path = tmp_path / "nested" / "summary.csv"
    export_to_csv({"candidate": _constant_result("candidate", value)}, path)

    with path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))

    assert row["mean"] == repr(value)
    assert row["min"] == repr(value)
    assert row["max"] == repr(value)
    assert float(row["mean"]) == value


def test_timeseries_csv_round_trips_values_under_the_matching_seed_headers(
    tmp_path: Path,
) -> None:
    result = _timeseries_result()
    path = tmp_path / "timeseries.csv"
    export_to_csv({"trace": result}, path, include_timeseries=True)

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert list(rows[0]) == ["step", "trace_seed17", "trace_seed29"]
    assert rows[0]["trace_seed17"] == repr(_ROUND_TRIP_VALUES[0])
    assert rows[0]["trace_seed29"] == repr(_ROUND_TRIP_VALUES[1])
    assert rows[1]["trace_seed17"] == repr(_ROUND_TRIP_VALUES[2])
    assert rows[1]["trace_seed29"] == repr(_ROUND_TRIP_VALUES[3])
    assert [[float(row["trace_seed17"]), float(row["trace_seed29"])] for row in rows] == [
        [_ROUND_TRIP_VALUES[0], _ROUND_TRIP_VALUES[1]],
        [_ROUND_TRIP_VALUES[2], _ROUND_TRIP_VALUES[3]],
    ]


def test_json_round_trips_summary_values_and_timeseries(tmp_path: Path) -> None:
    results = {
        f"value_{index}": _constant_result(f"value_{index}", value)
        for index, value in enumerate(_ROUND_TRIP_VALUES)
    }
    path = tmp_path / "nested" / "results.json"
    export_to_json(results, path, include_timeseries=True)

    payload = json.loads(path.read_text(encoding="utf-8"))
    for name, result in results.items():
        exported = payload[name]
        summary = result.summary[_METRIC]
        assert exported["seeds"] == result.seeds
        assert exported["summary"][_METRIC] == {
            "mean": summary.mean,
            "std": summary.std,
            "min": summary.min,
            "max": summary.max,
            "n_seeds": summary.n_seeds,
            "values": summary.values.tolist(),
        }
        assert exported["timeseries"][_METRIC] == result.metric_arrays[_METRIC].tolist()


@pytest.mark.parametrize("number", [math.nan, math.inf, -math.inf], ids=["nan", "inf", "-inf"])
@pytest.mark.parametrize(
    ("format_name", "surface", "include_timeseries"),
    [
        ("csv", "summary", False),
        ("csv", "values", False),
        ("csv", "timeseries", True),
        ("json", "summary", False),
        ("json", "values", False),
        ("json", "timeseries", True),
    ],
)
def test_nonfinite_export_rejects_before_any_destination_mutation(
    format_name: str,
    surface: str,
    include_timeseries: bool,
    number: float,
    tmp_path: Path,
) -> None:
    result = _result_with_nonfinite(surface, number)

    def export(path: Path) -> None:
        if format_name == "csv":
            export_to_csv(
                {"invalid": result},
                path,
                include_timeseries=include_timeseries,
            )
        else:
            export_to_json(
                {"invalid": result},
                path,
                include_timeseries=include_timeseries,
            )

    suffix = format_name
    existing = tmp_path / f"existing.{suffix}"
    sentinel = "existing artifact\n"
    existing.write_text(sentinel, encoding="utf-8")
    with pytest.raises(ValueError):
        export(existing)
    assert existing.read_text(encoding="utf-8") == sentinel

    absent = tmp_path / "not-created" / f"absent.{suffix}"
    with pytest.raises(ValueError):
        export(absent)
    assert not absent.exists()
    assert not absent.parent.exists()


@pytest.mark.parametrize("mode", ["summary_csv", "timeseries_csv", "json"])
@pytest.mark.parametrize(
    "case",
    [
        "empty_results",
        "duplicate_seeds",
        "empty_seeds",
        "bool_seed",
        "numpy_integer_seed",
        "negative_seed",
        "seed_above_uint32",
        "zero_seed_axis",
        "zero_step_axis",
        "metric_ndim",
        "metric_row_count",
        "summary_values_ndim",
        "summary_count",
        "summary_value_count",
        "nonfinite_unselected_array",
        "nonfinite_unselected_summary",
    ],
)
def test_shared_preflight_rejects_invalid_aggregate_before_filesystem_mutation(
    mode: str,
    case: str,
    tmp_path: Path,
) -> None:
    results = _preflight_invalid_results(case)
    suffix = "json" if mode == "json" else "csv"

    existing = tmp_path / f"existing.{suffix}"
    sentinel = "existing artifact\n"
    existing.write_text(sentinel, encoding="utf-8")
    with pytest.raises(ValueError):
        _export_mode(mode, results, existing)
    assert existing.read_text(encoding="utf-8") == sentinel

    absent = tmp_path / "not-created" / f"absent.{suffix}"
    with pytest.raises(ValueError):
        _export_mode(mode, results, absent)
    assert not absent.exists()
    assert not absent.parent.exists()


@pytest.mark.parametrize("mode", ["summary_csv", "timeseries_csv", "json"])
def test_shared_preflight_accepts_uint32_seed_boundaries(mode: str, tmp_path: Path) -> None:
    result = _timeseries_result()._replace(seeds=[0, (1 << 32) - 1])
    suffix = "json" if mode == "json" else "csv"
    path = tmp_path / "nested" / f"valid.{suffix}"

    _export_mode(mode, {"trace": result}, path)

    assert path.is_file()
    if mode == "json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["trace"]["seeds"] == [0, (1 << 32) - 1]


def test_timeseries_csv_rejects_seed_row_misalignment_without_overwriting(
    tmp_path: Path,
) -> None:
    result = _timeseries_result()._replace(seeds=[17])
    path = tmp_path / "timeseries.csv"
    path.write_text("existing artifact\n", encoding="utf-8")

    with pytest.raises(ValueError, match="seed count"):
        export_to_csv({"trace": result}, path, include_timeseries=True)

    assert path.read_text(encoding="utf-8") == "existing artifact\n"


def test_timeseries_csv_rejects_duplicate_seed_headers_without_overwriting(
    tmp_path: Path,
) -> None:
    result = _timeseries_result()._replace(seeds=[17, 17])
    path = tmp_path / "timeseries.csv"
    path.write_text("existing artifact\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unique seeds"):
        export_to_csv({"trace": result}, path, include_timeseries=True)

    assert path.read_text(encoding="utf-8") == "existing artifact\n"


def test_atomic_publish_failure_preserves_destination_and_removes_temporary_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "summary.csv"
    path.write_text("existing artifact\n", encoding="utf-8")

    def reject_replace(source: os.PathLike[str], destination: os.PathLike[str]) -> None:
        raise OSError(f"cannot publish {source!s} to {destination!s}")

    monkeypatch.setattr(export_module.os, "replace", reject_replace)
    with pytest.raises(OSError, match="cannot publish"):
        export_to_csv({"candidate": _constant_result("candidate")}, path)

    assert path.read_text(encoding="utf-8") == "existing artifact\n"
    assert list(tmp_path.iterdir()) == [path]


def test_display_only_tables_keep_four_decimal_presentation() -> None:
    results = {"candidate": _constant_result("candidate", 0.12345678901234566)}

    latex = generate_latex_table(results)
    markdown = generate_markdown_table(results)

    assert r"\textbf{0.1235} $\pm$ 0.0000" in latex
    assert "**0.1235** ± 0.0000" in markdown
