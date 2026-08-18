"""Hostile-safe validation for export utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alberta_framework.utils.experiments import AggregatedResults, MetricSummary
from alberta_framework.utils.export import (
    _checked_export_cells,
    _exported_number,
    export_to_csv,
    export_to_json,
    generate_latex_table,
    generate_markdown_table,
    generate_significance_table,
    save_experiment_report,
)
from alberta_framework.utils.statistics import SignificanceResult


class _StringSubclass(str):
    pass


class _HostilePath(Path):
    def __str__(self) -> str:  # pragma: no cover
        raise AssertionError("str hook")

    def __fspath__(self) -> str:  # pragma: no cover
        raise AssertionError("fspath hook")


class _HostileDict(dict[object, object]):
    def items(self) -> object:  # pragma: no cover
        raise AssertionError("items hook")


class _HostileArray(np.ndarray):
    def __getattribute__(self, name: str) -> object:  # pragma: no cover
        if name in {"dtype", "ndim", "shape", "size"}:
            raise AssertionError("array attribute hook")
        return super().__getattribute__(name)


def _constant_result(name: str = "a") -> AggregatedResults:
    values = np.asarray([1.0], dtype=np.float64)
    return AggregatedResults(
        config_name=name,
        seeds=[17],
        metric_arrays={"squared_error": values.reshape(1, 1)},
        summary={
            "squared_error": MetricSummary(
                mean=1.0,
                std=0.0,
                min=1.0,
                max=1.0,
                n_seeds=1,
                values=values,
            )
        },
    )


def test_exported_number_rejects_bool_without_repr() -> None:
    with pytest.raises(ValueError, match="refusing to export boolean"):
        _exported_number(True)
    with pytest.raises(ValueError, match="refusing to export boolean"):
        _exported_number(np.bool_(True))


def test_exported_number_rejects_bool_subclass() -> None:
    class HostileNPBool(np.bool_):
        def __repr__(self) -> str:  # pragma: no cover
            raise RuntimeError("repr hook")

    with pytest.raises(ValueError, match="refusing to export boolean"):
        _exported_number(HostileNPBool(True))


def test_export_to_csv_rejects_string_subclass_metric() -> None:
    results = {"a": _constant_result("a")}
    with pytest.raises(ValueError, match="metric"):
        export_to_csv(results, Path("/tmp/x.csv"), metric=_StringSubclass("squared_error"))


def test_export_to_csv_rejects_string_subclass_config_name() -> None:
    hostile_key = _StringSubclass("a")
    results = {hostile_key: _constant_result("a")}
    with pytest.raises(ValueError, match="config name"):
        export_to_csv(results, Path("/tmp/x.csv"))  # type: ignore[arg-type]


def test_export_to_csv_rejects_hostile_path() -> None:
    results = {"a": _constant_result("a")}
    hostile = object.__new__(_HostilePath)
    with pytest.raises(ValueError, match="filepath"):
        export_to_csv(results, hostile)


def test_export_to_json_rejects_hostile_path() -> None:
    results = {"a": _constant_result("a")}
    hostile = object.__new__(_HostilePath)
    with pytest.raises(ValueError, match="filepath"):
        export_to_json(results, hostile)


def test_generate_latex_rejects_string_subclass_metric() -> None:
    results = {"a": _constant_result("a")}
    with pytest.raises(ValueError, match="metric"):
        generate_latex_table(results, metric=_StringSubclass("squared_error"))


def test_save_report_rejects_string_subclass_experiment_name(tmp_path: Path) -> None:
    results = {"a": _constant_result("a")}
    with pytest.raises(ValueError, match="experiment_name"):
        save_experiment_report(results, tmp_path, _StringSubclass("exp"))


def test_save_report_rejects_hostile_output_dir() -> None:
    results = {"a": _constant_result("a")}
    hostile = object.__new__(_HostilePath)
    with pytest.raises(ValueError, match="output_dir"):
        save_experiment_report(results, hostile, "exp")


def test_preflight_rejects_string_subclass_metric_name() -> None:
    values = np.asarray([1.0], dtype=np.float64)
    bad = AggregatedResults(
        config_name="a",
        seeds=[17],
        metric_arrays={_StringSubclass("squared_error"): values.reshape(1, 1)},
        summary={
            "squared_error": MetricSummary(
                mean=1.0, std=0.0, min=1.0, max=1.0, n_seeds=1, values=values
            )
        },
    )
    with pytest.raises(ValueError, match="metric name"):
        export_to_csv({"a": bad}, Path("/tmp/x.csv"))


def test_exported_number_rejects_nonfinite_without_repr() -> None:
    with pytest.raises(ValueError, match="refusing to export non-finite"):
        _exported_number(float("nan"))
    with pytest.raises(ValueError, match="refusing to export non-finite"):
        _exported_number(float("inf"))


def test_exported_number_does_not_invoke_hostile_repr() -> None:
    class EvilStr(str):
        def __repr__(self) -> str:  # pragma: no cover
            raise RuntimeError("repr hook")

    evil = EvilStr("a")
    results = {evil: _constant_result("a")}
    with pytest.raises(ValueError, match="config name"):
        export_to_csv(results, Path("/tmp/x.csv"))  # type: ignore[arg-type]


def test_exported_number_rejects_float_subclass_without_float_hook() -> None:
    calls = 0

    class HostileFloat(float):
        def __float__(self) -> float:  # pragma: no cover
            nonlocal calls
            calls += 1
            raise AssertionError("float hook")

    with pytest.raises(ValueError, match="non-canonical"):
        _exported_number(HostileFloat(1.0))
    assert calls == 0


def test_preflight_rejects_hostile_outer_dictionary_before_iteration() -> None:
    results = _HostileDict({"a": _constant_result("a")})
    with pytest.raises(ValueError, match="exact dictionary"):
        export_to_csv(results, Path("/tmp/x.csv"))  # type: ignore[arg-type]


def test_preflight_rejects_hostile_metric_dictionary_before_iteration() -> None:
    result = _constant_result("a")._replace(
        metric_arrays=_HostileDict({"squared_error": np.ones((1, 1), dtype=np.float64)})
    )
    with pytest.raises(ValueError, match="metric_arrays must be an exact dictionary"):
        export_to_csv({"a": result}, Path("/tmp/x.csv"))  # type: ignore[arg-type]


def test_preflight_rejects_array_subclass_before_attribute_hooks() -> None:
    hostile = np.ones((1, 1), dtype=np.float64).view(_HostileArray)
    result = _constant_result("a")._replace(metric_arrays={"squared_error": hostile})
    with pytest.raises(ValueError, match="exact float64 NumPy array"):
        export_to_csv({"a": result}, Path("/tmp/x.csv"))


def test_preflight_binds_aggregate_name_to_dictionary_key() -> None:
    with pytest.raises(ValueError, match="config_name must match"):
        export_to_csv({"outer": _constant_result("inner")}, Path("/tmp/x.csv"))


def test_preflight_binds_summary_statistics_to_values() -> None:
    result = _constant_result("a")
    summary = result.summary["squared_error"]._replace(mean=2.0)
    result = result._replace(summary={"squared_error": summary})
    with pytest.raises(ValueError, match="statistics do not match values"):
        export_to_json({"a": result}, Path("/tmp/x.json"))


def test_preflight_uses_aggregate_metrics_float_formula(tmp_path: Path) -> None:
    values = np.asarray([0.1257302210933933, -0.1321048632913019], dtype=np.float64)
    summary = MetricSummary(
        mean=float(np.mean(values)),
        std=float(np.std(values, ddof=1)),
        min=float(np.min(values)),
        max=float(np.max(values)),
        n_seeds=2,
        values=values,
    )
    result = AggregatedResults(
        config_name="a",
        seeds=[17, 18],
        metric_arrays={"squared_error": values.reshape(2, 1)},
        summary={"squared_error": summary},
    )
    export_to_json({"a": result}, tmp_path / "result.json")


@pytest.mark.parametrize("include_timeseries", [0, np.bool_(True)])
def test_export_rejects_non_builtin_boolean_flag(include_timeseries: object) -> None:
    with pytest.raises(ValueError, match="include_timeseries must be an exact bool"):
        export_to_json(
            {"a": _constant_result("a")},
            Path("/tmp/x.json"),
            include_timeseries=include_timeseries,  # type: ignore[arg-type]
        )


def test_report_rejects_traversal_name_before_creating_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "not-created"
    with pytest.raises(ValueError, match="safe filename component"):
        save_experiment_report({"a": _constant_result("a")}, output_dir, "../escape")
    assert not output_dir.exists()


def _significance() -> SignificanceResult:
    return SignificanceResult(
        test_name="paired_t",
        statistic=1.0,
        p_value=0.01,
        significant=True,
        alpha=0.05,
        effect_size=0.5,
        method_a="a",
        method_b="b",
    )


def _forged_significance(**changes: object) -> SignificanceResult:
    """Bypass the public constructor only to retain sink-defense coverage."""

    fields = _significance()._asdict()
    fields.update(changes)
    return tuple.__new__(SignificanceResult, tuple(fields.values()))


def test_significance_rejects_hostile_dictionary_before_iteration() -> None:
    hostile = _HostileDict({("a", "b"): _significance()})
    with pytest.raises(ValueError, match="exact dictionary"):
        generate_significance_table(hostile)  # type: ignore[arg-type]


@pytest.mark.parametrize("format", ["html", _StringSubclass("latex")])
def test_significance_rejects_noncanonical_format(format: object) -> None:
    with pytest.raises(ValueError, match="format"):
        generate_significance_table(
            {("a", "b"): _significance()}, format=format  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("p_value", float("nan"), "non-finite"),
        ("p_value", 1.1, "canonical domains"),
        ("alpha", 0.0, "canonical domains"),
        ("significant", False, "must match"),
    ],
)
def test_significance_rejects_invalid_record_fields(
    field: str, value: object, message: str
) -> None:
    result = _forged_significance(**{field: value})
    with pytest.raises(ValueError, match=message):
        generate_significance_table({("a", "b"): result})


def test_display_tables_reject_non_builtin_direction_flag() -> None:
    results = {"a": _constant_result("a")}
    for render in (generate_latex_table, generate_markdown_table):
        with pytest.raises(ValueError, match="lower_is_better must be an exact bool"):
            render(results, lower_is_better=np.bool_(True))  # type: ignore[arg-type]


def test_export_cell_counter_rejects_signed_int32_overflow() -> None:
    assert _checked_export_cells(2**31 - 2, 1) == 2**31 - 1
    with pytest.raises(ValueError, match="fit signed int32"):
        _checked_export_cells(2**31 - 1, 1)
