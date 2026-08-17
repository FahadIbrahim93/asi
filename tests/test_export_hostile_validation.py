"""Hostile-safe validation for export utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alberta_framework.utils.experiments import AggregatedResults, MetricSummary
from alberta_framework.utils.export import (
    _exported_number,
    export_to_csv,
    export_to_json,
    generate_latex_table,
    save_experiment_report,
)


class _StringSubclass(str):
    pass


class _HostilePath(Path):
    def __str__(self) -> str:  # pragma: no cover
        raise AssertionError("str hook")

    def __fspath__(self) -> str:  # pragma: no cover
        raise AssertionError("fspath hook")


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
