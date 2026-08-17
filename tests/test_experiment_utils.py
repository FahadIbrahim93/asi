"""Contract tests for the generic multi-seed experiment aggregation helpers."""

from __future__ import annotations

from typing import Any, cast

import pytest

from alberta_framework.utils.experiments import (
    ExperimentConfig,
    SingleRunResult,
    aggregate_metrics,
    get_final_performance,
    run_multi_seed_experiment,
)

pytestmark = pytest.mark.unit


def _run(name: str, seed: int, values: list[float]) -> SingleRunResult:
    return SingleRunResult(
        config_name=name,
        seed=seed,
        metrics_history=[{"loss": value} for value in values],
        final_state=cast(Any, None),
    )


def test_aggregate_rejects_mixed_or_misaligned_runs() -> None:
    with pytest.raises(ValueError, match="same configuration"):
        aggregate_metrics([_run("a", 0, [1.0]), _run("b", 1, [1.0])])
    with pytest.raises(ValueError, match="same number"):
        aggregate_metrics([_run("a", 0, [1.0]), _run("a", 1, [1.0, 2.0])])
    duplicate_seed = [_run("a", 0, [1.0]), _run("a", 0, [2.0])]
    with pytest.raises(ValueError, match="distinct seed"):
        aggregate_metrics(duplicate_seed)


def test_aggregate_rejects_empty_metric_histories() -> None:
    empty = _run("a", 0, [])
    with pytest.raises(ValueError, match="metric step"):
        aggregate_metrics([empty])


def test_multi_seed_rejects_ambiguous_schedules_before_execution() -> None:
    config = ExperimentConfig("a", cast(Any, None), cast(Any, None), 1)
    with pytest.raises(ValueError, match="positive"):
        run_multi_seed_experiment([config], seeds=0)
    with pytest.raises(ValueError, match="unique"):
        run_multi_seed_experiment([config], seeds=[1, 1])
    with pytest.raises(ValueError, match="configs"):
        run_multi_seed_experiment([], seeds=[1])
    with pytest.raises(ValueError, match="names"):
        run_multi_seed_experiment([config, config], seeds=[1])


def test_final_performance_rejects_nonpositive_window() -> None:
    aggregated = aggregate_metrics([_run("a", 0, [1.0, 2.0])])
    with pytest.raises(ValueError, match="window"):
        get_final_performance({"a": aggregated}, metric="loss", window=0)
