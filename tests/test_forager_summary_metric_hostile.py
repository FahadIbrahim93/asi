"""Hostile string gate for Forager summary metric."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.forager import ForagerRunResult, summarize_forager_runs

pytestmark = pytest.mark.unit


class _HostileStr(str):
    calls = 0

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile eq must not run")

    def __hash__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile hash must not run")

    def __repr__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile repr must not run")


def _run(*, seed: int = 0) -> ForagerRunResult:
    return ForagerRunResult(
        agent="toy",
        privileged=False,
        seed=seed,
        steps=2,
        total_reward=0.0,
        mean_reward=0.0,
        final_window_mean_reward=0.0,
        final_ewm_reward=0.0,
        mean_ewm_reward=0.0,
        fov_last_10pct_ema_auc=0.0,
        mean_biome_regret=0.0,
        final_biome_regret=0.0,
        curve_steps=(1, 2),
        curve_ewm_reward=(0.0, 0.0),
        curve_window_reward=(0.0, 0.0),
        duration_s=1.0,
        frames_per_second=1.0,
        environment={"kind": "toy"},
        metric_contract={"metric": "mean_reward"},
        agent_metadata={"name": "toy", "privileged": False},
    )


def test_summarize_rejects_hostile_metric_before_in_and_repr() -> None:
    runs = [_run(seed=0), _run(seed=1)]
    hostile = _HostileStr("evil_metric")
    _HostileStr.calls = 0
    with pytest.raises(ValueError, match="unsupported Forager summary metric"):
        summarize_forager_runs(runs, metric=hostile)  # type: ignore[arg-type]
    assert _HostileStr.calls == 0

    with pytest.raises(ValueError, match="unsupported Forager summary metric"):
        summarize_forager_runs(runs, metric="unknown_metric")  # type: ignore[arg-type]

    # valid still works
    summary = summarize_forager_runs(runs, metric="mean_reward")
    assert summary.metric == "mean_reward"
