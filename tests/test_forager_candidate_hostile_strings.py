"""Hostile string gate for forager comparison candidate."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.forager import ForagerRunResult
from alberta_framework.benchmarks.forager_results import build_forager_comparison_report

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

    def __str__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile str must not run")


def _run(*, agent: str = "toy", seed: int = 0) -> ForagerRunResult:
    return ForagerRunResult(
        agent=agent,
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
        agent_metadata={"name": agent, "privileged": False},
    )


def test_candidate_rejects_hostile_before_membership() -> None:
    runs = [_run(agent="toy", seed=0), _run(agent="toy", seed=1)]
    hostile = _HostileStr("evil")
    _HostileStr.calls = 0
    with pytest.raises(ValueError, match="exact string"):
        build_forager_comparison_report(runs, candidate=hostile)  # type: ignore[arg-type]
    assert _HostileStr.calls == 0

    with pytest.raises(ValueError, match="candidate is absent"):
        build_forager_comparison_report(runs, candidate="missing")
    assert _HostileStr.calls == 0

    report = build_forager_comparison_report(runs, candidate="toy")
    assert report.candidate == "toy"


def test_candidate_and_runs_are_admitted_before_traversal() -> None:
    class HostileRuns(list[ForagerRunResult]):
        def __iter__(self):  # type: ignore[no-untyped-def]  # pragma: no cover
            raise AssertionError("hostile run traversal")

    hostile = _HostileStr("toy")
    with pytest.raises(ValueError, match="exact string"):
        build_forager_comparison_report(HostileRuns(), candidate=hostile)  # type: ignore[arg-type]

    run = _run()
    object.__setattr__(run, "agent", hostile)
    _HostileStr.calls = 0
    with pytest.raises(ValueError, match="agent must be a non-empty string"):
        build_forager_comparison_report([run], candidate="toy")
    assert _HostileStr.calls == 0
