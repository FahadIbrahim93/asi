"""Hostile input and boundary validation for Forager comparison dataclasses."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.forager import ForagerBenchmarkSummary, ForagerRunResult
from alberta_framework.benchmarks.forager_results import (
    ForagerComparisonReport,
    ForagerPairedComparison,
)


def _run(*, agent: str, seed: int, reward: float) -> ForagerRunResult:
    return ForagerRunResult(
        agent=agent,
        privileged=False,
        seed=seed,
        steps=1,
        total_reward=reward,
        mean_reward=reward,
        final_window_mean_reward=reward,
        final_ewm_reward=reward,
        mean_ewm_reward=reward,
        fov_last_10pct_ema_auc=reward,
        mean_biome_regret=0.0,
        final_biome_regret=0.0,
        curve_steps=(1,),
        curve_ewm_reward=(reward,),
        curve_window_reward=(reward,),
        duration_s=1.0,
        frames_per_second=1.0,
        environment={"kind": "toy"},
        metric_contract={"metric": "mean_reward"},
        agent_metadata={"name": agent},
    )


def _summary(agent: str, reward: float) -> ForagerBenchmarkSummary:
    runs = (_run(agent=agent, seed=1, reward=reward),)
    return ForagerBenchmarkSummary(
        agent=agent,
        privileged=False,
        seeds=(1,),
        metric="mean_reward",
        mean=reward,
        ci_low=reward,
        ci_high=reward,
        confidence=0.95,
        runs=runs,
    )


def test_forager_paired_comparison_valid_construction() -> None:
    comp = ForagerPairedComparison(
        candidate="cand_1",
        baseline="base_1",
        candidate_privileged=False,
        baseline_privileged=False,
        metric="mean_reward",
        seeds=(1, 2, 3),
        mean_difference=0.5,
        ci_low=0.2,
        ci_high=0.8,
        confidence=0.95,
    )
    assert comp.candidate == "cand_1"
    assert comp.seeds == (1, 2, 3)


def test_forager_paired_comparison_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="candidate must be a non-empty string"):
        ForagerPairedComparison(
            candidate="",
            baseline="base_1",
            candidate_privileged=False,
            baseline_privileged=False,
            metric="mean_reward",
            seeds=(1, 2),
            mean_difference=0.5,
            ci_low=0.2,
            ci_high=0.8,
            confidence=0.95,
        )

    with pytest.raises(ValueError, match="ci_low must not exceed ci_high"):
        ForagerPairedComparison(
            candidate="cand_1",
            baseline="base_1",
            candidate_privileged=False,
            baseline_privileged=False,
            metric="mean_reward",
            seeds=(1,),
            mean_difference=0.5,
            ci_low=0.8,
            ci_high=0.2,
            confidence=0.95,
        )

    with pytest.raises(TypeError, match="candidate_privileged must be an exact boolean"):
        ForagerPairedComparison(
            candidate="cand_1",
            baseline="base_1",
            candidate_privileged=1,  # type: ignore[arg-type]
            baseline_privileged=False,
            metric="mean_reward",
            seeds=(1, 2),
            mean_difference=0.5,
            ci_low=0.2,
            ci_high=0.8,
            confidence=0.95,
        )

    with pytest.raises(ValueError, match="seeds must be unique"):
        ForagerPairedComparison(
            candidate="cand_1",
            baseline="base_1",
            candidate_privileged=False,
            baseline_privileged=False,
            metric="mean_reward",
            seeds=(1, 1),
            mean_difference=0.5,
            ci_low=0.2,
            ci_high=0.8,
            confidence=0.95,
        )

    with pytest.raises(ValueError, match="mean_difference must be a finite float"):
        ForagerPairedComparison(
            candidate="cand_1",
            baseline="base_1",
            candidate_privileged=False,
            baseline_privileged=False,
            metric="mean_reward",
            seeds=(1, 2),
            mean_difference=float("nan"),
            ci_low=0.2,
            ci_high=0.8,
            confidence=0.95,
        )


def test_forager_comparison_report_rejects_invalid_inputs() -> None:
    with pytest.raises(TypeError, match="summaries must be an exact dictionary"):
        ForagerComparisonReport(
            candidate="cand_1",
            metric="mean_reward",
            summaries=None,  # type: ignore[arg-type]
            paired_comparisons=(),
            unpaired_methods=(),
        )


def test_forager_comparison_literals_reject_before_hooks() -> None:
    calls = 0

    class Hostile:
        def __eq__(self, _other: object) -> bool:
            nonlocal calls
            calls += 1
            raise AssertionError("comparison hook reached")

        def __repr__(self) -> str:
            nonlocal calls
            calls += 1
            raise AssertionError("repr hook reached")

    with pytest.raises(ValueError, match="metric is not a valid ForagerMetric"):
        ForagerPairedComparison(
            candidate="candidate",
            baseline="baseline",
            candidate_privileged=False,
            baseline_privileged=False,
            metric=Hostile(),  # type: ignore[arg-type]
            seeds=(0,),
            mean_difference=0.0,
            ci_low=0.0,
            ci_high=0.0,
            confidence=0.95,
        )
    assert calls == 0

    with pytest.raises(TypeError, match="paired_comparisons must be an exact tuple"):
        ForagerComparisonReport(
            candidate="cand_1",
            metric="mean_reward",
            summaries={},
            paired_comparisons=[],  # type: ignore[arg-type]
            unpaired_methods=(),
        )


def test_report_revalidates_nested_summaries_and_comparisons() -> None:
    candidate = _summary("candidate", 1.0)
    baseline = _summary("baseline", 0.0)
    comparison = ForagerPairedComparison(
        candidate="candidate",
        baseline="baseline",
        candidate_privileged=False,
        baseline_privileged=False,
        metric="mean_reward",
        seeds=(1,),
        mean_difference=1.0,
        ci_low=1.0,
        ci_high=1.0,
        confidence=0.95,
    )
    report = ForagerComparisonReport(
        candidate="candidate",
        metric="mean_reward",
        summaries={"candidate": candidate, "baseline": baseline},
        paired_comparisons=(comparison,),
        unpaired_methods=(),
    )
    assert report.paired_comparisons[0].mean_difference == 1.0

    object.__setattr__(candidate, "mean", 2.0)
    with pytest.raises(ValueError, match="mean must reconstruct"):
        ForagerComparisonReport(
            candidate="candidate",
            metric="mean_reward",
            summaries={"candidate": candidate, "baseline": baseline},
            paired_comparisons=(comparison,),
            unpaired_methods=(),
        )
