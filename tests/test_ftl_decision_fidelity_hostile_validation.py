"""Hostile input, leftover identity, and boundary validation for FTL decision-fidelity records."""

from __future__ import annotations

import numpy as np
import pytest

from alberta_framework.evaluation.ftl_decision_fidelity import (
    BootstrapEstimate,
    ConditionAggregate,
    DecisionFidelityConfig,
    DecisionFidelityReport,
    DecisionMetrics,
    DecisionProbeSet,
    PairedComparison,
    SeedDecisionResult,
)


def _make_dummy_estimate() -> BootstrapEstimate:
    return BootstrapEstimate(
        estimate=0.1,
        lower=0.05,
        upper=0.15,
        confidence_level=0.95,
        resamples=100,
        sample_size=10,
    )


def _make_dummy_metrics(condition: str = "oracle") -> DecisionMetrics:
    return DecisionMetrics(
        condition=condition,
        normalized_regret=0.1,
        domain_a_normalized_regret=0.1,
        domain_b_normalized_regret=0.1,
        oracle_pick_rate=0.9,
        reward_mae=0.05,
        return_mae=0.05,
        normalized_return_mae=0.05,
    )


def _make_dummy_aggregate(condition: str = "oracle") -> ConditionAggregate:
    est = _make_dummy_estimate()
    return ConditionAggregate(
        condition=condition,
        normalized_regret=est,
        domain_a_normalized_regret=est,
        domain_b_normalized_regret=est,
        oracle_pick_rate=est,
        reward_mae=est,
        return_mae=est,
        normalized_return_mae=est,
    )


def test_seed_decision_result_validation() -> None:
    metrics = (_make_dummy_metrics(),)
    result = SeedDecisionResult(seed=42, metrics=metrics)
    assert result.seed == 42
    assert result.metrics == metrics

    # Hostile / invalid seeds
    with pytest.raises(ValueError, match="seed must be an integer"):
        SeedDecisionResult(seed=True, metrics=metrics)

    with pytest.raises(ValueError, match="seed must be an integer"):
        SeedDecisionResult(seed=-1, metrics=metrics)

    with pytest.raises(ValueError, match="seed must be an integer"):
        SeedDecisionResult(seed="42", metrics=metrics)  # type: ignore[arg-type]

    # Hostile / invalid metrics
    with pytest.raises(ValueError, match="metrics must be a tuple of DecisionMetrics"):
        SeedDecisionResult(seed=42, metrics=[_make_dummy_metrics()])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="metrics must be a tuple of DecisionMetrics"):
        SeedDecisionResult(seed=42, metrics=(metrics[0], "invalid"))  # type: ignore[arg-type]


def test_condition_aggregate_validation() -> None:
    est = _make_dummy_estimate()
    agg = ConditionAggregate(
        condition="oracle",
        normalized_regret=est,
        domain_a_normalized_regret=est,
        domain_b_normalized_regret=est,
        oracle_pick_rate=est,
        reward_mae=est,
        return_mae=est,
        normalized_return_mae=est,
    )
    assert agg.condition == "oracle"

    # Hostile / invalid condition string
    with pytest.raises(ValueError, match="condition must be a non-empty string"):
        ConditionAggregate(
            condition="",
            normalized_regret=est,
            domain_a_normalized_regret=est,
            domain_b_normalized_regret=est,
            oracle_pick_rate=est,
            reward_mae=est,
            return_mae=est,
            normalized_return_mae=est,
        )

    with pytest.raises(ValueError, match="condition must be a non-empty string"):
        ConditionAggregate(
            condition=123,  # type: ignore[arg-type]
            normalized_regret=est,
            domain_a_normalized_regret=est,
            domain_b_normalized_regret=est,
            oracle_pick_rate=est,
            reward_mae=est,
            return_mae=est,
            normalized_return_mae=est,
        )

    # Invalid metric estimate type
    with pytest.raises(ValueError, match="normalized_regret must be a BootstrapEstimate"):
        ConditionAggregate(
            condition="oracle",
            normalized_regret=0.1,  # type: ignore[arg-type]
            domain_a_normalized_regret=est,
            domain_b_normalized_regret=est,
            oracle_pick_rate=est,
            reward_mae=est,
            return_mae=est,
            normalized_return_mae=est,
        )


def test_paired_comparison_validation() -> None:
    est = _make_dummy_estimate()
    comp = PairedComparison(
        name="delta_regret",
        definition="model - oracle",
        estimate=est,
    )
    assert comp.name == "delta_regret"
    assert comp.definition == "model - oracle"
    assert comp.estimate == est

    with pytest.raises(ValueError, match="name must be a non-empty string"):
        PairedComparison(name="", definition="model - oracle", estimate=est)

    with pytest.raises(ValueError, match="name must be a non-empty string"):
        PairedComparison(name=None, definition="model - oracle", estimate=est)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="definition must be a non-empty string"):
        PairedComparison(name="delta", definition="", estimate=est)

    with pytest.raises(ValueError, match="estimate must be a BootstrapEstimate"):
        PairedComparison(name="delta", definition="def", estimate=0.5)  # type: ignore[arg-type]


def test_decision_fidelity_report_validation() -> None:
    config = DecisionFidelityConfig()
    seeds = (42, 43)
    seed_results = (
        SeedDecisionResult(seed=42, metrics=(_make_dummy_metrics(),)),
        SeedDecisionResult(seed=43, metrics=(_make_dummy_metrics(),)),
    )
    aggregates = (_make_dummy_aggregate(),)
    comparisons = (
        PairedComparison(
            name="delta",
            definition="model - oracle",
            estimate=_make_dummy_estimate(),
        ),
    )

    report = DecisionFidelityReport(
        config=config,
        seeds=seeds,
        seed_results=seed_results,
        aggregates=aggregates,
        comparisons=comparisons,
    )
    assert report.config == config
    assert report.seeds == seeds

    max_seed_result = SeedDecisionResult(seed=2**31 - 1, metrics=(_make_dummy_metrics(),))
    max_seed_report = DecisionFidelityReport(
        config=config,
        seeds=(2**31 - 1,),
        seed_results=(max_seed_result,),
        aggregates=aggregates,
        comparisons=comparisons,
    )
    assert max_seed_report.seeds == (2**31 - 1,)

    with pytest.raises(ValueError, match="config must be a DecisionFidelityConfig"):
        DecisionFidelityReport(
            config="invalid",  # type: ignore[arg-type]
            seeds=seeds,
            seed_results=seed_results,
            aggregates=aggregates,
            comparisons=comparisons,
        )

    with pytest.raises(ValueError, match="seeds must be a tuple of non-negative int32 seeds"):
        DecisionFidelityReport(
            config=config,
            seeds=[42, 43],  # type: ignore[arg-type]
            seed_results=seed_results,
            aggregates=aggregates,
            comparisons=comparisons,
        )

    with pytest.raises(ValueError, match="seeds must be a tuple of non-negative int32 seeds"):
        DecisionFidelityReport(
            config=config,
            seeds=(42, -1),
            seed_results=seed_results,
            aggregates=aggregates,
            comparisons=comparisons,
        )

    with pytest.raises(ValueError, match="seed_results must be a tuple of SeedDecisionResult"):
        DecisionFidelityReport(
            config=config,
            seeds=seeds,
            seed_results=[seed_results[0]],  # type: ignore[arg-type]
            aggregates=aggregates,
            comparisons=comparisons,
        )


def _make_dummy_probe_set(*, seed: object = 42) -> DecisionProbeSet:
    return DecisionProbeSet(
        seed=seed,  # type: ignore[arg-type]
        initial_observations=np.zeros((1, 2), dtype=np.float32),
        goals=np.zeros((1, 2), dtype=np.float32),
        domain_indices=np.zeros((1,), dtype=np.int64),
        action_sequences=np.zeros((1, 1, 1), dtype=np.float32),
        true_next_observations=np.zeros((1, 1, 1, 2), dtype=np.float64),
        true_rewards=np.zeros((1, 1, 1), dtype=np.float64),
        true_returns=np.zeros((1, 1), dtype=np.float64),
    )


def test_decision_probe_set_rejects_leftover_seed_identities() -> None:
    probes = _make_dummy_probe_set(seed=42)
    assert probes.seed == 42

    with pytest.raises(ValueError, match="seed must be an integer"):
        _make_dummy_probe_set(seed=True)

    with pytest.raises(ValueError, match="seed must be an integer"):
        _make_dummy_probe_set(seed=-1)

    with pytest.raises(ValueError, match="seed must be an integer"):
        _make_dummy_probe_set(seed="42")
