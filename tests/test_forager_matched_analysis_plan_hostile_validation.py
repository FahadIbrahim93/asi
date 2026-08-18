"""Hostile input and boundary validation for matched analysis plan dataclasses."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.forager_matched_protocol import (
    ForagerMatchedProtocolError,
    MatchedAnalysisPlan,
    PrimaryBootstrapAnalysis,
    SecondarySignFlipAnalysis,
)


def _make_primary(**kwargs: object) -> PrimaryBootstrapAnalysis:
    valid_sha = "0" * 64
    base: dict[str, object] = {
        "method": "paired_percentile_bootstrap_lower_bound",
        "resamples": 1000,
        "seed": 42,
        "confidence": 0.95,
        "primary_margin": 0.0,
        "rng_algorithm": "PCG64",
        "quantile_method": "linear",
        "implementation_sha256": valid_sha,
        "gate": "lower_bound_strictly_greater_than_margin",
    }
    base.update(kwargs)
    return PrimaryBootstrapAnalysis(**base)  # type: ignore[arg-type]


def _make_secondary(**kwargs: object) -> SecondarySignFlipAnalysis:
    valid_sha = "0" * 64
    base: dict[str, object] = {
        "method": "paired_sign_flip",
        "monte_carlo_resamples": 1000,
        "seed": 42,
        "exact_max_pairs": 20,
        "rng_algorithm": "PCG64",
        "implementation_sha256": valid_sha,
        "alternative": "greater",
        "multiplicity_method": "holm",
        "familywise_alpha": 0.05,
    }
    base.update(kwargs)
    return SecondarySignFlipAnalysis(**base)  # type: ignore[arg-type]


def test_primary_bootstrap_analysis_rejects_invalid_inputs() -> None:
    with pytest.raises(
        ForagerMatchedProtocolError,
        match="resamples must be a positive integer",
    ):
        _make_primary(resamples=0)

    with pytest.raises(
        ForagerMatchedProtocolError,
        match=r"confidence must be a float in \(0\.0, 1\.0\)",
    ):
        _make_primary(confidence=1.0)


def test_secondary_sign_flip_analysis_rejects_invalid_inputs() -> None:
    with pytest.raises(ForagerMatchedProtocolError, match="exact_max_pairs must be 20"):
        _make_secondary(exact_max_pairs=10)

    with pytest.raises(
        ForagerMatchedProtocolError,
        match=r"familywise_alpha must be a float in \(0\.0, 1\.0\)",
    ):
        _make_secondary(familywise_alpha=0.0)


def test_matched_analysis_plan_rejects_invalid_inputs() -> None:
    primary = _make_primary()
    secondary = _make_secondary()
    with pytest.raises(ForagerMatchedProtocolError, match="metric must be a non-empty string"):
        MatchedAnalysisPlan(
            metric="",
            metric_implementation_sha256="0" * 64,
            metric_direction="maximize",
            primary=primary,
            secondary=secondary,
        )

    with pytest.raises(ForagerMatchedProtocolError, match="metric_direction must be maximize"):
        MatchedAnalysisPlan(
            metric="return",
            metric_implementation_sha256="0" * 64,
            metric_direction="minimize",  # type: ignore[arg-type]
            primary=primary,
            secondary=secondary,
        )
