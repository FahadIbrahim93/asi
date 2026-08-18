"""Hostile input and boundary validation for selection plan and hypothesis dataclasses."""

from __future__ import annotations

from typing import Any

import pytest

from alberta_framework.benchmarks.forager_matched_protocol import (
    EvaluationPanel,
    ForagerMatchedProtocolError,
    MatchedHypothesis,
    SelectionGroup,
    SelectionPlan,
    SelectionSlot,
)


def _make_plan(**kwargs: Any) -> SelectionPlan:
    valid_sha = "0" * 64
    base: dict[str, Any] = {
        "metric": "return",
        "metric_implementation_sha256": valid_sha,
        "candidate_universe_sha256": valid_sha,
        "direction": "maximize",
        "statistic": "mean",
        "statistic_implementation_sha256": valid_sha,
        "confidence": 0.95,
        "bootstrap_resamples": 1000,
        "bootstrap_seed": 42,
        "bootstrap_rng_identity": "numpy_generator_pcg64",
        "bootstrap_rng_implementation_sha256": valid_sha,
        "resampling_unit": "candidate_seed_block",
        "quantile_method": "linear",
        "bootstrap_interval": "two_sided_equal_tail",
        "conservative_endpoint": "lower",
        "endpoint_quantile": "(1-confidence)/2",
        "tie_break": "candidate_id_ascending",
        "groups": (
            SelectionGroup(
                selection_group="alberta",
                candidate_ids=("c1", "c2"),
                advance_count=1,
            ),
        ),
    }
    base.update(kwargs)
    return SelectionPlan(**base)


def test_selection_plan_rejects_invalid_inputs() -> None:
    with pytest.raises(ForagerMatchedProtocolError, match="metric must be a non-empty string"):
        _make_plan(metric="")

    with pytest.raises(ForagerMatchedProtocolError, match="direction must be maximize"):
        _make_plan(direction="minimize")

    with pytest.raises(
        ForagerMatchedProtocolError,
        match=r"confidence must be a float in \(0\.0, 1\.0\)",
    ):
        _make_plan(confidence=1.5)

    with pytest.raises(
        ForagerMatchedProtocolError,
        match="bootstrap_resamples must be a positive integer",
    ):
        _make_plan(bootstrap_resamples=0)


def test_evaluation_panel_rejects_invalid_inputs() -> None:
    slot = SelectionSlot("alberta", 1)
    with pytest.raises(ForagerMatchedProtocolError, match="require_complete_blocks must be True"):
        EvaluationPanel(
            selection_slots=(slot,),
            fixed_descriptive_candidate_ids=(),
            alberta_primary_slot=slot,
            primary_nonprivileged_external_baseline_slot=slot,
            require_complete_blocks=False,  # type: ignore[arg-type]
            pairing_failure_policy="fail_closed",
        )


def test_matched_hypothesis_rejects_invalid_inputs() -> None:
    slot = SelectionSlot("alberta", 1)
    with pytest.raises(ForagerMatchedProtocolError, match="paired must be True"):
        MatchedHypothesis(
            hypothesis_id="h1",
            intervention_slot=slot,
            comparator_slot=slot,
            estimand="paired_mean_difference",
            method="paired_sign_flip",
            alternative="greater",
            difference_order="intervention_minus_comparator",
            paired=False,  # type: ignore[arg-type]
        )
