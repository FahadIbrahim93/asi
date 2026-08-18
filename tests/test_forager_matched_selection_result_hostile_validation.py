"""Hostile input and boundary validation for selection result and sealed validation dataclasses."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.forager_matched_protocol import (
    ForagerMatchedProtocolError,
    ForagerMatchedSelectionResult,
    MultiplicityPolicy,
    RankedSelectionGroup,
    ResolvedHypothesis,
    ResolvedSelectionSlot,
    SealedProtocolValidation,
)


def test_selection_result_rejects_invalid_inputs() -> None:
    valid_sha = "0" * 64
    with pytest.raises(ForagerMatchedProtocolError, match="invalid schema_version"):
        ForagerMatchedSelectionResult(
            schema_version="invalid_schema",  # type: ignore[arg-type]
            open_protocol_sha256=valid_sha,
            selection_plan_sha256=valid_sha,
            tuning_seeds=(1, 2),
            ranked_groups=(RankedSelectionGroup("alberta", ("c1",), valid_sha),),
        )

    with pytest.raises(ForagerMatchedProtocolError, match="tuning_seeds item must be an integer"):
        ForagerMatchedSelectionResult(
            schema_version="alberta.forager_matched_selection_result.v1",
            open_protocol_sha256=valid_sha,
            selection_plan_sha256=valid_sha,
            tuning_seeds=(True,),
            ranked_groups=(RankedSelectionGroup("alberta", ("c1",), valid_sha),),
        )


def test_resolved_hypothesis_rejects_invalid_inputs() -> None:
    with pytest.raises(
        ForagerMatchedProtocolError,
        match="hypothesis_id must be a non-empty string",
    ):
        ResolvedHypothesis(
            hypothesis_id="",
            intervention_candidate_id="c1",
            comparator_candidate_id="c2",
            method="paired_sign_flip",
            alternative="greater",
            difference_order="intervention_minus_comparator",
        )


def test_sealed_protocol_validation_rejects_invalid_inputs() -> None:
    valid_sha = "0" * 64
    slot = ResolvedSelectionSlot("alberta", 1, "c1")
    hyp = ResolvedHypothesis(
        hypothesis_id="h1",
        intervention_candidate_id="c1",
        comparator_candidate_id="c2",
        method="paired_sign_flip",
        alternative="greater",
        difference_order="intervention_minus_comparator",
    )
    with pytest.raises(
        ForagerMatchedProtocolError,
        match="primary_intervention_candidate_id must be non-empty string",
    ):
        SealedProtocolValidation(
            open_protocol_sha256=valid_sha,
            selection_result_sha256=valid_sha,
            resolved_slots=(slot,),
            evaluation_candidate_ids=("c1", "c2"),
            primary_intervention_candidate_id="",
            primary_comparator_candidate_id="c2",
            resolved_hypotheses=(hyp,),
        )


def test_multiplicity_policy_rejects_invalid_inputs() -> None:
    with pytest.raises(
        ForagerMatchedProtocolError,
        match=r"alpha must be a float in \(0\.0, 1\.0\)",
    ):
        MultiplicityPolicy(
            method="holm",
            alpha=1.5,
            hypothesis_ids=("h1",),
            primary_excluded=True,
        )

    with pytest.raises(ForagerMatchedProtocolError, match="primary_excluded must be True"):
        MultiplicityPolicy(
            method="holm",
            alpha=0.05,
            hypothesis_ids=("h1",),
            primary_excluded=False,  # type: ignore[arg-type]
        )
