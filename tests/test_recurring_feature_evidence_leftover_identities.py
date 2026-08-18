"""Constructor boundaries for recurring-feature evidence record identities."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from alberta_framework.recurring_feature_gate import (
    PAIRWISE_PROBE_SCOPE,
    PHASE_TASKS,
    TASK_NAMES,
    TASK_PAIRS,
    FeatureMemoryBudget,
    PhaseEvidence,
    RecurringFeatureGateResult,
    RecurringFeatureProtocol,
    RecurringFeatureSeedEvidence,
    RecurringFeatureVariantEvidence,
    TaskRecoveryEvidence,
)


def _phase() -> PhaseEvidence:
    return PhaseEvidence(0, "A", 1, 0.25, None)


def _recovery() -> TaskRecoveryEvidence:
    return TaskRecoveryEvidence("A", None, (2, None))


def _seed() -> RecurringFeatureSeedEvidence:
    return RecurringFeatureSeedEvidence(
        seed=30,
        final_heldout_nmse=(0.1, 0.2, 0.3, math.inf),
        active_pairs=((0, 1),),
        candidate_pairs=((0, 1), (2, 3)),
        phase_evidence=(_phase(),),
        task_recovery=(_recovery(),),
        steps_seen=1,
    )


def _variant() -> RecurringFeatureVariantEvidence:
    return RecurringFeatureVariantEvidence("retained", 0.999, (_seed(),))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("phase_index", True),
        ("task", True),
        ("occurrence", True),
        ("prequential_nmse", True),
        ("recovery_steps", True),
    ),
)
def test_phase_evidence_rejects_leftover_identities(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        replace(_phase(), **{field: value})


def test_task_recovery_rejects_leftover_identities() -> None:
    with pytest.raises(ValueError):
        replace(_recovery(), acquisition_steps=True)
    with pytest.raises(ValueError):
        replace(_recovery(), recurrence_steps=(True,))


def test_seed_evidence_rejects_nested_leftover_identities() -> None:
    with pytest.raises(ValueError):
        replace(_seed(), seed=True)
    with pytest.raises(ValueError):
        replace(_seed(), final_heldout_nmse=(True,))
    with pytest.raises(ValueError):
        replace(_seed(), active_pairs=((True, 1),))


def test_variant_and_result_reject_leftover_string_and_float_identities() -> None:
    with pytest.raises(ValueError):
        replace(_variant(), name=True)
    with pytest.raises(ValueError):
        replace(_variant(), utility_retention_decay=True)
    result = RecurringFeatureGateResult(
        protocol=RecurringFeatureProtocol(),
        memory_budget=FeatureMemoryBudget(3, 15, 4),
        retained=_variant(),
        no_retention=RecurringFeatureVariantEvidence("no_retention", None, (_seed(),)),
        scope=PAIRWISE_PROBE_SCOPE,
    )
    with pytest.raises(ValueError):
        replace(result, scope=True)


def test_unrecovered_and_degenerate_nmse_sentinels_remain_legal() -> None:
    phase = PhaseEvidence(0, "A", 1, math.inf, None)
    recovery = TaskRecoveryEvidence("A", None, (None,))
    seed = replace(
        _seed(),
        final_heldout_nmse=(math.inf,),
        phase_evidence=(phase,),
        task_recovery=(recovery,),
    )
    assert math.isinf(seed.final_heldout_nmse[0])
    assert seed.phase_evidence[0].recovery_steps is None


def _decision_result() -> RecurringFeatureGateResult:
    protocol = RecurringFeatureProtocol(heldout_samples=512)
    occurrences = {task: 0 for task in TASK_NAMES}
    phases = []
    for index, task in enumerate(PHASE_TASKS):
        occurrences[task] += 1
        phases.append(PhaseEvidence(index, task, occurrences[task], 0.25, None))
    recoveries = tuple(
        TaskRecoveryEvidence(task, None, (None,) * (occurrences[task] - 1))
        for task in TASK_NAMES
    )
    seed = RecurringFeatureSeedEvidence(
        seed=30,
        final_heldout_nmse=(0.1, 0.1, 0.1, 1.0),
        active_pairs=TASK_PAIRS[:3],
        candidate_pairs=tuple(
            (left, right)
            for left in range(protocol.feature_dim)
            for right in range(left + 1, protocol.feature_dim)
        ),
        phase_evidence=tuple(phases),
        task_recovery=recoveries,
        steps_seen=protocol.total_steps,
    )
    return RecurringFeatureGateResult(
        protocol=protocol,
        memory_budget=FeatureMemoryBudget(3, 15, 4),
        retained=RecurringFeatureVariantEvidence("retained", 0.999, (seed,)),
        no_retention=RecurringFeatureVariantEvidence("no_retention", None, (seed,)),
    )


def test_decision_revalidates_forged_nested_evidence() -> None:
    result = _decision_result()
    phase = result.retained.seeds[0].phase_evidence[0]
    object.__setattr__(phase, "prequential_nmse", True)

    with pytest.raises(ValueError, match="prequential_nmse"):
        result.decision()


def test_decision_revalidates_forged_protocol_before_diagnostics() -> None:
    result = _decision_result()
    object.__setattr__(result.protocol, "feature_dim", True)

    with pytest.raises(ValueError, match="feature_dim must be a built-in integer"):
        result.decision()


def test_decision_revalidates_and_cross_binds_memory_budget() -> None:
    with pytest.raises(ValueError, match="active_pair_slots"):
        FeatureMemoryBudget(True, 15, 4)

    result = _decision_result()
    object.__setattr__(result.memory_budget, "active_pair_slots", True)
    with pytest.raises(ValueError, match="active_pair_slots"):
        result.decision()

    mismatched = replace(
        _decision_result(),
        memory_budget=FeatureMemoryBudget(2, 15, 4),
    )
    with pytest.raises(ValueError, match="capacities must match"):
        mismatched.decision()
