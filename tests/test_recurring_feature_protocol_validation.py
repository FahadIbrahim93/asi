"""Tests for RecurringFeatureProtocol, criteria, and seed validation."""

import pytest

from alberta_framework.recurring_feature_gate import (
    RecurringFeatureGateCriteria,
    RecurringFeatureProtocol,
    run_recurring_feature_gate,
)


def test_protocol_validate_rejects_booleans_and_nans() -> None:
    # Boolean integers
    with pytest.raises(ValueError, match="feature_dim"):
        RecurringFeatureProtocol(feature_dim=True).validate()  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="steps_per_phase"):
        RecurringFeatureProtocol(steps_per_phase=True).validate()  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="active_pair_budget"):
        RecurringFeatureProtocol(active_pair_budget=True).validate()  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="heldout_samples"):
        RecurringFeatureProtocol(heldout_samples=True).validate()  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="recovery_window"):
        RecurringFeatureProtocol(recovery_window=True).validate()  # type: ignore[arg-type]

    # Boolean / NaN floats
    with pytest.raises(ValueError, match="target_amplitude"):
        RecurringFeatureProtocol(target_amplitude=True).validate()  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="target_amplitude"):
        RecurringFeatureProtocol(target_amplitude=float("nan")).validate()

    with pytest.raises(ValueError, match="step_size_output"):
        RecurringFeatureProtocol(step_size_output=True).validate()  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="step_size_output"):
        RecurringFeatureProtocol(step_size_output=float("nan")).validate()

    with pytest.raises(ValueError, match="utility_decay"):
        RecurringFeatureProtocol(utility_decay=True).validate()  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="utility_decay"):
        RecurringFeatureProtocol(utility_decay=float("nan")).validate()

    with pytest.raises(ValueError, match="recovery_nmse_threshold"):
        RecurringFeatureProtocol(recovery_nmse_threshold=True).validate()  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="recovery_nmse_threshold"):
        RecurringFeatureProtocol(recovery_nmse_threshold=float("nan")).validate()

    # Boolean flags
    with pytest.raises(ValueError, match="refresh_candidates"):
        RecurringFeatureProtocol(refresh_candidates=1).validate()  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="use_obgd"):
        RecurringFeatureProtocol(use_obgd=1).validate()  # type: ignore[arg-type]

    # Valid protocol
    RecurringFeatureProtocol().validate()


def test_criteria_rejects_booleans_and_nans() -> None:
    with pytest.raises(ValueError, match="minimum_seeds"):
        RecurringFeatureGateCriteria(minimum_seeds=False)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="minimum_heldout_samples"):
        RecurringFeatureGateCriteria(minimum_heldout_samples=False)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="minimum_retained_all_critical_rate"):
        RecurringFeatureGateCriteria(minimum_retained_all_critical_rate=float("nan"))

    with pytest.raises(ValueError, match="require_recurrence_faster_than_acquisition"):
        RecurringFeatureGateCriteria(require_recurrence_faster_than_acquisition=1)  # type: ignore[arg-type]

    # Valid criteria
    criteria = RecurringFeatureGateCriteria()
    assert criteria.minimum_seeds == 30


def test_run_recurring_feature_gate_rejects_boolean_seeds() -> None:
    protocol = RecurringFeatureProtocol(steps_per_phase=10, recovery_window=5, heldout_samples=16)
    with pytest.raises(ValueError, match="seed"):
        run_recurring_feature_gate((True,), protocol=protocol)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="seed"):
        run_recurring_feature_gate((False,), protocol=protocol)  # type: ignore[arg-type]
