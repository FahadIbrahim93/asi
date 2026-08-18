"""Leftover-identity gates for multiagent AggregateEvidence records."""

from __future__ import annotations

import numpy as np
import pytest

from alberta_framework.evaluation.continual_multiagent import (
    AggregateEvidence,
    BootstrapInterval,
)


def _interval(estimate: float = 0.1) -> BootstrapInterval:
    return BootstrapInterval(
        estimate=estimate,
        lower=0.0,
        upper=0.2,
        confidence_level=0.95,
        resamples=10,
        sample_size=2,
    )


def _legal(**overrides: object) -> AggregateEvidence:
    payload: dict[str, object] = {
        "seeds": (1, 2),
        "frozen_prequential_reward": 0.1,
        "learner_only_prequential_reward": 0.2,
        "joint_adaptive_prequential_reward": 0.3,
        "reward_uplift_over_frozen": 0.1,
        "reward_uplift_interval": _interval(),
        "partner_uplift": 0.05,
        "partner_uplift_interval": _interval(0.05),
        "joint_adaptive_phase_rewards": np.zeros(3, dtype=np.float64),
        "joint_adaptive_performance_matrix": np.zeros((3, 2), dtype=np.float64),
        "recurrent_a_probe_reward": 0.0,
        "mean_forgetting": 0.0,
        "max_forgetting": 0.0,
        "mean_interference_forgetting": 0.0,
        "recurrence_recovery_fraction": 1.0,
        "mean_recurrence_recovery_steps": 0.0,
        "mean_stability_gap": 0.0,
        "maximum_update_latency_ms": 1.0,
        "state_scalars": 4,
        "state_bytes": 16,
        "action_scalars_per_step": 2,
        "budgets_identical": True,
        "all_values_finite": True,
    }
    payload.update(overrides)
    return AggregateEvidence(**payload)  # type: ignore[arg-type]


def test_aggregate_evidence_rejects_leftover_identities() -> None:
    """Public aggregate records must not keep leftover bool/seed identities."""

    with pytest.raises(ValueError, match="frozen_prequential_reward"):
        _legal(frozen_prequential_reward=True)
    with pytest.raises(ValueError, match="mean_forgetting"):
        _legal(mean_forgetting=True)
    with pytest.raises(ValueError, match="seeds"):
        _legal(seeds=(True, 2))
    with pytest.raises(ValueError, match="budgets_identical"):
        _legal(budgets_identical=1)
    with pytest.raises(ValueError, match="all_values_finite"):
        _legal(all_values_finite=0)
    with pytest.raises(ValueError, match="state_scalars"):
        _legal(state_scalars=True)

    legal = _legal()
    assert legal.seeds == (1, 2)
    assert type(legal.frozen_prequential_reward) is float
    assert type(legal.budgets_identical) is bool
    assert type(legal.all_values_finite) is bool
    assert legal.budgets_identical is True
    unrecovered = _legal(mean_recurrence_recovery_steps=float("inf"))
    assert unrecovered.mean_recurrence_recovery_steps == float("inf")


def test_aggregate_evidence_binds_nested_intervals_arrays_and_flags() -> None:
    with pytest.raises(ValueError, match="sample_size must match seeds"):
        _legal(seeds=(1,))
    with pytest.raises(ValueError, match="reward interval estimate"):
        _legal(reward_uplift_interval=_interval(0.2))
    with pytest.raises(ValueError, match="partner interval estimate"):
        _legal(partner_uplift_interval=_interval(0.2))
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        _legal(joint_adaptive_phase_rewards=np.zeros(2, dtype=np.float64))
    with pytest.raises(ValueError, match=r"shape \(3, 2\)"):
        _legal(joint_adaptive_performance_matrix=np.zeros((2, 2), dtype=np.float64))
    with pytest.raises(ValueError, match="only finite"):
        _legal(joint_adaptive_phase_rewards=np.full(3, np.nan, dtype=np.float64))
    with pytest.raises(ValueError, match="must match joint_adaptive"):
        _legal(recurrent_a_probe_reward=0.2)
    with pytest.raises(ValueError, match="all_values_finite must match"):
        _legal(all_values_finite=False)


def test_aggregate_evidence_revalidates_replaced_interval_before_attribute_hooks() -> None:
    class HostileInterval:
        def __getattribute__(self, name: str) -> object:  # pragma: no cover - must not run
            raise AssertionError(f"untrusted attribute hook executed: {name}")

    with pytest.raises(ValueError, match="reward_uplift_interval"):
        _legal(reward_uplift_interval=HostileInterval())


def test_aggregate_evidence_owns_read_only_array_snapshots() -> None:
    phase_rewards = np.zeros(3, dtype=np.float64)
    result = _legal(joint_adaptive_phase_rewards=phase_rewards)
    phase_rewards[0] = 1.0
    assert result.joint_adaptive_phase_rewards[0] == 0.0
    assert not result.joint_adaptive_phase_rewards.flags.writeable
    assert not result.joint_adaptive_performance_matrix.flags.writeable
