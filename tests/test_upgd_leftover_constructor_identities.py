"""Leftover UPGD constructor identities must not persist as bool or NaN."""

from __future__ import annotations

import math

import pytest

from alberta_framework.core.upgd import UPGDLearner


@pytest.mark.parametrize(
    "kwargs",
    [
        {"utility_decay": False},
        {"utility_decay": True},
        {"utility_decay": float("nan")},
        {"sparsity": True},
        {"sparsity": False},
        {"sparsity": float("nan")},
        {"head_step_size_multiplier": True},
        {"head_step_size_multiplier": float("nan")},
        {"head_step_size_multiplier": float("inf")},
        {"adaptive_kappa_base": True},
        {"adaptive_kappa_base": float("nan")},
        {"readout_adaptive_gate_width": True},
        {"readout_adaptive_gate_width": float("nan")},
    ],
)
def test_leftover_upgd_identities_reject_bool_and_nonfinite(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        UPGDLearner(n_heads=2, hidden_sizes=(4,), **kwargs)


def test_legal_leftover_upgd_identities_stay_canonical() -> None:
    learner = UPGDLearner(
        n_heads=2,
        hidden_sizes=(4,),
        utility_decay=0.0,
        sparsity=0.0,
        head_step_size_multiplier=2,
        adaptive_kappa_base=0.5,
        readout_adaptive_gate_width=1.0,
    )
    assert type(learner._utility_decay) is float and learner._utility_decay == 0.0
    assert type(learner._sparsity) is float and learner._sparsity == 0.0
    assert (
        type(learner._head_step_size_multiplier) is float
        and learner._head_step_size_multiplier == 2.0
    )
    assert type(learner._adaptive_kappa_base) is float
    assert learner._adaptive_kappa_base == 0.5
    assert type(learner._readout_adaptive_gate_width) is float
    assert learner._readout_adaptive_gate_width == 1.0
    assert math.isfinite(learner._head_step_size_multiplier)
