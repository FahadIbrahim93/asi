"""Tests for UPGDLearner constructor scalar validation."""

import pytest

from alberta_framework.core.upgd import UPGDLearner


def test_upgd_constructor_scalars_reject_booleans_and_nans() -> None:
    # utility_decay
    with pytest.raises(ValueError, match="utility_decay"):
        UPGDLearner(n_heads=1, utility_decay=False)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="utility_decay"):
        UPGDLearner(n_heads=1, utility_decay=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="utility_decay"):
        UPGDLearner(n_heads=1, utility_decay=float("nan"))

    with pytest.raises(ValueError, match="utility_decay"):
        UPGDLearner(n_heads=1, utility_decay=1.0)

    # sparsity
    with pytest.raises(ValueError, match="sparsity"):
        UPGDLearner(n_heads=1, sparsity=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="sparsity"):
        UPGDLearner(n_heads=1, sparsity=False)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="sparsity"):
        UPGDLearner(n_heads=1, sparsity=float("nan"))

    # head_step_size_multiplier
    with pytest.raises(ValueError, match="head_step_size_multiplier"):
        UPGDLearner(n_heads=1, head_step_size_multiplier=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="head_step_size_multiplier"):
        UPGDLearner(n_heads=1, head_step_size_multiplier=float("nan"))

    with pytest.raises(ValueError, match="head_step_size_multiplier"):
        UPGDLearner(n_heads=1, head_step_size_multiplier=float("inf"))

    with pytest.raises(ValueError, match="head_step_size_multiplier"):
        UPGDLearner(n_heads=1, head_step_size_multiplier=0.0)

    # adaptive_kappa_base
    with pytest.raises(ValueError, match="adaptive_kappa_base"):
        UPGDLearner(n_heads=1, adaptive_kappa_base=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="adaptive_kappa_base"):
        UPGDLearner(n_heads=1, adaptive_kappa_base=float("nan"))

    with pytest.raises(ValueError, match="adaptive_kappa_base"):
        UPGDLearner(n_heads=1, adaptive_kappa_base=0.0)

    # readout_adaptive_gate_width
    with pytest.raises(ValueError, match="readout_adaptive_gate_width"):
        UPGDLearner(n_heads=1, readout_adaptive_gate_width=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="readout_adaptive_gate_width"):
        UPGDLearner(n_heads=1, readout_adaptive_gate_width=float("nan"))

    with pytest.raises(ValueError, match="readout_adaptive_gate_width"):
        UPGDLearner(n_heads=1, readout_adaptive_gate_width=0.0)

    # Valid construction
    learner = UPGDLearner(
        n_heads=1,
        utility_decay=0.0,
        sparsity=0.0,
        head_step_size_multiplier=1.0,
        adaptive_kappa_base=0.5,
        readout_adaptive_gate_width=0.3,
    )
    assert learner._utility_decay == 0.0
    assert learner._sparsity == 0.0
