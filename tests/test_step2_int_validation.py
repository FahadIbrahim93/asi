"""Host-boundary leftover-identity gates for public Step 2 smoke records."""

from __future__ import annotations

import json

import pytest

from alberta_framework.steps.step2 import (
    Step2AssociativeConfig,
    Step2AssociativeSmokeResult,
    Step2KernelConfig,
    Step2SmokeResult,
)

pytestmark = pytest.mark.unit


def _legal_step2_smoke_result(**overrides: object) -> Step2SmokeResult:
    payload: dict[str, object] = {
        "config": Step2KernelConfig(),
        "steps": 8,
        "seed": 0,
        "final_window_mse": 0.25,
        "metrics_shape": (8, 2),
        "finite": True,
        "learner_config": {"ok": True},
    }
    payload.update(overrides)
    return Step2SmokeResult(**payload)  # type: ignore[arg-type]


def _legal_step2_associative_smoke_result(
    **overrides: object,
) -> Step2AssociativeSmokeResult:
    payload: dict[str, object] = {
        "config": Step2AssociativeConfig(),
        "steps": 8,
        "seed": 0,
        "initial_window_nll": 1.5,
        "final_window_nll": 0.75,
        "metrics_shape": (8,),
        "finite": True,
        "learner_config": {"ok": True},
    }
    payload.update(overrides)
    return Step2AssociativeSmokeResult(**payload)  # type: ignore[arg-type]


def test_step2_smoke_result_rejects_leftover_identities() -> None:
    """Public Step 2 smoke records must not keep leftover bool/NaN identities."""

    with pytest.raises(ValueError, match="steps"):
        _legal_step2_smoke_result(steps=True)
    with pytest.raises(ValueError, match="steps"):
        _legal_step2_smoke_result(steps=float("nan"))
    with pytest.raises(ValueError, match="seed"):
        _legal_step2_smoke_result(seed=True)
    with pytest.raises(ValueError, match="finite"):
        _legal_step2_smoke_result(finite=1)
    with pytest.raises(ValueError, match="final_window_mse"):
        _legal_step2_smoke_result(final_window_mse=True)
    with pytest.raises(ValueError, match="final_window_mse"):
        _legal_step2_smoke_result(final_window_mse=float("nan"))

    legal = _legal_step2_smoke_result()
    dumped = json.dumps(
        {
            "steps": legal.steps,
            "seed": legal.seed,
            "finite": legal.finite,
            "final_window_mse": legal.final_window_mse,
        },
        allow_nan=False,
    )
    assert '"steps": 8' in dumped
    assert '"seed": 0' in dumped
    assert '"finite": true' in dumped
    assert '"final_window_mse": 0.25' in dumped
    assert '"steps": true' not in dumped
    assert '"seed": true' not in dumped
    assert '"finite": 1' not in dumped
    assert '"final_window_mse": true' not in dumped


def test_step2_associative_smoke_result_rejects_leftover_identities() -> None:
    """Public Step 2 associative smoke records must not keep leftover identities."""

    with pytest.raises(ValueError, match="steps"):
        _legal_step2_associative_smoke_result(steps=True)
    with pytest.raises(ValueError, match="seed"):
        _legal_step2_associative_smoke_result(seed=True)
    with pytest.raises(ValueError, match="finite"):
        _legal_step2_associative_smoke_result(finite=1)
    with pytest.raises(ValueError, match="initial_window_nll"):
        _legal_step2_associative_smoke_result(initial_window_nll=True)
    with pytest.raises(ValueError, match="final_window_nll"):
        _legal_step2_associative_smoke_result(final_window_nll=float("nan"))

    legal = _legal_step2_associative_smoke_result()
    dumped = json.dumps(
        {
            "steps": legal.steps,
            "seed": legal.seed,
            "finite": legal.finite,
            "initial_window_nll": legal.initial_window_nll,
            "final_window_nll": legal.final_window_nll,
        },
        allow_nan=False,
    )
    assert '"steps": 8' in dumped
    assert '"finite": true' in dumped
    assert '"steps": true' not in dumped
    assert '"finite": 1' not in dumped
