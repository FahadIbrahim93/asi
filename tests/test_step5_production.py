"""Tests for leftover identities on public Step 5 smoke records."""

from __future__ import annotations

import json

import pytest

from alberta_framework.steps.step5 import Step5AverageRewardTDConfig, Step5SmokeResult


def _legal_step5_smoke_result(**overrides: object) -> Step5SmokeResult:
    payload: dict[str, object] = {
        "config": Step5AverageRewardTDConfig(),
        "steps": 8,
        "seed": 0,
        "predictions_shape": (8,),
        "td_errors_shape": (8,),
        "average_rewards_shape": (8,),
        "finite": True,
        "learner_config": {"ok": True},
    }
    payload.update(overrides)
    return Step5SmokeResult(**payload)  # type: ignore[arg-type]


def test_step5_smoke_result_rejects_leftover_identities() -> None:
    """Public Step 5 smoke records must not keep leftover bool/int identities."""

    with pytest.raises(ValueError, match="steps"):
        _legal_step5_smoke_result(steps=True)
    with pytest.raises(ValueError, match="steps"):
        _legal_step5_smoke_result(steps=float("nan"))
    with pytest.raises(ValueError, match="seed"):
        _legal_step5_smoke_result(seed=True)
    with pytest.raises(ValueError, match="finite"):
        _legal_step5_smoke_result(finite=1)

    legal = _legal_step5_smoke_result()
    dumped = json.dumps(
        {
            "steps": legal.steps,
            "seed": legal.seed,
            "finite": legal.finite,
        },
        allow_nan=False,
    )
    assert '"steps": 8' in dumped
    assert '"seed": 0' in dumped
    assert '"finite": true' in dumped
    assert '"steps": true' not in dumped
    assert '"seed": true' not in dumped
    assert '"finite": 1' not in dumped
