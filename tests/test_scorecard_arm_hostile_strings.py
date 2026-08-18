"""Hostile string gates for scorecard arm and environment selection."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.reference_life_scorecard import (
    build_development_plan,
    parameter_change_check,
)

pytestmark = pytest.mark.unit


class _HostileStr(str):
    calls = 0

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile eq must not run")

    def __hash__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile hash must not run")

    def __bool__(self) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile bool must not run")

    def __repr__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile repr must not run")

    def __str__(self) -> str:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile str must not run")


def test_arm_definition_rejects_hostile_before_membership() -> None:
    plan = build_development_plan()
    hostile = _HostileStr("sarsa")
    _HostileStr.calls = 0
    with pytest.raises(ValueError, match="exact string"):
        plan.arm_definition(hostile)  # type: ignore[arg-type]
    assert _HostileStr.calls == 0

    _HostileStr.calls = 0
    with pytest.raises(ValueError, match="unsupported scorecard arm"):
        plan.arm_definition("unknown_arm_xyz")
    assert _HostileStr.calls == 0
    assert plan.arm_definition("sarsa")["arm"] == "sarsa"


def test_protocol_rejects_hostile_before_membership() -> None:
    plan = build_development_plan()
    hostile = _HostileStr("switching_two_state")
    _HostileStr.calls = 0
    with pytest.raises(ValueError, match="exact string"):
        plan.protocol(hostile)  # type: ignore[arg-type]
    assert _HostileStr.calls == 0

    with pytest.raises(ValueError, match="unsupported environment"):
        plan.protocol("unknown_env")

    payload = plan.protocol("switching_two_state")
    assert isinstance(payload, dict)


def test_parameter_change_check_rejects_hostile_before_in() -> None:
    hostile = _HostileStr("prototype")
    _HostileStr.calls = 0
    with pytest.raises(ValueError, match="exact string"):
        parameter_change_check(hostile, 0)  # type: ignore[arg-type]
    assert _HostileStr.calls == 0

    assert parameter_change_check("prototype", 1)["passed"] is True
    assert parameter_change_check("random", 0)["passed"] is True
