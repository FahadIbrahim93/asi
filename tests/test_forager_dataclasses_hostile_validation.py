"""Hostile input and boundary validation for Forager benchmark dataclasses."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.forager import (
    ForagerFeatureState,
    PaperBaseline,
    PaperReferenceTarget,
)


def test_forager_feature_state_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="last_action must be an integer"):
        ForagerFeatureState(
            last_action=True,
            last_reward=0.0,
            reward_traces=(),
        )

    with pytest.raises(ValueError, match="last_reward must be a finite float"):
        ForagerFeatureState(
            last_action=0,
            last_reward=float("nan"),
            reward_traces=(),
        )

    with pytest.raises(ValueError, match="reward_traces must be a tuple"):
        ForagerFeatureState(
            last_action=0,
            last_reward=0.0,
            reward_traces=[],  # type: ignore[arg-type]
        )


def test_paper_baseline_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="name must be a non-empty string"):
        PaperBaseline(
            name="",
            family="family_1",
            role="learning_baseline",
            state_construction="raw",
            selected_hyperparameters={},
            in_tree_implementation=True,
            source="paper",
        )


def test_paper_baseline_role_rejects_before_comparison_hook() -> None:
    calls = 0

    class Hostile:
        def __eq__(self, _other: object) -> bool:
            nonlocal calls
            calls += 1
            raise AssertionError("comparison hook reached")

    with pytest.raises(ValueError, match="role must be an exact string"):
        PaperBaseline(
            name="base",
            family="family",
            role=Hostile(),  # type: ignore[arg-type]
            state_construction="raw",
            selected_hyperparameters={},
            in_tree_implementation=True,
            source="paper",
        )
    assert calls == 0

    with pytest.raises(ValueError, match="role is invalid"):
        PaperBaseline(
            name="base",
            family="family_1",
            role="invalid_role",  # type: ignore[arg-type]
            state_construction="raw",
            selected_hyperparameters={},
            in_tree_implementation=True,
            source="paper",
        )

    with pytest.raises(ValueError, match="in_tree_implementation must be a boolean"):
        PaperBaseline(
            name="base",
            family="family_1",
            role="learning_baseline",
            state_construction="raw",
            selected_hyperparameters={},
            in_tree_implementation="true",  # type: ignore[arg-type]
            source="paper",
        )


def test_paper_reference_target_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="central_estimate must be finite"):
        PaperReferenceTarget(
            method="method_1",
            metric="metric_1",
            central_estimate=float("inf"),
        )

    with pytest.raises(ValueError, match="privileged must be a boolean"):
        PaperReferenceTarget(
            method="method_1",
            metric="metric_1",
            central_estimate=1.0,
            privileged="no",  # type: ignore[arg-type]
        )


def test_numeric_records_reject_before_hostile_metaclass_dispatch() -> None:
    class HostileType(type):
        calls = 0

        def __hash__(cls) -> int:  # pragma: no cover - must not execute
            type(cls).calls += 1
            raise AssertionError("metaclass hash hook must not run")

        def __eq__(cls, _other: object) -> bool:  # pragma: no cover - must not execute
            type(cls).calls += 1
            raise AssertionError("metaclass equality hook must not run")

    class HostileValue(metaclass=HostileType):
        pass

    hostile = HostileValue()
    checks = (
        lambda: ForagerFeatureState(last_action=0, last_reward=hostile, reward_traces=()),
        lambda: ForagerFeatureState(last_action=0, last_reward=0.0, reward_traces=(hostile,)),
        lambda: PaperReferenceTarget(
            method="method_1",
            metric="metric_1",
            central_estimate=hostile,
        ),
    )
    for check in checks:
        with pytest.raises(ValueError):
            check()
    assert HostileType.calls == 0
