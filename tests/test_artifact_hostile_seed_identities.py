"""Exact-type hostile-int seed rejection in artifact validators."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from alberta_framework.evaluation.continual_multiagent_artifact import (
    _validate_seed_and_aggregate_consistency,
)
from alberta_framework.evaluation.ftl_decision_artifact import (
    _validate_and_extract_seed_vectors,
)
from alberta_framework.evaluation.recurring_feature_artifact import (
    _extract_seed_metrics,
)


class _HostileInt(int):
    calls = 0

    def _called(self) -> None:
        type(self).calls += 1
        raise AssertionError("hostile integer hook ran")

    __bool__ = _called
    __index__ = _called
    __int__ = _called
    __repr__ = _called
    __str__ = _called

    def __eq__(self, other: object) -> bool:
        del other
        self._called()

    __hash__ = int.__hash__


def _invalid_seeds() -> tuple[object, ...]:
    return (True, np.bool_(False), 4.0, "4", _HostileInt(4))


def _assert_each_rejected(validate: Callable[[object, list[str]], None]) -> None:
    _HostileInt.calls = 0
    for seed in _invalid_seeds():
        errors: list[str] = []
        validate(seed, errors)
        assert any("seed must be an integer" in error for error in errors)
    assert _HostileInt.calls == 0


def test_multiagent_artifact_rejects_hostile_seed_identities() -> None:
    def validate(seed: object, errors: list[str]) -> None:
        _validate_seed_and_aggregate_consistency(
            {
                "seed_summaries": [{"seed": seed, "conditions": {}}],
                "aggregate": {},
                "configuration": {},
            },
            errors,
        )

    _assert_each_rejected(validate)


def test_ftl_decision_artifact_rejects_hostile_seed_identities() -> None:
    def validate(seed: object, errors: list[str]) -> None:
        _validate_and_extract_seed_vectors(
            [{"seed": seed, "conditions": {}}],
            errors,
        )

    _assert_each_rejected(validate)


def test_recurring_feature_artifact_rejects_hostile_seed_identities() -> None:
    def validate(seed: object, errors: list[str]) -> None:
        _extract_seed_metrics(
            [{"seed": seed, "retained": {}, "no_retention": {}}],
            errors,
        )

    _assert_each_rejected(validate)
