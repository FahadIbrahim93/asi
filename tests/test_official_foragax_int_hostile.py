"""Hostile integer validation for official foragax."""

from __future__ import annotations

from pathlib import Path

import pytest

from alberta_framework.benchmarks.official_foragax import (
    OfficialForagaxBatchRunRequest,
    OfficialForagaxRunRequest,
    OfficialForagaxValidationError,
    _semantic_environment,
)

pytestmark = pytest.mark.unit


class _HostileInt(int):
    calls = 0

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile eq")

    def __lt__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile lt")

    def __le__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile le")


def test_reward_delay_rejects_hostile_before_lt() -> None:
    _HostileInt.calls = 0
    hostile = _HostileInt(0)
    with pytest.raises(OfficialForagaxValidationError, match="reward_delay is invalid"):
        _semantic_environment(
            {
                "environment": {
                    "env_id": "ForagaxTwoBiomeLarge-v1",
                    "reward_delay": hostile,
                }
            }
        )
    assert _HostileInt.calls == 0


def test_batch_indices_rejects_hostile_before_range() -> None:
    _HostileInt.calls = 0
    hostile = _HostileInt(1)
    with pytest.raises(OfficialForagaxValidationError, match="batch indices"):
        OfficialForagaxBatchRunRequest(
            repository=Path("repo"),
            execution_commit="0" * 40,
            config_path=Path("config.json"),
            interpreter=Path("python"),
            output_dir=Path("out"),
            indices=(0, hostile),  # type: ignore[arg-type]
        )
    assert _HostileInt.calls == 0


def test_batch_entry_rejects_hostile_before_eq() -> None:
    _HostileInt.calls = 0
    hostile = _HostileInt(1)
    with pytest.raises(OfficialForagaxValidationError, match="index must be"):
        OfficialForagaxRunRequest(
            repository=Path("repo"),
            execution_commit="0" * 40,
            config_path=Path("config.json"),
            interpreter=Path("python"),
            output_dir=Path("out"),
            index=hostile,  # type: ignore[arg-type]
        )
    assert _HostileInt.calls == 0


def test_hostile_not_in_error_message() -> None:
    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    try:
        if type(hostile) is not int:
            raise ValueError("must be an integer")
    except ValueError as exc:
        assert "!r" not in str(exc)
        assert _HostileInt.calls == 0
