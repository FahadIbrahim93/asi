"""Hostile integer validation for foragax open screen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from alberta_framework.benchmarks.foragax_open_screen import (
    ProcessCapture,
    ScreenError,
    _validate_config_payload,
)

pytestmark = pytest.mark.unit


class _HostileInt(int):
    calls = 0

    def __lt__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile lt")

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile eq")


def test_ppo_rollout_rejects_hostile_before_eq() -> None:
    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    with pytest.raises(ScreenError, match="returncode must be an integer"):
        ProcessCapture(returncode=hostile, stdout=b"", stderr=b"")  # type: ignore[arg-type]
    assert _HostileInt.calls == 0


def test_ppo_schedule_rejects_negative_factors_before_multiplication(tmp_path: Path) -> None:
    payload = {
        "problem": "Foragax",
        "total_steps": 10,
        "agent": "PPO",
        "metaParameters": {
            "environment": {"env_id": "env", "aperture_size": 9},
            "experiment": {"seed_offset": 0},
            "rollout_steps": -2,
            "num_updates": -5,
        },
    }
    raw = json.dumps(payload).encode()
    path = tmp_path / "config.json"
    path.write_bytes(raw)
    with pytest.raises(ScreenError, match="rollout_steps is not explicit"):
        _validate_config_payload(
            path,
            hashlib.sha256(raw).hexdigest(),
            "config.json",
            {"env_id": "env", "aperture_size": 9},
            10,
            "1.0",
        )
