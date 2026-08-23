"""Cardinality bounds for working-memory and fixed-trace decay rate sequences.

Peer modules bound configuration-item cardinalities before per-item walks
(_MAX_HISTORY_CONFIGURATION_ITEMS in history_features.py, _MAX_HORDE_DEMONS in
types.py), but WorkingMemoryConfig and FixedTraceStateBuilderConfig walked
decay-rate sequences of any size before validation. Issue #2220.
"""

from __future__ import annotations

import pytest

from alberta_framework.core import working_memory
from alberta_framework.core.state_builder import FixedTraceStateBuilderConfig
from alberta_framework.core.working_memory import WorkingMemoryConfig

pytestmark = pytest.mark.unit

_MAX = 4096


def test_max_working_memory_decay_rates_is_4096() -> None:
    assert working_memory._MAX_WORKING_MEMORY_DECAY_RATES == _MAX


def test_config_rejects_oversized_decay_rates() -> None:
    with pytest.raises(ValueError, match="observation_decay_rates"):
        WorkingMemoryConfig(
            observation_dim=2,
            observation_decay_rates=(0.5,) * (_MAX + 1),
        )


def test_config_accepts_boundary_decay_rates() -> None:
    cfg = WorkingMemoryConfig(
        observation_dim=2,
        observation_decay_rates=(0.5,) * _MAX,
    )
    assert len(cfg.observation_decay_rates) == _MAX


def _wm_payload(obs_rates: list[float]) -> dict:
    return {
        "type": "WorkingMemoryConfig",
        "observation_dim": 2,
        "action_dim": 0,
        "reward_dim": 1,
        "observation_decay_rates": obs_rates,
        "action_decay_rates": [0.5],
        "reward_decay_rates": [0.5],
        "include_current_observation": True,
        "include_current_action": False,
        "include_current_reward": False,
        "include_traces": True,
        "include_innovations": False,
        "gated_update": False,
        "gate_threshold": 0.5,
        "gate_temperature": 0.1,
    }


def test_from_config_rejects_oversized_serialized_list() -> None:
    with pytest.raises(ValueError):
        WorkingMemoryConfig.from_config(_wm_payload([0.5] * (_MAX + 1)))


class _HostileList(list[object]):
    calls = 0

    def __len__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile list length")

    def __iter__(self):  # type: ignore[no-untyped-def]
        type(self).calls += 1
        raise AssertionError("hostile list iteration")


def test_from_config_rejects_hostile_list_subclass_before_hooks() -> None:
    _HostileList.calls = 0
    payload = _wm_payload([0.5, 0.9])
    payload["observation_decay_rates"] = _HostileList([0.5, 0.9])
    with pytest.raises(ValueError, match="observation_decay_rates"):
        WorkingMemoryConfig.from_config(payload)
    assert _HostileList.calls == 0


def test_fixed_trace_rejects_oversized_decay_rates() -> None:
    with pytest.raises(ValueError, match="observation_decay_rates"):
        FixedTraceStateBuilderConfig(
            observation_dim=2,
            observation_decay_rates=(0.5,) * (_MAX + 1),
        )


def test_fixed_trace_accepts_boundary_decay_rates() -> None:
    cfg = FixedTraceStateBuilderConfig(
        observation_dim=2,
        observation_decay_rates=(0.5,) * _MAX,
    )
    assert len(cfg.observation_decay_rates) == _MAX


def _ft_payload(obs_rates: list[float]) -> dict:
    return {
        "type": "FixedTraceStateBuilder",
        "observation_dim": 2,
        "n_actions": 0,
        "observation_decay_rates": obs_rates,
        "action_decay_rates": [0.5],
        "outcome_decay_rates": [0.5],
        "include_raw_observation": True,
    }


def test_fixed_trace_from_config_rejects_oversized_list() -> None:
    with pytest.raises(ValueError):
        FixedTraceStateBuilderConfig.from_config(_ft_payload([0.5] * (_MAX + 1)))


def test_fixed_trace_from_config_rejects_hostile_list_subclass() -> None:
    _HostileList.calls = 0
    payload = _ft_payload([0.5, 0.9])
    payload["observation_decay_rates"] = _HostileList([0.5, 0.9])
    with pytest.raises(ValueError):
        FixedTraceStateBuilderConfig.from_config(payload)
    assert _HostileList.calls == 0
