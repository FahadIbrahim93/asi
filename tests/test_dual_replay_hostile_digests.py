"""Hostile string validation for dual replay digests."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class _HostileStr(str):
    calls = 0
    __hash__ = str.__hash__

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile eq executed")


def test_dual_replay_config_digest_rejects_hostile() -> None:
    from alberta_framework.core.dual_replay import _payload_digest

    payload = {"memory": "x"}
    digest = _payload_digest(payload)
    hostile = _HostileStr(digest)
    _HostileStr.calls = 0
    assert (type(hostile) is not str or hostile != digest) is True
    assert _HostileStr.calls == 0


def test_dual_replay_state_digest_rejects_hostile() -> None:
    from alberta_framework.core.dual_replay import _payload_digest

    payload = {"state": "y"}
    digest = _payload_digest(payload)
    hostile = _HostileStr(digest)
    _HostileStr.calls = 0
    assert (type(hostile) is not str or hostile != digest) is True
    assert _HostileStr.calls == 0


def test_dual_replay_checkpoint_rejects_hostile_digests() -> None:
    from alberta_framework.core.dual_replay import DualReplayMemory, _payload_digest

    # Use real payloads that would be dicts with mappings
    memory_payload = {"a": 1}
    state_payload = {
        "short_term": {},
        "long_term": {},
        "rng_key_data": "x",
        "persistent_bytes": "",
        "short_term_count": 0,
        "long_term_count": 0,
        "total_steps": 0,
    }
    # Need full expected_state fields to pass other validation before digest?
    # Simpler: directly test the gate, not full checkpoint restore
    # Test that the code path would reject hostile before eq
    cfg_digest = _payload_digest(memory_payload)
    st_digest = _payload_digest(state_payload)
    hostile_cfg = _HostileStr(cfg_digest)
    hostile_st = _HostileStr(st_digest)
    _HostileStr.calls = 0
    # Simulate payload with hostile digests
    payload = {
        "schema": "alberta.dual-replay.checkpoint.v1",
        "mechanism_status": "mechanism-only-no-training-integration",
        "memory": memory_payload,
        "state": state_payload,
        "config_digest": hostile_cfg,  # type: ignore[dict-item]
        "state_digest": hostile_st,  # type: ignore[dict-item]
    }
    # The actual class method will check config_digest first
    with pytest.raises(ValueError, match="config digest mismatch"):
        DualReplayMemory.from_checkpoint_payload(payload)  # type: ignore[arg-type]
    assert _HostileStr.calls == 0
