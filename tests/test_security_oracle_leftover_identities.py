"""Leftover-identity gates for security oracle-experience records."""

from __future__ import annotations

import json
from types import MappingProxyType

import pytest

from alberta_framework.security import (
    SecurityAction,
    SecurityFeatureSchema,
    SecurityOracleExperience,
    SecurityRolloutStep,
    security_rollout_step_to_oracle_experience,
    validate_security_oracle_experience,
)


def _legal(**overrides: object) -> SecurityOracleExperience:
    payload: dict[str, object] = {
        "state": (0.5,),
        "action": SecurityAction.PASS,
        "reward": 0.0,
        "outcome": {"label": "safe"},
    }
    payload.update(overrides)
    return SecurityOracleExperience(**payload)  # type: ignore[arg-type]


def test_security_oracle_experience_rejects_leftover_identities() -> None:
    """Public oracle records must not keep leftover bool/string identities."""

    with pytest.raises(ValueError, match="reward"):
        _legal(reward=True)
    with pytest.raises(ValueError, match="reward"):
        _legal(reward=False)
    with pytest.raises(ValueError, match="reward"):
        _legal(reward="FIXED")
    with pytest.raises(ValueError, match="action"):
        _legal(action=0)
    with pytest.raises(ValueError, match="action"):
        _legal(action=True)
    with pytest.raises(ValueError, match="schema"):
        _legal(schema=True)
    with pytest.raises(ValueError, match="exact tuple"):
        _legal(state=[0.5])
    with pytest.raises(ValueError, match=r"state\[0\]"):
        _legal(state=(True,))

    legal = _legal()
    dumped = json.dumps(legal.to_dict(), allow_nan=False)
    assert dumped.count('"reward": 0.0') == 1
    assert '"reward": true' not in dumped
    assert '"reward": "FIXED"' not in dumped
    assert '"schema": true' not in dumped
    assert '"state": [true]' not in dumped
    assert legal.action is SecurityAction.PASS
    assert type(legal.reward) is float


def test_oracle_experience_snapshots_nested_payloads_on_real_conversion_path() -> None:
    nested = ["source"]
    step = SecurityRolloutStep(
        state=(0.5,),
        action=SecurityAction.BLOCK,
        reward=1.0,
        next_state=(0.25,),
        terminated=False,
        policy_metadata={"is_malicious": True, "nested": nested},
    )
    record = security_rollout_step_to_oracle_experience(step)
    nested.append("changed")

    validate_security_oracle_experience(
        [record], SecurityFeatureSchema(names=("risk",))
    )
    assert type(record.outcome) is MappingProxyType
    assert type(record.policy_metadata) is MappingProxyType
    assert record.to_dict()["policy_metadata"]["nested"] == ["source"]
    assert record.to_dict()["outcome"] == {
        "label": "true_positive",
        "terminated": False,
        "truncated": False,
    }

    outcome_nested = ["source"]
    metadata_nested = ["source"]
    direct = _legal(
        outcome={"label": "safe", "nested": outcome_nested},
        policy_metadata={"nested": metadata_nested},
    )
    outcome_nested.append("changed")
    metadata_nested.append("changed")
    assert direct.to_dict()["outcome"]["nested"] == ["source"]
    assert direct.to_dict()["policy_metadata"]["nested"] == ["source"]


def test_oracle_experience_cross_binds_derived_label_to_action_metadata() -> None:
    with pytest.raises(ValueError, match="does not match action metadata"):
        _legal(
            action=SecurityAction.BLOCK,
            outcome={"label": "false_negative"},
            policy_metadata={"is_malicious": True},
        )


def test_oracle_experience_rejects_hostile_nested_type_without_hooks() -> None:
    class _HostileMeta(type):
        calls = 0

        def __eq__(cls, other: object) -> bool:
            cls.calls += 1
            raise AssertionError("metaclass equality hook executed")

    class _Hostile(metaclass=_HostileMeta):
        pass

    with pytest.raises(ValueError, match="exact JSON values"):
        _legal(outcome={"label": "safe", "nested": _Hostile()})
    assert _HostileMeta.calls == 0


def test_oracle_experience_enforces_nested_resource_limits() -> None:
    nested: object = "leaf"
    for _ in range(34):
        nested = [nested]
    with pytest.raises(ValueError, match="nesting limit"):
        _legal(outcome={"label": "safe", "nested": nested})
    with pytest.raises(ValueError, match="oversized string"):
        _legal(outcome={"label": "safe", "nested": "x" * 65_537})


def test_oracle_validator_rejects_hostile_outer_container_without_hooks() -> None:
    class _HostileRecords(list[SecurityOracleExperience]):
        def __iter__(self):
            raise AssertionError("container iteration hook executed")

    with pytest.raises(ValueError, match="exact list or tuple"):
        validate_security_oracle_experience(
            _HostileRecords([_legal()]), SecurityFeatureSchema(names=("risk",))
        )
