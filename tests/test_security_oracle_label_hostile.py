"""Hostile string validation for security oracle label."""

from __future__ import annotations

import pytest

from alberta_framework.security import (
    SecurityAction,
    SecurityFeatureSchema,
    SecurityOracleExperience,
    validate_security_oracle_experience,
)

pytestmark = pytest.mark.unit


class _HostileStr(str):
    calls = 0

    __hash__ = str.__hash__

    def __bool__(self) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile bool executed")

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile eq executed")

    def __len__(self) -> int:
        type(self).calls += 1
        raise AssertionError("hostile len executed")

    def __str__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("hostile str executed")

    def __repr__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("hostile repr executed")


def _schema() -> SecurityFeatureSchema:
    return SecurityFeatureSchema(names=("x",))


def test_validate_oracle_experience_rejects_hostile_label_before_bool() -> None:
    schema = _schema()
    hostile = _HostileStr("safe")
    _HostileStr.calls = 0
    record = SecurityOracleExperience(
        state=(0.5,),
        action=SecurityAction.PASS,
        reward=0.0,
        outcome={"label": hostile},  # type: ignore[dict-item]
    )
    with pytest.raises(ValueError, match="missing outcome label"):
        validate_security_oracle_experience([record], schema)
    assert _HostileStr.calls == 0


def test_validate_oracle_experience_accepts_builtin_label() -> None:
    schema = _schema()
    record = SecurityOracleExperience(
        state=(0.5,),
        action=SecurityAction.PASS,
        reward=0.0,
        outcome={"label": "safe"},
    )
    # Should not raise for valid builtin label
    validate_security_oracle_experience([record], schema)


def test_validate_oracle_experience_rejects_empty_label() -> None:
    schema = _schema()
    record = SecurityOracleExperience(
        state=(0.5,),
        action=SecurityAction.PASS,
        reward=0.0,
        outcome={"label": ""},
    )
    with pytest.raises(ValueError, match="missing outcome label"):
        validate_security_oracle_experience([record], schema)


def test_hostile_empty_string_still_rejects_before_len() -> None:
    # Empty hostile subclass with empty value; type check should fail before len/bool
    hostile = _HostileStr("")
    _HostileStr.calls = 0
    schema = _schema()
    record = SecurityOracleExperience(
        state=(0.5,),
        action=SecurityAction.PASS,
        reward=0.0,
        outcome={"label": hostile},  # type: ignore[dict-item]
    )
    with pytest.raises(ValueError, match="missing outcome label"):
        validate_security_oracle_experience([record], schema)
    assert _HostileStr.calls == 0
