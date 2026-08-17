"""Hostile-safe validation for security contracts."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from alberta_framework.security import (
    SecurityAction,
    SecurityFeatureSchema,
    SecurityRolloutStep,
    ThroughputMeter,
    coerce_security_action,
    to_security_gym_action,
)


class _HostileInt(int):
    def __index__(self) -> int:  # pragma: no cover
        raise AssertionError("index hook")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("repr hook")


class _HostileFloat(float):
    calls = 0

    def as_integer_ratio(self) -> tuple[int, int]:  # type: ignore[override]
        type(self).calls += 1
        raise RuntimeError("ratio hook")


class _StringSubclass(str):
    pass


class _RaisingRepr:
    def __repr__(self) -> str:  # pragma: no cover
        raise RuntimeError("repr hook")


def test_coerce_rejects_hostile_int_without_hook() -> None:
    with pytest.raises(ValueError, match="security action"):
        coerce_security_action(_HostileInt(1))  # type: ignore[arg-type]


def test_coerce_rejects_string_subclass() -> None:
    with pytest.raises(ValueError, match="security action"):
        coerce_security_action(_StringSubclass("pass"))  # type: ignore[arg-type]


def test_coerce_rejects_hostile_repr() -> None:
    with pytest.raises(ValueError, match="security action"):
        coerce_security_action(_RaisingRepr())  # type: ignore[arg-type]


def test_to_gym_rejects_hostile_float_without_ratio() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="risk_score"):
        to_security_gym_action("pass", risk_score=_HostileFloat(0.5))  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0


def test_to_gym_rejects_bool_risk() -> None:
    with pytest.raises(ValueError, match="risk_score"):
        to_security_gym_action("pass", risk_score=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="risk_score"):
        to_security_gym_action("pass", risk_score=np.bool_(True))  # type: ignore[arg-type]


def test_to_gym_rejects_string_subclass_risk() -> None:
    with pytest.raises(ValueError, match="risk_score"):
        to_security_gym_action("pass", risk_score=_StringSubclass("0.5"))  # type: ignore[arg-type]


def test_tick_rejects_hostile_int_without_hook() -> None:
    meter = ThroughputMeter()
    with pytest.raises(ValueError, match="n_events"):
        meter.tick(_HostileInt(1))  # type: ignore[arg-type]


def test_tick_rejects_bool() -> None:
    meter = ThroughputMeter()
    with pytest.raises(ValueError, match="n_events"):
        meter.tick(True)  # type: ignore[arg-type]


def test_schema_from_dict_preserves_mapping_proxy() -> None:
    schema = SecurityFeatureSchema(names=("a", "b"))
    payload = schema.to_dict()
    restored = SecurityFeatureSchema.from_dict(MappingProxyType(payload))
    assert restored == schema


def test_schema_from_dict_rejects_string_subclass_key() -> None:
    schema = SecurityFeatureSchema(names=("a", "b"))
    payload = schema.to_dict()
    hostile: dict[Any, Any] = {_StringSubclass("names"): payload["names"]}
    for k, v in payload.items():
        if k != "names":
            hostile[k] = v
    with pytest.raises(ValueError, match="exact strings"):
        SecurityFeatureSchema.from_dict(hostile)  # type: ignore[arg-type]


def test_schema_from_dict_rejects_hostile_mapping() -> None:
    from collections.abc import Mapping

    class HostileMapping(Mapping[str, Any]):  # type: ignore[type-arg]
        def __getitem__(self, key: str) -> Any:
            raise RuntimeError("hook")

        def __iter__(self):  # type: ignore[no-untyped-def]
            raise RuntimeError("iter hook")

        def __len__(self) -> int:
            return 0

    with pytest.raises(ValueError, match="mapping"):
        SecurityFeatureSchema.from_dict(HostileMapping())  # type: ignore[arg-type]


def test_rollout_from_dict_rejects_hostile_float() -> None:
    step = SecurityRolloutStep(
        state=(0.0, 1.0),
        action=SecurityAction.PASS,
        reward=0.0,
        next_state=(0.0, 1.0),
        terminated=False,
    )
    payload = step.to_dict()
    _HostileFloat.calls = 0
    payload["reward"] = _HostileFloat(0.5)
    with pytest.raises(ValueError, match="reward"):
        SecurityRolloutStep.from_dict(payload)  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0


def test_numpy_risk_canonicalizes() -> None:
    res = to_security_gym_action("alert", risk_score=np.float64(5.5))
    assert res["risk_score"] == (5.5,)
    res2 = to_security_gym_action("pass", risk_score=np.int32(5))
    assert res2["risk_score"] == (5.0,)
