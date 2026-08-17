"""Validation hardening for experiential-memory policy."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from alberta_framework.core.experiential_memory import (
    ExperientialMemory,
    ExperientialMemoryConfig,
)
from alberta_framework.core.experiential_memory_policy import (
    ExperientialMemoryPolicy,
)


def _memory(
    *,
    action_dim: int = 2,
    capacity: int = 4,
    **overrides: Any,
) -> ExperientialMemory:
    cfg = ExperientialMemoryConfig(
        capacity=capacity,
        observation_dim=2,
        key_dim=2,
        action_dim=action_dim,
        outcome_dim=1,
        top_k=2,
        min_neighbors=1,
        **overrides,  # type: ignore[arg-type]
    )
    return ExperientialMemory(cfg)


def _policy(**overrides: Any) -> ExperientialMemoryPolicy:
    mem = overrides.pop("memory", _memory())
    return ExperientialMemoryPolicy(mem)


class _HostileInt(int):
    def __index__(self) -> int:  # pragma: no cover
        raise AssertionError("index hook executed")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("repr executed")


class _RaisingRepr:
    def __repr__(self) -> str:  # pragma: no cover
        raise RuntimeError("repr hook must not run")


class _HostileFloat(float):
    calls = 0

    def as_integer_ratio(self) -> tuple[int, int]:  # type: ignore[override]
        type(self).calls += 1
        raise RuntimeError("ratio hook")


def test_policy_init_rejects_subclass_memory() -> None:
    class MemorySubclass(ExperientialMemory):  # type: ignore[type-arg]
        pass

    base = _memory()
    # Bypass __init__ to create subclass instance without validation
    subclass = object.__new__(MemorySubclass)
    object.__setattr__(subclass, "_config", base.config)
    object.__setattr__(subclass, "_persistent_bytes", base.persistent_bytes)
    object.__setattr__(subclass, "_slot_bytes", base.slot_bytes)
    with pytest.raises(TypeError, match="exact ExperientialMemory"):
        ExperientialMemoryPolicy(subclass)  # type: ignore[arg-type]


def test_policy_init_rejects_hostile_repr() -> None:
    with pytest.raises(TypeError):
        ExperientialMemoryPolicy(_RaisingRepr())  # type: ignore[arg-type]


def test_policy_from_config_preserves_mapping_proxy() -> None:
    policy = _policy()
    payload = policy.to_config()
    restored = ExperientialMemoryPolicy.from_config(MappingProxyType(payload))
    assert restored.to_config() == payload


def test_policy_from_config_rejects_string_subclass_key() -> None:
    policy = _policy()
    payload = policy.to_config()

    class StringSubclass(str):
        pass

    hostile: dict[Any, Any] = {StringSubclass("schema"): payload["schema"]}
    for key, value in payload.items():
        if key != "schema":
            hostile[key] = value
    with pytest.raises(ValueError, match="exact strings"):
        ExperientialMemoryPolicy.from_config(hostile)  # type: ignore[arg-type]


def test_policy_from_config_rejects_hostile_mapping() -> None:
    from collections.abc import Mapping

    class HostileMapping(Mapping[str, Any]):  # type: ignore[type-arg]
        def __getitem__(self, key: str) -> Any:
            raise RuntimeError("hook")

        def __iter__(self):  # type: ignore[no-untyped-def]
            raise RuntimeError("iter hook")

        def __len__(self) -> int:
            return 0

    with pytest.raises(ValueError, match="mapping"):
        ExperientialMemoryPolicy.from_config(HostileMapping())  # type: ignore[arg-type]


def test_policy_from_config_rejects_hostile_inner_mapping() -> None:
    policy = _policy()
    payload = policy.to_config()

    from collections.abc import Mapping

    class HostileInner(Mapping[str, Any]):  # type: ignore[type-arg]
        def __getitem__(self, key: str) -> Any:
            raise RuntimeError("hook")

        def __iter__(self):  # type: ignore[no-untyped-def]
            raise RuntimeError("hook")

        def __len__(self) -> int:
            return 1

    bad = dict(payload)
    bad["memory"] = HostileInner()  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="readable mapping"):
        ExperientialMemoryPolicy.from_config(bad)


def test_policy_from_config_rejects_wrong_schema_type() -> None:
    policy = _policy()
    payload = dict(policy.to_config())
    payload["schema"] = "wrong"
    with pytest.raises(ValueError, match="schema|v1"):
        ExperientialMemoryPolicy.from_config(payload)
    payload2 = dict(policy.to_config())
    payload2["type"] = "Wrong"
    with pytest.raises(ValueError, match="type|invalid"):
        ExperientialMemoryPolicy.from_config(payload2)


def test_policy_from_config_rejects_noncanonical_extra_field() -> None:
    policy = _policy()
    payload = dict(policy.to_config())
    payload["extra"] = 1
    with pytest.raises(ValueError, match="fields do not match"):
        ExperientialMemoryPolicy.from_config(payload)


def test_policy_resource_declaration_validates_action_dim() -> None:
    mem = _memory(action_dim=3)
    policy = ExperientialMemoryPolicy(mem)
    decl = policy.resource_declaration()
    assert decl.n_actions == 3
    assert decl.score_mass_values_interpreted_per_proposal == 3


def test_policy_resource_declaration_rejects_hostile_action_dim() -> None:
    # Construct a memory with hostile action_dim via direct config bypass
    # is prevented by ExperientialMemoryConfig validation, so we test the policy
    # gate by injecting a hostile int through a spoofed config object.
    cfg = ExperientialMemoryConfig(
        capacity=4,
        observation_dim=2,
        key_dim=2,
        action_dim=2,
        outcome_dim=1,
    )
    # Simulate hostile value reaching resource_declaration by monkeying the
    # underlying config's action_dim via object.__setattr__ (frozen dataclass)
    hostile = _HostileInt(2)
    object.__setattr__(cfg, "action_dim", hostile)  # type: ignore[arg-type]
    mem = ExperientialMemory.__new__(ExperientialMemory)
    object.__setattr__(mem, "_config", cfg)
    object.__setattr__(mem, "_persistent_bytes", 1024)
    object.__setattr__(mem, "_slot_bytes", 64)
    policy = ExperientialMemoryPolicy.__new__(ExperientialMemoryPolicy)
    object.__setattr__(policy, "_memory", mem)
    with pytest.raises(ValueError, match="must be an integer"):
        policy.resource_declaration()


@pytest.mark.parametrize(
    "np_type",
    [
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.longlong,  # noqa: E501
        np.ulonglong,
    ],
)
def test_policy_numpy_action_dim_canonicalized_via_memory(np_type: type) -> None:
    mem = _memory(action_dim=np_type(2))  # type: ignore[arg-type]
    assert mem.config.action_dim == 2
    assert type(mem.config.action_dim) is int
    policy = ExperientialMemoryPolicy(mem)
    assert policy.resource_declaration().n_actions == 2


def test_policy_float_hostile_ratio_suppressed_via_memory() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="distance_scale"):
        _memory(distance_scale=_HostileFloat(0.5))  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0


def test_policy_to_from_config_roundtrip() -> None:
    policy = _policy()
    payload = policy.to_config()
    restored = ExperientialMemoryPolicy.from_config(dict(payload))
    assert restored.to_config() == payload
    assert restored.memory.config == policy.memory.config


def test_policy_from_config_rejects_bad_memory_object_type() -> None:
    policy = _policy()
    payload = dict(policy.to_config())
    payload["memory"] = "not a mapping"
    with pytest.raises(ValueError, match="mapping"):
        ExperientialMemoryPolicy.from_config(payload)
