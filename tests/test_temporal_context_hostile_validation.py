"""Hostile-safe validation for TemporalContextFeaturizer."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from alberta_framework.core.temporal_context import (
    TemporalContextConfig,
    TemporalContextFeaturizer,
)


class _HostileInt(int):
    def __index__(self) -> int:  # pragma: no cover
        raise AssertionError("index hook executed")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("repr hook")


class _RaisingRepr:
    def __repr__(self) -> str:  # pragma: no cover
        raise RuntimeError("repr hook")


class _HostileFloat(float):
    calls = 0

    def as_integer_ratio(self) -> tuple[int, int]:  # type: ignore[override]
        type(self).calls += 1
        raise RuntimeError("ratio hook")


class _StringSubclass(str):
    pass


def test_temporal_rejects_hostile_int_input_dim() -> None:
    with pytest.raises(ValueError, match="input_dim"):
        TemporalContextConfig(input_dim=_HostileInt(4))  # type: ignore[arg-type]


def test_temporal_rejects_hostile_repr_input_dim() -> None:
    with pytest.raises(ValueError, match="input_dim"):
        TemporalContextConfig(input_dim=_RaisingRepr())  # type: ignore[arg-type]


def test_temporal_rejects_bool_input_dim() -> None:
    with pytest.raises(ValueError, match="input_dim"):
        TemporalContextConfig(input_dim=True)  # type: ignore[arg-type]


def test_temporal_rejects_hostile_float_ema_without_ratio() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="ema_decay"):
        TemporalContextConfig(input_dim=4, ema_decay=_HostileFloat(0.5))  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0


def test_temporal_rejects_hostile_period_float() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="period"):
        TemporalContextConfig(input_dim=2, periods=(_HostileFloat(50.0),))  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0


def test_temporal_rejects_string_subclass_periods() -> None:
    with pytest.raises(ValueError, match="period"):
        TemporalContextConfig(input_dim=2, periods=(_StringSubclass("50"),))  # type: ignore[arg-type]


def test_temporal_from_config_preserves_mapping_proxy() -> None:
    cfg = TemporalContextConfig(input_dim=4)
    payload = cfg.to_config()
    restored = TemporalContextConfig.from_config(MappingProxyType(payload))
    assert restored.to_config() == payload


def test_temporal_from_config_rejects_string_subclass_key() -> None:
    cfg = TemporalContextConfig(input_dim=4)
    payload = cfg.to_config()
    hostile: dict[Any, Any] = {_StringSubclass("input_dim"): 4}
    for k, v in payload.items():
        if k != "input_dim":
            hostile[k] = v
    with pytest.raises(ValueError, match="exact strings"):
        TemporalContextConfig.from_config(hostile)  # type: ignore[arg-type]


def test_temporal_from_config_rejects_hostile_mapping() -> None:
    from collections.abc import Mapping

    class HostileMapping(Mapping[str, Any]):  # type: ignore[type-arg]
        def __getitem__(self, key: str) -> Any:
            raise RuntimeError("hook")

        def __iter__(self):  # type: ignore[no-untyped-def]
            raise RuntimeError("iter hook")

        def __len__(self) -> int:
            return 0

    with pytest.raises(ValueError, match="mapping"):
        TemporalContextConfig.from_config(HostileMapping())  # type: ignore[arg-type]


def test_temporal_numpy_int_float_canonicalizes() -> None:
    for np_type in [
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.longlong,
        np.ulonglong,
    ]:
        cfg = TemporalContextConfig(input_dim=np_type(4))  # type: ignore[arg-type]
        assert cfg.input_dim == 4
        assert type(cfg.input_dim) is int
    cfg2 = TemporalContextConfig(input_dim=4, ema_decay=np.float64(0.5))
    assert cfg2.ema_decay == pytest.approx(0.5)
    cfg3 = TemporalContextConfig(input_dim=4, periods=(np.float32(50.0),))
    assert cfg3.periods[0] == pytest.approx(50.0)


def test_temporal_valid_config_passes() -> None:
    cfg = TemporalContextConfig(input_dim=4, ema_decay=0.95, periods=(50.0, 100.0))
    featurizer = TemporalContextFeaturizer(cfg)
    state = featurizer.init()
    assert state.observation_ema.shape == (4,)
