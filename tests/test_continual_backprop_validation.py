"""Validation hardening for continual backprop (int/float/bool + mappings)."""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from alberta_framework.core.continual_backprop import (
    CBPMLPLearner,
    CBPMultiHeadMLPLearner,
    ContinualBackpropConfig,
    ContinualBackpropTracker,
)

_INT32_MAX = 2**31 - 1


class _LyingIntSubclass(int):
    def __int__(self) -> int:  # pragma: no cover
        return 2

    def __index__(self) -> int:  # pragma: no cover
        return 2


class _RaisingIntSubclass(int):
    def __int__(self) -> int:  # pragma: no cover
        raise RuntimeError("conversion hook must not run")

    def __index__(self) -> int:  # pragma: no cover
        raise RuntimeError("conversion hook must not run")


class _RaisingRepr:
    def __repr__(self) -> str:  # pragma: no cover
        raise RuntimeError("repr hook must not run")


def _cbp_cfg(**overrides: object) -> ContinualBackpropConfig:
    base: dict[str, object] = {
        "decay_rate": 0.99,
        "replacement_rate": 1e-4,
        "maturity_threshold": 100,
        "enabled": True,
    }
    base.update(overrides)
    return ContinualBackpropConfig(**base)  # type: ignore[arg-type]


def test_cbp_int_validators_reject_hostile_subclass_without_running_hook() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        _cbp_cfg(maturity_threshold=_LyingIntSubclass(5))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be an integer"):
        _cbp_cfg(maturity_threshold=_RaisingIntSubclass(5))  # type: ignore[arg-type]


def test_cbp_int_validators_do_not_run_repr_hook() -> None:
    with pytest.raises(ValueError):
        _cbp_cfg(maturity_threshold=_RaisingRepr())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ContinualBackpropConfig(
            decay_rate=0.99, replacement_rate=1e-4, maturity_threshold=_RaisingRepr(), enabled=True  # type: ignore[arg-type]
        )


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
def test_cbp_int_validators_canonicalize_numpy_scalars(np_type: type) -> None:
    cfg = _cbp_cfg(maturity_threshold=np_type(10))
    assert cfg.maturity_threshold == 10
    assert type(cfg.maturity_threshold) is int


@pytest.mark.parametrize("value", [
    True, np.bool_(True), 4.0, np.float64(4.0), "4", None, -1, _INT32_MAX + 1
])
def test_cbp_int_validators_reject_non_integer_and_out_of_range(value: object) -> None:
    with pytest.raises(ValueError, match="must be"):
        _cbp_cfg(maturity_threshold=value)  # type: ignore[arg-type]


def test_cbp_float_validators_reject_hostile_subclass_without_running_hook() -> None:
    class HostileFloat(float):
        def as_integer_ratio(self) -> tuple[int, int]:  # pragma: no cover
            raise RuntimeError("untrusted ratio hook executed")

    class ClassSpoof:
        @property
        def __class__(self):  # type: ignore[no-untyped-def]
            return float

        def __float__(self) -> float:  # pragma: no cover
            return 0.5

    with pytest.raises(ValueError, match="must be a finite real number"):
        _cbp_cfg(decay_rate=HostileFloat(0.5))
    with pytest.raises(ValueError, match="must be a finite real number"):
        _cbp_cfg(decay_rate=ClassSpoof())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be a finite real number"):
        ContinualBackpropTracker(config=_cbp_cfg(), sparsity=HostileFloat(0.5))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be a finite real number"):
        CBPMultiHeadMLPLearner(n_heads=2, hidden_sizes=(4,), sparsity=HostileFloat(0.5))  # type: ignore[arg-type]


def test_cbp_float_validators_reject_hostile_ratio_without_calling() -> None:
    class HostileFloat(float):
        calls = 0

        def as_integer_ratio(self) -> tuple[int, int]:  # pragma: no cover
            type(self).calls += 1
            raise RuntimeError("ratio hook")

    with pytest.raises(ValueError, match="decay_rate"):
        _cbp_cfg(decay_rate=HostileFloat(0.5))  # type: ignore[arg-type]
    assert HostileFloat.calls == 0
    HostileFloat.calls = 0
    with pytest.raises(ValueError, match="replacement_rate"):
        _cbp_cfg(replacement_rate=HostileFloat(0.5))  # type: ignore[arg-type]
    assert HostileFloat.calls == 0


@pytest.mark.parametrize(
    "np_type",
    [np.float16, np.float32, np.float64],
)
def test_cbp_float_validators_canonicalize_numpy_float_scalars(np_type: type) -> None:
    cfg = _cbp_cfg(decay_rate=np_type(0.5))
    assert cfg.decay_rate == pytest.approx(0.5)
    assert type(cfg.decay_rate) is float


def test_cbp_float_validators_reject_nonfinite_and_domain() -> None:
    class HostileFloat(float):
        def as_integer_ratio(self) -> tuple[int, int]:  # pragma: no cover
            raise RuntimeError

    for field, bad in [
        ("decay_rate", float("nan")),
        ("decay_rate", float("inf")),
        ("decay_rate", 1.0),
        ("decay_rate", -0.1),
        ("decay_rate", HostileFloat(0.5)),
        ("replacement_rate", -0.1),
        ("replacement_rate", 1.5),
        ("replacement_rate", float("nan")),
        ("sparsity", -0.1),
        ("sparsity", 1.5),
    ]:
        if field == "sparsity":
            with pytest.raises(ValueError, match=field):
                ContinualBackpropTracker(config=_cbp_cfg(), sparsity=bad)  # type: ignore[arg-type]
        else:
            with pytest.raises(ValueError, match=field):
                _cbp_cfg(**{field: bad})  # type: ignore[arg-type]


def test_cbp_float_validators_reject_bool_and_string() -> None:
    for bad in [True, np.bool_(True), "0.5", None]:
        with pytest.raises(ValueError, match="must be"):
            _cbp_cfg(decay_rate=bad)  # type: ignore[arg-type]


def test_cbp_float_validators_reject_subnormal_flush_to_zero() -> None:
    with pytest.raises(ValueError, match="must remain nonzero"):
        _cbp_cfg(decay_rate=1e-45)
    with pytest.raises(ValueError, match="must remain nonzero"):
        ContinualBackpropTracker(config=_cbp_cfg(), sparsity=1e-45)  # type: ignore[arg-type]
    _cbp_cfg(decay_rate=1e-44)


def test_cbp_float_validators_accept_valid_values() -> None:
    cfg = _cbp_cfg(decay_rate=0.5, replacement_rate=0.1, maturity_threshold=10)
    assert cfg.decay_rate == 0.5
    assert cfg.replacement_rate == 0.1
    tracker = ContinualBackpropTracker(config=cfg, sparsity=0.5)
    assert tracker.sparsity == 0.5
    learner = CBPMultiHeadMLPLearner(
        n_heads=2, hidden_sizes=(4,), sparsity=0.5, leaky_relu_slope=0.01
    )
    assert learner._sparsity == 0.5
    # Fraction accepted
    from fractions import Fraction

    cfg2 = _cbp_cfg(decay_rate=Fraction(1, 2))
    assert cfg2.decay_rate == 0.5


def test_cbp_bool_validators_reject_non_bool() -> None:
    for bad in [1, 0, "true", np.bool_(True), 1.0]:
        with pytest.raises(ValueError, match="must be an actual bool"):
            _cbp_cfg(enabled=bad)  # type: ignore[arg-type]
    for bad in [1, 0, "true", np.bool_(True)]:
        with pytest.raises(ValueError, match="must be an actual bool"):
            CBPMultiHeadMLPLearner(n_heads=2, hidden_sizes=(4,), use_layer_norm=bad)  # type: ignore[arg-type]


def test_cbp_bool_validators_do_not_run_repr() -> None:
    with pytest.raises(ValueError):
        _cbp_cfg(enabled=_RaisingRepr())  # type: ignore[arg-type]


def test_cbp_mapping_loaders_preserve_markers_and_exact_keys() -> None:
    cfg = _cbp_cfg()
    payload = cfg.to_config()
    restored = ContinualBackpropConfig.from_config(MappingProxyType(payload))
    assert restored == cfg
    with pytest.raises(ValueError, match="fields do not match"):
        ContinualBackpropConfig.from_config({**payload, "unknown": 1})
    with pytest.raises(ValueError, match="fields do not match"):
        ContinualBackpropConfig.from_config({"decay_rate": 0.5})

    learner = CBPMultiHeadMLPLearner(n_heads=2, hidden_sizes=(4,))
    lcfg = learner.to_config()
    assert CBPMultiHeadMLPLearner.from_config(MappingProxyType(lcfg))._cbp_config == learner.config
    with pytest.raises(ValueError, match="fields do not match"):
        CBPMultiHeadMLPLearner.from_config({**lcfg, "unknown": 1})

    single = CBPMLPLearner(hidden_sizes=(4,))
    scfg = single.to_config()
    restored = CBPMLPLearner.from_config(MappingProxyType(scfg))
    assert restored.learner._cbp_config == single.learner.config

    class StringSubclass(str):
        pass

    with pytest.raises(ValueError, match="exact strings"):
        ContinualBackpropConfig.from_config({StringSubclass("decay_rate"): 0.5, **payload})
    with pytest.raises(ValueError, match="exact strings"):
        CBPMultiHeadMLPLearner.from_config(
            {StringSubclass("type"): "CBPMultiHeadMLPLearner", **lcfg}
        )

    class HostileMapping(dict):  # type: ignore[type-arg]
        def __iter__(self):  # type: ignore[override]
            raise RuntimeError("iter hook")

        def __getitem__(self, key):  # type: ignore[override]
            raise RuntimeError("get hook")

    with pytest.raises(ValueError, match="could not be read"):
        ContinualBackpropConfig.from_config(HostileMapping(payload))  # type: ignore[arg-type]


def test_cbp_maturity_threshold_domain() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        _cbp_cfg(maturity_threshold=-1)
    _cbp_cfg(maturity_threshold=0)
    _cbp_cfg(maturity_threshold=_INT32_MAX)
    with pytest.raises(ValueError, match="must be an integer"):
        _cbp_cfg(maturity_threshold=_INT32_MAX + 1)


def test_cbp_decay_rate_upper_exclusive() -> None:
    with pytest.raises(ValueError, match="decay_rate"):
        _cbp_cfg(decay_rate=1.0)
    _cbp_cfg(decay_rate=0.999)


