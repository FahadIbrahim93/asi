"""Validation hardening for UPGD learner (int/float bounds + resources)."""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest
from jax import random as jr

from alberta_framework.core.upgd import UPGDLearner

_INT32_MAX = 2**31 - 1


class _HostileInt(int):
    def __index__(self) -> int:  # pragma: no cover
        raise AssertionError("index hook must not run")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("repr hook must not run")


class _HostileFloat(float):
    calls = 0

    def as_integer_ratio(self) -> tuple[int, int]:  # pragma: no cover
        type(self).calls += 1
        raise RuntimeError("ratio hook")


class _ClassSpoof:
    @property
    def __class__(self) -> type:  # type: ignore[no-untyped-def]
        return float  # type: ignore[return-value]

    def __float__(self) -> float:  # pragma: no cover
        return 0.1


class _RaisingRepr:
    def __repr__(self) -> str:  # pragma: no cover
        raise RuntimeError("repr hook must not run")


class _HostileMapping(dict):  # type: ignore[type-arg]
    def __iter__(self):  # type: ignore[override]
        raise RuntimeError("hostile iter")

    def __getitem__(self, key):  # type: ignore[override]
        raise RuntimeError("hostile getitem")

    def keys(self):  # type: ignore[override]
        raise RuntimeError("hostile keys")


def _learner(**overrides: object) -> UPGDLearner:
    base: dict[str, object] = {"n_heads": 2, "hidden_sizes": (4,)}
    base.update(overrides)
    return UPGDLearner(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "ctor",
    [
        lambda v: _learner(n_heads=v),
        lambda v: _learner(hidden_sizes=(v,)),  # type: ignore[arg-type]
        lambda v: _learner(perturbation_interval=v),
        lambda v: _learner(perturbation_warmup_steps=v),
        lambda v: _learner(head_loss_pressure_warmup_steps=v),
        lambda v: _learner(unit_maturity_threshold=v),
    ],
)
def test_upgd_int_validators_reject_hostile_without_hook(ctor) -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        ctor(_HostileInt(4))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "ctor",
    [
        lambda v: _learner(n_heads=v),
        lambda v: _learner(perturbation_interval=v),
    ],
)
def test_upgd_int_validators_do_not_run_repr_hook(ctor) -> None:
    with pytest.raises(ValueError):
        ctor(_RaisingRepr())  # type: ignore[arg-type]


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
        np.longlong,
        np.ulonglong,
    ],
)
def test_upgd_int_validators_canonicalize_numpy_scalars(np_type: type) -> None:
    cfg = _learner(n_heads=np_type(2), perturbation_interval=np_type(2))
    assert cfg._n_heads == 2
    assert type(cfg._n_heads) is int
    assert cfg._perturbation_interval == 2
    assert type(cfg._perturbation_interval) is int


@pytest.mark.parametrize(
    "ctor",
    [
        lambda v: _learner(n_heads=v),
        lambda v: _learner(perturbation_interval=v),
    ],
)
@pytest.mark.parametrize(
    "value",
    [True, np.bool_(True), 4.0, np.float64(4.0), "4", None, 0, -1, _INT32_MAX + 1],
)
def test_upgd_int_validators_reject_non_integer_and_out_of_range(
    ctor, value: object
) -> None:
    with pytest.raises(ValueError, match="must be"):
        ctor(value)  # type: ignore[arg-type]


def test_upgd_float_validators_reject_hostile_ratio() -> None:
    class HostileFloat(float):
        calls = 0

        def as_integer_ratio(self) -> tuple[int, int]:  # pragma: no cover
            type(self).calls += 1
            raise RuntimeError("ratio hook")

    with pytest.raises(ValueError, match="must be a finite"):
        _learner(step_size=HostileFloat(0.01))  # type: ignore[arg-type]
    assert HostileFloat.calls == 0
    with pytest.raises(ValueError, match="must be a finite"):
        _learner(utility_decay=HostileFloat(0.5))  # type: ignore[arg-type]
    assert HostileFloat.calls == 0


def test_upgd_float_validators_reject_spoof_and_nonfinite() -> None:
    for field, bad in [
        ("step_size", float("nan")),
        ("step_size", float("inf")),
        ("step_size", -0.1),
        ("utility_decay", float("nan")),
        ("utility_decay", 1.0),
        ("utility_decay", -0.1),
        ("utility_decay", _ClassSpoof()),  # type: ignore[arg-type]
        ("sparsity", 1.5),
        ("sparsity", -0.1),
        ("sparsity", _HostileFloat(0.5)),
        ("head_step_size_multiplier", 0.0),
        ("head_step_size_multiplier", _ClassSpoof()),  # type: ignore[arg-type]
    ]:
        with pytest.raises(ValueError, match=field):
            _learner(**{field: bad})  # type: ignore[arg-type]


def test_upgd_hidden_sizes_hostile_and_range() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        _learner(hidden_sizes=(_HostileInt(4),))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        _learner(hidden_sizes=(_RaisingRepr(),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be an integer"):
        _learner(hidden_sizes=(0,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be an integer"):
        _learner(hidden_sizes=(_INT32_MAX + 1,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be a tuple"):
        _learner(hidden_sizes="4")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be an integer"):
        _learner(hidden_sizes=(True,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be an integer"):
        _learner(hidden_sizes=(4.0,))  # type: ignore[arg-type]


def test_upgd_require_float32_resource_boundaries() -> None:
    from alberta_framework.core.upgd import _require_float32_resource

    legal = _INT32_MAX // 4
    _require_float32_resource("test", vector_scalars=legal)
    with pytest.raises(ValueError, match="byte count"):
        _require_float32_resource("test", vector_scalars=legal + 1)
    with pytest.raises(ValueError, match="scalar count"):
        _require_float32_resource("test", vector_scalars=_INT32_MAX + 1)


def test_upgd_init_resource_preflight_without_allocation() -> None:
    # heads product alone should trigger byte overflow without allocating JAX arrays
    legal = _INT32_MAX // 4
    # n_heads=legal with hidden 2 => heads scalars = legal*2, bytes = 8*legal > INT32
    with pytest.raises(ValueError, match="byte count"):
        _learner(n_heads=legal, hidden_sizes=(2,))
    # feature_dim huge should trigger scalar overflow at init without allocation
    learner = _learner(n_heads=2, hidden_sizes=(4, 4))
    with pytest.raises(ValueError, match="scalar count"):
        learner.init(_INT32_MAX, jr.PRNGKey(0))
    # hidden_sizes huge scalar overflow at creation
    with pytest.raises(ValueError, match="scalar count"):
        _learner(n_heads=2, hidden_sizes=(_INT32_MAX,))


def test_upgd_feature_dim_hostile_and_range() -> None:
    learner = _learner()
    with pytest.raises(ValueError, match="must be an integer"):
        learner.init(_HostileInt(4), jr.PRNGKey(0))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        learner.init(_RaisingRepr(), jr.PRNGKey(0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be an integer"):
        learner.init(0, jr.PRNGKey(0))
    with pytest.raises(ValueError, match="must be an integer"):
        learner.init(_INT32_MAX + 1, jr.PRNGKey(0))


def test_upgd_bool_validators_reject_non_bool() -> None:
    with pytest.raises(ValueError, match="must be a bool"):
        _learner(use_layer_norm="true")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be a bool"):
        _learner(track_unit_utilities=1)  # type: ignore[arg-type]


def test_upgd_choice_validators_reject_invalid() -> None:
    with pytest.raises(ValueError, match="is unsupported"):
        _learner(perturbation_noise="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="is unsupported"):
        _learner(loss_normalization="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="is unsupported"):
        _learner(head_gradient_scale="invalid")  # type: ignore[arg-type]


def test_upgd_mapping_loaders_preserve_markers_and_exact_keys() -> None:
    learner = _learner()
    payload = learner.to_config()
    restored = UPGDLearner.from_config(MappingProxyType(payload))
    assert restored._n_heads == learner._n_heads
    assert restored._hidden_sizes == learner._hidden_sizes
    with pytest.raises(ValueError, match="type"):
        UPGDLearner.from_config({**payload, "type": "wrong"})
    # String subclass key should be rejected
    class StringSubclass(str):
        pass

    bad = {StringSubclass("type"): "UPGDLearner", "n_heads": 2, "hidden_sizes": [4]}
    with pytest.raises(ValueError, match="exact strings"):
        UPGDLearner.from_config(bad)  # type: ignore[arg-type]
    # Hostile mapping that raises on iter should be rejected
    hostile = _HostileMapping({"type": "UPGDLearner", "n_heads": 2, "hidden_sizes": [4]})
    with pytest.raises(ValueError, match="payload could not be read"):
        UPGDLearner.from_config(hostile)  # type: ignore[arg-type]


def test_upgd_valid_construction_and_roundtrip() -> None:
    learner = _learner(
        n_heads=2,
        hidden_sizes=(8, 8),
        step_size=0.01,
        utility_decay=0.9,
        sparsity=0.5,
    )
    assert learner._n_heads == 2
    assert learner._hidden_sizes == (8, 8)
    state = learner.init(4, jr.PRNGKey(0))
    assert state.trunk_params.weights[0].shape == (8, 4)
    payload = learner.to_config()
    restored = UPGDLearner.from_config(payload)
    assert restored._n_heads == learner._n_heads
    assert restored._hidden_sizes == learner._hidden_sizes
    assert restored._step_size == learner._step_size
    # numpy canonicalization should survive roundtrip
    learner2 = _learner(n_heads=np.int32(3), hidden_sizes=(np.int64(5),))
    assert type(learner2._n_heads) is int
    assert learner2._hidden_sizes == (5,)
