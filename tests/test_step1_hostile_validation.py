"""Hostile validation for Step 1 kernel facade."""

from fractions import Fraction
from typing import Any, cast

import numpy as np
import pytest

from alberta_framework.steps.step1 import Step1KernelConfig


class _EvilStr(str):
    def __str__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__str__ must not be called")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__repr__ must not be called")


class _StringSubclass(str):
    pass


class _HostileInt(int):
    calls = 0

    def __index__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileInt.__index__ must not be called")

    def __repr__(self) -> str:
        type(self).calls += 1
        raise AssertionError("HostileInt.__repr__ must not be called")


class _HostileFloat(float):
    calls = 0

    def as_integer_ratio(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileFloat.as_integer_ratio must not be called")

    def __float__(self) -> float:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileFloat.__float__ must not be called")

    def __repr__(self) -> str:
        type(self).calls += 1
        raise AssertionError("HostileFloat.__repr__ must not be called")


def test_rejects_string_subclass_for_feature_dim() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        Step1KernelConfig(feature_dim=_StringSubclass("4"))  # type: ignore[arg-type]


def test_hostile_str_for_feature_dim_without_repr_leak() -> None:
    evil = _EvilStr("4")
    with pytest.raises(ValueError, match="must be an integer") as exc:
        Step1KernelConfig(feature_dim=evil)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_rejects_bool_and_hostile_int() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        Step1KernelConfig(feature_dim=True)  # type: ignore[arg-type]
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="must be an integer") as exc:
        Step1KernelConfig(feature_dim=_HostileInt(4))  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    assert "HostileInt" not in str(exc.value)


def test_rejects_hostile_float_without_hook_and_repr_leak() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="must be a real number") as exc:
        Step1KernelConfig(step_size=_HostileFloat(0.01))  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0
    assert "HostileFloat" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_does_not_invoke_hostile_value_when_name_is_evil_via_sink() -> None:
    from alberta_framework.steps.step1 import finite_real_and_float32

    evil = _EvilStr("x")
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="must be an exact string"):
        finite_real_and_float32(evil, _HostileFloat(1.0))  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0


def test_rejects_plain_string_for_step_size() -> None:
    with pytest.raises(ValueError, match="must be a real number"):
        Step1KernelConfig(step_size="0.01")  # type: ignore[arg-type]


def test_rejects_string_subclass_for_step_size() -> None:
    with pytest.raises(ValueError, match="must be a real number"):
        Step1KernelConfig(step_size=_StringSubclass("0.01"))  # type: ignore[arg-type]


def test_rejects_negative_step_size_without_repr() -> None:
    with pytest.raises(ValueError, match="must be non-negative") as exc:
        Step1KernelConfig(step_size=-0.01)
    assert "-0.01" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_rejects_non_tuple_int_without_repr_via_feature_dim() -> None:
    # feature_dim out of bounds also sanitized
    with pytest.raises(ValueError, match="must be positive") as exc:
        Step1KernelConfig(feature_dim=0)
    assert "!r" not in str(exc.value)


def test_rejects_unknown_optimizer_without_repr() -> None:
    # unknown optimizer message sanitized, no !r leak
    with pytest.raises(ValueError, match="unknown Step 1 optimizer") as exc:
        Step1KernelConfig(optimizer="evil")  # type: ignore[arg-type]
    assert "evil" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_valid_configs_still_pass() -> None:
    cfg = Step1KernelConfig(feature_dim=4, num_relevant=2, step_size=0.01)
    assert cfg.feature_dim == 4
    assert cfg.step_size == pytest.approx(0.01)
    cfg2 = Step1KernelConfig(
        feature_dim=10,
        num_relevant=3,
        step_size=0.02,
        meta_step_size=0.02,
    )
    assert cfg2.num_relevant == 3


def test_numpy_scalars_pass() -> None:
    cfg = Step1KernelConfig(
        feature_dim=cast(Any, np.int32(10)),
        num_relevant=cast(Any, np.int32(3)),
        step_size=cast(Any, np.float32(0.01)),
        meta_step_size=cast(Any, np.float64(0.02)),
    )
    assert cfg.feature_dim == 10
    cfg2 = Step1KernelConfig(step_size=cast(Any, Fraction(1, 100)))
    assert cfg2.step_size == pytest.approx(0.01)


def test_float_subclass_with_lying_ratio_is_rejected() -> None:
    class RatioFloat(float):
        calls = 0

        def as_integer_ratio(self) -> tuple[int, int]:
            type(self).calls += 1
            return (1, 100)

    with pytest.raises(ValueError, match="must be a real number"):
        Step1KernelConfig(step_size=RatioFloat(0.01))
    assert RatioFloat.calls == 0
