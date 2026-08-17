"""Hostile-safe validation for float32 helpers."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from alberta_framework._float32 import round_real_to_float32, round_real_to_float32_with_ratio


class _HostileFloat(float):
    calls = 0

    def as_integer_ratio(self) -> tuple[int, int]:
        type(self).calls += 1
        raise RuntimeError("ratio hook")


class _HostileInt(int):
    def __index__(self) -> int:  # pragma: no cover
        raise AssertionError("index hook")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("repr hook")


class _StringSubclass(str):
    pass


def test_rejects_bool() -> None:
    with pytest.raises(TypeError, match="actual non-bool real"):
        round_real_to_float32(True)
    with pytest.raises(TypeError, match="actual non-bool real"):
        round_real_to_float32(np.bool_(True))


def test_rejects_hostile_float_without_ratio() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(TypeError, match="actual non-bool real"):
        round_real_to_float32(_HostileFloat(0.5))
    assert _HostileFloat.calls == 0


def test_rejects_hostile_float_with_ratio_via_with_ratio() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(TypeError, match="actual non-bool real"):
        round_real_to_float32_with_ratio(_HostileFloat(1.5))
    assert _HostileFloat.calls == 0


def test_rejects_string_subclass() -> None:
    with pytest.raises(TypeError, match="actual non-bool real"):
        round_real_to_float32(_StringSubclass("1.0"))


def test_rejects_hostile_int() -> None:
    with pytest.raises(TypeError, match="actual non-bool real"):
        round_real_to_float32(_HostileInt(1))


def test_rejects_decimal() -> None:
    import decimal

    with pytest.raises(TypeError, match="actual non-bool real"):
        round_real_to_float32(decimal.Decimal("1.5"))


def test_valid_int_roundtrip() -> None:
    assert round_real_to_float32(1) == 1.0
    _, _, r = round_real_to_float32_with_ratio(42)
    assert r == 42.0


def test_valid_float_roundtrip() -> None:
    assert round_real_to_float32(0.5) == 0.5
    n, d, r = round_real_to_float32_with_ratio(0.5)
    assert (n, d) == (1, 2)


def test_valid_fraction() -> None:
    f = Fraction(3, 2)
    assert round_real_to_float32(f) == 1.5
    n, d, _ = round_real_to_float32_with_ratio(f)
    assert (n, d) == (3, 2)


def test_valid_numpy_scalars() -> None:
    assert round_real_to_float32(np.float64(0.5)) == 0.5
    assert round_real_to_float32(np.int32(5)) == 5.0
    assert round_real_to_float32(np.float32(1.5)) == 1.5


def test_rejects_hostile_ratio_return_type() -> None:
    class BadRatio(float):
        def as_integer_ratio(self) -> tuple[int, int]:
            return ["not", "tuple"]  # type: ignore[return-value]

    # Hostile subclass is rejected before ratio hook, so still TypeError
    with pytest.raises(TypeError, match="actual non-bool real"):
        round_real_to_float32(BadRatio(1.0))


def test_negative_zero_preserved() -> None:
    # -0.0 should round to -0.0 with negative_zero flag
    n, d, r = round_real_to_float32_with_ratio(-0.0)
    assert n == 0 and d == 1
    assert r == -0.0 and math.copysign(1.0, r) < 0
