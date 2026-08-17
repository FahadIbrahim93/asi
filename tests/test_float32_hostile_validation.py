"""Hostile-safe validation for float32 helpers."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from alberta_framework._float32 import round_real_to_float32, round_real_to_float32_with_ratio
from alberta_framework.core._float32_scalars import (
    validated_float32_scalar,
    validated_float32_scalar_with_ratio,
)


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


def test_rejects_class_spoof_without_invoking_class_property() -> None:
    calls = 0

    class ClassSpoof:
        @property
        def __class__(self) -> type[float]:  # pragma: no cover
            nonlocal calls
            calls += 1
            raise AssertionError("class hook")

    with pytest.raises(TypeError, match="actual non-bool real"):
        round_real_to_float32(ClassSpoof())
    assert calls == 0


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (3, 3.0),
        (np.int8(3), 3.0),
        (np.int16(3), 3.0),
        (np.int32(3), 3.0),
        (np.int64(3), 3.0),
        (np.longlong(3), 3.0),
        (np.uint8(3), 3.0),
        (np.uint16(3), 3.0),
        (np.uint32(3), 3.0),
        (np.uint64(3), 3.0),
        (np.ulonglong(3), 3.0),
        (0.25, 0.25),
        (Fraction(1, 4), 0.25),
        (np.float16(0.25), 0.25),
        (np.float32(0.25), 0.25),
        (np.float64(0.25), 0.25),
        (np.longdouble(0.25), 0.25),
    ],
)
def test_supported_exact_scalar_families_return_builtin_float(
    value: object, expected: float
) -> None:
    rounded = round_real_to_float32(value)
    assert type(rounded) is float
    assert rounded == expected


def test_shared_helper_preserves_builtin_float_storage_compatibility() -> None:
    value = 0.1
    stored, numerator, denominator = validated_float32_scalar_with_ratio("value", value)
    assert stored is value
    assert (numerator, denominator) == value.as_integer_ratio()
    assert stored != float(np.float32(value))


def test_shared_helper_canonicalizes_nonbuiltin_scalars() -> None:
    for value in (Fraction(1, 10), np.float64(0.1), np.int64(3)):
        stored = validated_float32_scalar("value", value)
        assert type(stored) is float
        assert stored == float(np.float32(value))


def test_shared_helper_rejects_hostile_name_before_formatting() -> None:
    calls = 0

    class HostileName(str):
        def __format__(self, format_spec: str) -> str:  # pragma: no cover
            nonlocal calls
            calls += 1
            raise AssertionError("format hook")

    with pytest.raises(ValueError, match="name must be an exact string"):
        validated_float32_scalar(HostileName("value"), 0.5)
    assert calls == 0


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("positive", np.bool_(True), "positive must be an exact bool"),
        ("upper_inclusive", 1, "upper_inclusive must be an exact bool"),
        ("lower", object(), "lower must be None or a finite canonical real"),
        ("lower", float("nan"), "lower must be None or a finite canonical real"),
        ("upper", float("inf"), "upper must be None or a finite canonical real"),
    ],
)
def test_shared_helper_rejects_noncanonical_policy(
    keyword: str, value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validated_float32_scalar("value", 0.5, **{keyword: value})  # type: ignore[arg-type]


def test_shared_helper_rejects_hostile_bound_without_comparison_hook() -> None:
    calls = 0

    class HostileBound(float):
        def __gt__(self, other: object) -> bool:  # pragma: no cover
            nonlocal calls
            calls += 1
            raise AssertionError("comparison hook")

        def as_integer_ratio(self) -> tuple[int, int]:  # pragma: no cover
            nonlocal calls
            calls += 1
            raise AssertionError("ratio hook")

    with pytest.raises(ValueError, match="lower must be None or a finite canonical real"):
        validated_float32_scalar("value", 0.5, lower=HostileBound(0.0))
    assert calls == 0


def test_shared_helper_rejects_bound_class_spoof_without_class_hook() -> None:
    calls = 0

    class ClassSpoof:
        @property
        def __class__(self) -> type[float]:  # pragma: no cover
            nonlocal calls
            calls += 1
            raise AssertionError("class hook")

    with pytest.raises(ValueError, match="lower must be"):
        validated_float32_scalar("value", 0.5, lower=ClassSpoof())
    assert calls == 0


@pytest.mark.parametrize(
    "bound",
    [
        pytest.param(10**10_000, id="integer"),
        pytest.param(Fraction(10**10_000, 3), id="fraction"),
    ],
)
def test_shared_helper_accepts_huge_finite_bounds_with_bounded_errors(bound: object) -> None:
    assert validated_float32_scalar("value", 0.5, upper=bound) == 0.5
    with pytest.raises(ValueError, match="finite exact bound"):
        validated_float32_scalar("value", 0.5, lower=bound)


def test_shared_helper_accepts_exact_canonical_bound_families() -> None:
    one_third = Fraction(1, 3)
    assert validated_float32_scalar("value", one_third, lower=one_third) == float(
        np.float32(one_third)
    )
    assert validated_float32_scalar("value", 1, lower=np.int32(1), upper=np.float64(1.0)) == 1.0


def test_exact_fraction_bound_survives_binary64_and_float32_narrowing() -> None:
    just_above_one = Fraction(2**54 + 1, 2**54)
    with pytest.raises(ValueError, match="must remain"):
        validated_float32_scalar("value", just_above_one, lower=just_above_one)


def test_shared_helper_rejects_empty_or_impossible_policy_domains() -> None:
    with pytest.raises(ValueError, match="non-empty interval"):
        validated_float32_scalar("value", 0.5, lower=1.0, upper=1.0, upper_inclusive=False)
    with pytest.raises(ValueError, match="permit a positive value"):
        validated_float32_scalar("value", 0.5, positive=True, upper=0.0)


def test_shared_helper_checks_exact_and_narrowed_endpoints() -> None:
    half_minimum_subnormal = Fraction(1, 2**150)
    with pytest.raises(ValueError, match="remain positive once narrowed"):
        validated_float32_scalar("value", half_minimum_subnormal, positive=True)

    # Inclusive zero permits an exactly positive host value that canonically
    # narrows to zero; consumers that require nonzero must request positive.
    assert validated_float32_scalar("value", half_minimum_subnormal, lower=0.0) == 0.0

    above_one = Fraction(1, 1) + Fraction(1, 2**80)
    with pytest.raises(ValueError, match=r"must be in \[0.0, 1.0\]"):
        validated_float32_scalar("value", above_one, lower=0.0, upper=1.0)

    below_one_rounding_up = Fraction(1, 1) - Fraction(1, 2**80)
    with pytest.raises(ValueError, match="once narrowed"):
        validated_float32_scalar(
            "value", below_one_rounding_up, lower=0.0, upper=1.0, upper_inclusive=False
        )


def test_shared_helper_rejects_float32_overflow_midpoint() -> None:
    maximum = (2**24 - 1) * 2**104
    midpoint = Fraction(maximum + 2**103)
    assert validated_float32_scalar("value", midpoint - 1) == float(np.finfo(np.float32).max)
    with pytest.raises(ValueError, match="remain finite once narrowed"):
        validated_float32_scalar("value", midpoint)
