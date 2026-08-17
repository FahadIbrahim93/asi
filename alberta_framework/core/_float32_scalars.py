"""Shared validation for configuration scalars that are consumed as float32.

A configuration scalar is checked in both domains that matter: the exact host
value (so a lying ``as_integer_ratio`` or a value that only *rounds* into
range cannot pass) and its binary32 rounding (so a host-finite value that
narrows to infinity, zero, or the excluded end of a half-open interval is
refused before it can freeze an EMA or divide by zero at the sink).  Only an
actual built-in ``float`` is stored as-is — JAX narrows it once, exactly as
validated here — while ints and other reals are stored as the validated
binary32 value.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple, cast

from alberta_framework._float32 import round_real_to_float32_with_ratio


class _Bound(NamedTuple):
    display: float
    numerator: int
    denominator: int


def _optional_bound(name: str, value: object) -> _Bound | None:
    if value is None:
        return None
    try:
        numerator, denominator, _ = round_real_to_float32_with_ratio(value)
        display = float(cast(Any, value))
    except Exception as error:
        raise ValueError(f"{name} must be None or a finite canonical real") from error
    if not math.isfinite(display):
        raise ValueError(f"{name} must be None or a finite canonical real")
    return _Bound(display, numerator, denominator)


def _compare_ratios(
    left_numerator: int,
    left_denominator: int,
    right_numerator: int,
    right_denominator: int,
) -> int:
    left = left_numerator * right_denominator
    right = right_numerator * left_denominator
    return (left > right) - (left < right)


def _validate_policy(
    name: object,
    *,
    positive: object,
    lower: object | None,
    upper: object | None,
    upper_inclusive: object,
) -> tuple[str, bool, _Bound | None, _Bound | None, bool]:
    """Validate trusted policy metadata before it can run formatting or comparison hooks."""
    if type(name) is not str:
        raise ValueError("name must be an exact string")
    if type(positive) is not bool:
        raise ValueError("positive must be an exact bool; built-in bool required")
    if type(upper_inclusive) is not bool:
        raise ValueError("upper_inclusive must be an exact bool; built-in bool required")
    checked_lower = _optional_bound("lower", lower)
    checked_upper = _optional_bound("upper", upper)
    if checked_lower is not None and checked_upper is not None:
        order = _compare_ratios(
            checked_lower.numerator,
            checked_lower.denominator,
            checked_upper.numerator,
            checked_upper.denominator,
        )
    else:
        order = -1
    if order > 0 or (order == 0 and not upper_inclusive):
        raise ValueError("float32 scalar bounds must describe a non-empty interval")
    if positive and checked_upper is not None and checked_upper.numerator <= 0:
        raise ValueError("positive float32 scalar policy must permit a positive value")
    return name, positive, checked_lower, checked_upper, upper_inclusive


def validated_float32_scalar(
    name: object,
    value: object,
    *,
    positive: object = False,
    lower: object | None = None,
    upper: object | None = None,
    upper_inclusive: object = True,
) -> float:
    """Return the canonical stored value of one float32-consumed scalar or fail closed.

    Raises:
        ValueError: If ``value`` is not an actual non-bool real, does not narrow
            to a finite binary32, or leaves the declared domain either as the
            exact host value or once narrowed to binary32.
    """
    stored, _, _ = validated_float32_scalar_with_ratio(
        name,
        value,
        positive=positive,
        lower=lower,
        upper=upper,
        upper_inclusive=upper_inclusive,
    )
    return stored


def validated_float32_scalar_with_ratio(
    name: object,
    value: object,
    *,
    positive: object = False,
    lower: object | None = None,
    upper: object | None = None,
    upper_inclusive: object = True,
) -> tuple[float, int, int]:
    """Validate once and also return the exact host numerator and denominator."""
    name, positive, lower, upper, upper_inclusive = _validate_policy(
        name,
        positive=positive,
        lower=lower,
        upper=upper,
        upper_inclusive=upper_inclusive,
    )
    try:
        numerator, denominator, narrowed = round_real_to_float32_with_ratio(value)
    except Exception as error:
        raise ValueError(f"{name} must be a finite real number") from error
    if not math.isfinite(narrowed):
        raise ValueError(f"{name} must remain finite once narrowed to float32")

    narrowed_numerator, narrowed_denominator = narrowed.as_integer_ratio()

    def narrowed_in_domain() -> bool:
        if positive and narrowed_numerator <= 0:
            return False
        if lower is not None and _compare_ratios(
            narrowed_numerator,
            narrowed_denominator,
            lower.numerator,
            lower.denominator,
        ) < 0:
            return False
        if upper is not None:
            comparison = _compare_ratios(
                narrowed_numerator,
                narrowed_denominator,
                upper.numerator,
                upper.denominator,
            )
            return comparison <= 0 if upper_inclusive else comparison < 0
        return True

    def exact_in_domain() -> bool:
        if positive and numerator <= 0:
            return False
        if lower is not None and _compare_ratios(
            numerator, denominator, lower.numerator, lower.denominator
        ) < 0:
            return False
        if upper is not None:
            comparison = _compare_ratios(
                numerator, denominator, upper.numerator, upper.denominator
            )
            if comparison > 0 or (comparison == 0 and not upper_inclusive):
                return False
        return True

    domain = _describe_domain(
        positive,
        None if lower is None else lower.display,
        None if upper is None else upper.display,
        upper_inclusive,
    )
    if not exact_in_domain():
        raise ValueError(f"{name} must be {domain}")
    if not narrowed_in_domain():
        raise ValueError(f"{name} must remain {domain} once narrowed to float32")
    stored = value if type(value) is float else narrowed
    return stored, numerator, denominator


def _describe_domain(
    positive: bool,
    lower: object | None,
    upper: object | None,
    upper_inclusive: bool,
) -> str:
    if upper is not None:
        floor: object = lower if lower is not None else "-inf"
        bracket = "]" if upper_inclusive else ")"
        return f"in [{floor}, {upper}{bracket}"
    if positive:
        return "positive"
    if lower is not None:
        return f">= {lower}"
    return "finite"


__all__ = ["validated_float32_scalar", "validated_float32_scalar_with_ratio"]
