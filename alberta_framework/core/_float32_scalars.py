"""Shared validation for configuration scalars consumed as float32."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, cast

from alberta_framework._float32 import round_real_to_float32_with_ratio


def validated_float32_scalar(
    name: str,
    value: object,
    *,
    positive: bool = False,
    lower: float | None = None,
    upper: float | None = None,
    upper_inclusive: bool = True,
) -> float:
    """Return the canonical stored float32-consumed scalar or fail closed."""
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
    name: str,
    value: object,
    *,
    positive: bool = False,
    lower: float | None = None,
    upper: float | None = None,
    upper_inclusive: bool = True,
) -> tuple[float, int, int]:
    """Validate a scalar and return its exact host ratio."""
    actual_type = type(value)
    if issubclass(actual_type, bool) or not issubclass(actual_type, Real):
        raise ValueError(f"{name} must be a finite real number")
    real = cast(Any, value)
    try:
        numerator, denominator, narrowed = round_real_to_float32_with_ratio(real)
    except Exception as error:
        raise ValueError(f"{name} must be a finite real number") from error
    if not math.isfinite(narrowed):
        raise ValueError(f"{name} must remain finite once narrowed to float32")

    def narrowed_in_domain(candidate: float) -> bool:
        if positive and candidate <= 0.0:
            return False
        if lower is not None and candidate < lower:
            return False
        if upper is not None:
            return bool(candidate <= upper) if upper_inclusive else bool(candidate < upper)
        return True

    def ratio_compares_to(bound: float) -> int:
        bound_numerator, bound_denominator = bound.as_integer_ratio()
        left = numerator * bound_denominator
        right = bound_numerator * denominator
        return (left > right) - (left < right)

    def exact_in_domain() -> bool:
        if positive and numerator <= 0:
            return False
        if lower is not None and ratio_compares_to(lower) < 0:
            return False
        if upper is not None:
            comparison = ratio_compares_to(upper)
            if comparison > 0 or (comparison == 0 and not upper_inclusive):
                return False
        return True

    domain = _describe_domain(positive, lower, upper, upper_inclusive)
    if not exact_in_domain():
        raise ValueError(f"{name} must be {domain}")
    if not narrowed_in_domain(narrowed):
        raise ValueError(f"{name} must remain {domain} once narrowed to float32")
    stored = real if type(real) is float else narrowed
    return stored, numerator, denominator


def _describe_domain(
    positive: bool,
    lower: float | None,
    upper: float | None,
    upper_inclusive: bool,
) -> str:
    if upper is not None:
        floor = lower if lower is not None else "-inf"
        bracket = "]" if upper_inclusive else ")"
        return f"in [{floor}, {upper}{bracket}"
    if positive:
        return "positive"
    if lower is not None:
        return f">= {lower}"
    return "finite"


__all__ = ["validated_float32_scalar", "validated_float32_scalar_with_ratio"]
