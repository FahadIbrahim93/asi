"""Exact host-scalar conversion helpers for float32 JAX sinks."""

from __future__ import annotations

import math
import struct
from numbers import Integral, Real
from typing import cast


def _round_quotient_ties_to_even(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    doubled_remainder = remainder * 2
    if doubled_remainder > denominator or (
        doubled_remainder == denominator and quotient % 2 == 1
    ):
        return quotient + 1
    return quotient


def _float32_from_ratio(
    numerator: int,
    denominator: int,
    *,
    negative_zero: bool,
) -> float:
    """Round an exact ratio to IEEE 754 binary32 with ties-to-even."""
    if denominator < 0:
        numerator = -numerator
        denominator = -denominator
    if denominator == 0:
        raise ValueError("ratio denominator must be nonzero")

    negative = numerator < 0 or (numerator == 0 and negative_zero)
    magnitude = abs(numerator)
    sign_bits = int(negative) << 31
    if magnitude == 0:
        bits = sign_bits
    else:
        exponent = magnitude.bit_length() - denominator.bit_length()
        if exponent >= 0:
            scaled_numerator = magnitude
            scaled_denominator = denominator << exponent
        else:
            scaled_numerator = magnitude << (-exponent)
            scaled_denominator = denominator
        if scaled_numerator < scaled_denominator:
            exponent -= 1
        elif scaled_numerator >= (scaled_denominator << 1):
            exponent += 1

        unbiased_exponent = exponent
        if unbiased_exponent > 127:
            raise OverflowError("value overflows IEEE 754 binary32")

        if unbiased_exponent >= -126:
            shift = 23 - exponent
            if shift >= 0:
                scaled_num = magnitude << shift
                scaled_den = denominator
            else:
                scaled_num = magnitude
                scaled_den = denominator << (-shift)
            significand_full = _round_quotient_ties_to_even(scaled_num, scaled_den)
            if significand_full == 1 << 24:
                unbiased_exponent += 1
                if unbiased_exponent > 127:
                    raise OverflowError("value overflows IEEE 754 binary32")
                significand_full >>= 1
            biased_exponent = unbiased_exponent + 127
            significand_bits = significand_full & ((1 << 23) - 1)
            bits = sign_bits | (biased_exponent << 23) | significand_bits
        else:
            scaled_num = magnitude << 149
            scaled_den = denominator
            significand = _round_quotient_ties_to_even(scaled_num, scaled_den)
            if significand >= (1 << 23):
                bits = sign_bits | (1 << 23)
            else:
                bits = sign_bits | significand
    return float(struct.unpack("!f", bits.to_bytes(4, byteorder="big"))[0])


def _real_ratio(value: Real) -> tuple[int, int, bool]:
    """Return one normalized exact ratio and its zero-sign metadata."""
    actual_type = type(value)
    if issubclass(actual_type, bool) or not issubclass(actual_type, Real):
        raise TypeError("value must be an actual non-bool real")
    if issubclass(actual_type, Integral):
        ratio: object = (int(cast(Integral, value)), 1)
    else:
        ratio_method = getattr(actual_type, "as_integer_ratio", None)
        if not callable(ratio_method):
            raise TypeError("real value must expose as_integer_ratio")
        ratio = ratio_method(value)
    if type(ratio) is not tuple or len(ratio) != 2:
        raise TypeError("as_integer_ratio must return an integer pair")
    numerator_raw, denominator_raw = ratio
    num_type = type(numerator_raw)
    den_type = type(denominator_raw)
    if (
        issubclass(num_type, bool)
        or not issubclass(num_type, Integral)
        or issubclass(den_type, bool)
        or not issubclass(den_type, Integral)
    ):
        raise TypeError("as_integer_ratio must return an integer pair")
    numerator = int(numerator_raw)
    denominator = int(denominator_raw)
    if denominator < 0:
        numerator = -numerator
        denominator = -denominator
    if denominator == 0:
        raise ValueError("ratio denominator must be nonzero")
    negative_zero = numerator == 0 and math.copysign(1.0, float(value)) < 0.0
    return numerator, denominator, negative_zero


def round_real_to_float32_with_ratio(value: Real) -> tuple[int, int, float]:
    """Read one exact ratio and return it with its binary32 rounding."""
    numerator, denominator, negative_zero = _real_ratio(value)
    narrowed = _float32_from_ratio(
        numerator,
        denominator,
        negative_zero=negative_zero,
    )
    return numerator, denominator, narrowed


def round_real_to_float32(value: Real) -> float:
    """Round a standard exact-ratio real directly to IEEE binary32.

    Integer and ``as_integer_ratio`` inputs are rounded with IEEE
    round-to-nearest, ties-to-even semantics without an intermediate binary64
    conversion. Real implementations that cannot expose an exact ratio are
    rejected instead of being silently double-rounded.
    """
    return round_real_to_float32_with_ratio(value)[2]


__all__ = ["round_real_to_float32", "round_real_to_float32_with_ratio"]
