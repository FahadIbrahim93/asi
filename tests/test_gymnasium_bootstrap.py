"""Tests for Gymnasium VALUE bootstrap algebra that do not need Gymnasium."""

from __future__ import annotations

import math

from alberta_framework.streams.gymnasium import _discounted_bootstrap


def test_zero_gamma_skips_inf_bootstrap() -> None:
    """gamma=0 is no bootstrap; 0 * inf V(s') is NaN.

    Fail-closed: a zero discount does not multiply the later value.
    """
    assert _discounted_bootstrap(0.0, float("inf")) == 0.0
    assert math.isfinite(_discounted_bootstrap(0.0, float("inf")))


def test_positive_gamma_keeps_finite_and_inf_products() -> None:
    assert _discounted_bootstrap(0.5, 10.0) == 5.0
    assert math.isinf(_discounted_bootstrap(0.99, float("inf")))
