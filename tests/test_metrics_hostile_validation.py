"""Hostile-safe validation for metrics utilities."""

from __future__ import annotations

import numpy as np
import pytest

from alberta_framework.utils.metrics import (
    StabilityGap,
    compare_learners,
    compute_cumulative_error,
    compute_recovery_lengths,
    compute_running_mean,
    compute_stability_gap,
    extract_metric,
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

    def as_integer_ratio(self) -> tuple[int, int]:
        type(self).calls += 1
        raise RuntimeError("ratio hook")


class _StringSubclass(str):
    pass


def test_rejects_hostile_float_threshold_without_ratio() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="threshold"):
        compute_recovery_lengths([0.1, 0.9, 0.9], [0], _HostileFloat(0.8), window_size=1)
    assert _HostileFloat.calls == 0


def test_rejects_string_subclass_error_key() -> None:
    with pytest.raises(ValueError, match="error_key"):
        compute_cumulative_error([{"squared_error": 1.0}], _StringSubclass("squared_error"))


def test_rejects_string_subclass_metric_compare() -> None:
    with pytest.raises(ValueError, match="metric"):
        compare_learners({"a": [{"squared_error": 1.0}]}, metric=_StringSubclass("squared_error"))


def test_rejects_string_subclass_key_extract() -> None:
    with pytest.raises(ValueError, match="key"):
        extract_metric([{"x": 1.0}], _StringSubclass("x"))


def test_rejects_string_subclass_learner_name() -> None:
    with pytest.raises(ValueError, match="learner name"):
        compare_learners({_StringSubclass("a"): [{"squared_error": 1.0}]})


def test_rejects_hostile_int_window_size() -> None:
    with pytest.raises(ValueError, match="window_size"):
        compute_running_mean([1.0, 2.0, 3.0], window_size=_HostileInt(2))


def test_rejects_hostile_int_change_points() -> None:
    with pytest.raises(ValueError, match="change_points"):
        compute_recovery_lengths([0.1, 0.9, 0.9], [_HostileInt(0)], 0.8, window_size=1)


def test_rejects_bool_threshold() -> None:
    with pytest.raises(ValueError, match="threshold"):
        compute_recovery_lengths([0.1, 0.9], [0], True, window_size=1)


def test_rejects_hostile_float_reference_performance() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="reference_performance"):
        compute_stability_gap([0.5, 0.6], _HostileFloat(0.8))
    assert _HostileFloat.calls == 0


def test_stability_gap_rejects_hostile_float_mean() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="mean"):
        StabilityGap(mean=_HostileFloat(0.1), maximum=0.2, per_step=np.array([0.1]))
    assert _HostileFloat.calls == 0


def test_rejects_hostile_repr_window_size() -> None:
    with pytest.raises(ValueError, match="window_size"):
        compute_running_mean([1.0, 2.0], window_size=_RaisingRepr())  # type: ignore[arg-type]


def test_numpy_and_fraction_threshold_canonicalizes() -> None:
    result = compute_recovery_lengths([0.1, 0.9, 0.9], [0], np.float64(0.8), window_size=1)
    assert int(result[0]) == 2
    from fractions import Fraction

    result2 = compute_recovery_lengths([0.1, 0.9, 0.9], [0], Fraction(4, 5), window_size=1)  # type: ignore[arg-type]
    assert int(result2[0]) == 2
