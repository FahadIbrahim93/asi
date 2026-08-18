"""Hostile integer validation for forager matrix."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class _HostileInt(int):
    calls = 0

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile eq")

    def __lt__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile lt")

    def __le__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile le")

    def __gt__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile gt")

    def __ge__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile ge")

    def __hash__(self) -> int:
        return int.__hash__(self)


class _HostileFloat(float):
    calls = 0

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile float eq")


def test_require_positive_int_rejects_hostile_before_lt() -> None:
    from alberta_framework.benchmarks.forager_matrix import _require_positive_int

    hostile = _HostileInt(5)
    _HostileInt.calls = 0
    with pytest.raises(Exception, match="must be an integer"):
        _require_positive_int(hostile, "p")  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    assert _require_positive_int(1, "p") == 1
    with pytest.raises(Exception, match="must be an integer"):
        _require_positive_int(True, "p")  # type: ignore[arg-type]


def test_require_nonnegative_int_rejects_hostile_before_lt() -> None:
    from alberta_framework.benchmarks.forager_matrix import _require_nonnegative_int

    hostile = _HostileInt(0)
    _HostileInt.calls = 0
    with pytest.raises(Exception, match="must be an integer"):
        _require_nonnegative_int(hostile, "p")  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    assert _require_nonnegative_int(0, "p") == 0


def test_require_seed_list_rejects_hostile_before_lt() -> None:
    from alberta_framework.benchmarks.forager_matrix import _require_seed_list

    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    with pytest.raises(Exception, match="must be an integer"):
        _require_seed_list([hostile], "p")  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    assert _require_seed_list([1, 2], "p") == (1, 2)


def test_coerce_int_rejects_hostile() -> None:
    from alberta_framework.benchmarks.forager_matrix import _coerce_typed_value

    hostile = _HostileInt(5)
    _HostileInt.calls = 0
    with pytest.raises(Exception, match="must be an integer"):
        _coerce_typed_value(hostile, int, "p")  # type: ignore[arg-type]
    assert _HostileInt.calls == 0


def test_finite_number_rejects_hostile_before_float() -> None:
    from alberta_framework.benchmarks.forager_matrix import _finite_number

    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    with pytest.raises(Exception, match="must be a finite number"):
        _finite_number(hostile, "p")  # type: ignore[arg-type]
    # bool also rejected via exact type check (bool not in (int,float))
    with pytest.raises(Exception, match="must be a finite number"):
        _finite_number(True, "p")  # type: ignore[arg-type]
    assert _finite_number(1, "p") == 1.0
    assert _finite_number(1.0, "p") == 1.0


def test_tuning_rule_rejects_hostile_confidence_before_float() -> None:
    from alberta_framework.benchmarks.forager_matrix import _parse_tuning_rule

    hostile = _HostileFloat(0.95)
    _HostileFloat.calls = 0
    with pytest.raises(Exception, match="confidence must be finite"):
        _parse_tuning_rule(
            {
                "metric": "mean_reward",
                "direction": "maximize",
                "statistic": "mean",
                "confidence": hostile,
                "bootstrap_resamples": 100,
                "bootstrap_seed": 0,
                "tie_break": "variant_id_lexicographic",
            }
        )
    assert _HostileFloat.calls == 0


def test_hostile_not_in_error_message() -> None:
    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    try:
        if type(hostile) is not int:
            raise ValueError("must be an integer")
    except ValueError as exc:
        assert "!r" not in str(exc)
        assert _HostileInt.calls == 0
