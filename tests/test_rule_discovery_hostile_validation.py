"""Hostile validation for rule discovery int gates.

Search ints must stay inside [minimum, 2**31-1] so a 10**12 generations
value cannot allocate or hang the micro-suite screen.
"""

from __future__ import annotations

import pathlib

import pytest

from alberta_framework.benchmarks.rule_discovery import (
    _SEARCH_CANDIDATE_EVALS_MAX,
    _SEARCH_INT_MAX_BY_NAME,
    _SEARCH_STREAM_STEPS_MAX,
    _require_search_int,
    _require_search_work_unit,
)


class _HostileInt(int):
    calls = 0

    def __repr__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("repr hook")

    def __str__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("str hook")


class _EvilStr(str):
    calls = 0

    def __repr__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("repr hook")


def test_require_search_int_rejects_hostile_without_hooks() -> None:
    evil = _HostileInt(1)
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="must be an integer") as exc:
        _require_search_int("my_param", evil, minimum=1)  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    assert "!r" not in str(exc.value)
    assert "HostileInt" not in str(exc.value)


def test_require_search_int_rejects_string_subclass_name() -> None:
    evil = _EvilStr("my_param")
    _EvilStr.calls = 0
    with pytest.raises(ValueError, match="exact string"):
        _require_search_int(evil, 1, minimum=0)  # type: ignore[arg-type]
    assert _EvilStr.calls == 0


def test_require_search_int_rejects_below_minimum_sanitized() -> None:
    with pytest.raises(ValueError, match="must be an integer") as exc:
        _require_search_int("my_param", 0, minimum=1)
    assert "!r" not in str(exc.value)
    assert "my_param" in str(exc.value)


def test_source_has_no_repr_leak() -> None:
    text = pathlib.Path("alberta_framework/benchmarks/rule_discovery.py").read_text()
    assert "!r" not in text


def test_valid_int_passes() -> None:
    assert _require_search_int("my_param", 5, minimum=1) == 5


def test_require_search_int_rejects_unbounded_int() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        _require_search_int("task_length", 10**12, minimum=1)


def test_require_search_int_last_fit_and_first_overflow() -> None:
    for name, maximum in _SEARCH_INT_MAX_BY_NAME.items():
        assert _require_search_int(name, maximum, minimum=0) == maximum
        with pytest.raises(ValueError, match="must be an integer"):
            _require_search_int(name, maximum + 1, minimum=0)


def test_require_search_work_unit_combined_product() -> None:
    _require_search_work_unit(n_random=_SEARCH_CANDIDATE_EVALS_MAX)
    with pytest.raises(ValueError, match="candidate evaluations"):
        _require_search_work_unit(n_random=_SEARCH_CANDIDATE_EVALS_MAX + 1)
    _require_search_work_unit(
        n_random=0, n_tasks=2, task_length=_SEARCH_STREAM_STEPS_MAX // 2
    )
    with pytest.raises(ValueError, match="stream steps"):
        _require_search_work_unit(n_random=0, n_tasks=2, task_length=5_000_001)
