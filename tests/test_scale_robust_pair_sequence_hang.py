"""Scale-robust pair sequences reject oversized host collections before hang.

Origin ``ConditionSeedRecord`` rebuilt every integer pair with no count bound.
A cheap ``((0, 1),) * 25_000_000`` pointer-repeat took 4.107s on origin/main.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import cast

import pytest

from alberta_framework.evaluation.scale_robust_feature import (
    _MAX_INTEGER_PAIRS,
    CONDITION_PRIMARY,
    ConditionSeedRecord,
    count_relevant_context_pairs,
    count_relevant_context_pairs_by_task,
)

pytestmark = pytest.mark.unit

_PAIR = (0, 12)


class _HostileMeta(type):
    calls = 0

    def __eq__(cls, other: object) -> bool:
        del other
        cls.calls += 1
        raise AssertionError("hostile metaclass equality must not run")

    def __hash__(cls) -> int:
        cls.calls += 1
        raise AssertionError("hostile metaclass hashing must not run")


class _HostileSequence(tuple[object, ...], metaclass=_HostileMeta):
    pass


def test_frozen_pair_count_bound() -> None:
    assert _MAX_INTEGER_PAIRS == 4096


def test_last_fit_pair_count_is_accepted() -> None:
    pairs = (_PAIR,) * _MAX_INTEGER_PAIRS
    record = ConditionSeedRecord(
        seed=0,
        condition=CONDITION_PRIMARY,
        phases=(),
        end_segment_5_active_pairs=pairs,
        end_segment_7_active_pairs=(),
        final_active_pairs=(),
    )
    assert len(record.end_segment_5_active_pairs) == _MAX_INTEGER_PAIRS
    assert count_relevant_context_pairs(pairs) == 1
    assert count_relevant_context_pairs_by_task(pairs) == (1, 0)


@pytest.mark.parametrize("count", [4097, 25_000_000])
def test_condition_record_rejects_oversized_pair_pointer_repeat(count: int) -> None:
    started = time.perf_counter()
    with pytest.raises(ValueError, match="4096-pair limit"):
        ConditionSeedRecord(
            seed=0,
            condition=CONDITION_PRIMARY,
            phases=(),
            end_segment_5_active_pairs=(_PAIR,) * count,
            end_segment_7_active_pairs=(),
            final_active_pairs=(),
        )
    assert time.perf_counter() - started < 0.5


@pytest.mark.parametrize("count", [4097, 25_000_000])
def test_pair_counters_reject_oversized_pointer_repeat(count: int) -> None:
    pairs = (_PAIR,) * count
    started = time.perf_counter()
    with pytest.raises(ValueError, match="4096-pair limit"):
        count_relevant_context_pairs(pairs)
    with pytest.raises(ValueError, match="4096-pair limit"):
        count_relevant_context_pairs_by_task(pairs)
    assert time.perf_counter() - started < 0.5


def test_pair_admission_rejects_hostile_sequence_without_type_hooks() -> None:
    _HostileMeta.calls = 0
    hostile = _HostileSequence((_PAIR,))
    hostile_pairs = cast(Sequence[tuple[int, int]], hostile)
    with pytest.raises(TypeError, match="exact tuple or list"):
        count_relevant_context_pairs(hostile_pairs)
    with pytest.raises(TypeError, match="exact tuple or list"):
        count_relevant_context_pairs_by_task(hostile_pairs)
    assert _HostileMeta.calls == 0


def test_pair_counters_reject_hostile_nested_values_before_comparison() -> None:
    class HostileInt(int):
        def __lt__(self, other: object) -> bool:
            del other
            raise AssertionError("hostile comparison must not run")

        def __hash__(self) -> int:
            raise AssertionError("hostile hashing must not run")

    pairs = ((HostileInt(0), 12),)
    with pytest.raises(ValueError, match="must contain integers"):
        count_relevant_context_pairs(pairs)
    with pytest.raises(ValueError, match="must contain integers"):
        count_relevant_context_pairs_by_task(pairs)
