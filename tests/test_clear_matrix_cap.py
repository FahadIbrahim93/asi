"""Protocol ceilings for CLEAR accuracy-matrix enumeration."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.clear_qualification import (
    MAX_METRIC_MATRIX_ROWS,
    ClearQualificationError,
    _metric_values,
)


def test_documented_protocol_ceiling() -> None:
    assert MAX_METRIC_MATRIX_ROWS == 10


def test_rejects_oversized_accuracy_matrix() -> None:
    with pytest.raises(ClearQualificationError, match="accuracy matrix"):
        _metric_values([[0.0] for _ in range(MAX_METRIC_MATRIX_ROWS + 1)])


def test_rejects_short_or_ragged_matrix_before_nested_enumeration() -> None:
    with pytest.raises(ClearQualificationError, match="exact 10x10"):
        _metric_values([[0.0] * 10 for _ in range(9)])
    with pytest.raises(ClearQualificationError, match="exact 10x10"):
        _metric_values([[0.0] * 10 for _ in range(9)] + [[0.0] * 11])
