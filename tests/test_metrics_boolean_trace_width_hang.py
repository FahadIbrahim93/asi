# mypy: disable-error-code="arg-type"
"""Boolean-trace walks reject oversized host lists before the width hang."""

from __future__ import annotations

import numpy as np
import pytest

from alberta_framework.utils import metrics
from alberta_framework.utils.metrics import (
    compute_cumulative_error,
    compute_running_mean,
)

pytestmark = pytest.mark.unit


def test_running_mean_rejects_origin_hang_class_before_trace_walk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(metrics, "_BOOLEAN_TRACE_MAX_NODES", 8)
    with pytest.raises(ValueError, match="boolean-trace value limit"):
        compute_running_mean([0.0] * 9, window_size=2)


def test_cumulative_error_rejects_oversized_metrics_history_before_walk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(metrics, "_BOOLEAN_TRACE_MAX_NODES", 8)
    with pytest.raises(ValueError, match="boolean-trace value limit"):
        compute_cumulative_error([{"squared_error": 1.0}] * 9)


def test_object_array_rejects_oversized_width_before_flat_walk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(metrics, "_BOOLEAN_TRACE_MAX_NODES", 8)
    values = np.empty(9, dtype=object)
    values.fill(0.0)
    with pytest.raises(ValueError, match="boolean-trace value limit"):
        metrics._reject_boolean_numeric_trace(values, name="values")


def test_index_vector_rejects_oversized_width_before_item_walk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(metrics, "_BOOLEAN_TRACE_MAX_NODES", 8)
    with pytest.raises(ValueError, match="boolean-trace value limit"):
        metrics._require_index_vector([0] * 9, name="indices")


def test_nested_shared_trace_uses_one_traversal_wide_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Small aliased containers must not amplify beyond the global budget."""
    monkeypatch.setattr(metrics, "_BOOLEAN_TRACE_MAX_NODES", 8)
    leaf = [0.0, 1.0]
    middle = [leaf, leaf]
    root = [middle, middle]
    with pytest.raises(ValueError, match="boolean-trace value limit"):
        metrics._reject_boolean_numeric_trace(root, name="values")


def test_trace_budget_counts_the_root_and_rejects_the_first_non_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(metrics, "_BOOLEAN_TRACE_MAX_NODES", 4)
    metrics._reject_boolean_numeric_trace([0.0, 1.0, 2.0], name="values")
    with pytest.raises(ValueError, match="boolean-trace value limit"):
        metrics._reject_boolean_numeric_trace([0.0, 1.0, 2.0, 3.0], name="values")
    with pytest.raises(ValueError, match="boolean-trace value limit"):
        metrics._reject_boolean_numeric_trace(
            np.asarray([0.0, 1.0, 2.0, 3.0], dtype=object),
            name="values",
        )


def test_running_mean_accepts_public_last_fit() -> None:
    result = compute_running_mean([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], window_size=3)
    assert result.shape == (6,)
