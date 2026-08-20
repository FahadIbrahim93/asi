# mypy: disable-error-code="arg-type"
"""Resource boundaries for numeric-trace validation."""

from __future__ import annotations

import numpy as np
import pytest

from alberta_framework.utils import metrics
from alberta_framework.utils.metrics import (
    compute_cumulative_error,
    compute_running_mean,
)

pytestmark = pytest.mark.unit


def test_running_mean_rejects_oversized_trace_before_numeric_work(
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


def test_index_list_rejects_oversized_width_before_item_walk(
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


def test_trace_budget_counts_root_and_rejects_first_non_fit(
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


def test_dense_numeric_array_is_not_a_python_traversal_node_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dense typed leaves keep their public numeric-array compatibility."""
    monkeypatch.setattr(metrics, "_BOOLEAN_TRACE_MAX_NODES", 8)
    view = np.broadcast_to(np.asarray(1.0, dtype=np.float64), (9,))
    result = compute_running_mean(view, window_size=2)
    assert np.isnan(result[0])
    np.testing.assert_array_equal(result[1:], np.ones(8, dtype=np.float64))


def test_dense_numeric_broadcast_view_rejects_before_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(metrics, "_NUMERIC_TRACE_MAX_VALUES", 8)
    view = np.broadcast_to(np.asarray(1.0, dtype=np.float64), (9,))
    with pytest.raises(ValueError, match="dense numeric value limit"):
        compute_running_mean(view, window_size=2)


def test_nested_dense_view_rejects_before_coercion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(metrics, "_NUMERIC_TRACE_MAX_VALUES", 8)
    view = np.broadcast_to(np.asarray(1.0, dtype=np.float64), (9,))

    def fail_asarray(*args: object, **kwargs: object) -> object:
        raise AssertionError("coercion ran before the nested dense-value gate")

    monkeypatch.setattr(metrics.np, "asarray", fail_asarray)
    with pytest.raises(ValueError, match="dense numeric value limit"):
        metrics._numeric_array([view], name="values")


def test_shared_dense_views_use_one_aggregate_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(metrics, "_NUMERIC_TRACE_MAX_VALUES", 8)
    view = np.broadcast_to(np.asarray(1.0, dtype=np.float64), (5,))
    with pytest.raises(ValueError, match="dense numeric value limit"):
        metrics._reject_boolean_numeric_trace([view, view], name="values")


def test_numeric_trace_rejects_hostile_metaclass_without_hash_or_equality() -> None:
    class HostileMeta(type):
        calls = 0

        def __hash__(cls) -> int:
            type(cls).calls += 1
            raise AssertionError("hostile metaclass hash")

        def __eq__(cls, other: object) -> bool:
            type(cls).calls += 1
            raise AssertionError("hostile metaclass equality")

    class HostileValue(metaclass=HostileMeta):
        pass

    with pytest.raises(ValueError, match="exact real numeric values"):
        metrics._reject_boolean_numeric_trace(HostileValue(), name="values")
    assert HostileMeta.calls == 0


def test_metric_history_rejects_wide_record_before_key_iteration() -> None:
    record = {f"k{index}": 1.0 for index in range(4_097)}
    record["squared_error"] = 1.0
    with pytest.raises(ValueError, match="metric-record key limit"):
        compute_cumulative_error([record])


def test_dense_index_array_is_not_a_python_traversal_node_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(metrics, "_BOOLEAN_TRACE_MAX_NODES", 8)
    view = np.arange(9, dtype=np.int64)
    result = metrics._require_index_vector(view, name="indices")
    np.testing.assert_array_equal(result, view)


def test_dense_index_broadcast_view_rejects_before_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(metrics, "_NUMERIC_TRACE_MAX_VALUES", 8)
    view = np.broadcast_to(np.asarray(0, dtype=np.int64), (9,))
    with pytest.raises(ValueError, match="dense numeric value limit"):
        metrics._require_index_vector(view, name="indices")


def test_compare_learners_shares_one_history_item_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(metrics, "_BOOLEAN_TRACE_MAX_NODES", 8)
    shared = [{"squared_error": 1.0}, {"squared_error": 2.0}]
    with pytest.raises(ValueError, match="aggregate metric-record item limit"):
        metrics.compare_learners({"a": shared, "b": shared})
