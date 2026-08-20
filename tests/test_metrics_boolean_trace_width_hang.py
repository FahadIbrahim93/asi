# mypy: disable-error-code="arg-type"
"""Boolean-trace walks reject oversized host lists before the width hang."""

from __future__ import annotations

import time

import numpy as np
import pytest

from alberta_framework.utils import metrics
from alberta_framework.utils.metrics import (
    _BOOLEAN_TRACE_MAX_NODES,
    compute_cumulative_error,
    compute_running_mean,
)

pytestmark = pytest.mark.unit


def test_running_mean_rejects_origin_hang_class_before_trace_walk() -> None:
    started = time.perf_counter()
    with pytest.raises(ValueError, match="boolean-trace value limit"):
        compute_running_mean([0.0] * (_BOOLEAN_TRACE_MAX_NODES + 1), window_size=2)
    assert time.perf_counter() - started < 0.25


def test_cumulative_error_rejects_oversized_metrics_history_before_walk() -> None:
    started = time.perf_counter()
    with pytest.raises(ValueError, match="boolean-trace value limit"):
        compute_cumulative_error([{"squared_error": 1.0}] * (_BOOLEAN_TRACE_MAX_NODES + 1))
    assert time.perf_counter() - started < 0.25


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


def test_running_mean_accepts_public_last_fit() -> None:
    result = compute_running_mean([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], window_size=3)
    assert result.shape == (6,)


def test_running_mean_rejects_oversized_broadcast_view_before_numeric_work() -> None:
    view = np.broadcast_to(
        np.asarray(0.0, dtype=np.float64), (_BOOLEAN_TRACE_MAX_NODES + 1,)
    )
    with pytest.raises(ValueError, match="boolean-trace value limit"):
        compute_running_mean(view, window_size=2)


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
