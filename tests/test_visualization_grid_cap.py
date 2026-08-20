"""Protocol ceilings for visualization grid and smoothing lengths."""

from __future__ import annotations

import pytest

from alberta_framework.utils.visualization import (
    _VIS_GRID_MAX,
    plot_hyperparameter_heatmap,
    plot_learning_curves,
)


def test_documented_protocol_ceiling() -> None:
    assert _VIS_GRID_MAX == 10_000


def test_heatmap_rejects_oversized_axes() -> None:
    with pytest.raises(ValueError, match="visualization heatmap budget"):
        plot_hyperparameter_heatmap(
            {},
            param1_name="a",
            param1_values=list(range(_VIS_GRID_MAX + 1)),
            param2_name="b",
            param2_values=[0],
        )


def test_heatmap_rejects_oversized_cartesian_grid_before_plotting() -> None:
    with pytest.raises(ValueError, match="step-units exceed"):
        plot_hyperparameter_heatmap(
            {},
            param1_name="a",
            param1_values=list(range(101)),
            param2_name="b",
            param2_values=list(range(100)),
        )


def test_heatmap_does_not_dispatch_to_hostile_sequence_length() -> None:
    class HostileList(list[object]):
        def __len__(self) -> int:
            raise AssertionError("untrusted __len__ executed")

    with pytest.raises(TypeError, match="exact list"):
        plot_hyperparameter_heatmap(
            {},
            param1_name="a",
            param1_values=HostileList([0]),  # type: ignore[arg-type]
            param2_name="b",
            param2_values=[0],
        )


def test_heatmap_rejects_empty_axis_before_matplotlib() -> None:
    with pytest.raises(ValueError, match="param1_values length"):
        plot_hyperparameter_heatmap(
            {},
            param1_name="a",
            param1_values=[],
            param2_name="b",
            param2_values=[0],
        )


def test_learning_curves_reject_oversized_window() -> None:
    with pytest.raises(ValueError, match="window_size"):
        plot_learning_curves({}, window_size=_VIS_GRID_MAX + 1)
