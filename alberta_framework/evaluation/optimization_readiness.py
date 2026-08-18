"""Prospective optimization-readiness diagnostic for development evaluation.

The equations follow Wang et al., arXiv:2605.09044v1.  This module evaluates
already-collected mini-batch gradients; it neither trains nor reads benchmark
data, and therefore cannot itself create a performance result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

OPTIMIZATION_READINESS_PROTOCOL = MappingProxyType(
    {
        "schema": "asi.optimization-readiness.protocol.v1",
        "paper_revision": "arXiv:2605.09044v1",
        "paper_revision_date": "2026-05-09",
        "population_gradient_estimator": "full_validation_set_gradient",
        "reliability_estimator": "independently_sampled_minibatch_gradients",
        "reference_reliability_batch_count": 128,
        "reference_reliability_batch_size": 4,
        "diagnostics": (
            "optimization_readiness",
            "gradient_norm",
            "representation_energy_rank_0.99",
            "curvature_energy_rank_0.99",
            "parameter_norm",
        ),
        "target": "future_relative_loss_reduction_after_matched_updates",
        "future_gain_steps": (1, 10, 100),
        "primary_comparison": "pairwise_checkpoint_ranking_accuracy",
        "matched_axes": (
            "seed",
            "checkpoint",
            "task",
            "updates",
            "observations",
            "mini_batch_size",
        ),
        "execution_protocol_required": True,
        "matrix_shape_and_dtype_accounting_required": True,
        "working_set_bytes_accounting_required": True,
        "persistent_bytes_accounting_required": True,
        "environment_steps_accounting_required": True,
        "model_queries_accounting_required": True,
        "timing_is_telemetry_only": True,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "completed_result_exists": False,
    }
)


@dataclass(frozen=True)
class OptimizationReadiness:
    """Equation-level diagnostic values for one checkpoint/task pair."""

    gradient_squared_norm: float
    expected_batch_gradient_squared_norm: float
    gradient_strength: float
    gradient_reliability: float
    optimization_readiness: float
    gradient_norm: float
    batch_count: int
    parameter_count: int


def _finite_array(
    value: object, *, name: str, dimensions: int
) -> NDArray[np.float64]:
    if type(value) is not np.ndarray:
        raise ValueError(f"{name} must be an exact numpy.ndarray")
    raw = value
    if raw.ndim != dimensions or any(size < 1 for size in raw.shape):
        raise ValueError(f"{name} must be a non-empty {dimensions}-dimensional array")
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(f"{name} must have a real numeric dtype")
    resolved = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(resolved)):
        raise ValueError(f"{name} must contain only finite values")
    return resolved


def _squared_norm(value: NDArray[np.float64], *, name: str) -> float:
    scale = float(np.max(np.abs(value)))
    if scale == 0.0:
        return 0.0
    scaled_squared_norm = float(np.sum(np.square(value / scale)))
    if scale > math.sqrt(np.finfo(np.float64).max / scaled_squared_norm):
        raise ValueError(f"{name} squared norm must fit in a finite float64")
    result = scale * scale * scaled_squared_norm
    if not math.isfinite(result):
        raise ValueError(f"{name} squared norm must fit in a finite float64")
    return result


def estimate_optimization_readiness(
    *,
    loss: float,
    full_validation_gradient: object,
    batch_gradients: object,
    include_reliability: bool = True,
) -> OptimizationReadiness:
    """Estimate OR using the paper's full-data and mini-batch estimators.

    ``full_validation_gradient`` estimates the population gradient separately.
    The rows of ``batch_gradients`` must come from independently sampled
    mini-batches and estimate the expected squared mini-batch-gradient norm.
    Setting ``include_reliability=False`` is the predeclared
    gradient-strength-only mechanism-off reduction.
    """
    if type(loss) is not float and type(loss) is not int:
        raise ValueError("loss must be a finite non-negative real number")
    try:
        resolved_loss = float(loss)
    except (OverflowError, ValueError) as exc:
        raise ValueError("loss must be a finite non-negative real number") from exc
    if not math.isfinite(resolved_loss) or resolved_loss < 0.0:
        raise ValueError("loss must be a finite non-negative real number")
    if type(include_reliability) is not bool:
        raise ValueError("include_reliability must be a bool")
    gradient = _finite_array(
        full_validation_gradient,
        name="full_validation_gradient",
        dimensions=1,
    )
    gradients = _finite_array(batch_gradients, name="batch_gradients", dimensions=2)
    if gradients.shape[1] != gradient.shape[0]:
        raise ValueError(
            "full_validation_gradient and batch_gradients must share a parameter axis"
        )
    gradient_squared_norm = _squared_norm(
        gradient, name="full_validation_gradient"
    )
    row_squared_norms = np.asarray(
        [_squared_norm(row, name="batch gradient") for row in gradients],
        dtype=np.float64,
    )
    expected_squared_norm = float(np.mean(row_squared_norms))
    strength = gradient_squared_norm / resolved_loss if resolved_loss > 0.0 else 0.0
    reliability = (
        gradient_squared_norm / expected_squared_norm if expected_squared_norm > 0.0 else 0.0
    )
    if not include_reliability:
        readiness = strength
    elif resolved_loss > 0.0 and expected_squared_norm > 0.0:
        readiness = strength * reliability
    else:
        readiness = 0.0
    return OptimizationReadiness(
        gradient_squared_norm=gradient_squared_norm,
        expected_batch_gradient_squared_norm=expected_squared_norm,
        gradient_strength=strength,
        gradient_reliability=reliability,
        optimization_readiness=readiness,
        gradient_norm=math.sqrt(gradient_squared_norm),
        batch_count=int(gradients.shape[0]),
        parameter_count=int(gradients.shape[1]),
    )


def energy_rank(matrix: object, *, threshold: float = 0.99) -> int:
    """Return the smallest singular-value count reaching squared-energy mass."""
    if type(threshold) is not float:
        raise ValueError("threshold must be a float in (0, 1]")
    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be a float in (0, 1]")
    resolved = _finite_array(matrix, name="matrix", dimensions=2)
    try:
        singular_values = np.linalg.svd(resolved, compute_uv=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("matrix singular-value decomposition did not converge") from exc
    leading = float(singular_values[0])
    if leading == 0.0:
        return 0
    squared = np.square(singular_values / leading)
    total = float(np.sum(squared))
    target = np.nextafter(threshold * total, -math.inf)
    return int(np.searchsorted(np.cumsum(squared), target, side="left") + 1)
