"""Prospective optimization-readiness diagnostic for development evaluation.

The equations and empirical estimator follow Wang et al., arXiv:2605.09044v1.
This module evaluates caller-provided measurements; it neither trains nor reads
benchmark data, and therefore cannot itself create a performance result.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

PROTOCOL_SCHEMA = "asi.optimization-readiness.protocol.v1"
RESOURCE_SCHEMA = "asi.optimization-readiness.resources.v1"
RESULT_SCHEMA = "asi.optimization-readiness.development-result.v1"

OPTIMIZATION_READINESS_PROTOCOL = MappingProxyType(
    {
        "schema": PROTOCOL_SCHEMA,
        "paper_revision": "arXiv:2605.09044v1",
        "paper_revision_date": "2026-05-09",
        "population_gradient_estimator": "full_validation_set_gradient",
        "reliability_estimator": "independently_sampled_minibatch_gradients",
        "reference_reliability_batch_count": 128,
        "reference_reliability_batch_size": 4,
        "official_code_revision": "none-cited-in-arxiv-v1-as-of-2026-08-17",
        "estimator": "appendix-c.1-full-gradient-plus-independent-mini-batches",
        "paper_defaults": MappingProxyType(
            {"validation_observations": 10_000, "mini_batch_size": 4, "batch_count": 128}
        ),
        "asi_protocol_differences": (
            "caller_supplies_precomputed_full_and_mini_batch_gradients",
            "sample_counts_are_bound_by_each_development_protocol",
            "resource_nonpromotion_and_outcome_retention_receipts_are_added",
        ),
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
            "diagnostic_batch_count",
            "allowed_boundary_information",
            "allowed_task_information",
        ),
        "resource_fields": (
            "persistent_bytes",
            "environment_steps",
            "data_steps",
            "model_queries",
            "timing_seconds",
        ),
        "timing_is_telemetry_only": True,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "outcome_retention_required": True,
        "execution_protocol_required": True,
        "matrix_shape_and_dtype_accounting_required": True,
        "working_set_bytes_accounting_required": True,
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


@dataclass(frozen=True)
class ProspectiveDevelopmentProtocol:
    """Matched axes and permitted information for one prospective comparison arm."""

    seed: int
    checkpoint: str
    task: str
    updates: int
    observations: int
    mini_batch_size: int
    diagnostic_batch_count: int
    allowed_boundary_information: tuple[str, ...]
    allowed_task_information: tuple[str, ...]


@dataclass(frozen=True)
class ResourceReceipt:
    """Resource accounting required for every prospective result."""

    persistent_bytes: int
    environment_steps: int
    data_steps: int
    model_queries: int
    timing_seconds: float


@dataclass(frozen=True)
class DevelopmentResultReceipt:
    """Strictly validated, permanently nonpromoting prospective result."""

    comparison_id: str
    arm_id: str
    protocol: ProspectiveDevelopmentProtocol
    resources: ResourceReceipt
    optimization_readiness: float
    gradient_norm: float
    representation_energy_rank_0_99: int
    curvature_energy_rank_0_99: int
    parameter_norm: float
    future_relative_loss_reduction: float
    outcome: str


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
    outputs = (strength, reliability, readiness)
    if not all(math.isfinite(value) for value in outputs):
        raise ValueError("optimization-readiness outputs must be finite")
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
    if not math.isfinite(total):
        raise ValueError("matrix singular-value energy must be finite")
    found = int(np.searchsorted(np.cumsum(squared), target, side="left") + 1)
    return min(found, int(singular_values.shape[0]))


def _strict_mapping(value: object, *, name: str, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    actual = set(value)
    if actual != keys:
        raise ValueError(f"{name} keys must be exactly {sorted(keys)}")
    return value


def _strict_nonnegative_int(value: object, *, name: str, positive: bool = False) -> int:
    if type(value) is not int or value < int(positive):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a {qualifier} int")
    return value


def _strict_finite_float(value: object, *, name: str, nonnegative: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value) or (nonnegative and value < 0.0):
        qualifier = " finite nonnegative" if nonnegative else " finite"
        raise ValueError(f"{name} must be a{qualifier} float")
    return value


def _strict_string_tuple(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) is not list or any(type(item) is not str or not item for item in value):
        raise ValueError(f"{name} must be a list of non-empty strings")
    resolved = tuple(value)
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{name} must not contain duplicates")
    return resolved


def validate_development_result(payload: object) -> DevelopmentResultReceipt:
    """Fail closed while adopting one prospective development-result payload."""
    outer = _strict_mapping(
        payload,
        name="result",
        keys={
            "schema",
            "comparison_id",
            "arm_id",
            "protocol",
            "resources",
            "metrics",
            "outcome",
            "outcome_retained",
            "development_only",
            "scientific_promotion_allowed",
        },
    )
    if outer["schema"] != RESULT_SCHEMA:
        raise ValueError("result schema is not supported")
    for name in ("comparison_id", "arm_id"):
        if type(outer[name]) is not str or not outer[name]:
            raise ValueError(f"{name} must be a non-empty string")
    if type(outer["outcome"]) is not str or outer["outcome"] not in {
        "supported",
        "rejected",
        "inconclusive",
    }:
        raise ValueError("outcome must be supported, rejected, or inconclusive")
    if outer["outcome_retained"] is not True:
        raise ValueError("outcome_retained must permanently remain True")
    if outer["development_only"] is not True:
        raise ValueError("development_only must permanently remain True")
    if outer["scientific_promotion_allowed"] is not False:
        raise ValueError("scientific_promotion_allowed must permanently remain False")

    protocol_payload = _strict_mapping(
        outer["protocol"],
        name="protocol",
        keys={
            "schema",
            "seed",
            "checkpoint",
            "task",
            "updates",
            "observations",
            "mini_batch_size",
            "diagnostic_batch_count",
            "allowed_boundary_information",
            "allowed_task_information",
        },
    )
    if protocol_payload["schema"] != PROTOCOL_SCHEMA:
        raise ValueError("protocol schema is not supported")
    for name in ("checkpoint", "task"):
        if type(protocol_payload[name]) is not str or not protocol_payload[name]:
            raise ValueError(f"{name} must be a non-empty string")
    protocol = ProspectiveDevelopmentProtocol(
        seed=_strict_nonnegative_int(protocol_payload["seed"], name="seed"),
        checkpoint=protocol_payload["checkpoint"],
        task=protocol_payload["task"],
        updates=_strict_nonnegative_int(protocol_payload["updates"], name="updates", positive=True),
        observations=_strict_nonnegative_int(
            protocol_payload["observations"], name="observations", positive=True
        ),
        mini_batch_size=_strict_nonnegative_int(
            protocol_payload["mini_batch_size"], name="mini_batch_size", positive=True
        ),
        diagnostic_batch_count=_strict_nonnegative_int(
            protocol_payload["diagnostic_batch_count"],
            name="diagnostic_batch_count",
            positive=True,
        ),
        allowed_boundary_information=_strict_string_tuple(
            protocol_payload["allowed_boundary_information"],
            name="allowed_boundary_information",
        ),
        allowed_task_information=_strict_string_tuple(
            protocol_payload["allowed_task_information"], name="allowed_task_information"
        ),
    )

    resource_payload = _strict_mapping(
        outer["resources"],
        name="resources",
        keys={
            "schema",
            "persistent_bytes",
            "environment_steps",
            "data_steps",
            "model_queries",
            "timing_seconds",
            "timing_is_telemetry_only",
        },
    )
    if resource_payload["schema"] != RESOURCE_SCHEMA:
        raise ValueError("resource schema is not supported")
    if resource_payload["timing_is_telemetry_only"] is not True:
        raise ValueError("timing_is_telemetry_only must permanently remain True")
    resources = ResourceReceipt(
        persistent_bytes=_strict_nonnegative_int(
            resource_payload["persistent_bytes"], name="persistent_bytes"
        ),
        environment_steps=_strict_nonnegative_int(
            resource_payload["environment_steps"], name="environment_steps"
        ),
        data_steps=_strict_nonnegative_int(resource_payload["data_steps"], name="data_steps"),
        model_queries=_strict_nonnegative_int(
            resource_payload["model_queries"], name="model_queries"
        ),
        timing_seconds=_strict_finite_float(
            resource_payload["timing_seconds"], name="timing_seconds", nonnegative=True
        ),
    )

    metric_payload = _strict_mapping(
        outer["metrics"],
        name="metrics",
        keys={
            "optimization_readiness",
            "gradient_norm",
            "representation_energy_rank_0_99",
            "curvature_energy_rank_0_99",
            "parameter_norm",
            "future_relative_loss_reduction",
        },
    )
    return DevelopmentResultReceipt(
        comparison_id=outer["comparison_id"],
        arm_id=outer["arm_id"],
        protocol=protocol,
        resources=resources,
        optimization_readiness=_strict_finite_float(
            metric_payload["optimization_readiness"],
            name="optimization_readiness",
            nonnegative=True,
        ),
        gradient_norm=_strict_finite_float(
            metric_payload["gradient_norm"], name="gradient_norm", nonnegative=True
        ),
        representation_energy_rank_0_99=_strict_nonnegative_int(
            metric_payload["representation_energy_rank_0_99"],
            name="representation_energy_rank_0_99",
        ),
        curvature_energy_rank_0_99=_strict_nonnegative_int(
            metric_payload["curvature_energy_rank_0_99"],
            name="curvature_energy_rank_0_99",
        ),
        parameter_norm=_strict_finite_float(
            metric_payload["parameter_norm"], name="parameter_norm", nonnegative=True
        ),
        future_relative_loss_reduction=_strict_finite_float(
            metric_payload["future_relative_loss_reduction"],
            name="future_relative_loss_reduction",
        ),
        outcome=outer["outcome"],
    )


def validate_matched_development_results(
    payloads: Sequence[object],
) -> tuple[DevelopmentResultReceipt, ...]:
    """Validate two or more receipts and require all predeclared matched axes."""
    if type(payloads) not in {list, tuple} or len(payloads) < 2:
        raise ValueError("payloads must contain at least two development results")
    receipts = tuple(validate_development_result(payload) for payload in payloads)
    first = receipts[0]
    if len({receipt.arm_id for receipt in receipts}) != len(receipts):
        raise ValueError("development result arm_id values must be unique")
    for receipt in receipts[1:]:
        if receipt.comparison_id != first.comparison_id:
            raise ValueError("comparison_id must match across development results")
        if receipt.protocol != first.protocol:
            raise ValueError("all predeclared matched protocol axes must be equal")
    return receipts
