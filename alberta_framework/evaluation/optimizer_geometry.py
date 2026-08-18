"""Bounded development evaluation for continual optimizer geometry controls.

This module deliberately stops before IPMNIST. It binds three paper revisions,
runs their smallest useful matrix-geometry slices on one frozen synthetic stream,
and emits a strict, permanently nonpromoting result. The FOGO and FLAD arms are
mechanism probes rather than complete optimizer ports; those differences are part
of the validated protocol rather than being left implicit.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

GEOMETRY_RESULT_SCHEMA = "asi.optimizer-geometry.streaming-matrix-result.v1"
FROZEN_GEOMETRY_CONFIG = MappingProxyType(
    {
        "seed": 20_260_817,
        "updates": 8,
        "rows": 3,
        "columns": 2,
        "newton_schulz_steps": 5,
        "muon_dual_steps": 2,
        "muon_dual_learning_rate": 0.25,
        "allowed_boundary_information": "none",
        "target_definition": "sum_of_stream_updates_projected_away_from_fixed_e0",
    }
)
GEOMETRY_PROTOCOL = MappingProxyType(
    {
        "schema": "asi.optimizer-geometry.protocol.v2",
        "paper_revisions": (
            "arXiv:2605.08949v2",
            "arXiv:2606.10406v1",
            "arXiv:2601.07636v1",
        ),
        "stage": "frozen_small_streaming_matrix_pre_ipmnist",
        "protocol_differences": (
            "Muon-OGD is the paper-v2 NS5 and two-step dual update on one fixed constraint",
            "FOGO is only its long-term orthogonal-correction equation; no codebook, random "
            "projection, slow-fast streams, or proximal lift",
            "FLAD is only the ideal gradient-orthogonal decomposition in equation 6; no EMA "
            "approximation, Hessian-vector product, sharpness objective, or schedule",
            "the synthetic target is a mechanism diagnostic defined by the protected complement; "
            "its outcome is not comparative performance evidence",
        ),
        "matched_axes": (
            "seed",
            "ordered_matrices",
            "updates",
            "observations",
            "allowed_boundary_information",
        ),
        "mechanism_off": "empty_basis_or_zero_gradient_exact_reduction",
        "finite_kernel_preflight_required": True,
        "persistent_bytes_accounting_required": True,
        "environment_steps_accounting_required": True,
        "model_queries_accounting_required": True,
        "timing_is_telemetry_only": True,
        "development_only": True,
        "scientific_promotion_allowed": False,
    }
)

_ARM_SPECS = (
    ("muon_ns5_empty_constraints", "muon_ogd_v2_dual", "empty_constraints"),
    ("muon_ogd_v2_dual", "muon_ogd_v2_dual", "active_constraint"),
    ("fogo_empty_basis", "fogo_v1_long_term_correction", "empty_basis"),
    ("fogo_projection", "fogo_v1_long_term_correction", "active_basis"),
    ("flad_zero_gradient", "flad_v1_ideal_noise_component", "zero_gradient"),
    ("flad_ideal_noise", "flad_v1_ideal_noise_component", "active_gradient"),
)
_CONTROL_BY_CANDIDATE = {
    "muon_ogd_v2_dual": "muon_ns5_empty_constraints",
    "fogo_projection": "fogo_empty_basis",
    "flad_ideal_noise": "flad_zero_gradient",
}

_MAX_MATRIX_ELEMENTS = 1_000_000


def _trusted_array(value: object, *, name: str) -> Array:
    actual_type = type(value)
    if not (actual_type is np.ndarray or issubclass(actual_type, (jax.Array, jax.core.Tracer))):
        raise ValueError(f"{name} must be an exact NumPy or JAX array")
    result = jnp.asarray(value)
    if result.size > _MAX_MATRIX_ELEMENTS or not jnp.issubdtype(result.dtype, jnp.floating):
        raise ValueError(f"{name} must be a bounded floating array")
    return result


def _unwrap_transaction(result: tuple[Array, Array], *, name: str) -> Array:
    safe, valid = result
    if type(valid) is not jax.core.Tracer and not isinstance(valid, jax.core.Tracer):
        if not bool(valid):
            raise ValueError(f"{name} must be finite")
    return safe


def orthogonal_correction_transaction(update: Array, protected_basis: Array) -> tuple[Array, Array]:
    """Return a finite orthogonal correction and caller-visible validity bit."""
    vector = _trusted_array(update, name="update")
    basis = _trusted_array(protected_basis, name="protected_basis")
    if vector.ndim != 1 or basis.ndim != 2 or basis.shape[1] != vector.shape[0]:
        raise ValueError("update must be a vector and basis rows must match its width")
    coordinates = basis @ vector
    projection = basis.T @ coordinates
    candidate = vector - projection
    valid = (
        jnp.all(jnp.isfinite(vector))
        & jnp.all(jnp.isfinite(basis))
        & jnp.all(jnp.isfinite(coordinates))
        & jnp.all(jnp.isfinite(projection))
        & jnp.all(jnp.isfinite(candidate))
    )
    return jnp.where(valid, candidate, jnp.zeros_like(candidate)), valid


def orthogonal_correction(update: Array, protected_basis: Array) -> Array:
    """Project a vector away from row-wise orthonormal protected directions."""
    return _unwrap_transaction(
        orthogonal_correction_transaction(update, protected_basis),
        name="orthogonal correction",
    )


def spectral_matrix_sign_transaction(matrix: Array, *, steps: int = 5) -> tuple[Array, Array]:
    """Apply Muon-OGD v2's cubic NS5 matrix-sign approximation.

    The pinned paper defines ``f(X) = 3/2 X - 1/2 XX^T X``. Frobenius
    normalization places every singular value in its convergence interval and
    preserves an exact zero for a zero matrix.
    """
    value = _trusted_array(matrix, name="matrix")
    if (
        value.ndim != 2
        or value.size == 0
        or not jnp.issubdtype(value.dtype, jnp.floating)
        or type(steps) is not int
        or steps < 1
        or steps > 32
    ):
        raise ValueError("matrix must be non-empty and steps a positive integer")
    norm = jnp.linalg.norm(value)
    valid = jnp.all(jnp.isfinite(value)) & jnp.isfinite(norm)
    x = value / jnp.maximum(norm, jnp.asarray(1e-12, dtype=value.dtype))
    if x.shape[0] > x.shape[1]:
        x = x.T
        transposed = True
    else:
        transposed = False
    for _ in range(steps):
        a = x @ x.T
        next_x = 1.5 * x - 0.5 * a @ x
        valid = valid & jnp.all(jnp.isfinite(a)) & jnp.all(jnp.isfinite(next_x))
        x = next_x
    candidate = x.T if transposed else x
    valid = valid & jnp.all(jnp.isfinite(candidate))
    safe = jnp.where(valid, candidate, jnp.zeros_like(candidate))
    return safe, valid


def spectral_matrix_sign(matrix: Array, *, steps: int = 5) -> Array:
    """Apply Muon-OGD v2's cubic NS5 matrix-sign approximation."""
    return _unwrap_transaction(
        spectral_matrix_sign_transaction(matrix, steps=steps), name="matrix sign"
    )


def flad_noise_component_transaction(perturbation: Array, gradient: Array) -> tuple[Array, Array]:
    """Return a finite FLAD noise component and caller-visible validity bit."""
    delta = _trusted_array(perturbation, name="perturbation")
    direction = _trusted_array(gradient, name="gradient")
    if delta.shape != direction.shape or delta.ndim != 1 or delta.size < 1:
        raise ValueError("perturbation and gradient must be non-empty equal-width vectors")
    squared_norm = jnp.vdot(direction, direction).real
    numerator = jnp.vdot(direction, delta).real
    active = squared_norm > 0.0
    denominator = jnp.where(active, squared_norm, jnp.ones_like(squared_norm))
    coefficient = numerator / denominator
    projection = direction * coefficient * active.astype(delta.dtype)
    candidate = delta - projection
    valid = (
        jnp.all(jnp.isfinite(delta))
        & jnp.all(jnp.isfinite(direction))
        & jnp.isfinite(squared_norm)
        & jnp.isfinite(numerator)
        & jnp.isfinite(coefficient)
        & jnp.all(jnp.isfinite(projection))
        & jnp.all(jnp.isfinite(candidate))
    )
    safe = jnp.where(valid, candidate, jnp.zeros_like(candidate))
    return safe, valid


def muon_ogd_dual_update(
    momentum: Array,
    constraints: Array,
    dual: Array,
    *,
    dual_learning_rate: float,
    dual_steps: int,
    newton_schulz_steps: int = 5,
) -> tuple[Array, Array]:
    """Run the bounded matrix form of Muon-OGD v2 Algorithm 1."""
    value = _trusted_array(momentum, name="momentum")
    protected = _trusted_array(constraints, name="constraints")
    multipliers = _trusted_array(dual, name="dual")
    if value.ndim != 2 or value.size == 0 or not jnp.issubdtype(value.dtype, jnp.floating):
        raise ValueError("momentum must be a non-empty floating matrix")
    if protected.ndim != 3 or protected.shape[1:] != value.shape:
        raise ValueError("constraints must have shape (count, rows, columns)")
    if multipliers.ndim != 1 or multipliers.shape[0] != protected.shape[0]:
        raise ValueError("dual must contain one multiplier per constraint")
    if type(dual_learning_rate) is not float or not math.isfinite(dual_learning_rate):
        raise ValueError("dual_learning_rate must be a finite float")
    if dual_learning_rate < 0.0 or type(dual_steps) is not int or dual_steps < 1:
        raise ValueError("dual learning rate must be non-negative and dual_steps positive")
    for _ in range(dual_steps):
        shifted = value + jnp.einsum("k,kij->ij", multipliers, protected)
        matrix_sign = spectral_matrix_sign(shifted, steps=newton_schulz_steps)
        conflicts = jnp.einsum("kij,ij->k", protected, matrix_sign)
        multipliers = multipliers - dual_learning_rate * conflicts
    shifted = value + jnp.einsum("k,kij->ij", multipliers, protected)
    return spectral_matrix_sign(shifted, steps=newton_schulz_steps), multipliers


def flad_noise_component(perturbation: Array, gradient: Array) -> Array:
    """Remove the ideal FLAD gradient-aligned perturbation component safely."""
    return _unwrap_transaction(
        flad_noise_component_transaction(perturbation, gradient),
        name="FLAD decomposition",
    )


def _frozen_stream() -> Array:
    config = FROZEN_GEOMETRY_CONFIG
    key = jr.key(cast(int, config["seed"]), impl="threefry2x32")
    shape = (
        cast(int, config["updates"]),
        cast(int, config["rows"]),
        cast(int, config["columns"]),
    )
    return jr.normal(key, shape, dtype=jnp.float32)


def _stream_sha256(stream: Array) -> str:
    canonical = np.asarray(stream, dtype="<f4")
    descriptor = f"float32:{canonical.shape}".encode()
    return hashlib.sha256(descriptor + canonical.tobytes(order="C")).hexdigest()


def _protected_geometry(rows: int, columns: int) -> tuple[Array, Array]:
    vector = jnp.zeros((rows * columns,), dtype=jnp.float32).at[0].set(1.0)
    return vector.reshape((1, rows, columns)), vector.reshape((1, rows * columns))


def _outcome(delta: float) -> str:
    if delta < -1e-7:
        return "improved"
    if delta > 1e-7:
        return "worse"
    return "tied"


def _protocol_payload() -> dict[str, object]:
    return {
        "schema": GEOMETRY_PROTOCOL["schema"],
        "paper_revisions": list(cast(tuple[str, ...], GEOMETRY_PROTOCOL["paper_revisions"])),
        "stage": GEOMETRY_PROTOCOL["stage"],
        "protocol_differences": list(
            cast(tuple[str, ...], GEOMETRY_PROTOCOL["protocol_differences"])
        ),
        "matched_axes": list(cast(tuple[str, ...], GEOMETRY_PROTOCOL["matched_axes"])),
        "mechanism_off": GEOMETRY_PROTOCOL["mechanism_off"],
        "finite_kernel_preflight_required": True,
        "persistent_bytes_accounting_required": True,
        "environment_steps_accounting_required": True,
        "model_queries_accounting_required": True,
        "timing_is_telemetry_only": True,
        "development_only": True,
        "scientific_promotion_allowed": False,
    }


def _execute_frozen_stream(*, measure_timing: bool) -> dict[str, object]:
    config = FROZEN_GEOMETRY_CONFIG
    stream = _frozen_stream()
    updates = cast(int, config["updates"])
    rows = cast(int, config["rows"])
    columns = cast(int, config["columns"])
    ns_steps = cast(int, config["newton_schulz_steps"])
    dual_steps = cast(int, config["muon_dual_steps"])
    dual_learning_rate = cast(float, config["muon_dual_learning_rate"])
    constraints, basis = _protected_geometry(rows, columns)
    target = jnp.sum(
        jnp.stack(
            [
                orthogonal_correction(matrix.reshape(-1), basis).reshape((rows, columns))
                for matrix in stream
            ]
        ),
        axis=0,
    )
    arm_records: list[dict[str, object]] = []
    for arm, mechanism, mode in _ARM_SPECS:
        started = time.perf_counter_ns() if measure_timing else 0
        state = jnp.zeros((rows, columns), dtype=jnp.float32)
        dual = jnp.zeros((constraints.shape[0],), dtype=jnp.float32)
        processed_updates: list[Array] = []
        for matrix in stream:
            if arm == "muon_ns5_empty_constraints":
                processed = spectral_matrix_sign(matrix, steps=ns_steps)
            elif arm == "muon_ogd_v2_dual":
                processed, dual = muon_ogd_dual_update(
                    matrix,
                    constraints,
                    dual,
                    dual_learning_rate=dual_learning_rate,
                    dual_steps=dual_steps,
                    newton_schulz_steps=ns_steps,
                )
            elif arm == "fogo_empty_basis":
                processed = orthogonal_correction(
                    matrix.reshape(-1), jnp.zeros((0, rows * columns), dtype=matrix.dtype)
                ).reshape((rows, columns))
            elif arm == "fogo_projection":
                processed = orthogonal_correction(matrix.reshape(-1), basis).reshape(
                    (rows, columns)
                )
            elif arm == "flad_zero_gradient":
                processed = flad_noise_component(
                    matrix.reshape(-1), jnp.zeros((rows * columns,), dtype=matrix.dtype)
                ).reshape((rows, columns))
            else:
                processed = flad_noise_component(matrix.reshape(-1), basis[0]).reshape(
                    (rows, columns)
                )
            processed_updates.append(processed)
            state = state + processed
        if measure_timing:
            state.block_until_ready()
        elapsed = time.perf_counter_ns() - started if measure_timing else 0
        stacked = jnp.stack(processed_updates)
        final_error = float(jnp.mean(jnp.square(state - target)))
        interference = float(jnp.mean(jnp.abs(stacked[:, 0, 0])))
        mean_update_norm = float(jnp.mean(jnp.linalg.norm(stacked, axis=(1, 2))))
        persistent_bytes = int(state.nbytes)
        if arm == "muon_ogd_v2_dual":
            persistent_bytes += int(dual.nbytes + constraints.nbytes)
        elif mode == "active_basis":
            persistent_bytes += int(basis.nbytes)
        elif mode == "active_gradient":
            persistent_bytes += int(basis[0].nbytes)
        arm_records.append(
            {
                "arm": arm,
                "mechanism": mechanism,
                "mode": mode,
                "metrics": {
                    "final_target_mse": final_error,
                    "mean_protected_interference": interference,
                    "mean_update_frobenius_norm": mean_update_norm,
                },
                "resources": {
                    "persistent_bytes": persistent_bytes,
                    "observations": updates,
                    "updates": updates,
                    "data_steps": updates,
                    "environment_steps": 0,
                    "model_queries": 0,
                    "timing_ns": elapsed,
                    "timing_qualified": False,
                },
            }
        )
    records_by_arm = {cast(str, record["arm"]): record for record in arm_records}
    comparisons: list[dict[str, object]] = []
    for candidate, control in _CONTROL_BY_CANDIDATE.items():
        candidate_metrics = cast(Mapping[str, float], records_by_arm[candidate]["metrics"])
        control_metrics = cast(Mapping[str, float], records_by_arm[control]["metrics"])
        delta = candidate_metrics["final_target_mse"] - control_metrics["final_target_mse"]
        comparisons.append(
            {
                "candidate": candidate,
                "control": control,
                "final_target_mse_delta": delta,
                "outcome": _outcome(delta),
            }
        )
    return {
        "schema": GEOMETRY_RESULT_SCHEMA,
        "protocol": _protocol_payload(),
        "config": dict(FROZEN_GEOMETRY_CONFIG),
        "stream": {
            "generator": "jax.random.threefry2x32.normal.float32",
            "sha256": _stream_sha256(stream),
            "shape": [updates, rows, columns],
        },
        "policy": {
            "status": "development-only-nonpromoting",
            "development_only": True,
            "scientific_promotion_allowed": False,
            "negative_outcomes_retained": True,
        },
        "arms": arm_records,
        "comparisons": comparisons,
    }


def run_streaming_matrix_evaluation() -> dict[str, object]:
    """Run and strictly validate the literal frozen development slice."""
    result = _execute_frozen_stream(measure_timing=True)
    validate_streaming_matrix_result(result)
    return result


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{name} must be a string-keyed mapping")
    result = cast(dict[object, object], value)
    if not all(type(key) is str for key in result):
        raise ValueError(f"{name} must be a string-keyed mapping")
    return cast(dict[str, object], result)


def _sequence(value: object, *, name: str) -> list[object]:
    if type(value) is not list:
        raise ValueError(f"{name} must be a list")
    return cast(list[object], value)


def _same_float(actual: object, expected: object, *, name: str) -> None:
    if type(actual) is not float or type(expected) is not float:
        raise ValueError(f"{name} must be a float")
    if not math.isfinite(actual) or actual != expected:
        raise ValueError(f"{name} does not match the frozen evaluation")


def _exact_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        actual_dict = cast(dict[object, object], actual)
        expected_dict = cast(dict[object, object], expected)
        if set(actual_dict) != set(expected_dict):
            return False
        return all(_exact_equal(actual_dict[key], value) for key, value in expected_dict.items())
    if type(expected) is list:
        actual_list = cast(list[object], actual)
        expected_list = cast(list[object], expected)
        return len(actual_list) == len(expected_list) and all(
            _exact_equal(left, right)
            for left, right in zip(actual_list, expected_list, strict=True)
        )
    if type(expected) in (str, int, float, bool, type(None)):
        return bool(actual == expected)
    return False


def validate_streaming_matrix_result(result: object) -> None:
    """Fail closed unless ``result`` is an exact current frozen-run record."""
    actual_result = _mapping(result, name="result")
    expected = _execute_frozen_stream(measure_timing=False)
    required_top = {"schema", "protocol", "config", "stream", "policy", "arms", "comparisons"}
    if set(actual_result) != required_top or not _exact_equal(
        actual_result["schema"], GEOMETRY_RESULT_SCHEMA
    ):
        raise ValueError("result fields or schema do not match the frozen protocol")
    for field in ("protocol", "config", "stream", "policy"):
        actual_mapping = _mapping(actual_result[field], name=field)
        expected_mapping = _mapping(expected[field], name=f"expected.{field}")
        if not _exact_equal(actual_mapping, expected_mapping):
            raise ValueError(f"{field} does not match the frozen protocol")
    arms = _sequence(actual_result["arms"], name="arms")
    expected_arms = _sequence(expected["arms"], name="expected.arms")
    if len(arms) != len(_ARM_SPECS):
        raise ValueError("arms must contain every frozen candidate and control exactly once")
    for index, (raw_arm, raw_expected_arm) in enumerate(zip(arms, expected_arms, strict=True)):
        arm = _mapping(raw_arm, name=f"arms[{index}]")
        expected_arm = _mapping(raw_expected_arm, name=f"expected.arms[{index}]")
        if set(arm) != {"arm", "mechanism", "mode", "metrics", "resources"}:
            raise ValueError(f"arms[{index}] fields do not match the schema")
        for field in ("arm", "mechanism", "mode"):
            if not _exact_equal(arm[field], expected_arm[field]):
                raise ValueError(f"arms[{index}].{field} does not match the frozen plan")
        metrics = _mapping(arm["metrics"], name=f"arms[{index}].metrics")
        expected_metrics = _mapping(expected_arm["metrics"], name=f"expected.arms[{index}].metrics")
        if set(metrics) != set(expected_metrics):
            raise ValueError(f"arms[{index}].metrics fields do not match the schema")
        for metric, expected_value in expected_metrics.items():
            _same_float(metrics[metric], expected_value, name=f"arms[{index}].metrics.{metric}")
        resources = _mapping(arm["resources"], name=f"arms[{index}].resources")
        expected_resources = _mapping(
            expected_arm["resources"], name=f"expected.arms[{index}].resources"
        )
        if set(resources) != set(expected_resources):
            raise ValueError(f"arms[{index}].resources fields do not match the schema")
        for resource, expected_value in expected_resources.items():
            if resource == "timing_ns":
                if type(resources[resource]) is not int or cast(int, resources[resource]) < 0:
                    raise ValueError(f"arms[{index}].resources.timing_ns must be non-negative")
            elif not _exact_equal(resources[resource], expected_value):
                raise ValueError(f"arms[{index}].resources.{resource} is not canonical")
    comparisons = _sequence(actual_result["comparisons"], name="comparisons")
    expected_comparisons = _sequence(expected["comparisons"], name="expected.comparisons")
    if len(comparisons) != len(expected_comparisons):
        raise ValueError("comparisons must contain every matched A/B pair")
    for index, (raw_comparison, raw_expected_comparison) in enumerate(
        zip(comparisons, expected_comparisons, strict=True)
    ):
        comparison = _mapping(raw_comparison, name=f"comparisons[{index}]")
        expected_comparison = _mapping(
            raw_expected_comparison, name=f"expected.comparisons[{index}]"
        )
        if set(comparison) != {"candidate", "control", "final_target_mse_delta", "outcome"}:
            raise ValueError(f"comparisons[{index}] fields do not match the schema")
        for field in ("candidate", "control", "outcome"):
            if not _exact_equal(comparison[field], expected_comparison[field]):
                raise ValueError(f"comparisons[{index}].{field} is not canonical")
        _same_float(
            comparison["final_target_mse_delta"],
            expected_comparison["final_target_mse_delta"],
            name=f"comparisons[{index}].final_target_mse_delta",
        )
