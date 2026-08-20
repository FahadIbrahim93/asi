"""Bounded nonlinear representation/eNTK Optimization Readiness diagnostic.

This development-only adapter evaluates an exact two-layer ReLU regression
checkpoint.  It constructs the checkpoint's hidden representation and analytic
empirical neural tangent feature matrix, reports their 99% energy ranks, and
measures one full-gradient step.  It is a real model-bound diagnostic, not a
paper reproduction or a promotion path.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Final, cast

import numpy as np

SCHEMA: Final[str] = "asi.optimization-readiness.entk.v1"
LEARNING_RATE: Final[float] = 1e-3
ENERGY_THRESHOLD: Final[float] = 0.99
MAX_OBSERVATIONS: Final[int] = 256
MAX_INPUT_DIMENSION: Final[int] = 32
MAX_HIDDEN_UNITS: Final[int] = 64
MAX_PARAMETERS: Final[int] = 4096
MAX_PEAK_BYTES: Final[int] = 256 << 20
MAX_SVD_WORK_UNITS: Final[int] = 100_000_000
MAX_ROLLOUT_WORK_UNITS: Final[int] = 200_000_000
FUTURE_HORIZONS: Final[tuple[int, ...]] = (1, 10, 100)
MAX_JSON_NODES: Final[int] = 4096
MAX_JSON_STRING_BYTES: Final[int] = 64 << 10
MAX_JSON_INTEGER: Final[int] = (1 << 63) - 1


@dataclasses.dataclass(frozen=True, slots=True)
class MLPCheckpoint:
    """Caller-owned parameters for one scalar-output two-layer ReLU MLP."""

    input_weights: np.ndarray
    hidden_bias: np.ndarray
    output_weights: np.ndarray
    output_bias: np.ndarray


def _utf8_length(value: str, *, name: str) -> int:
    try:
        return len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must be valid UTF-8") from exc


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or _utf8_length(value, name=name) > 256:
        raise ValueError(f"{name} must be bounded non-empty exact text")
    return value


def _array(value: object, *, name: str, ndim: int) -> np.ndarray:
    if type(value) is not np.ndarray or value.dtype != np.dtype(np.float64):
        raise ValueError(f"{name} must be an exact float64 ndarray")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    return value


def _checkpoint_arrays(checkpoint: object) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if type(checkpoint) is not MLPCheckpoint:
        raise ValueError("checkpoint must be an exact MLPCheckpoint")
    w1 = _array(checkpoint.input_weights, name="input_weights", ndim=2)
    b1 = _array(checkpoint.hidden_bias, name="hidden_bias", ndim=1)
    w2 = _array(checkpoint.output_weights, name="output_weights", ndim=1)
    bias = _array(checkpoint.output_bias, name="output_bias", ndim=0)
    if w1.shape[1:] != b1.shape or b1.shape != w2.shape:
        raise ValueError("checkpoint hidden axes differ")
    return w1, b1, w2, bias


def _preflight(
    inputs: object, labels: object, checkpoint: object
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    dict[str, int],
]:
    x = _array(inputs, name="validation_inputs", ndim=2)
    y = _array(labels, name="validation_labels", ndim=1)
    w1, b1, w2, bias = _checkpoint_arrays(checkpoint)
    observations, dimension = (int(size) for size in x.shape)
    hidden = int(b1.size)
    if not 2 <= observations <= MAX_OBSERVATIONS:
        raise ValueError("observations lie outside the bounded eNTK protocol")
    if not 1 <= dimension <= MAX_INPUT_DIMENSION:
        raise ValueError("input dimension lies outside the bounded eNTK protocol")
    if not 1 <= hidden <= MAX_HIDDEN_UNITS:
        raise ValueError("hidden units lie outside the bounded eNTK protocol")
    if y.shape != (observations,) or w1.shape != (dimension, hidden):
        raise ValueError("dataset and checkpoint axes differ")
    parameters = dimension * hidden + hidden + hidden + 1
    if parameters > MAX_PARAMETERS:
        raise ValueError("parameter count exceeds the bounded eNTK protocol")
    representation_bytes = observations * hidden * np.dtype(np.float64).itemsize
    jacobian_bytes = observations * parameters * np.dtype(np.float64).itemsize
    caller_bytes = x.nbytes + y.nbytes + w1.nbytes + b1.nbytes + w2.nbytes + bias.nbytes
    itemsize = np.dtype(np.float64).itemsize
    vector_bytes = observations * itemsize
    parameter_bytes = parameters * itemsize
    # Caller arrays and immutable snapshots plus the initial diagnostic remain live while
    # each independent rollout executes.  The rollout allowance covers preactivation,
    # representation, masks/weighted masks, both Jacobian constructions, residuals, and
    # parameter/gradient vectors.  Each SVD allowance additionally follows the established
    # bounded energy-rank convention: two float64 copies, a finite mask, rank temporaries,
    # and conservative quadratic LAPACK workspace.
    persistent_bytes = (
        2 * caller_bytes
        + representation_bytes
        + jacobian_bytes
        + 2 * vector_bytes
        + parameter_bytes
    )
    rollout_peak_bytes = persistent_bytes + (
        3 * jacobian_bytes
        + 4 * representation_bytes
        + 4 * vector_bytes
        + 3 * parameter_bytes
    )

    def svd_extra_bytes(columns: int) -> int:
        rank = min(observations, columns)
        elements = observations * columns
        return 17 * elements + 24 * rank * rank + 8 * max(observations, columns) + 96 * rank

    peak_bytes = max(
        rollout_peak_bytes,
        persistent_bytes + svd_extra_bytes(hidden),
        persistent_bytes + svd_extra_bytes(parameters),
    )
    representation_svd_work = observations * hidden * min(observations, hidden)
    jacobian_svd_work = observations * parameters * min(observations, parameters)
    svd_work_units = representation_svd_work + jacobian_svd_work
    optimizer_updates = sum(FUTURE_HORIZONS)
    terminal_evaluations = len(FUTURE_HORIZONS)
    rollout_work_units = (
        optimizer_updates + terminal_evaluations
    ) * observations * parameters
    diagnostic_work_units = observations * parameters
    if peak_bytes > MAX_PEAK_BYTES:
        raise ValueError("eNTK peak bytes exceed the bounded protocol")
    if svd_work_units > MAX_SVD_WORK_UNITS:
        raise ValueError("eNTK SVD work exceeds the bounded protocol")
    if rollout_work_units > MAX_ROLLOUT_WORK_UNITS:
        raise ValueError("eNTK rollout work exceeds the bounded protocol")
    return x, y, (w1, b1, w2, bias), {
        "observations": observations,
        "input_dimension": dimension,
        "hidden_units": hidden,
        "parameter_count": parameters,
        "caller_numeric_bytes": caller_bytes,
        "representation_bytes": representation_bytes,
        "jacobian_bytes": jacobian_bytes,
        "peak_numeric_bytes": peak_bytes,
        "svd_work_units": svd_work_units,
        "diagnostic_work_units": diagnostic_work_units,
        "rollout_work_units": rollout_work_units,
        "total_logical_work_units": (
            svd_work_units + diagnostic_work_units + rollout_work_units
        ),
        "gradient_evaluations": 1 + optimizer_updates,
        "optimizer_updates": optimizer_updates,
        "model_queries": (1 + optimizer_updates + terminal_evaluations) * observations,
    }


def _snapshot(value: np.ndarray) -> np.ndarray:
    result = value.copy(order="C")
    if not np.isfinite(result).all():
        raise ValueError("model inputs and parameters must be finite")
    return result


def _array_identity(domain: str, value: np.ndarray) -> dict[str, object]:
    digest = hashlib.sha256(domain.encode("ascii"))
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return {"dtype": value.dtype.str, "shape": list(value.shape), "sha256": digest.hexdigest()}


def _checkpoint_identity(arrays: tuple[np.ndarray, ...]) -> dict[str, object]:
    names = ("input_weights", "hidden_bias", "output_weights", "output_bias")
    return {
        name: _array_identity(name, value)
        for name, value in zip(names, arrays, strict=True)
    }


def _flatten(arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    return np.concatenate(tuple(value.reshape(-1) for value in arrays))


def _unpack(
    flat: np.ndarray, dimension: int, hidden: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w1_end = dimension * hidden
    b1_end = w1_end + hidden
    w2_end = b1_end + hidden
    return (
        flat[:w1_end].reshape(dimension, hidden),
        flat[w1_end:b1_end],
        flat[b1_end:w2_end],
        flat[w2_end:].reshape(()),
    )


def _model(
    inputs: np.ndarray, arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w1, b1, w2, bias = arrays
    preactivation = inputs @ w1 + b1
    hidden = np.maximum(preactivation, 0.0)
    predictions = hidden @ w2 + bias
    mask = preactivation > 0.0
    w1_jacobian = (inputs[:, :, None] * (mask * w2)[:, None, :]).reshape(inputs.shape[0], -1)
    jacobian = np.concatenate(
        (w1_jacobian, mask * w2, hidden, np.ones((inputs.shape[0], 1), dtype=np.float64)),
        axis=1,
    )
    return predictions, hidden, jacobian


def _predict(
    inputs: np.ndarray, arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    w1, b1, w2, bias = arrays
    return cast(np.ndarray, np.maximum(inputs @ w1 + b1, 0.0) @ w2 + bias)


def _energy_rank(value: np.ndarray) -> int:
    scale = float(np.max(np.abs(value)))
    if scale == 0.0:
        return 0
    singular_values = np.linalg.svd(value / scale, compute_uv=False)
    leading = float(singular_values[0])
    energy = np.square(singular_values / leading)
    total = float(np.sum(energy))
    if not math.isfinite(total):
        raise ValueError("eNTK singular-value energy must be finite")
    target = np.nextafter(ENERGY_THRESHOLD * total, -math.inf)
    found = int(np.searchsorted(np.cumsum(energy), target, side="left") + 1)
    return min(found, int(singular_values.shape[0]))


def _gradient(
    inputs: np.ndarray,
    labels: np.ndarray,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    predictions, _, jacobian = _model(inputs, arrays)
    return cast(np.ndarray, 2.0 * (jacobian.T @ (predictions - labels)) / inputs.shape[0])


def _future_gain(
    inputs: np.ndarray,
    labels: np.ndarray,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    initial_loss: float,
    dimension: int,
    hidden: int,
) -> dict[str, float]:
    initial = _flatten(arrays)
    gains: dict[str, float] = {}
    for horizon in FUTURE_HORIZONS:
        parameters = initial.copy()
        for _ in range(horizon):
            current = _unpack(parameters, dimension, hidden)
            parameters -= LEARNING_RATE * _gradient(inputs, labels, current)
        terminal = _predict(inputs, _unpack(parameters, dimension, hidden))
        terminal_loss = float(np.mean(np.square(terminal - labels)))
        gains[str(horizon)] = initial_loss - terminal_loss
    return gains


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _runtime() -> dict[str, str]:
    numpy_build = json.dumps(
        np.__config__.CONFIG,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "numpy_build_sha256": hashlib.sha256(numpy_build).hexdigest(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def _json_preflight(value: object) -> None:
    stack = [(value, 0)]
    seen: set[int] = set()
    nodes = 0
    strings = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES or depth > 16:
            raise ValueError("eNTK payload exceeds its JSON work bound")
        if item is None or type(item) is bool:
            continue
        if type(item) is int:
            if not -MAX_JSON_INTEGER - 1 <= item <= MAX_JSON_INTEGER:
                raise ValueError("eNTK payload contains an unbounded integer")
            continue
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("eNTK payload contains nonfinite JSON")
            continue
        if type(item) is str:
            strings += _utf8_length(item, name="eNTK JSON string")
            if strings > MAX_JSON_STRING_BYTES:
                raise ValueError("eNTK payload exceeds its string-byte bound")
            continue
        if type(item) not in (dict, list):
            raise ValueError("eNTK payload must contain exact JSON primitives")
        identity = id(item)
        if identity in seen:
            raise ValueError("eNTK payload cannot contain aliases or cycles")
        seen.add(identity)
        if type(item) is dict:
            mapping = cast(dict[object, object], item)
            if len(mapping) > 128:
                raise ValueError("eNTK JSON container exceeds its item bound")
            for key, child in mapping.items():
                if type(key) is not str:
                    raise ValueError("eNTK JSON keys must be exact strings")
                strings += _utf8_length(key, name="eNTK JSON key")
                if strings > MAX_JSON_STRING_BYTES:
                    raise ValueError("eNTK payload exceeds its string-byte bound")
                stack.append((child, depth + 1))
        else:
            sequence = cast(list[object], item)
            if len(sequence) > 128:
                raise ValueError("eNTK JSON container exceeds its item bound")
            stack.extend((child, depth + 1) for child in sequence)


def _canonical(value: object) -> bytes:
    _json_preflight(value)
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def execute_entk_readiness(
    validation_inputs: object,
    validation_labels: object,
    checkpoint: object,
    *,
    task_id: str,
    checkpoint_id: str,
) -> dict[str, object]:
    """Execute the bounded nonlinear representation/eNTK diagnostic."""
    resolved_task = _text(task_id, name="task_id")
    resolved_checkpoint = _text(checkpoint_id, name="checkpoint_id")
    source_before = _source_sha256()
    raw_x, raw_y, raw_arrays, resources = _preflight(
        validation_inputs, validation_labels, checkpoint
    )
    x, y = _snapshot(raw_x), _snapshot(raw_y)
    arrays = cast(
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        tuple(_snapshot(value) for value in raw_arrays),
    )
    predictions, representation, jacobian = _model(x, arrays)
    residual = predictions - y
    initial_loss = float(np.mean(np.square(residual)))
    gradient = 2.0 * (jacobian.T @ residual) / x.shape[0]
    future_gain = _future_gain(
        x,
        y,
        arrays,
        initial_loss,
        resources["input_dimension"],
        resources["hidden_units"],
    )
    metrics: dict[str, object] = {
        "initial_loss": initial_loss,
        "future_loss_gain": future_gain,
        "gradient_norm": float(np.linalg.norm(gradient)),
        "representation_frobenius_norm": float(np.linalg.norm(representation)),
        "entk_feature_frobenius_norm": float(np.linalg.norm(jacobian)),
        "representation_energy_rank_99": _energy_rank(representation),
        "entk_energy_rank_99": _energy_rank(jacobian),
    }
    scalar_metrics = tuple(value for key, value in metrics.items() if key != "future_loss_gain")
    if not all(
        type(value) is int or (type(value) is float and math.isfinite(value))
        for value in (*scalar_metrics, *future_gain.values())
    ):
        raise RuntimeError("eNTK diagnostic produced invalid metrics")
    if _source_sha256() != source_before:
        raise RuntimeError("eNTK source changed during execution")
    result: dict[str, object] = {
        "schema": SCHEMA,
        "protocol": {
            "model": "two_layer_relu_scalar_regression",
            "learning_rate": LEARNING_RATE,
            "energy_threshold": ENERGY_THRESHOLD,
            "future_horizons": list(FUTURE_HORIZONS),
            "rng": "none_full_batch_deterministic",
            "relu_zero_derivative": 0.0,
            "task_id": resolved_task,
            "checkpoint_id": resolved_checkpoint,
        },
        "identity": {
            "source_sha256": source_before,
            "runtime": _runtime(),
            "dataset": {
                "inputs": _array_identity("validation_inputs", x),
                "labels": _array_identity("validation_labels", y),
            },
            "checkpoint": _checkpoint_identity(arrays),
        },
        "metrics": metrics,
        "resources": resources,
        "policy": {
            "development_only": True,
            "scientific_promotion_allowed": False,
            "timing_is_measured": False,
        },
    }
    return cast(dict[str, object], json.loads(_canonical(result)))


def validate_entk_readiness(
    value: object,
    validation_inputs: object,
    validation_labels: object,
    checkpoint: object,
) -> dict[str, object]:
    """Strictly validate identities and replay the complete nonlinear diagnostic."""
    _json_preflight(value)
    if type(value) is not dict or set(value) != {
        "schema", "protocol", "identity", "metrics", "resources", "policy"
    }:
        raise ValueError("eNTK result fields differ from the schema")
    raw = cast(dict[str, object], value)
    protocol = raw["protocol"]
    identity = raw["identity"]
    if type(protocol) is not dict or set(protocol) != {
        "model",
        "learning_rate",
        "energy_threshold",
        "future_horizons",
        "rng",
        "relu_zero_derivative",
        "task_id",
        "checkpoint_id",
    }:
        raise ValueError("eNTK protocol fields differ")
    if type(identity) is not dict or set(identity) != {
        "source_sha256", "runtime", "dataset", "checkpoint"
    }:
        raise ValueError("eNTK identity fields differ")
    raw_x, raw_y, raw_arrays, _ = _preflight(validation_inputs, validation_labels, checkpoint)
    x, y = _snapshot(raw_x), _snapshot(raw_y)
    arrays = cast(
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        tuple(_snapshot(value) for value in raw_arrays),
    )
    dataset = identity["dataset"]
    if type(dataset) is not dict or dataset != {
        "inputs": _array_identity("validation_inputs", x),
        "labels": _array_identity("validation_labels", y),
    }:
        raise ValueError("supplied arrays do not match the dataset identity")
    if identity["checkpoint"] != _checkpoint_identity(arrays):
        raise ValueError("supplied checkpoint does not match the checkpoint identity")
    expected = execute_entk_readiness(
        x,
        y,
        MLPCheckpoint(*arrays),
        task_id=cast(str, protocol["task_id"]),
        checkpoint_id=cast(str, protocol["checkpoint_id"]),
    )
    if _canonical(raw) != _canonical(expected):
        raise ValueError("eNTK result differs from strict replay")
    return expected


__all__ = ["MLPCheckpoint", "execute_entk_readiness", "validate_entk_readiness"]
