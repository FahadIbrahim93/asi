"""Bounded, model-bound prospective Optimization Readiness execution.

This is a permanently nonpromoting development diagnostic.  It executes a
fully specified linear squared-loss model from an explicit checkpoint against
one supplied 10,000-example task.  The validator repeats the execution from
the bound inputs instead of accepting reported metrics or counters.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import stat
import sys
from importlib.metadata import version
from pathlib import Path
from types import MappingProxyType
from typing import Final

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.evaluation.optimization_readiness import (
    energy_rank,
    estimate_appendix_c1_optimization_readiness,
)

EXECUTION_SCHEMA: Final[str] = "asi.optimization-readiness.execution.v1"
AUTHORIZATION_TRANSITION_APPROVED: Final[bool] = False
EXECUTION_AUTHORIZED: Final[bool] = False
_OBSERVATIONS: Final[int] = 10_000
_MAX_PARAMETERS: Final[int] = 64
_BATCH_SIZE: Final[int] = 4
_ROLLOUTS: Final[int] = 128
_DIAGNOSTIC_BATCHES: Final[int] = 128
_GAIN_STEPS: Final[tuple[int, ...]] = (1, 10, 100)
_STEP_SIZE: Final[float] = 1e-3
_MAX_LIVE_BYTES: Final[int] = 256 * 1024 * 1024
_MAX_PREFLIGHT_WORK_UNITS: Final[int] = 500_000_000
_MAX_JSON_NODES: Final[int] = 4_096
_MAX_JSON_STRING_BYTES: Final[int] = 64 * 1024
_SAMPLING_PROVENANCE: Final[str] = (
    "executor_derived_independent_with_replacement_jax_threefry2x32"
)
_SOURCE_FILES: Final[tuple[str, ...]] = (
    "pyproject.toml",
    "uv.lock",
    "alberta_framework/evaluation/optimization_readiness.py",
    "alberta_framework/evaluation/optimization_readiness_executor.py",
)

FROZEN_EXECUTION_PLAN = MappingProxyType(
    {
        "authorization_transition_approved": AUTHORIZATION_TRANSITION_APPROVED,
        "execution_authorized": EXECUTION_AUTHORIZED,
        "model": "linear_squared_loss_no_bias",
        "full_validation_observations": _OBSERVATIONS,
        "diagnostic_batch_count": _DIAGNOSTIC_BATCHES,
        "diagnostic_batch_size": _BATCH_SIZE,
        "diagnostic_sampling": "independent_with_replacement",
        "future_gain_steps": _GAIN_STEPS,
        "future_gain_rollout_count": _ROLLOUTS,
        "future_gain_batch_size": _BATCH_SIZE,
        "future_gain_step_size": _STEP_SIZE,
        "optimizer": "plain_sgd",
        "rng_impl": "threefry2x32",
        "sampling_provenance": _SAMPLING_PROVENANCE,
        "representation": "linear_model_input_matrix",
        "curvature": "exact_full_validation_squared_loss_hessian",
        "paper_parity": "bounded_linear_adapter_not_scr_or_permuted_mnist",
        "preflight_work_unit_scope": (
            "safety_only_parameter_data_incidence_not_scalar_flops_or_comparison_receipt"
        ),
    }
)


def _authorization_identity() -> dict[str, bool]:
    return {
        "authorization_transition_approved": AUTHORIZATION_TRANSITION_APPROVED,
        "execution_authorized": EXECUTION_AUTHORIZED,
    }


def _require_execution_authorized() -> None:
    if AUTHORIZATION_TRANSITION_APPROVED is not True or EXECUTION_AUTHORIZED is not True:
        raise PermissionError(
            "optimization-readiness execution requires a separately reviewed authorization "
            "transition and literal execution authorization"
        )


def _exact_string(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be an exact string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must be valid UTF-8") from exc
    if not encoded or len(encoded) > 4_096 or "\x00" in value:
        raise ValueError(f"{name} must be a bounded non-empty UTF-8 string")
    return value


def _assert_plain_json(value: object) -> None:
    """Reject subclasses, aliases, cycles, and aggregate-unbounded JSON trees."""
    seen_containers: set[int] = set()
    node_count = 0
    string_bytes = 0

    def visit(item: object, *, depth: int) -> None:
        nonlocal node_count, string_bytes
        node_count += 1
        if node_count > _MAX_JSON_NODES:
            raise ValueError("execution artifact exceeds the aggregate node limit")
        if depth > 8:
            raise ValueError("execution artifact must be bounded plain JSON")
        if item is None or type(item) is bool:
            return
        if type(item) is int:
            if not -(1 << 63) <= item <= (1 << 63) - 1:
                raise ValueError("execution artifact must be bounded plain JSON")
            return
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("execution artifact must be bounded plain JSON")
            return
        if type(item) is str:
            resolved = _exact_string(item, name="execution artifact string")
            string_bytes += len(resolved.encode("utf-8"))
            if string_bytes > _MAX_JSON_STRING_BYTES:
                raise ValueError("execution artifact exceeds the aggregate string-byte limit")
            return
        if type(item) is list:
            identity = id(item)
            if identity in seen_containers:
                raise ValueError("execution artifact contains an aliased or cyclic container")
            seen_containers.add(identity)
            if list.__len__(item) > 512:
                raise ValueError("execution artifact must be bounded plain JSON")
            for child in list.__iter__(item):
                visit(child, depth=depth + 1)
            return
        if type(item) is dict:
            identity = id(item)
            if identity in seen_containers:
                raise ValueError("execution artifact contains an aliased or cyclic container")
            seen_containers.add(identity)
            if dict.__len__(item) > 128:
                raise ValueError("execution artifact must be bounded plain JSON")
            for key, child in dict.items(item):
                if type(key) is not str:
                    raise ValueError("execution artifact must be bounded plain JSON")
                visit(key, depth=depth + 1)
                visit(child, depth=depth + 1)
            return
        raise ValueError("execution artifact must be bounded plain JSON")

    visit(value, depth=0)


def _preflight_arrays(
    validation_inputs: object,
    validation_labels: object,
    checkpoint_parameters: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Check trusted ndarray metadata and total work before scanning values."""
    named = (
        ("validation_inputs", validation_inputs, 2),
        ("validation_labels", validation_labels, 1),
        ("checkpoint_parameters", checkpoint_parameters, 1),
    )
    for name, value, dimensions in named:
        if type(value) is not np.ndarray:
            raise ValueError(f"{name} must be an exact numpy.ndarray")
        if value.ndim != dimensions or any(axis < 1 for axis in value.shape):
            raise ValueError(f"{name} must be a non-empty {dimensions}-dimensional array")
        if value.dtype != np.dtype(np.float64):
            raise ValueError(f"{name} must have exact float64 dtype")
    inputs = validation_inputs
    labels = validation_labels
    checkpoint = checkpoint_parameters
    assert isinstance(inputs, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert isinstance(checkpoint, np.ndarray)
    if inputs.shape[0] != _OBSERVATIONS or labels.shape != (_OBSERVATIONS,):
        raise ValueError("the frozen task requires exactly 10,000 aligned observations")
    parameters = int(inputs.shape[1])
    if not 1 <= parameters <= _MAX_PARAMETERS:
        raise ValueError("parameter count exceeds the bounded model protocol")
    if checkpoint.shape != (parameters,):
        raise ValueError("checkpoint parameter axis must match validation inputs")

    total_updates = _ROLLOUTS * sum(_GAIN_STEPS)
    preflight_work_units = (
        _OBSERVATIONS * parameters
        + _DIAGNOSTIC_BATCHES * _BATCH_SIZE * parameters
        + total_updates * _BATCH_SIZE * parameters
        + len(_GAIN_STEPS) * _ROLLOUTS * _OBSERVATIONS * parameters
        + _OBSERVATIONS * parameters * parameters
    )
    if preflight_work_units > _MAX_PREFLIGHT_WORK_UNITS:
        raise ValueError("execution preflight work exceeds the bounded protocol")
    schedule_items = _BATCH_SIZE * (_DIAGNOSTIC_BATCHES + total_updates)
    caller_bytes = int(inputs.nbytes + labels.nbytes + checkpoint.nbytes)
    owned_bytes = caller_bytes
    schedule_bytes = schedule_items * np.dtype(np.int64).itemsize
    schedule_device_bytes = schedule_items * np.dtype(np.int32).itemsize
    diagnostic_bytes = _DIAGNOSTIC_BATCHES * parameters * np.dtype(np.float64).itemsize
    # Terminal loss evaluation retains two full observation-by-rollout arrays:
    # matmul output beside residual, then residual beside its squared temporary.
    terminal_loss_bytes = (
        2 * _OBSERVATIONS * _ROLLOUTS * np.dtype(np.float64).itemsize
    )
    rollout_state_bytes = (
        _ROLLOUTS
        * (parameters + 2 * _BATCH_SIZE * parameters + 2 * _BATCH_SIZE)
        * np.dtype(np.float64).itemsize
    )
    diagnostic_state_bytes = (
        parameters * parameters + 2 * parameters
    ) * np.dtype(np.float64).itemsize
    # The input-matrix SVD is conservatively charged with its float64 copy,
    # LAPACK copy, finite mask, and a quadratic workspace allowance.  Caller
    # inputs and our immutable snapshots remain live throughout.
    representation_svd_bytes = (
        int(inputs.nbytes)
        + 17 * int(inputs.size)
        + 24 * parameters * parameters
        + 8 * max(inputs.shape)
        + 96 * parameters
    )
    peak_bytes = (
        caller_bytes
        + owned_bytes
        + schedule_bytes
        + schedule_device_bytes
        + diagnostic_bytes
        + terminal_loss_bytes
        + rollout_state_bytes
        + diagnostic_state_bytes
        + representation_svd_bytes
    )
    if peak_bytes > _MAX_LIVE_BYTES:
        raise ValueError("execution total live memory exceeds the bounded protocol")
    return inputs, labels, checkpoint, {
        "parameter_count": parameters,
        "preflight_work_units": preflight_work_units,
        "persistent_bytes": owned_bytes + schedule_bytes,
        "peak_working_set_bytes": peak_bytes,
        "model_queries": (
            _OBSERVATIONS
            + _DIAGNOSTIC_BATCHES * _BATCH_SIZE
            + _ROLLOUTS
            * sum(step * _BATCH_SIZE + _OBSERVATIONS for step in _GAIN_STEPS)
        ),
        "parameter_updates": total_updates,
    }


def _snapshot(value: np.ndarray, *, name: str) -> np.ndarray:
    result = np.array(value, dtype=np.float64, order="C", copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    result.flags.writeable = False
    return result


def _array_identity(value: np.ndarray) -> dict[str, object]:
    digest = hashlib.sha256()
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.dtype.str.encode())
    digest.update(value.tobytes(order="C"))
    return {
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "sha256": digest.hexdigest(),
    }


def _dataset_identity(inputs: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    return {"inputs": _array_identity(inputs), "labels": _array_identity(labels)}


def _source_identity() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    identities: dict[str, str] = {}
    for relative in _SOURCE_FILES:
        path = root / relative
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise ValueError("source identity file cannot be opened safely") from exc
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= 2 * 1024 * 1024:
                raise ValueError("source identity file must be a bounded regular file")
            chunks: list[bytes] = []
            remaining = metadata.st_size + 1
            while remaining:
                chunk = os.read(descriptor, min(remaining, 64 * 1024))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            payload = b"".join(chunks)
            if len(payload) != metadata.st_size:
                raise ValueError("source identity file changed during its bounded read")
            after = os.fstat(descriptor)
            if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_size,
                metadata.st_mtime_ns,
            ):
                raise ValueError("source identity file changed during its bounded read")
        finally:
            os.close(descriptor)
        identities[relative] = hashlib.sha256(payload).hexdigest()
    return identities


def _runtime_identity() -> dict[str, object]:
    numpy_build = json.dumps(
        np.__config__.CONFIG, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "byteorder": sys.byteorder,
        "numpy": np.__version__,
        "numpy_build_sha256": hashlib.sha256(numpy_build).hexdigest(),
        "jax": jax.__version__,
        "jaxlib": version("jaxlib"),
        "jax_backend": jax.default_backend(),
        "jax_devices": [
            {"platform": device.platform, "device_kind": device.device_kind}
            for device in jax.devices()
        ],
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "agent_rng_impl": "jax.random.key(seed, impl='threefry2x32')",
    }


def _plan_payload() -> dict[str, object]:
    payload = dict(FROZEN_EXECUTION_PLAN)
    payload["future_gain_steps"] = list(_GAIN_STEPS)
    return payload


def _schedule(seed: int) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    root = jr.key(seed, impl="threefry2x32")
    diagnostic_key, rollout_root = jr.split(root)
    diagnostic = np.asarray(
        jr.randint(
            diagnostic_key,
            (_DIAGNOSTIC_BATCHES, _BATCH_SIZE),
            0,
            _OBSERVATIONS,
            dtype=jnp.int32,
        ),
        dtype=np.int64,
    )
    horizon_keys = jr.split(rollout_root, len(_GAIN_STEPS))
    rollouts: dict[int, np.ndarray] = {}
    for step, key in zip(_GAIN_STEPS, horizon_keys, strict=True):
        rollouts[step] = np.asarray(
            jr.randint(
                key,
                (_ROLLOUTS, step, _BATCH_SIZE),
                0,
                _OBSERVATIONS,
                dtype=jnp.int32,
            ),
            dtype=np.int64,
        )
    return diagnostic, rollouts


def _schedule_sha256(diagnostic: np.ndarray, rollouts: dict[int, np.ndarray]) -> str:
    digest = hashlib.sha256(diagnostic.tobytes(order="C"))
    for step in _GAIN_STEPS:
        digest.update(step.to_bytes(2, "big"))
        digest.update(rollouts[step].tobytes(order="C"))
    return digest.hexdigest()


def _loss_and_gradient(
    inputs: np.ndarray, labels: np.ndarray, parameters: np.ndarray
) -> tuple[float, np.ndarray]:
    residual = inputs @ parameters - labels
    loss = float(np.mean(np.square(residual)))
    gradient = np.asarray(2.0 * (inputs.T @ residual) / inputs.shape[0], dtype=np.float64)
    if not math.isfinite(loss) or not np.all(np.isfinite(gradient)):
        raise ValueError("model loss and gradient must remain finite")
    return loss, gradient


def _future_gain(
    inputs: np.ndarray,
    labels: np.ndarray,
    checkpoint: np.ndarray,
    schedule: np.ndarray,
    initial_loss: float,
) -> float:
    parameters = np.broadcast_to(checkpoint, (_ROLLOUTS, checkpoint.size)).copy()
    for update in range(schedule.shape[1]):
        indices = schedule[:, update, :]
        batch_inputs = inputs[indices]
        batch_labels = labels[indices]
        prediction = np.einsum("rbp,rp->rb", batch_inputs, parameters, optimize=False)
        residual = prediction - batch_labels
        gradient = 2.0 * np.einsum(
            "rbp,rb->rp", batch_inputs, residual, optimize=False
        ) / _BATCH_SIZE
        parameters -= _STEP_SIZE * gradient
    terminal_residual = inputs @ parameters.T - labels[:, None]
    terminal_losses = np.mean(np.square(terminal_residual), axis=0)
    if not np.all(np.isfinite(terminal_losses)):
        raise ValueError("future-gain rollout produced non-finite losses")
    if initial_loss == 0.0:
        return 0.0
    return float(np.mean((initial_loss - terminal_losses) / initial_loss))


def _execute_optimization_readiness(
    *,
    validation_inputs: object,
    validation_labels: object,
    checkpoint_parameters: object,
    seed: int,
    task_id: str,
    checkpoint_id: str,
) -> dict[str, object]:
    """Private executor used only behind the panel's reserved transaction."""
    if type(seed) is not int or not 0 <= seed <= (1 << 32) - 1:
        raise ValueError("seed must be a bounded nonnegative built-in int")
    resolved_task = _exact_string(task_id, name="task_id")
    resolved_checkpoint_id = _exact_string(checkpoint_id, name="checkpoint_id")
    raw_inputs, raw_labels, raw_checkpoint, resources = _preflight_arrays(
        validation_inputs, validation_labels, checkpoint_parameters
    )
    inputs = _snapshot(raw_inputs, name="validation_inputs")
    labels = _snapshot(raw_labels, name="validation_labels")
    checkpoint = _snapshot(raw_checkpoint, name="checkpoint_parameters")

    dataset = _dataset_identity(inputs, labels)
    checkpoint_identity = _array_identity(checkpoint)
    diagnostic_indices, rollout_indices = _schedule(seed)
    initial_loss, full_gradient = _loss_and_gradient(inputs, labels, checkpoint)
    batch_gradients = np.empty(
        (_DIAGNOSTIC_BATCHES, checkpoint.size), dtype=np.float64
    )
    for row, indices in enumerate(diagnostic_indices):
        _, batch_gradients[row] = _loss_and_gradient(
            inputs[indices], labels[indices], checkpoint
        )
    readiness = estimate_appendix_c1_optimization_readiness(
        loss=initial_loss,
        full_validation_gradient=full_gradient,
        batch_gradients=batch_gradients,
        full_validation_observations=_OBSERVATIONS,
        mini_batch_size=_BATCH_SIZE,
        sampling_provenance=(
            "caller_reported_independent_with_replacement_not_verified_from_gradients"
        ),
    )
    curvature = np.asarray(2.0 * (inputs.T @ inputs) / _OBSERVATIONS, dtype=np.float64)
    gains = {
        str(step): _future_gain(
            inputs, labels, checkpoint, rollout_indices[step], initial_loss
        )
        for step in _GAIN_STEPS
    }
    metrics: dict[str, object] = {
        "initial_validation_loss": initial_loss,
        "optimization_readiness": readiness.optimization_readiness,
        "gradient_strength": readiness.gradient_strength,
        "gradient_reliability": readiness.gradient_reliability,
        "gradient_norm": readiness.gradient_norm,
        "parameter_norm": float(np.linalg.norm(checkpoint)),
        "representation_energy_rank_0_99": energy_rank(inputs, threshold=0.99),
        "curvature_energy_rank_0_99": energy_rank(curvature, threshold=0.99),
        "future_relative_loss_reduction": gains,
    }
    scalar_metrics = (
        initial_loss,
        readiness.optimization_readiness,
        readiness.gradient_strength,
        readiness.gradient_reliability,
        readiness.gradient_norm,
        float(np.linalg.norm(checkpoint)),
    )
    if not all(math.isfinite(value) for value in scalar_metrics) or not all(
        math.isfinite(value) for value in gains.values()
    ):
        raise ValueError("execution metrics must be finite")
    return {
        "schema": EXECUTION_SCHEMA,
        "policy": {
            **_authorization_identity(),
            "development_only": True,
            "scientific_promotion_allowed": False,
            "timing_is_telemetry_only": True,
        },
        "plan": _plan_payload(),
        "execution": {
            "seed": seed,
            "task_id": resolved_task,
            "checkpoint_id": resolved_checkpoint_id,
        },
        "identity": {
            "authorization": _authorization_identity(),
            "source_sha256": _source_identity(),
            "runtime": _runtime_identity(),
            "dataset": dataset,
            "checkpoint": checkpoint_identity,
        },
        "sampling": {
            "rng_impl": "threefry2x32",
            "diagnostic_gradient_count": _DIAGNOSTIC_BATCHES,
            "schedule_sha256": _schedule_sha256(diagnostic_indices, rollout_indices),
        },
        "resources": {
            **resources,
            "environment_steps": 0,
            "data_steps": resources["model_queries"],
            "timing_seconds": 0.0,
            "timing_measured": False,
            "timing_is_telemetry_only": True,
        },
        "metrics": metrics,
    }


def _validate_optimization_readiness_execution(
    payload: object,
    *,
    validation_inputs: object,
    validation_labels: object,
    checkpoint_parameters: object,
) -> dict[str, object]:
    """Fail closed unless a supplied execution repeats exactly from its inputs."""
    if type(payload) is not dict:
        raise ValueError("execution artifact must be an exact dict")
    expected_keys = {
        "schema",
        "policy",
        "plan",
        "execution",
        "identity",
        "sampling",
        "resources",
        "metrics",
    }
    if dict.__len__(payload) != len(expected_keys):
        raise ValueError("execution artifact has unexpected keys")
    if any(
        type(key) is not str or key not in expected_keys
        for key in dict.keys(payload)
    ):
        raise ValueError("execution artifact has unexpected keys")
    _assert_plain_json(payload)
    if payload.get("schema") != EXECUTION_SCHEMA:
        raise ValueError("execution artifact schema is not supported")
    execution = payload.get("execution")
    identity = payload.get("identity")
    if type(execution) is not dict or set(execution) != {
        "seed",
        "task_id",
        "checkpoint_id",
    }:
        raise ValueError("execution descriptor is malformed")
    if type(identity) is not dict or set(identity) != {
        "authorization",
        "source_sha256",
        "runtime",
        "dataset",
        "checkpoint",
    }:
        raise ValueError("execution identity is malformed")
    seed = execution["seed"]
    if type(seed) is not int:
        raise ValueError("execution seed must be an exact int")
    task_id = _exact_string(execution["task_id"], name="task_id")
    checkpoint_id = _exact_string(execution["checkpoint_id"], name="checkpoint_id")

    raw_inputs, raw_labels, raw_checkpoint, _ = _preflight_arrays(
        validation_inputs, validation_labels, checkpoint_parameters
    )
    inputs = _snapshot(raw_inputs, name="validation_inputs")
    labels = _snapshot(raw_labels, name="validation_labels")
    checkpoint = _snapshot(raw_checkpoint, name="checkpoint_parameters")
    if identity["dataset"] != _dataset_identity(inputs, labels):
        raise ValueError("supplied arrays do not match the bound dataset identity")
    if identity["checkpoint"] != _array_identity(checkpoint):
        raise ValueError("supplied parameters do not match the bound checkpoint identity")
    recomputed = _execute_optimization_readiness(
        validation_inputs=inputs,
        validation_labels=labels,
        checkpoint_parameters=checkpoint,
        seed=seed,
        task_id=task_id,
        checkpoint_id=checkpoint_id,
    )
    if payload != recomputed:
        raise ValueError("execution artifact does not recompute exactly")
    return recomputed


__all__ = [
    "AUTHORIZATION_TRANSITION_APPROVED",
    "EXECUTION_SCHEMA",
    "EXECUTION_AUTHORIZED",
    "FROZEN_EXECUTION_PLAN",
]
