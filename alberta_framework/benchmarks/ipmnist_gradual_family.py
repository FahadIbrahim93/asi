"""Bounded matched micro-phases for gradual-transition development research.

This is an additive companion to :mod:`ipmnist_gradual`.  It exercises the
paper's output-interpolation and task-sampling definitions end to end without
recasting the retained input-interpolation result.  The adapter is deliberately
small, caller-fed, permanently nonpromoting, and incapable of writing results.
"""

from __future__ import annotations

import hashlib
import math
import operator
import platform
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from typing import Any, SupportsIndex, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework._seed_validation import require_jax_seed
from alberta_framework.benchmarks.ipmnist_gradual import output_interpolation
from alberta_framework.benchmarks.upgd_ipmnist import (
    ADAMW_PROTOCOL_HYPERPARAMETERS,
    IPMNISTConfig,
    LearnerUpdateResult,
    _make_adamw_learner,
    init_mlp_params,
    mlp_logits,
    validated_ipmnist_data,
)

_SCHEMA = "asi.ipmnist.gradual-family.micro-phase-result.v1"
_PRNG_IMPLEMENTATION = "threefry2x32"
_ARMS = ("abrupt", "output_interpolation", "task_sampling")
_ADAMW_IDENTITY = tuple(sorted(ADAMW_PROTOCOL_HYPERPARAMETERS.items()))
_MAX_RESOURCE_BYTES = 256 * 1024 * 1024
_MAX_RUN_STEPS = 1_000_000
_INT32_MAX = 2**31 - 1

GRADUAL_FAMILY_PROTOCOL = MappingProxyType(
    {
        "schema": "asi.ipmnist.gradual-family.micro-phase-protocol.v1",
        "paper_revision": "arXiv:2602.09234v2",
        "paper_revision_date": "2026-06-16",
        "paper_full_dataset_training": False,
        "output_interpolation_loss": "soft_target_cross_entropy",
        "output_interpolation_input": "new-task paired example",
        "task_sampling_count": "floor(alpha * phase_examples)",
        "task_sampling_old_count": "ceil((1-alpha) * phase_examples)",
        "abrupt_switch": "old task for alpha < 1/2; new task for alpha >= 1/2",
        "matched_axes": (
            "initial_parameters",
            "learner_hyperparameters",
            "phase_count",
            "phase_examples",
            "updates",
            "new_task_evaluation_queries",
        ),
        "adaptation_gaps": (
            "bounded paired micro-phases, not the paper's complete datasets or training horizon",
            "the output arm presents paired new-task inputs while targets move old-uniform-new",
            "caller-provided arrays are not an official IPMNIST dataset acquisition protocol",
        ),
        "retained_input_result_recast": False,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "negative_results_must_be_retained": True,
        "timing_is_telemetry_only": True,
    }
)


def _exact_int(name: str, value: object, *, minimum: int, maximum: int = _INT32_MAX) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an exact integer")
    resolved = operator.index(cast(SupportsIndex, value))
    if resolved < minimum or resolved > maximum:
        raise ValueError(f"{name} must lie in [{minimum}, {maximum}]")
    return resolved


def _digest_is_canonical(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _source_identity() -> tuple[tuple[str, str], ...]:
    root = Path(__file__).resolve().parents[2]
    relative_paths = (
        "alberta_framework/_seed_validation.py",
        "alberta_framework/benchmarks/ipmnist_gradual.py",
        "alberta_framework/benchmarks/ipmnist_gradual_family.py",
        "alberta_framework/benchmarks/upgd_ipmnist.py",
        "alberta_framework/core/_float32_scalars.py",
        "alberta_framework/core/baseline_optimizers.py",
        "alberta_framework/core/update_safety.py",
    )
    return tuple(
        (relative, hashlib.sha256((root / relative).read_bytes()).hexdigest())
        for relative in relative_paths
    )


def _runtime_identity() -> tuple[tuple[str, str], ...]:
    devices = sorted(
        (
            int(device.process_index),
            int(device.id),
            str(device.platform),
            str(device.device_kind),
        )
        for device in jax.devices()
    )
    if not 1 <= len(devices) <= 128:
        raise ValueError("runtime device inventory must contain 1 through 128 devices")
    return (
        ("python_implementation", platform.python_implementation()),
        ("python_version", platform.python_version()),
        ("platform_system", platform.system()),
        ("platform_machine", platform.machine()),
        ("jax_version", jax.__version__),
        ("jaxlib_version", metadata.version("jaxlib")),
        ("numpy_version", np.__version__),
        ("jax_backend", jax.default_backend()),
        ("jax_devices", repr(devices)),
        ("jax_enable_x64", repr(jax.config.jax_enable_x64)),
        ("jax_default_matmul_precision", repr(jax.config.jax_default_matmul_precision)),
        ("jax_random_seed_offset", repr(jax.config.jax_random_seed_offset)),
        ("jax_prng_implementation", _PRNG_IMPLEMENTATION),
    )


def _array_digest(domain: bytes, *arrays: np.ndarray | Array) -> str:
    digest = hashlib.sha256(domain)
    for value in arrays:
        array = np.asarray(jax.device_get(value))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True, slots=True)
class GradualMicroPhaseConfig:
    """A bounded family of ``K + 1`` complete, equal-size transition phases."""

    transition_intervals: int
    phase_examples: int
    input_dim: int
    hidden1: int
    hidden2: int
    n_classes: int

    def __post_init__(self) -> None:
        for name, minimum in (
            ("transition_intervals", 1),
            ("phase_examples", 1),
            ("input_dim", 1),
            ("hidden1", 1),
            ("hidden2", 1),
            ("n_classes", 2),
        ):
            object.__setattr__(self, name, _exact_int(name, getattr(self, name), minimum=minimum))
        if self.phase_count * self.phase_examples > _MAX_RUN_STEPS:
            raise ValueError("micro-phase run exceeds the 1000000-update ceiling")
        if self.parameter_count > _INT32_MAX:
            raise ValueError("parameter count must fit signed int32")

    @property
    def phase_count(self) -> int:
        return self.transition_intervals + 1

    @property
    def updates(self) -> int:
        return self.phase_count * self.phase_examples

    @property
    def parameter_count(self) -> int:
        return (
            self.input_dim * self.hidden1
            + self.hidden1 * self.hidden2
            + self.hidden2 * self.n_classes
            + self.hidden1
            + self.hidden2
            + self.n_classes
        )


@dataclass(frozen=True, slots=True)
class _ResourceEstimate:
    dataset_bytes: int
    schedule_bytes: int
    learner_bytes: int
    transient_schedule_bytes: int
    persistent_numeric_bytes: int
    working_set_bytes: int


def _resource_estimate(
    config: GradualMicroPhaseConfig, old_rows: int, new_rows: int
) -> _ResourceEstimate:
    """Return the one protocol resource calculation used by run and validation."""
    dataset_bytes = (old_rows + new_rows) * (config.input_dim + 1) * 4
    schedule_bytes = config.phase_count * config.phase_examples * (4 + 4 + 1)
    learner_bytes = config.parameter_count * 16 + 6 * 5 * 4
    transient_schedule_bytes = max(old_rows, new_rows) * 4 * 3
    persistent_numeric_bytes = dataset_bytes + schedule_bytes + learner_bytes
    return _ResourceEstimate(
        dataset_bytes=dataset_bytes,
        schedule_bytes=schedule_bytes,
        learner_bytes=learner_bytes,
        transient_schedule_bytes=transient_schedule_bytes,
        persistent_numeric_bytes=persistent_numeric_bytes,
        working_set_bytes=persistent_numeric_bytes + transient_schedule_bytes,
    )


@dataclass(frozen=True, slots=True)
class GradualMicroPhaseResult:
    schema: str
    arm_names: tuple[str, str, str]
    learner_name: str
    learner_hyperparameters: tuple[tuple[str, float], ...]
    prng_implementation: str
    seed: int
    config: GradualMicroPhaseConfig
    phase_alpha_numerators: tuple[int, ...]
    phase_alpha_denominator: int
    task_sampling_new_counts: np.ndarray
    dataset_rows: tuple[int, int]
    dataset_sha256: tuple[str, str]
    task_sampling_sha256: str
    source_sha256: tuple[tuple[str, str], ...]
    runtime_identity: tuple[tuple[str, str], ...]
    training_loss_sums: np.ndarray
    new_task_eval_correct_counts: np.ndarray
    new_task_eval_loss_sums: np.ndarray
    persistent_numeric_bytes: np.ndarray
    timing_ns: np.ndarray
    soft_target_updates_per_arm: tuple[int, int, int]
    observations_per_arm: int
    updates_per_arm: int
    data_steps_per_arm: int
    environment_steps_per_arm: int
    model_queries_per_arm: int
    parent_input_result_used: bool
    development_only: bool
    scientific_promotion_allowed: bool
    negative_results_must_be_retained: bool
    execution_attestation: bool

    def __post_init__(self) -> None:
        if type(self.schema) is not str or self.schema != _SCHEMA:
            raise ValueError(f"schema must be {_SCHEMA!r}")
        if type(self.arm_names) is not tuple or self.arm_names != _ARMS:
            raise ValueError("arm_names must identify the exact matched family")
        if type(self.learner_name) is not str or self.learner_name != "adamw_control":
            raise ValueError("learner identity must be the exact AdamW control")
        if (
            type(self.learner_hyperparameters) is not tuple
            or len(self.learner_hyperparameters) != len(_ADAMW_IDENTITY)
            or any(
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not str
                or type(item[1]) is not float
                or item != expected
                for item, expected in zip(
                    self.learner_hyperparameters, _ADAMW_IDENTITY, strict=True
                )
            )
        ):
            raise ValueError("learner identity must be the exact AdamW control")
        if (
            type(self.prng_implementation) is not str
            or self.prng_implementation != _PRNG_IMPLEMENTATION
        ):
            raise ValueError("prng_implementation must be threefry2x32")
        require_jax_seed(self.seed, name="seed")
        if type(self.config) is not GradualMicroPhaseConfig:
            raise ValueError("config must be an exact GradualMicroPhaseConfig")
        config = GradualMicroPhaseConfig(**{
            name: getattr(self.config, name)
            for name in (
                "transition_intervals", "phase_examples", "input_dim",
                "hidden1", "hidden2", "n_classes",
            )
        })
        if (
            type(self.phase_alpha_numerators) is not tuple
            or any(type(value) is not int for value in self.phase_alpha_numerators)
            or self.phase_alpha_numerators != tuple(range(config.phase_count))
        ):
            raise ValueError("phase alpha numerators must be the complete exact schedule")
        if (
            type(self.phase_alpha_denominator) is not int
            or self.phase_alpha_denominator != config.transition_intervals
        ):
            raise ValueError("phase alpha denominator must equal transition_intervals")
        if (
            type(self.dataset_rows) is not tuple
            or len(self.dataset_rows) != 2
            or any(
                type(value) is not int or value < config.phase_examples
                for value in self.dataset_rows
            )
        ):
            raise ValueError("dataset_rows must identify both bounded input datasets")
        if (
            type(self.dataset_sha256) is not tuple
            or len(self.dataset_sha256) != 2
            or not all(_digest_is_canonical(value) for value in self.dataset_sha256)
            or not _digest_is_canonical(self.task_sampling_sha256)
        ):
            raise ValueError("dataset and schedule identities must be canonical SHA-256 values")
        if self.source_sha256 != _source_identity():
            raise ValueError("current source identity drift")
        if self.runtime_identity != _runtime_identity():
            raise ValueError("current runtime identity drift")

        shapes = {
            "task_sampling_new_counts": (np.int32, (config.phase_count,)),
            "training_loss_sums": (np.float64, (len(_ARMS), config.phase_count)),
            "new_task_eval_correct_counts": (np.int32, (len(_ARMS), config.phase_count)),
            "new_task_eval_loss_sums": (np.float64, (len(_ARMS), config.phase_count)),
            "persistent_numeric_bytes": (np.int64, (len(_ARMS),)),
            "timing_ns": (np.int64, (len(_ARMS),)),
        }
        snapshots: dict[str, np.ndarray] = {}
        for name, (dtype, shape) in shapes.items():
            value = getattr(self, name)
            if type(value) is not np.ndarray or value.dtype != dtype or value.shape != shape:
                raise ValueError(f"{name} must be an exact {dtype.__name__} array of shape {shape}")
            snapshot = value.copy()
            snapshot.flags.writeable = False
            snapshots[name] = snapshot
        expected_counts = np.asarray(
            [index * config.phase_examples // config.transition_intervals
             for index in range(config.phase_count)],
            dtype=np.int32,
        )
        if not np.array_equal(snapshots["task_sampling_new_counts"], expected_counts):
            raise ValueError("task sampling counts must equal floor(alpha * phase_examples)")
        if not np.all(np.isfinite(snapshots["training_loss_sums"])) or np.any(
            snapshots["training_loss_sums"] < 0.0
        ):
            raise ValueError("training losses must be finite and nonnegative")
        eval_loss = snapshots["new_task_eval_loss_sums"]
        if not np.all(np.isfinite(eval_loss)) or np.any(eval_loss < 0.0):
            raise ValueError("evaluation losses must be finite and nonnegative")
        eval_correct = snapshots["new_task_eval_correct_counts"]
        if np.any(eval_correct < 0) or np.any(eval_correct > config.phase_examples):
            raise ValueError("evaluation correct counts must be valid integer numerators")
        if np.any(snapshots["persistent_numeric_bytes"] <= 0) or np.any(
            snapshots["persistent_numeric_bytes"] > _MAX_RESOURCE_BYTES
        ):
            raise ValueError("persistent numeric bytes must be positive and bounded")
        expected_persistent = _resource_estimate(
            config, self.dataset_rows[0], self.dataset_rows[1]
        ).persistent_numeric_bytes
        if not np.all(snapshots["persistent_numeric_bytes"] == expected_persistent):
            raise ValueError("persistent numeric bytes must equal the complete exact receipt")
        if np.any(snapshots["timing_ns"] < 0):
            raise ValueError("timing telemetry must be nonnegative")

        expected_counters = {
            "soft_target_updates_per_arm": (0, config.updates, 0),
            "observations_per_arm": 2 * config.updates,
            "updates_per_arm": config.updates,
            "data_steps_per_arm": 2 * config.updates,
            "environment_steps_per_arm": 0,
            "model_queries_per_arm": 3 * config.updates,
        }
        for name, expected in expected_counters.items():
            value = getattr(self, name)
            if name == "soft_target_updates_per_arm":
                valid = (
                    type(value) is tuple
                    and all(type(item) is int for item in value)
                    and value == expected
                )
            else:
                valid = type(value) is int and value == expected
            if not valid:
                raise ValueError(f"{name} must equal the protocol-derived value")
        flag_requirements = (
            self.parent_input_result_used is False,
            self.development_only is True,
            self.scientific_promotion_allowed is False,
            self.negative_results_must_be_retained is True,
            self.execution_attestation is False,
        )
        if not all(flag_requirements):
            raise ValueError("result must remain additive, nonpromoting, retained, and unattested")
        object.__setattr__(self, "config", config)
        for name, snapshot in snapshots.items():
            object.__setattr__(self, name, snapshot)


def _numeric_tree_bytes(value: object) -> int:
    total = 0
    for leaf in jax.tree_util.tree_leaves(value):
        if type(leaf) is np.ndarray or isinstance(leaf, jax.Array):
            array = np.asarray(jax.device_get(leaf))
            if not np.all(np.isfinite(array)):
                raise ValueError("learner state must remain finite")
            total += array.size * array.dtype.itemsize
    return total


def _checked_dataset_metadata(
    name: str, data_x: object, data_y: object, config: GradualMicroPhaseConfig
) -> tuple[np.ndarray | Array, np.ndarray | Array, int]:
    for field_name, value in ((f"{name}_x", data_x), (f"{name}_y", data_y)):
        actual_type = type(value)
        if not (actual_type is np.ndarray or isinstance(value, jax.Array)):
            raise ValueError(f"{field_name} must be an exact NumPy or JAX array")
    x = cast(np.ndarray | Array, data_x)
    y = cast(np.ndarray | Array, data_y)
    if len(x.shape) != 2 or x.shape[1] != config.input_dim:
        raise ValueError(f"{name}_x must be a matrix with input_dim columns")
    if y.shape != (x.shape[0],):
        raise ValueError(f"{name}_y must align with {name}_x")
    if x.shape[0] < config.phase_examples:
        raise ValueError(f"{name} dataset is smaller than one complete phase")
    elements = math.prod(x.shape) + math.prod(y.shape)
    if elements > _MAX_RESOURCE_BYTES // 4:
        raise ValueError(f"{name} materialized dataset exceeds 256 MiB")
    return x, y, elements


def _soft_cross_entropy(
    params: dict[str, Array], x: Array, target: Array
) -> tuple[Array, Array]:
    logits = mlp_logits(params, x)
    return -jnp.sum(target * jax.nn.log_softmax(logits)), logits


def _realized_schedule(
    seed: int,
    config: GradualMicroPhaseConfig,
    old_rows: int,
    new_rows: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[int], str]:
    root = jr.key(seed, impl=_PRNG_IMPLEMENTATION)
    _, schedule_key, _ = jr.split(root, 3)
    old_orders: list[np.ndarray] = []
    new_orders: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    new_counts: list[int] = []
    for phase in range(config.phase_count):
        phase_key = jr.fold_in(schedule_key, phase)
        old_key, new_key, mask_key = jr.split(phase_key, 3)
        old_order = np.asarray(
            jax.device_get(jr.permutation(old_key, old_rows)[:config.phase_examples]),
            dtype=np.int32,
        )
        new_order = np.asarray(
            jax.device_get(jr.permutation(new_key, new_rows)[:config.phase_examples]),
            dtype=np.int32,
        )
        count = phase * config.phase_examples // config.transition_intervals
        order = np.asarray(
            jax.device_get(jr.permutation(mask_key, config.phase_examples)), dtype=np.int32
        )
        mask = np.zeros((config.phase_examples,), dtype=np.bool_)
        mask[order[:count]] = True
        old_orders.append(old_order)
        new_orders.append(new_order)
        masks.append(mask)
        new_counts.append(count)
    digest = _array_digest(
        b"asi.ipmnist.gradual-family.schedule.v1\0",
        np.asarray(old_orders, dtype=np.int32),
        np.asarray(new_orders, dtype=np.int32),
        np.asarray(masks, dtype=np.bool_),
    )
    return old_orders, new_orders, masks, new_counts, digest


def run_gradual_micro_phase_family(
    old_x: object,
    old_y: object,
    new_x: object,
    new_y: object,
    *,
    learner_name: str,
    seed: int,
    config: GradualMicroPhaseConfig,
) -> GradualMicroPhaseResult:
    """Run abrupt, output-interpolation, and exact task-sampling arms.

    Every phase consumes ``phase_examples`` updates and then evaluates the
    current learner on ``phase_examples`` paired new-task examples.  The
    output arm alone trains with soft targets.  No result is retained here.
    """
    if type(learner_name) is not str or learner_name != "adamw_control":
        raise ValueError("learner_name must be the exact 'adamw_control' adapter")
    if type(config) is not GradualMicroPhaseConfig:
        raise ValueError("config must be an exact GradualMicroPhaseConfig")
    checked_config = GradualMicroPhaseConfig(**{
        name: getattr(config, name)
        for name in (
            "transition_intervals", "phase_examples", "input_dim",
            "hidden1", "hidden2", "n_classes",
        )
    })
    resolved_seed = require_jax_seed(seed, name="seed")
    raw_old_x, raw_old_y, _ = _checked_dataset_metadata(
        "old", old_x, old_y, checked_config
    )
    raw_new_x, raw_new_y, _ = _checked_dataset_metadata(
        "new", new_x, new_y, checked_config
    )
    resources = _resource_estimate(
        checked_config, int(raw_old_x.shape[0]), int(raw_new_x.shape[0])
    )
    if resources.working_set_bytes > _MAX_RESOURCE_BYTES:
        raise ValueError("aggregate working set exceeds 256 MiB")

    resolved_old_x, resolved_old_y = validated_ipmnist_data(
        raw_old_x,
        raw_old_y,
        input_dim=checked_config.input_dim,
        n_classes=checked_config.n_classes,
        min_length=checked_config.phase_examples,
    )
    resolved_new_x, resolved_new_y = validated_ipmnist_data(
        raw_new_x,
        raw_new_y,
        input_dim=checked_config.input_dim,
        n_classes=checked_config.n_classes,
        min_length=checked_config.phase_examples,
    )
    if resolved_old_x.shape != resolved_new_x.shape or not np.array_equal(
        resolved_old_x, resolved_new_x
    ):
        raise ValueError(
            "output interpolation requires row-aligned identical inputs across tasks"
        )
    old_x_array = jnp.asarray(resolved_old_x, dtype=jnp.float32)
    old_y_array = jnp.asarray(resolved_old_y, dtype=jnp.int32)
    new_x_array = jnp.asarray(resolved_new_x, dtype=jnp.float32)
    new_y_array = jnp.asarray(resolved_new_y, dtype=jnp.int32)

    init_fn, step_fn = _make_adamw_learner(dict(_ADAMW_IDENTITY))
    root = jr.key(resolved_seed, impl=_PRNG_IMPLEMENTATION)
    init_key, _, learner_key = jr.split(root, 3)
    model_config = IPMNISTConfig(
        n_tasks=checked_config.phase_count,
        task_length=checked_config.phase_examples,
        input_dim=checked_config.input_dim,
        hidden1=checked_config.hidden1,
        hidden2=checked_config.hidden2,
        n_classes=checked_config.n_classes,
    )
    initial_params = init_mlp_params(init_key, model_config)

    old_orders, new_orders, masks, new_counts, schedule_digest = _realized_schedule(
        resolved_seed,
        checked_config,
        int(old_x_array.shape[0]),
        int(new_x_array.shape[0]),
    )

    @jax.jit
    def checked_hard_step(
        params: dict[str, Array], state: Any, x: Array, label: Array, key: Array
    ) -> tuple[LearnerUpdateResult, Array, Array]:
        def loss_fn(candidate: dict[str, Array]) -> tuple[Array, Array]:
            logits = mlp_logits(candidate, x)
            return -jax.nn.log_softmax(logits)[label], logits

        (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        transaction = step_fn(params, state, grads, key)
        if type(transaction) is not LearnerUpdateResult:
            raise TypeError("AdamW learner did not return its checked transaction")
        post_loss, _ = loss_fn(transaction.params)
        return transaction, loss, post_loss

    @jax.jit
    def checked_soft_step(
        params: dict[str, Array], state: Any, x: Array, target: Array, key: Array
    ) -> tuple[LearnerUpdateResult, Array, Array]:
        (loss, _), grads = jax.value_and_grad(_soft_cross_entropy, has_aux=True)(params, x, target)
        transaction = step_fn(params, state, grads, key)
        if type(transaction) is not LearnerUpdateResult:
            raise TypeError("AdamW learner did not return its checked transaction")
        post_loss, _ = _soft_cross_entropy(transaction.params, x, target)
        return transaction, loss, post_loss

    @jax.jit
    def evaluate(params: dict[str, Array], x: Array, label: Array) -> tuple[Array, Array]:
        logits = mlp_logits(params, x)
        loss = -jax.nn.log_softmax(logits)[label]
        return (jnp.argmax(logits) == label).astype(jnp.int32), loss

    def run_arm(arm: str) -> tuple[list[float], list[int], list[float], int, int]:
        params = initial_params
        state = init_fn(params)
        key = learner_key
        phase_training_losses: list[float] = []
        phase_eval_correct: list[int] = []
        phase_eval_losses: list[float] = []
        started = time.perf_counter_ns()
        for phase in range(checked_config.phase_count):
            alpha = phase / checked_config.transition_intervals
            training_loss = 0.0
            for position in range(checked_config.phase_examples):
                old_index = int(old_orders[phase][position])
                new_index = int(new_orders[phase][position])
                if arm == "abrupt":
                    use_new = 2 * phase >= checked_config.transition_intervals
                    x = new_x_array[new_index] if use_new else old_x_array[old_index]
                    label = new_y_array[new_index] if use_new else old_y_array[old_index]
                    target = None
                elif arm == "output_interpolation":
                    x = new_x_array[new_index]
                    label = new_y_array[new_index]
                    target = output_interpolation(
                        int(old_y_array[new_index]),
                        int(new_y_array[new_index]),
                        alpha,
                        n_classes=checked_config.n_classes,
                    )
                else:
                    use_new = bool(masks[phase][position])
                    x = new_x_array[new_index] if use_new else old_x_array[old_index]
                    label = new_y_array[new_index] if use_new else old_y_array[old_index]
                    target = None
                key, step_key = jr.split(key)
                if target is None:
                    transaction, loss, post_loss = checked_hard_step(
                        params, state, x, label, step_key
                    )
                else:
                    transaction, loss, post_loss = checked_soft_step(
                        params, state, x, target, step_key
                    )
                if type(transaction) is not LearnerUpdateResult or not bool(
                    transaction.update_applied
                ):
                    raise ValueError("AdamW learner transaction became invalid")
                if not bool(jnp.isfinite(loss) & jnp.isfinite(post_loss)):
                    raise ValueError("training metric transaction became invalid")
                params, state = transaction.params, transaction.state
                training_loss += float(loss)
            phase_training_losses.append(training_loss)
            eval_correct = 0
            eval_loss = 0.0
            for position in range(checked_config.phase_examples):
                index = int(new_orders[phase][position])
                correct, loss = evaluate(params, new_x_array[index], new_y_array[index])
                if not bool(jnp.isfinite(loss)):
                    raise ValueError("evaluation metric transaction became invalid")
                eval_correct += int(correct)
                eval_loss += float(loss)
            phase_eval_correct.append(eval_correct)
            phase_eval_losses.append(eval_loss)
        for leaf in jax.tree_util.tree_leaves((params, state)):
            if hasattr(leaf, "block_until_ready"):
                leaf.block_until_ready()
        elapsed = time.perf_counter_ns() - started
        persistent = resources.dataset_bytes + resources.schedule_bytes + _numeric_tree_bytes(
            (initial_params, params, state)
        )
        if persistent > _MAX_RESOURCE_BYTES:
            raise ValueError("complete persistent numeric state exceeds 256 MiB")
        return phase_training_losses, phase_eval_correct, phase_eval_losses, persistent, elapsed

    outputs = tuple(run_arm(arm) for arm in _ARMS)
    result = GradualMicroPhaseResult(
        schema=_SCHEMA,
        arm_names=_ARMS,
        learner_name=learner_name,
        learner_hyperparameters=_ADAMW_IDENTITY,
        prng_implementation=_PRNG_IMPLEMENTATION,
        seed=resolved_seed,
        config=checked_config,
        phase_alpha_numerators=tuple(range(checked_config.phase_count)),
        phase_alpha_denominator=checked_config.transition_intervals,
        task_sampling_new_counts=np.asarray(new_counts, dtype=np.int32),
        dataset_rows=(int(old_x_array.shape[0]), int(new_x_array.shape[0])),
        dataset_sha256=(
            _array_digest(b"asi.ipmnist.gradual-family.old-dataset.v1\0", old_x_array, old_y_array),
            _array_digest(b"asi.ipmnist.gradual-family.new-dataset.v1\0", new_x_array, new_y_array),
        ),
        task_sampling_sha256=schedule_digest,
        source_sha256=_source_identity(),
        runtime_identity=_runtime_identity(),
        training_loss_sums=np.asarray([value[0] for value in outputs], dtype=np.float64),
        new_task_eval_correct_counts=np.asarray([value[1] for value in outputs], dtype=np.int32),
        new_task_eval_loss_sums=np.asarray([value[2] for value in outputs], dtype=np.float64),
        persistent_numeric_bytes=np.asarray([value[3] for value in outputs], dtype=np.int64),
        timing_ns=np.asarray([value[4] for value in outputs], dtype=np.int64),
        soft_target_updates_per_arm=(0, checked_config.updates, 0),
        observations_per_arm=2 * checked_config.updates,
        updates_per_arm=checked_config.updates,
        data_steps_per_arm=2 * checked_config.updates,
        environment_steps_per_arm=0,
        model_queries_per_arm=3 * checked_config.updates,
        parent_input_result_used=False,
        development_only=True,
        scientific_promotion_allowed=False,
        negative_results_must_be_retained=True,
        execution_attestation=False,
    )
    validate_gradual_micro_phase_result(result, old_x, old_y, new_x, new_y)
    return result


def validate_gradual_micro_phase_result(
    result: GradualMicroPhaseResult,
    old_x: object,
    old_y: object,
    new_x: object,
    new_y: object,
) -> None:
    """Revalidate an in-memory receipt against current data, source, and runtime."""
    if type(result) is not GradualMicroPhaseResult:
        raise ValueError("result must be an exact GradualMicroPhaseResult")
    # Reconstruction re-runs all field, source, runtime, counter, and array checks.
    checked = GradualMicroPhaseResult(**{
        field: getattr(result, field)
        for field in result.__dataclass_fields__
    })
    config = checked.config
    raw_old_x, raw_old_y, _ = _checked_dataset_metadata("old", old_x, old_y, config)
    raw_new_x, raw_new_y, _ = _checked_dataset_metadata("new", new_x, new_y, config)
    actual_rows = (int(raw_old_x.shape[0]), int(raw_new_x.shape[0]))
    if checked.dataset_rows != actual_rows:
        raise ValueError("dataset row identity mismatch")
    resources = _resource_estimate(config, actual_rows[0], actual_rows[1])
    if resources.working_set_bytes > _MAX_RESOURCE_BYTES:
        raise ValueError("aggregate working set exceeds 256 MiB")
    resolved_old_x, resolved_old_y = validated_ipmnist_data(
        raw_old_x, raw_old_y, input_dim=config.input_dim,
        n_classes=config.n_classes, min_length=config.phase_examples,
    )
    resolved_new_x, resolved_new_y = validated_ipmnist_data(
        raw_new_x, raw_new_y, input_dim=config.input_dim,
        n_classes=config.n_classes, min_length=config.phase_examples,
    )
    expected = (
        _array_digest(
            b"asi.ipmnist.gradual-family.old-dataset.v1\0", resolved_old_x, resolved_old_y
        ),
        _array_digest(
            b"asi.ipmnist.gradual-family.new-dataset.v1\0", resolved_new_x, resolved_new_y
        ),
    )
    if checked.dataset_sha256 != expected:
        raise ValueError("dataset content identity mismatch")
    if resolved_old_x.shape != resolved_new_x.shape or not np.array_equal(
        resolved_old_x, resolved_new_x
    ):
        raise ValueError(
            "output interpolation requires row-aligned identical inputs across tasks"
        )
    _, _, _, expected_counts, expected_schedule = _realized_schedule(
        checked.seed,
        config,
        resolved_old_x.shape[0],
        resolved_new_x.shape[0],
    )
    if not np.array_equal(
        checked.task_sampling_new_counts, np.asarray(expected_counts, dtype=np.int32)
    ):
        raise ValueError("realized task-sampling counts mismatch")
    if checked.task_sampling_sha256 != expected_schedule:
        raise ValueError("realized task-sampling schedule identity mismatch")
