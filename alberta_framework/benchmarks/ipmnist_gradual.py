"""Deterministic gradual-transition primitives for development IPMNIST lanes.

This module adapts the transition definitions in Liu and Mou,
``arXiv:2602.09234v2``.  It does not define or execute an evidence protocol.
The transition coefficient and task identity are evaluator-owned: learners see
only the resulting example/target, exactly as in the abrupt IPMNIST lane.
"""

from __future__ import annotations

import hashlib
import math
import operator
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, SupportsIndex, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework._seed_validation import require_jax_seed
from alberta_framework.benchmarks.upgd_ipmnist import (
    ADAMW_PROTOCOL_HYPERPARAMETERS,
    IPMNISTConfig,
    LearnerUpdateResult,
    _make_adamw_learner,
    build_schedule,
    cross_entropy_loss,
    init_mlp_params,
    validated_ipmnist_data,
)

TransitionMode = Literal[
    "abrupt", "input_interpolation", "output_interpolation", "task_sampling"
]
_MODES = frozenset(
    {"abrupt", "input_interpolation", "output_interpolation", "task_sampling"}
)
_INT32_MAX = 2**31 - 1
_MAX_ARRAY_ELEMENTS = 1_000_000
_MAX_RESOURCE_BYTES = 256 * 1024 * 1024
_MAX_RUN_STEPS = 1_000_000
_NUMPY_INTEGER_TYPES = (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.longlong,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.ulonglong,
)

GRADUAL_IPMNIST_PROTOCOL = MappingProxyType(
    {
        "schema": "asi.ipmnist.gradual-transition.protocol.v1",
        "paper_revision": "arXiv:2602.09234v2",
        "paper_revision_date": "2026-06-16",
        "adaptation_difference": (
            "single-pass ASI IPMNIST uses evaluator-owned per-example transitions; "
            "the paper iterates mini-batches within tasks without revealing boundaries"
        ),
        "matched_axes": ("seed", "updates", "observations", "example_order"),
        "learner_observes_transition_alpha": False,
        "learner_observes_task_boundary": False,
        "transaction_validity_required": True,
        "persistent_bytes_accounting_required": True,
        "environment_steps_accounting_required": True,
        "model_queries_accounting_required": True,
        "timing_is_telemetry_only": True,
        "development_only": True,
        "scientific_promotion_allowed": False,
    }
)


def _exact_index(name: str, value: object, *, minimum: int) -> int:
    if type(value) is not int and type(value) not in _NUMPY_INTEGER_TYPES:
        raise ValueError(f"{name} must be an integer")
    try:
        resolved = operator.index(cast(SupportsIndex, value))
    except Exception as error:
        raise ValueError(f"{name} must be an integer") from error
    if resolved < minimum or resolved > _INT32_MAX:
        raise ValueError(f"{name} must be in [{minimum}, {_INT32_MAX}]")
    return resolved


def _alpha(value: object) -> float:
    if type(value) is not float and type(value) is not int:
        raise ValueError("alpha must be a finite real number in [0, 1]")
    resolved = float(value)
    if not math.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
        raise ValueError("alpha must be a finite real number in [0, 1]")
    return resolved


@dataclass(frozen=True)
class GradualTransitionConfig:
    """One prospectively selectable transition rule.

    ``transition_steps`` counts interpolation intervals.  A width ``k`` uses
    coefficients ``0/k, 1/k, ..., k/k`` across ``k + 1`` update
    opportunities; the last is exactly the ordinary new-task example.
    Abrupt mode requires one step and is the mechanism-off reduction.
    """

    mode: TransitionMode
    transition_steps: int

    def __post_init__(self) -> None:
        if type(self.mode) is not str or self.mode not in _MODES:
            raise ValueError(f"mode must be one of {sorted(_MODES)}")
        resolved = _exact_index("transition_steps", self.transition_steps, minimum=1)
        if self.mode == "abrupt" and resolved != 1:
            raise ValueError("abrupt mode requires transition_steps=1")
        object.__setattr__(self, "transition_steps", resolved)


def transition_alpha(step: int, config: GradualTransitionConfig) -> float:
    """Return a deterministic uniform interpolation coefficient."""
    if type(config) is not GradualTransitionConfig:
        raise ValueError("config must be an exact GradualTransitionConfig")
    checked = GradualTransitionConfig(
        mode=config.mode, transition_steps=config.transition_steps
    )
    resolved_step = _exact_index("step", step, minimum=0)
    if checked.mode == "abrupt":
        return 1.0
    return min(resolved_step / checked.transition_steps, 1.0)


def input_interpolation_transaction(
    old: object, new: object, alpha: float
) -> tuple[Array, Array]:
    """Return a finite interpolation candidate and a traced validity bit."""
    resolved = _alpha(alpha)
    old_type = type(old)
    new_type = type(new)
    if not (
        old_type is np.ndarray or issubclass(old_type, (jax.Array, jax.core.Tracer))
    ) or not (
        new_type is np.ndarray or issubclass(new_type, (jax.Array, jax.core.Tracer))
    ):
        raise ValueError("old and new inputs must be exact NumPy or JAX arrays")
    old_array = jnp.asarray(old)
    new_array = jnp.asarray(new)
    if old_array.shape != new_array.shape:
        raise ValueError("old and new inputs must have identical shapes")
    if old_array.size < 1 or old_array.size > _MAX_ARRAY_ELEMENTS:
        raise ValueError("input size must be in [1, 1000000]")
    if old_array.dtype != new_array.dtype or not jnp.issubdtype(old_array.dtype, jnp.floating):
        raise ValueError("old and new inputs must share a floating dtype")
    candidate = (1.0 - resolved) * old_array + resolved * new_array
    valid = (
        jnp.all(jnp.isfinite(old_array))
        & jnp.all(jnp.isfinite(new_array))
        & jnp.all(jnp.isfinite(candidate))
    )
    safe = jnp.where(valid, candidate, jnp.zeros_like(candidate))
    return safe, valid


def input_interpolation(old: object, new: object, alpha: float) -> Array:
    """Apply paper equation ``x_alpha = (1-alpha)x_old + alpha*x_new``."""
    safe, valid = input_interpolation_transaction(old, new, alpha)
    if isinstance(valid, jax.core.Tracer):
        return jnp.where(valid, safe, jnp.full_like(safe, jnp.nan))
    if not bool(valid):
        raise ValueError("old and new inputs must produce only finite values")
    return safe


def output_interpolation(
    old_label: int, new_label: int, alpha: float, *, n_classes: int
) -> Array:
    """Interpolate old one-hot -> uniform -> new one-hot as paper section 4."""
    resolved_alpha = _alpha(alpha)
    classes = _exact_index("n_classes", n_classes, minimum=2)
    if classes > _MAX_ARRAY_ELEMENTS:
        raise ValueError("n_classes must be at most 1000000")
    old = _exact_index("old_label", old_label, minimum=0)
    new = _exact_index("new_label", new_label, minimum=0)
    if old >= classes or new >= classes:
        raise ValueError("labels must be smaller than n_classes")
    uniform = jnp.full((classes,), 1.0 / classes, dtype=jnp.float32)
    if resolved_alpha <= 0.5:
        old_target = jax.nn.one_hot(old, classes, dtype=jnp.float32)
        return (1.0 - 2.0 * resolved_alpha) * old_target + 2.0 * resolved_alpha * uniform
    new_target = jax.nn.one_hot(new, classes, dtype=jnp.float32)
    return (2.0 * resolved_alpha - 1.0) * new_target + (2.0 - 2.0 * resolved_alpha) * uniform


def task_sampling_mask(
    *, seed: int, transition_id: int, count: int, alpha: float
) -> np.ndarray:
    """Select exactly ``floor(alpha * count)`` positions from the new task.

    A Threefry root and transition fold make selection independent across
    transitions and reproducible across processes.  The mask is evaluator
    state, never learner-visible boundary information.
    """
    resolved_seed = require_jax_seed(seed, name="seed")
    resolved_transition = _exact_index("transition_id", transition_id, minimum=0)
    resolved_count = _exact_index("count", count, minimum=1)
    if resolved_count > _MAX_ARRAY_ELEMENTS:
        raise ValueError("count must be at most 1000000")
    new_count = math.floor(_alpha(alpha) * resolved_count)
    order = np.asarray(
        jax.device_get(
            jr.permutation(jr.fold_in(jr.key(resolved_seed), resolved_transition), resolved_count)
        )
    )
    mask = np.zeros(resolved_count, dtype=np.bool_)
    mask[order[:new_count]] = True
    return mask


@dataclass(frozen=True)
class GradualInputPairResult:
    """Host receipt for one matched abrupt/input-interpolation learner pair."""

    arm_names: tuple[str, str]
    learner_name: str
    seed: int
    config: IPMNISTConfig
    transition_steps: int
    schedule_sha256: str
    example_order_sha256: str
    correct_counts: np.ndarray
    loss_sums: np.ndarray
    persistent_numeric_bytes: np.ndarray
    timing_ns: np.ndarray
    observations: int
    updates: int
    data_steps: int
    environment_steps: int
    model_queries: int


def _array_sha256(domain: bytes, value: Array) -> str:
    raw = np.asarray(jax.device_get(value))
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(raw.dtype.str.encode("ascii"))
    digest.update(str(raw.shape).encode("ascii"))
    digest.update(raw.tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"


def _numeric_tree_bytes(value: object) -> int:
    total = 0
    for leaf in jax.tree_util.tree_leaves(value):
        actual_type = type(leaf)
        if not (
            actual_type is np.ndarray
            or issubclass(actual_type, (jax.Array, jax.core.Tracer))
        ):
            continue
        array = jnp.asarray(leaf)
        if not bool(jnp.all(jnp.isfinite(array))):
            raise ValueError("learner transaction produced non-finite state")
        total += int(array.size) * int(array.dtype.itemsize)
        if total > _MAX_RESOURCE_BYTES:
            raise ValueError("learner persistent numeric state exceeds 256 MiB")
    return total


def run_gradual_input_pair(
    data_x: object,
    data_y: object,
    *,
    learner_name: str,
    seed: int,
    config: IPMNISTConfig,
    transition_steps: int,
) -> GradualInputPairResult:
    """Run a matched abrupt/input-interpolation pair on one IPMNIST schedule.

    The two arms start from identical parameters, consume identical example,
    learner-RNG, update, and task-permutation schedules, and differ only in the
    evaluator-owned input presented during the first ``transition_steps`` of
    each task after the first.  The learner never observes task identity or
    interpolation coefficient.
    """
    if type(learner_name) is not str:
        raise ValueError("learner_name must be an exact string")
    if learner_name != "adamw_control":
        raise ValueError("this frozen adapter supports only adamw_control")
    if type(config) is not IPMNISTConfig:
        raise ValueError("config must be an exact IPMNISTConfig")
    # Reconstruct before reading derived fields: frozen dataclasses remain
    # forgeable through object.__setattr__.
    checked_config = IPMNISTConfig(**config.to_config())
    if checked_config.n_steps > _MAX_RUN_STEPS:
        raise ValueError("run horizon exceeds the 1000000-step ceiling")
    width = _exact_index("transition_steps", transition_steps, minimum=1)
    if width >= checked_config.task_length:
        raise ValueError("transition_steps must be smaller than task_length")
    resolved_seed = require_jax_seed(seed, name="seed")
    for name, value in (("data_x", data_x), ("data_y", data_y)):
        actual_type = type(value)
        if not (
            actual_type is np.ndarray
            or issubclass(actual_type, jax.Array)
        ):
            raise ValueError(f"{name} must be an exact NumPy or JAX array")
    raw_x = cast(np.ndarray | Array, data_x)
    raw_y = cast(np.ndarray | Array, data_y)
    if len(raw_x.shape) != 2 or len(raw_y.shape) != 1:
        raise ValueError("dataset metadata must describe one matrix and one vector")
    if raw_x.shape[0] != raw_y.shape[0]:
        raise ValueError("data_x and data_y row counts must match")
    dataset_elements = math.prod(raw_x.shape) + math.prod(raw_y.shape)
    # Materialization converts both arrays to four-byte protocol dtypes.
    if dataset_elements > _MAX_RESOURCE_BYTES // 4:
        raise ValueError("materialized dataset exceeds 256 MiB")
    schedule_elements = checked_config.n_tasks * (
        checked_config.input_dim + checked_config.task_length
    )
    # AdamW retains parameters plus first/second moments and one int32 count
    # for each of the six parameter leaves.  Dataset and full evaluator
    # schedule remain resident for the complete run.  Gate their exact sum
    # before any protocol materialization or device initialization.
    aggregate_persistent_bytes = (
        checked_config.parameter_count * 12
        + 6 * 4
        + dataset_elements * 4
        + schedule_elements * 4
    )
    if aggregate_persistent_bytes > _MAX_RESOURCE_BYTES:
        raise ValueError("aggregate persistent numeric allocation exceeds 256 MiB")
    resolved_x, resolved_y = validated_ipmnist_data(
        cast(np.ndarray | Array, data_x),
        cast(np.ndarray | Array, data_y),
        input_dim=checked_config.input_dim,
        n_classes=checked_config.n_classes,
        min_length=checked_config.task_length,
    )
    data_x_array = jnp.asarray(resolved_x, dtype=jnp.float32)
    data_y_array = jnp.asarray(resolved_y, dtype=jnp.int32)
    dataset_numeric_bytes = int(data_x_array.nbytes + data_y_array.nbytes)
    init_fn, step_fn = _make_adamw_learner(dict(ADAMW_PROTOCOL_HYPERPARAMETERS))

    @jax.jit
    def checked_step(
        params: dict[str, Array], state: Any, x: Array, y: Array, key: Array
    ) -> tuple[LearnerUpdateResult, Array, Array, Array]:
        (loss, logits), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
            params, x, y
        )
        transaction = step_fn(params, state, grads, key)
        if type(transaction) is not LearnerUpdateResult:
            raise TypeError("AdamW learner did not return its checked transaction")
        post_loss, _ = cross_entropy_loss(transaction.params, x, y)
        accuracy = (jnp.argmax(logits) == y).astype(jnp.float32)
        return transaction, accuracy, loss, post_loss

    root = jr.key(jnp.uint32(resolved_seed))
    key_init, key_schedule, key_noise = jr.split(root, 3)
    initial_params = init_mlp_params(key_init, checked_config)
    schedule = build_schedule(key_schedule, checked_config, int(data_x_array.shape[0]))
    transition_config = GradualTransitionConfig(
        mode="input_interpolation", transition_steps=width
    )

    def run_arm(gradual: bool) -> tuple[list[int], list[float], int, int]:
        params = initial_params
        state = init_fn(params)
        key = key_noise
        correct_counts: list[int] = []
        loss_sums: list[float] = []
        started = time.perf_counter_ns()
        for task in range(checked_config.n_tasks):
            current_permutation = schedule.permutations[task]
            previous_permutation = schedule.permutations[max(task - 1, 0)]
            examples = schedule.example_indices[task]
            task_correct = 0
            task_loss = 0.0
            for offset in range(checked_config.task_length):
                example = examples[offset]
                new_x = data_x_array[example][current_permutation]
                if gradual and task > 0 and offset < width:
                    old_x = data_x_array[example][previous_permutation]
                    x, valid = input_interpolation_transaction(
                        old_x, new_x, transition_alpha(offset, transition_config)
                    )
                    if not bool(valid):
                        raise ValueError("input interpolation transaction became invalid")
                else:
                    x = new_x
                y = data_y_array[example]
                key, step_key = jr.split(key)
                transaction, accuracy, loss, post_loss = checked_step(
                    params, state, x, y, step_key
                )
                if type(transaction) is not LearnerUpdateResult:
                    raise ValueError("AdamW learner did not return its checked transaction")
                if not bool(transaction.update_applied):
                    raise ValueError("AdamW learner update transaction became invalid")
                params = transaction.params
                state = transaction.state
                if not bool(jnp.isfinite(accuracy) & jnp.isfinite(loss) & jnp.isfinite(post_loss)):
                    raise ValueError("learner metric transaction became invalid")
                task_correct += int(accuracy)
                task_loss += float(loss)
                if not math.isfinite(task_loss):
                    raise ValueError("learner task-loss accumulation became invalid")
            correct_counts.append(task_correct)
            loss_sums.append(task_loss)
        for leaf in jax.tree_util.tree_leaves((params, state)):
            if hasattr(leaf, "block_until_ready"):
                leaf.block_until_ready()
        timing = time.perf_counter_ns() - started
        persistent = dataset_numeric_bytes + _numeric_tree_bytes((params, state, schedule))
        if persistent > _MAX_RESOURCE_BYTES:
            raise ValueError("complete persistent numeric state exceeds 256 MiB")
        return correct_counts, loss_sums, persistent, timing

    arm_outputs = (run_arm(False), run_arm(True))
    horizon = checked_config.n_steps
    return GradualInputPairResult(
        arm_names=("abrupt", "input_interpolation"),
        learner_name=learner_name,
        seed=resolved_seed,
        config=checked_config,
        transition_steps=width,
        schedule_sha256=_array_sha256(
            b"asi.ipmnist.gradual.permutations.v1\0", schedule.permutations
        ),
        example_order_sha256=_array_sha256(
            b"asi.ipmnist.gradual.example-order.v1\0", schedule.example_indices
        ),
        correct_counts=np.asarray([output[0] for output in arm_outputs], dtype=np.int32),
        loss_sums=np.asarray([output[1] for output in arm_outputs], dtype=np.float64),
        persistent_numeric_bytes=np.asarray(
            [output[2] for output in arm_outputs], dtype=np.int64
        ),
        timing_ns=np.asarray([output[3] for output in arm_outputs], dtype=np.int64),
        observations=horizon,
        updates=horizon,
        data_steps=horizon,
        environment_steps=0,
        model_queries=2 * horizon,
    )
