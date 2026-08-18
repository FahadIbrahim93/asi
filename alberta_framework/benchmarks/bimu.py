"""Strict development-only BiMU lane with a tiny end-to-end executable.

The 1000-task Permuted-MNIST configuration is represented exactly enough to
launch a separately authorized run, but pytest exercises only a five-task
synthetic slice.  A result from this module is permanently nonpromoting and is
never paper-comparable: the official dataset bytes/dataloader and the paper's
five-run aggregate are not bound here.

Equation and implementation provenance:

* Cottart et al., arXiv:2605.30198v1, equations 2, 6, and 7.
* Official CC-BY-4.0 implementation commit
  ``1b8a1a1fb892fbe89401390b3ff9611d7f3a5168``.

The official implementation supplies the experiment-only likelihood, KL, and
gradient scaling semantics that are not part of the unscaled main-text
equations.  This module keeps those scalars explicit and reports their values.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

PAPER_REVISION: Final = "arXiv:2605.30198v1"
OFFICIAL_CODE_URL: Final = (
    "https://github.com/kellian-cottart/active-continual-learning-bayesianbinn"
)
OFFICIAL_CODE_COMMIT: Final = "1b8a1a1fb892fbe89401390b3ff9611d7f3a5168"
RESULT_SCHEMA: Final = "asi.bimu.development_result.v1"
PROTOCOL_SCHEMA: Final = "asi.bimu.protocol.v2"

NONPROMOTING_POLICY: Final[dict[str, object]] = {
    "evidence_class": "development_comparator",
    "development_only": True,
    "scientific_promotion_allowed": False,
    "sota_claim_allowed": False,
}

_REMAINING_PAPER_GAPS: Final[tuple[str, ...]] = (
    "official MNIST bytes and dataloader order are not content-bound",
    "one result is one development seed, not the paper's five-run aggregate",
    "Concrete uniforms use float32-safe endpoint clipping instead of the official 1e-10 literal",
)

BIMU_PROTOCOL = MappingProxyType(
    {
        "schema": PROTOCOL_SCHEMA,
        "paper_revision": PAPER_REVISION,
        "paper_revision_date": "2026-05-28",
        "official_code_url": OFFICIAL_CODE_URL,
        "official_code_commit": OFFICIAL_CODE_COMMIT,
        "official_code_license": "CC-BY-4.0",
        "lane": "binary_bayesian_permuted_mnist",
        "weight_domain": (-1, 1),
        "paper_primary_metric": "mean_test_accuracy_over_last_5_tasks",
        "asi_metric": "whole_stream_pre_update_online_accuracy",
        "metrics_are_not_interchangeable": True,
        "learner_observes_task_boundary": False,
        "finite_kernel_preflight_required": True,
        "matched_axes": ("seed", "updates", "observations", "label_queries"),
        "persistent_bytes_accounting_required": True,
        "environment_steps_accounting_required": True,
        "model_queries_accounting_required": True,
        "timing_is_telemetry_only": True,
        "development_only": True,
        "scientific_promotion_allowed": False,
    }
)

_INIT_DOMAIN = 101
_TASK_PERMUTATION_DOMAIN = 211
_EXAMPLE_ORDER_DOMAIN = 307
_QUERY_DOMAIN = 401
_TRAIN_DOMAIN = 503
_TEST_DOMAIN = 601
_INT32_MAX = 2**31 - 1
_MAX_VECTOR_ELEMENTS = 1_000_000


def _exact_positive_int(value: object, name: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_float(
    value: object,
    name: str,
    *,
    positive: bool = False,
    lower: float | None = None,
    upper: float | None = None,
) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite float")
    resolved = value
    if positive and resolved <= 0.0:
        raise ValueError(f"{name} must be positive")
    if lower is not None and resolved < lower:
        raise ValueError(f"{name} must be >= {lower}")
    if upper is not None and resolved > upper:
        raise ValueError(f"{name} must be <= {upper}")
    return resolved


@dataclass(frozen=True)
class BiMUConfig:
    """One explicit BiMU stream configuration.

    The defaults are the official 1000-task/100-unit configuration.  The
    runner accepts smaller values for CI-cheap development slices without
    relabeling them as paper comparisons.
    """

    input_dim: int = 784
    hidden_units: int = 100
    n_classes: int = 10
    n_tasks: int = 1000
    train_examples_per_task: int = 60_000
    test_examples_per_task: int = 10_000
    train_samples: int = 5
    test_samples: int = 5
    query_samples: int = 5
    temperature: float = 1.0
    likelihood_multiplier: float = 161.3
    kl_multiplier: float = 3.76
    alpha_max: float = 0.0023
    memory_window: int | None = 700
    gradient_scale: float = 4.9
    query_threshold: float = 0.0

    def __post_init__(self) -> None:
        _exact_positive_int(self.input_dim, "input_dim")
        _exact_positive_int(self.hidden_units, "hidden_units")
        _exact_positive_int(self.n_classes, "n_classes", minimum=2)
        _exact_positive_int(self.n_tasks, "n_tasks", minimum=5)
        _exact_positive_int(self.train_examples_per_task, "train_examples_per_task")
        _exact_positive_int(self.test_examples_per_task, "test_examples_per_task")
        _exact_positive_int(self.train_samples, "train_samples")
        _exact_positive_int(self.test_samples, "test_samples")
        _exact_positive_int(self.query_samples, "query_samples", minimum=2)
        _finite_float(self.temperature, "temperature", positive=True)
        _finite_float(self.likelihood_multiplier, "likelihood_multiplier", positive=True)
        _finite_float(self.kl_multiplier, "kl_multiplier", positive=True)
        _finite_float(self.alpha_max, "alpha_max", positive=True)
        if self.memory_window is not None:
            _exact_positive_int(self.memory_window, "memory_window")
        _finite_float(self.gradient_scale, "gradient_scale", positive=True)
        _finite_float(self.query_threshold, "query_threshold", lower=0.0, upper=1.0)
        if self.trainable_scalar_count > _MAX_VECTOR_ELEMENTS:
            raise ValueError("BiMU trainable state exceeds the development allocation bound")
        if self.n_tasks * self.input_dim > _MAX_VECTOR_ELEMENTS:
            raise ValueError("BiMU task permutation schedule exceeds the allocation bound")
        if self.n_tasks * self.train_examples_per_task > _INT32_MAX:
            raise ValueError("BiMU observation horizon must fit in signed int32")

    @property
    def learner_observes_task_boundary(self) -> bool:
        return False

    @property
    def matches_paper_configuration(self) -> bool:
        return self == BIMU_PAPER_CONFIG

    @property
    def trainable_scalar_count(self) -> int:
        return self.input_dim * self.hidden_units + self.hidden_units * self.n_classes

    def to_protocol_payload(self) -> dict[str, object]:
        return {
            "schema": PROTOCOL_SCHEMA,
            "paper_revision": PAPER_REVISION,
            "official_code_url": OFFICIAL_CODE_URL,
            "official_code_commit": OFFICIAL_CODE_COMMIT,
            "dataset": "standardised Permuted-MNIST",
            "architecture": "bias-free binary Bayesian MLP with layer normalization",
            "activation": "reverse_binary_gate_width_1",
            "input_dim": self.input_dim,
            "hidden_units": self.hidden_units,
            "n_classes": self.n_classes,
            "n_tasks": self.n_tasks,
            "train_examples_per_task": self.train_examples_per_task,
            "test_examples_per_task": self.test_examples_per_task,
            "train_batch_size": 1,
            "test_batch_size": 500,
            "epochs_per_task": 1,
            "train_samples": self.train_samples,
            "test_samples": self.test_samples,
            "query_samples": self.query_samples,
            "temperature": self.temperature,
            "likelihood_multiplier": self.likelihood_multiplier,
            "kl_multiplier": self.kl_multiplier,
            "alpha_max": self.alpha_max,
            "memory_window": self.memory_window,
            "gradient_scale": self.gradient_scale,
            "query_threshold": self.query_threshold,
            "learner_observes_task_boundary": False,
            "matches_paper_configuration": self.matches_paper_configuration,
        }


BIMU_PAPER_CONFIG: Final = BiMUConfig()


@chex.dataclass(frozen=True)
class BiMUState:
    """Natural parameters for the two bias-free binary Bayesian layers."""

    input_hidden: Array
    hidden_output: Array


def _make_state(input_hidden: Array, hidden_output: Array) -> BiMUState:
    return BiMUState(  # type: ignore[call-arg]
        input_hidden=input_hidden,
        hidden_output=hidden_output,
    )


def _floating_array(value: object, *, name: str, ndim: int | None = None) -> Array:
    """Perform only trace-time-safe shape/dtype validation.

    Value-level finite checks belong at the host runner boundary.  Keeping a
    Python ``bool`` conversion out of this function fixes the original
    ``TracerBoolConversionError`` under :func:`jax.jit`.
    """
    actual_type = type(value)
    if not (actual_type is np.ndarray or issubclass(actual_type, (jax.Array, jax.core.Tracer))):
        raise ValueError(f"{name} must be an exact NumPy or JAX array")
    array = jnp.asarray(value)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    if (
        array.size < 1
        or array.size > _MAX_VECTOR_ELEMENTS
        or not jnp.issubdtype(array.dtype, jnp.floating)
    ):
        raise ValueError(f"{name} must be a non-empty floating array")
    valid = jnp.all(jnp.isfinite(array))
    if not isinstance(valid, jax.core.Tracer) and not bool(valid):
        raise ValueError(f"{name} must contain only finite values")
    return array


def posterior_probability_transaction(natural_parameter: object) -> tuple[Array, Array]:
    """Return a finite posterior and a caller-visible transaction validity bit."""
    state = _floating_array(natural_parameter, name="natural_parameter")
    logit = 2.0 * state
    result = jax.nn.sigmoid(logit)
    valid = (
        jnp.all(jnp.isfinite(state))
        & jnp.all(jnp.isfinite(logit))
        & jnp.all(jnp.isfinite(result))
    )
    return jnp.where(valid, result, jnp.full_like(result, 0.5)), valid


def posterior_probability(natural_parameter: object) -> Array:
    """Return ``P(weight=+1) = sigmoid(2 lambda)`` (paper equation 2)."""
    safe, valid = posterior_probability_transaction(natural_parameter)
    if isinstance(valid, jax.core.Tracer):
        return jnp.where(valid, safe, jnp.full_like(safe, jnp.nan))
    if not bool(valid):
        raise ValueError("posterior probability must be finite")
    return safe


def sample_binary_weights(natural_parameter: object, key: Array) -> Array:
    """Draw exact ``{-1,+1}`` Bernoulli weights from a natural parameter."""
    state = _floating_array(natural_parameter, name="natural_parameter")
    positive = jr.bernoulli(key, posterior_probability(state), shape=state.shape)
    return 2.0 * positive.astype(state.dtype) - 1.0


def concrete_binary_weights(
    natural_parameter: object, key: Array, *, temperature: float
) -> Array:
    """Draw the paper's differentiable binary Concrete/Gumbel relaxation."""
    state = _floating_array(natural_parameter, name="natural_parameter")
    epsilon = jnp.asarray(1e-7, dtype=state.dtype)
    uniform = jr.uniform(key, state.shape, dtype=state.dtype)
    uniform = jnp.clip(uniform, epsilon, 1.0 - epsilon)
    logistic_noise = jnp.log(uniform) - jnp.log1p(-uniform)
    return jnp.tanh((state + 0.5 * logistic_noise) / temperature)


def bimu_update_transaction(
    natural_parameter: object,
    loss_gradient: object,
    prior_natural_parameter: object,
    *,
    memory_window: int | None,
    alpha_max: float,
    likelihood_multiplier: float = 1.0,
    kl_multiplier: float = 1.0,
    gradient_scale: float = 1.0,
) -> tuple[Array, Array]:
    """Apply equations 6--7 plus the official experiment scaling semantics.

    ``memory_window=None`` is the mechanism-off reduction: only controlled
    relaxation toward the prior is removed.  Metaplastic step sizing remains.
    Scalar configuration validation is performed by :class:`BiMUConfig` so
    this numerical kernel remains JIT-compatible.
    """
    state = _floating_array(natural_parameter, name="natural_parameter")
    gradient = _floating_array(loss_gradient, name="loss_gradient")
    prior = _floating_array(prior_natural_parameter, name="prior_natural_parameter")
    if state.shape != gradient.shape or state.shape != prior.shape:
        raise ValueError("state, gradient, and prior must have identical shapes")
    _finite_float(alpha_max, "alpha_max", positive=True)
    _finite_float(likelihood_multiplier, "likelihood_multiplier", positive=True)
    _finite_float(kl_multiplier, "kl_multiplier", positive=True)
    _finite_float(gradient_scale, "gradient_scale", positive=True)
    if memory_window is not None:
        _exact_positive_int(memory_window, "memory_window")
        if memory_window > _INT32_MAX:
            raise ValueError("memory_window must fit in signed int32")
    scaled_gradient = likelihood_multiplier * gradient
    uncertainty = kl_multiplier * (1.0 - jnp.tanh(state) ** 2)
    reciprocal_eta = (
        uncertainty
        + 2.0 * jnp.tanh(state) * scaled_gradient
        + 1.0 / alpha_max
        + 2.0 * jnp.abs(scaled_gradient)
    )
    forgetting = (
        jnp.zeros_like(state)
        if memory_window is None
        else (state - prior) * uncertainty / memory_window
    )
    numerator = gradient_scale * scaled_gradient + forgetting
    candidate = state - numerator / reciprocal_eta
    source_valid = (
        jnp.all(jnp.isfinite(state))
        & jnp.all(jnp.isfinite(gradient))
        & jnp.all(jnp.isfinite(prior))
    )
    intermediate_valid = (
        jnp.all(jnp.isfinite(scaled_gradient))
        & jnp.all(jnp.isfinite(uncertainty))
        & jnp.all(jnp.isfinite(reciprocal_eta))
        & jnp.all(jnp.isfinite(forgetting))
        & jnp.all(jnp.isfinite(numerator))
    )
    valid = source_valid & intermediate_valid & jnp.all(jnp.isfinite(candidate))
    fallback = jnp.where(source_valid, state, jnp.zeros_like(state))
    return jnp.where(valid, candidate, fallback), valid


def bimu_update(
    natural_parameter: object,
    loss_gradient: object,
    prior_natural_parameter: object,
    *,
    memory_window: int | None,
    alpha_max: float,
    likelihood_multiplier: float = 1.0,
    kl_multiplier: float = 1.0,
    gradient_scale: float = 1.0,
) -> Array:
    """Compatibility wrapper; traced invalid transactions become explicit NaNs."""
    safe, valid = bimu_update_transaction(
        natural_parameter,
        loss_gradient,
        prior_natural_parameter,
        memory_window=memory_window,
        alpha_max=alpha_max,
        likelihood_multiplier=likelihood_multiplier,
        kl_multiplier=kl_multiplier,
        gradient_scale=gradient_scale,
    )
    if isinstance(valid, jax.core.Tracer):
        return jnp.where(valid, safe, jnp.full_like(safe, jnp.nan))
    if not bool(valid):
        raise ValueError("BiMU update must produce only finite values")
    return safe


@jax.custom_jvp
def _reverse_binary_gate(value: Array) -> Array:
    return (jnp.abs(value) > 1.0).astype(value.dtype)


@_reverse_binary_gate.defjvp
def _reverse_binary_gate_jvp(
    primals: tuple[Array], tangents: tuple[Array]
) -> tuple[Array, Array]:
    (value,), (tangent,) = primals, tangents
    output = _reverse_binary_gate(value)
    surrogate = (
        ((value > 0.5) & (value < 1.5)).astype(value.dtype)
        - ((value > -1.5) & (value < -0.5)).astype(value.dtype)
    )
    return output, tangent * surrogate


def _layer_normalize(value: Array) -> Array:
    mean = jnp.mean(value)
    variance = jnp.mean((value - mean) ** 2)
    return (value - mean) * jax.lax.rsqrt(variance + 1e-5)


def _forward(weights: BiMUState, features: Array) -> Array:
    hidden = _layer_normalize(weights.input_hidden @ features)
    hidden = _reverse_binary_gate(hidden)
    return _layer_normalize(weights.hidden_output @ hidden)


def _sample_state(state: BiMUState, key: Array, *, concrete: bool, temperature: float) -> BiMUState:
    first_key, second_key = jr.split(key)
    if concrete:
        return _make_state(
            input_hidden=concrete_binary_weights(
                state.input_hidden, first_key, temperature=temperature
            ),
            hidden_output=concrete_binary_weights(
                state.hidden_output, second_key, temperature=temperature
            ),
        )
    return _make_state(
        input_hidden=sample_binary_weights(state.input_hidden, first_key),
        hidden_output=sample_binary_weights(state.hidden_output, second_key),
    )


def _loss(weights: BiMUState, features: Array, label: Array) -> Array:
    return -jax.nn.log_softmax(_forward(weights, features))[label]


def _zero_state_like(state: BiMUState) -> BiMUState:
    return _make_state(
        input_hidden=jnp.zeros_like(state.input_hidden),
        hidden_output=jnp.zeros_like(state.hidden_output),
    )


def _add_state(left: BiMUState, right: BiMUState) -> BiMUState:
    return _make_state(
        input_hidden=left.input_hidden + right.input_hidden,
        hidden_output=left.hidden_output + right.hidden_output,
    )


@partial(jax.jit, static_argnames=("n_samples",))
def _concrete_mean_gradient(
    state: BiMUState,
    features: Array,
    label: Array,
    key: Array,
    *,
    temperature: float,
    n_samples: int,
) -> BiMUState:
    """Return the official Concrete gradient estimator averaged over samples."""
    total = _zero_state_like(state)
    sample_keys = jr.split(key, n_samples)
    for sample_key in sample_keys:
        concrete_state = _sample_state(
            state, sample_key, concrete=True, temperature=temperature
        )
        raw_gradient = jax.grad(_loss)(concrete_state, features, label)

        def correct(gradient: Array, natural: Array, relaxed: Array) -> Array:
            derivative_relaxed = 1.0 - relaxed * relaxed + 1e-7
            derivative_mean = 1.0 - jnp.tanh(natural) ** 2 + 1e-7
            return gradient * derivative_relaxed / (temperature * derivative_mean)

        corrected = _make_state(
            input_hidden=correct(
                raw_gradient.input_hidden, state.input_hidden, concrete_state.input_hidden
            ),
            hidden_output=correct(
                raw_gradient.hidden_output, state.hidden_output, concrete_state.hidden_output
            ),
        )
        total = _add_state(total, corrected)
    return _make_state(
        input_hidden=total.input_hidden / n_samples,
        hidden_output=total.hidden_output / n_samples,
    )


@partial(jax.jit, static_argnames=("n_samples",))
def _binary_predictions(
    state: BiMUState, features: Array, key: Array, *, n_samples: int
) -> Array:
    keys = jr.split(key, n_samples)
    logits = jnp.stack(
        [
            _forward(
                _sample_state(state, item, concrete=False, temperature=1.0), features
            )
            for item in keys
        ]
    )
    return jnp.argmax(logits, axis=1)


def _majority_prediction(predictions: Array, n_classes: int) -> tuple[int, float]:
    counts = np.bincount(np.asarray(predictions, dtype=np.int64), minlength=n_classes)
    prediction = int(np.argmax(counts))
    variation_ratio = 1.0 - float(counts[prediction]) / int(predictions.size)
    return prediction, variation_ratio


def _apply_gradient(state: BiMUState, gradient: BiMUState, config: BiMUConfig) -> BiMUState:
    prior = _zero_state_like(state)
    candidate = _make_state(
        input_hidden=bimu_update(
            state.input_hidden,
            gradient.input_hidden,
            prior.input_hidden,
            memory_window=config.memory_window,
            alpha_max=config.alpha_max,
            likelihood_multiplier=config.likelihood_multiplier,
            kl_multiplier=config.kl_multiplier,
            gradient_scale=config.gradient_scale,
        ),
        hidden_output=bimu_update(
            state.hidden_output,
            gradient.hidden_output,
            prior.hidden_output,
            memory_window=config.memory_window,
            alpha_max=config.alpha_max,
            likelihood_multiplier=config.likelihood_multiplier,
            kl_multiplier=config.kl_multiplier,
            gradient_scale=config.gradient_scale,
        ),
    )
    has_gradient = jnp.logical_or(
        jnp.any(jnp.abs(gradient.input_hidden) > 0.0),
        jnp.any(jnp.abs(gradient.hidden_output) > 0.0),
    )
    return _make_state(
        input_hidden=jnp.where(has_gradient, candidate.input_hidden, state.input_hidden),
        hidden_output=jnp.where(has_gradient, candidate.hidden_output, state.hidden_output),
    )


def _initialize_state(config: BiMUConfig, key: Array) -> BiMUState:
    first_key, second_key = jr.split(key)
    first_limit = 1.0 / math.sqrt(config.input_dim)
    second_limit = 1.0 / math.sqrt(config.hidden_units)
    return _make_state(
        input_hidden=jr.uniform(
            first_key,
            (config.hidden_units, config.input_dim),
            minval=-first_limit,
            maxval=first_limit,
            dtype=jnp.float32,
        ),
        hidden_output=jr.uniform(
            second_key,
            (config.n_classes, config.hidden_units),
            minval=-second_limit,
            maxval=second_limit,
            dtype=jnp.float32,
        ),
    )


def build_task_schedule(config: BiMUConfig, *, seed: int) -> tuple[tuple[int, ...], ...]:
    """Return environment-only feature permutations for every task."""
    _exact_positive_int(seed, "seed", minimum=0)
    root = jr.key(seed)
    return tuple(
        tuple(
            int(value)
            for value in np.asarray(
                jr.permutation(jr.fold_in(root, _TASK_PERMUTATION_DOMAIN + task), config.input_dim)
            )
        )
        for task in range(config.n_tasks)
    )


def _validated_dataset(
    features: object,
    labels: object,
    *,
    expected_examples: int,
    input_dim: int,
    n_classes: int,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    if type(features) is not np.ndarray or type(labels) is not np.ndarray:
        raise ValueError(f"{name} data must be exact NumPy arrays")
    resolved_features = np.asarray(features)
    resolved_labels = np.asarray(labels)
    if resolved_features.shape != (expected_examples, input_dim):
        raise ValueError(f"{name}_features has the wrong shape")
    if resolved_features.dtype.kind != "f" or not np.all(np.isfinite(resolved_features)):
        raise ValueError(f"{name}_features must contain finite floating values")
    if resolved_labels.shape != (expected_examples,) or resolved_labels.dtype.kind not in {
        "i",
        "u",
    }:
        raise ValueError(f"{name}_labels has the wrong shape or dtype")
    if np.any(resolved_labels < 0) or np.any(resolved_labels >= n_classes):
        raise ValueError(f"{name}_labels must be in the configured class range")
    return (
        np.asarray(resolved_features, dtype=np.float32),
        np.asarray(resolved_labels, dtype=np.int32),
    )


def _state_sha256(state: BiMUState) -> str:
    digest = hashlib.sha256()
    for name, value in (
        ("input_hidden", state.input_hidden),
        ("hidden_output", state.hidden_output),
    ):
        host = np.asarray(value, dtype=np.float32)
        digest.update(name.encode("ascii"))
        digest.update(str(host.shape).encode("ascii"))
        digest.update(host.tobytes(order="C"))
    return digest.hexdigest()


def _dataset_sha256(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for name, value, dtype in (
        ("train_features", train_features, np.dtype("<f4")),
        ("train_labels", train_labels, np.dtype("<i4")),
        ("test_features", test_features, np.dtype("<f4")),
        ("test_labels", test_labels, np.dtype("<i4")),
    ):
        host = np.asarray(value, dtype=dtype)
        digest.update(name.encode("ascii"))
        digest.update(str(host.shape).encode("ascii"))
        digest.update(host.tobytes(order="C"))
    return digest.hexdigest()


def _implementation_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _comparison_payload() -> dict[str, object]:
    return {
        "paper_comparable": False,
        "development_only": True,
        "remaining_paper_gaps": list(_REMAINING_PAPER_GAPS),
    }


def run_bimu_development(
    train_features: object,
    train_labels: object,
    test_features: object,
    test_labels: object,
    *,
    config: BiMUConfig = BIMU_PAPER_CONFIG,
    seed: int,
) -> dict[str, Any]:
    """Run one deterministic, task-boundary-private BiMU development life.

    The caller supplies already standardized base data.  Each task gets a
    deterministic input permutation and example order derived from the one
    Threefry root.  The learner receives only the permuted example and, when
    queried, its label; it never receives a task identifier or reset signal.
    """
    if type(config) is not BiMUConfig:
        raise ValueError("config must be a BiMUConfig")
    _exact_positive_int(seed, "seed", minimum=0)
    train_x, train_y = _validated_dataset(
        train_features,
        train_labels,
        expected_examples=config.train_examples_per_task,
        input_dim=config.input_dim,
        n_classes=config.n_classes,
        name="train",
    )
    test_x, test_y = _validated_dataset(
        test_features,
        test_labels,
        expected_examples=config.test_examples_per_task,
        input_dim=config.input_dim,
        n_classes=config.n_classes,
        name="test",
    )
    started = time.perf_counter()
    root = jr.key(seed)
    state = _initialize_state(config, jr.fold_in(root, _INIT_DOMAIN))
    initial_sha256 = _state_sha256(state)
    task_permutations = build_task_schedule(config, seed=seed)
    schedule_records: list[dict[str, object]] = []
    per_task_test_accuracy: list[float] = []
    online_correct = 0
    observations = 0
    label_queries = 0
    optimizer_updates = 0
    model_forward_queries = 0
    global_step = 0

    for task, permutation_tuple in enumerate(task_permutations):
        permutation = np.asarray(permutation_tuple, dtype=np.int32)
        order_key = jr.fold_in(root, _EXAMPLE_ORDER_DOMAIN + task)
        example_order = np.asarray(
            jr.permutation(order_key, config.train_examples_per_task), dtype=np.int32
        )
        task_queries: list[bool] = []
        for example_index in example_order:
            features = jnp.asarray(train_x[int(example_index), permutation], dtype=jnp.float32)
            label = int(train_y[int(example_index)])
            query_key = jr.fold_in(root, _QUERY_DOMAIN + global_step)
            predictions = _binary_predictions(
                state, features, query_key, n_samples=config.query_samples
            )
            prediction, variation_ratio = _majority_prediction(
                predictions, config.n_classes
            )
            queried = variation_ratio >= config.query_threshold
            task_queries.append(queried)
            online_correct += int(prediction == label)
            observations += 1
            model_forward_queries += config.query_samples
            if queried:
                gradient = _concrete_mean_gradient(
                    state,
                    features,
                    jnp.asarray(label, dtype=jnp.int32),
                    jr.fold_in(root, _TRAIN_DOMAIN + global_step),
                    temperature=config.temperature,
                    n_samples=config.train_samples,
                )
                state = _apply_gradient(state, gradient, config)
                if not all(
                    np.all(np.isfinite(np.asarray(value)))
                    for value in (state.input_hidden, state.hidden_output)
                ):
                    raise ValueError("BiMU state became non-finite")
                label_queries += 1
                optimizer_updates += 1
                model_forward_queries += config.train_samples
            global_step += 1

        task_correct = 0
        for test_index in range(config.test_examples_per_task):
            features = jnp.asarray(test_x[test_index, permutation], dtype=jnp.float32)
            predictions = _binary_predictions(
                state,
                features,
                jr.fold_in(
                    root,
                    _TEST_DOMAIN + task * config.test_examples_per_task + test_index,
                ),
                n_samples=config.test_samples,
            )
            prediction, _ = _majority_prediction(predictions, config.n_classes)
            task_correct += int(prediction == int(test_y[test_index]))
            model_forward_queries += config.test_samples
        per_task_test_accuracy.append(task_correct / config.test_examples_per_task)
        schedule_records.append(
            {
                "task": task,
                "permutation": list(permutation_tuple),
                "example_order": [int(value) for value in example_order],
                "query_decisions": task_queries,
            }
        )

    schedule_sha256 = hashlib.sha256(
        json.dumps(schedule_records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    final_sha256 = _state_sha256(state)
    parameter_bytes = config.trainable_scalar_count * np.dtype(np.float32).itemsize
    # The official optimizer persists ``step`` and ``seen`` in addition to the
    # natural parameters.  This lane represents them canonically as uint32;
    # the configured paper horizon fits without overflow.
    optimizer_state_bytes = 2 * np.dtype(np.uint32).itemsize
    persistent_bytes = parameter_bytes + optimizer_state_bytes
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": "complete",
        "seed": seed,
        "protocol": config.to_protocol_payload(),
        "evidence_policy": dict(NONPROMOTING_POLICY),
        "dataset_sha256": _dataset_sha256(train_x, train_y, test_x, test_y),
        "implementation_sha256": _implementation_sha256(),
        "schedule_sha256": schedule_sha256,
        "initial_state_sha256": initial_sha256,
        "final_state_sha256": final_sha256,
        "metrics": {
            "paper_late_five_test_accuracy": late_window_mean(
                per_task_test_accuracy, window=5
            ),
            "asi_whole_stream_online_accuracy": online_correct / observations,
            "per_task_test_accuracy": per_task_test_accuracy,
        },
        "counters": {
            "environment_steps": observations,
            "observations": observations,
            "label_queries": label_queries,
            "optimizer_updates": optimizer_updates,
            "model_forward_queries": model_forward_queries,
        },
        "resources": {
            "trainable_scalar_count": config.trainable_scalar_count,
            "parameter_numeric_bytes": parameter_bytes,
            "optimizer_state_numeric_bytes": optimizer_state_bytes,
            "initial_persistent_numeric_bytes": persistent_bytes,
            "final_persistent_numeric_bytes": persistent_bytes,
            "state_changed": initial_sha256 != final_sha256,
        },
        "timing": {
            "wall_clock_seconds": time.perf_counter() - started,
            "qualified": False,
            "role": "telemetry_only",
        },
        "comparison": _comparison_payload(),
    }
    validate_bimu_result(result)
    return result


def late_window_mean(
    task_accuracies: list[float] | tuple[float, ...], *, window: int = 5
) -> float:
    """Compute the paper's late-task metric, never whole-stream accuracy."""
    if type(window) is not int or window < 1 or window > _MAX_VECTOR_ELEMENTS:
        raise ValueError("window must be an integer in [1, 1000000]")
    if (
        (type(task_accuracies) is not list and type(task_accuracies) is not tuple)
        or len(task_accuracies) > _MAX_VECTOR_ELEMENTS
    ):
        raise ValueError("task_accuracies must be an exact bounded list or tuple")
    if any(type(value) is not int and type(value) is not float for value in task_accuracies):
        raise ValueError("task_accuracies must contain exact real numbers")
    values = np.asarray(task_accuracies)
    if values.ndim != 1 or values.size < window or values.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("task_accuracies must be a numeric vector at least window long")
    resolved = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(resolved)) or np.any((resolved < 0.0) | (resolved > 1.0)):
        raise ValueError("task_accuracies must be finite and in [0, 1]")
    return float(np.mean(resolved[-window:]))


_TOP_LEVEL_FIELDS: Final = {
    "schema",
    "status",
    "seed",
    "protocol",
    "evidence_policy",
    "dataset_sha256",
    "implementation_sha256",
    "schedule_sha256",
    "initial_state_sha256",
    "final_state_sha256",
    "metrics",
    "counters",
    "resources",
    "timing",
    "comparison",
}


def _exact_mapping(value: object, fields: set[str], name: str) -> Mapping[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{name} fields do not match the strict schema")
    return cast(Mapping[str, object], value)


def _payload_config(value: object) -> BiMUConfig:
    fields = set(BIMU_PAPER_CONFIG.to_protocol_payload())
    payload = _exact_mapping(value, fields, "protocol")
    if payload["schema"] != PROTOCOL_SCHEMA:
        raise ValueError("protocol schema drifted")
    if payload["paper_revision"] != PAPER_REVISION:
        raise ValueError("paper revision drifted")
    if payload["official_code_url"] != OFFICIAL_CODE_URL:
        raise ValueError("official code URL drifted")
    if payload["official_code_commit"] != OFFICIAL_CODE_COMMIT:
        raise ValueError("official code commit drifted")
    if payload["dataset"] != "standardised Permuted-MNIST":
        raise ValueError("dataset declaration drifted")
    if payload["architecture"] != "bias-free binary Bayesian MLP with layer normalization":
        raise ValueError("architecture declaration drifted")
    if payload["activation"] != "reverse_binary_gate_width_1":
        raise ValueError("activation declaration drifted")
    for field, expected in (
        ("train_batch_size", 1),
        ("test_batch_size", 500),
        ("epochs_per_task", 1),
    ):
        if payload[field] != expected or type(payload[field]) is not int:
            raise ValueError(f"{field} drifted")
    if payload["learner_observes_task_boundary"] is not False:
        raise ValueError("learner task-boundary policy drifted")
    memory_value = payload["memory_window"]
    memory_window = (
        None
        if memory_value is None
        else _exact_positive_int(memory_value, "memory_window")
    )
    config = BiMUConfig(
        input_dim=_exact_positive_int(payload["input_dim"], "input_dim"),
        hidden_units=_exact_positive_int(payload["hidden_units"], "hidden_units"),
        n_classes=_exact_positive_int(payload["n_classes"], "n_classes", minimum=2),
        n_tasks=_exact_positive_int(payload["n_tasks"], "n_tasks", minimum=5),
        train_examples_per_task=_exact_positive_int(
            payload["train_examples_per_task"], "train_examples_per_task"
        ),
        test_examples_per_task=_exact_positive_int(
            payload["test_examples_per_task"], "test_examples_per_task"
        ),
        train_samples=_exact_positive_int(payload["train_samples"], "train_samples"),
        test_samples=_exact_positive_int(payload["test_samples"], "test_samples"),
        query_samples=_exact_positive_int(
            payload["query_samples"], "query_samples", minimum=2
        ),
        temperature=_finite_float(payload["temperature"], "temperature", positive=True),
        likelihood_multiplier=_finite_float(
            payload["likelihood_multiplier"], "likelihood_multiplier", positive=True
        ),
        kl_multiplier=_finite_float(
            payload["kl_multiplier"], "kl_multiplier", positive=True
        ),
        alpha_max=_finite_float(payload["alpha_max"], "alpha_max", positive=True),
        memory_window=memory_window,
        gradient_scale=_finite_float(
            payload["gradient_scale"], "gradient_scale", positive=True
        ),
        query_threshold=_finite_float(
            payload["query_threshold"], "query_threshold", lower=0.0, upper=1.0
        ),
    )
    if dict(payload) != config.to_protocol_payload():
        raise ValueError("protocol payload is not canonical")
    return config


def _unit_interval(value: object, name: str) -> float:
    resolved = _finite_float(value, name, lower=0.0, upper=1.0)
    return resolved


def validate_bimu_result(value: object) -> None:
    """Fail closed on any result, accounting, provenance, or policy drift."""
    payload = _exact_mapping(value, _TOP_LEVEL_FIELDS, "result")
    if payload["schema"] != RESULT_SCHEMA or payload["status"] != "complete":
        raise ValueError("result identity or completion status drifted")
    seed = _exact_positive_int(payload["seed"], "seed", minimum=0)
    del seed
    config = _payload_config(payload["protocol"])
    if payload["evidence_policy"] != NONPROMOTING_POLICY:
        raise ValueError("evidence policy must remain permanently nonpromoting")
    for field in (
        "dataset_sha256",
        "implementation_sha256",
        "schedule_sha256",
        "initial_state_sha256",
        "final_state_sha256",
    ):
        digest = payload[field]
        if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    if payload["implementation_sha256"] != _implementation_sha256():
        raise ValueError("result was produced by different BiMU implementation bytes")

    metrics = _exact_mapping(
        payload["metrics"],
        {
            "paper_late_five_test_accuracy",
            "asi_whole_stream_online_accuracy",
            "per_task_test_accuracy",
        },
        "metrics",
    )
    raw_per_task = metrics["per_task_test_accuracy"]
    if type(raw_per_task) is not list or len(cast(list[object], raw_per_task)) != config.n_tasks:
        raise ValueError("per-task test accuracy length drifted")
    per_task = [
        _unit_interval(item, f"per_task_test_accuracy[{index}]")
        for index, item in enumerate(cast(list[object], raw_per_task))
    ]
    paper_metric = _unit_interval(
        metrics["paper_late_five_test_accuracy"], "paper_late_five_test_accuracy"
    )
    if paper_metric != late_window_mean(per_task, window=5):
        raise ValueError("paper late-five metric is not canonical")
    _unit_interval(
        metrics["asi_whole_stream_online_accuracy"], "asi_whole_stream_online_accuracy"
    )

    counters = _exact_mapping(
        payload["counters"],
        {
            "environment_steps",
            "observations",
            "label_queries",
            "optimizer_updates",
            "model_forward_queries",
        },
        "counters",
    )
    parsed_counters = {
        name: _exact_positive_int(counters[name], name, minimum=0) for name in counters
    }
    expected_steps = config.n_tasks * config.train_examples_per_task
    if parsed_counters["environment_steps"] != expected_steps:
        raise ValueError("environment-step count drifted")
    if parsed_counters["observations"] != expected_steps:
        raise ValueError("observation count drifted")
    if parsed_counters["optimizer_updates"] != parsed_counters["label_queries"]:
        raise ValueError("one-pass query/update accounting drifted")
    if not 0 <= parsed_counters["label_queries"] <= expected_steps:
        raise ValueError("label-query count is impossible")
    expected_forwards = (
        expected_steps * config.query_samples
        + parsed_counters["label_queries"] * config.train_samples
        + config.n_tasks * config.test_examples_per_task * config.test_samples
    )
    if parsed_counters["model_forward_queries"] != expected_forwards:
        raise ValueError("model-forward query count drifted")

    resources = _exact_mapping(
        payload["resources"],
        {
            "trainable_scalar_count",
            "parameter_numeric_bytes",
            "optimizer_state_numeric_bytes",
            "initial_persistent_numeric_bytes",
            "final_persistent_numeric_bytes",
            "state_changed",
        },
        "resources",
    )
    scalar_count = _exact_positive_int(
        resources["trainable_scalar_count"], "trainable_scalar_count"
    )
    if scalar_count != config.trainable_scalar_count:
        raise ValueError("trainable scalar accounting drifted")
    parameter_bytes = scalar_count * np.dtype(np.float32).itemsize
    if _exact_positive_int(resources["parameter_numeric_bytes"], "parameter_numeric_bytes") != (
        parameter_bytes
    ):
        raise ValueError("parameter byte accounting drifted")
    optimizer_state_bytes = 2 * np.dtype(np.uint32).itemsize
    if _exact_positive_int(
        resources["optimizer_state_numeric_bytes"], "optimizer_state_numeric_bytes"
    ) != optimizer_state_bytes:
        raise ValueError("optimizer-state byte accounting drifted")
    expected_bytes = parameter_bytes + optimizer_state_bytes
    for field in ("initial_persistent_numeric_bytes", "final_persistent_numeric_bytes"):
        if _exact_positive_int(resources[field], field) != expected_bytes:
            raise ValueError("persistent byte accounting drifted")
    if type(resources["state_changed"]) is not bool:
        raise ValueError("state_changed must be a bool")
    state_changed = payload["initial_state_sha256"] != payload["final_state_sha256"]
    if resources["state_changed"] is not state_changed:
        raise ValueError("state_changed contradicts the state digests")
    if parsed_counters["optimizer_updates"] == 0 and state_changed:
        raise ValueError("state changed despite zero optimizer updates")

    timing = _exact_mapping(
        payload["timing"], {"wall_clock_seconds", "qualified", "role"}, "timing"
    )
    _finite_float(timing["wall_clock_seconds"], "wall_clock_seconds", lower=0.0)
    if timing["qualified"] is not False or timing["role"] != "telemetry_only":
        raise ValueError("timing must remain unqualified telemetry")
    if payload["comparison"] != _comparison_payload():
        raise ValueError("paper-comparison fail-closed declaration drifted")
