# mypy: disable-error-code="call-arg"
"""Bounded recurrent sequence-model control with latent imagination.

This development mechanism closes a deliberately narrow gap between ASI's
one-step dreaming utilities and a Dreamer-family control loop.  It owns a
fixed-capacity transition replay, trains a recurrent latent dynamics model on
contiguous sequences, rolls that model forward under a learned categorical
actor, computes finite-horizon lambda returns, and updates learned actor and
value heads from those imagined trajectories.

It is not a reproduction of DreamerV3, WMAR, or Continual-Dreamer: there is no
stochastic categorical latent, pixel encoder/decoder, distributional critic,
or paper benchmark result.  The mechanism is L0, permanently nonpromoting,
and exists to make the missing causal path executable under exact counters and
bounded memory.  ``imagination_enabled=False`` is an exact mechanism-off
reduction: with matched authoritative transitions it leaves replay and model
updates byte-identical while performing no imagination, lambda-return, actor,
or value update.
"""

from __future__ import annotations

import dataclasses
import math
import operator
from collections.abc import Mapping
from numbers import Real
from typing import Any, NamedTuple, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

EVIDENCE_LEVEL = "L0"
SCIENTIFIC_PROMOTION_ALLOWED = False
PRNG_IMPLEMENTATION = "threefry2x32"
SCHEMA = "asi.dreamer_sequence_control.v1"
_INT32_MAX = 2**31 - 1
_MAX_STATE_NBYTES = 256 * 1024 * 1024
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_ACTUAL_INTS = frozenset(
    {
        int,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.longlong,
        np.ulonglong,
    }
)


def _int(name: str, value: object, minimum: int, maximum: int = _INT32_MAX) -> int:
    if type(value) not in _ACTUAL_INTS:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    result = operator.index(cast(SupportsIndex, value))
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return result


def _float(
    name: str,
    value: object,
    *,
    minimum: float,
    maximum: float,
    minimum_inclusive: bool = True,
) -> float:
    if type(value) is bool or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    lower = result >= minimum if minimum_inclusive else result > minimum
    narrowed = float(np.float32(result))
    if (
        not math.isfinite(result)
        or not lower
        or result > maximum
        or not math.isfinite(narrowed)
    ):
        raise ValueError(f"{name} lies outside its finite float32 bounds")
    return narrowed


def _copy_mapping(value: object, *, name: str) -> dict[str, Any]:
    if not issubclass(type(value), Mapping):
        raise ValueError(f"{name} must be a mapping")
    try:
        result = dict(cast(Mapping[str, Any], value))
    except Exception as error:
        raise ValueError(f"{name} must be a readable mapping") from error
    if any(type(key) is not str for key in result):
        raise ValueError(f"{name} keys must be exact strings")
    return result


@dataclasses.dataclass(frozen=True, slots=True)
class DreamerSequenceConfig:
    """Fixed replay, recurrent-model, imagination, and optimizer contract."""

    observation_dim: int
    n_actions: int
    latent_dim: int = 16
    replay_capacity: int = 256
    sequence_length: int = 8
    imagination_horizon: int = 5
    model_learning_rate: float = 1.0e-3
    actor_learning_rate: float = 3.0e-4
    value_learning_rate: float = 3.0e-4
    discount: float = 0.99
    lambda_: float = 0.95
    entropy_scale: float = 1.0e-3
    gradient_clip_norm: float = 10.0
    max_input_magnitude: float = 10_000.0
    imagination_enabled: bool = True
    max_transitions: int = _INT32_MAX // 64

    def __post_init__(self) -> None:
        for name in (
            "observation_dim",
            "n_actions",
            "latent_dim",
            "replay_capacity",
            "sequence_length",
            "imagination_horizon",
        ):
            object.__setattr__(self, name, _int(name, getattr(self, name), 1))
        object.__setattr__(
            self, "max_transitions", _int("max_transitions", self.max_transitions, 1)
        )
        if self.replay_capacity < 2:
            raise ValueError("replay_capacity must be at least 2")
        if self.sequence_length > self.replay_capacity:
            raise ValueError("sequence_length cannot exceed replay_capacity")
        if self.max_transitions > _INT32_MAX // self.imagination_horizon:
            raise ValueError("max_transitions times imagination_horizon must fit signed int32")
        for name in ("model_learning_rate", "actor_learning_rate", "value_learning_rate"):
            object.__setattr__(
                self,
                name,
                _float(name, getattr(self, name), minimum=0.0, maximum=1.0),
            )
        object.__setattr__(
            self,
            "discount",
            _float(
                "discount",
                self.discount,
                minimum=0.0,
                maximum=1.0,
                minimum_inclusive=False,
            ),
        )
        object.__setattr__(
            self, "lambda_", _float("lambda_", self.lambda_, minimum=0.0, maximum=1.0)
        )
        object.__setattr__(
            self,
            "entropy_scale",
            _float("entropy_scale", self.entropy_scale, minimum=0.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "gradient_clip_norm",
            _float(
                "gradient_clip_norm",
                self.gradient_clip_norm,
                minimum=0.0,
                maximum=_FLOAT32_MAX,
                minimum_inclusive=False,
            ),
        )
        object.__setattr__(
            self,
            "max_input_magnitude",
            _float(
                "max_input_magnitude",
                self.max_input_magnitude,
                minimum=0.0,
                maximum=_FLOAT32_MAX,
                minimum_inclusive=False,
            ),
        )
        if type(self.imagination_enabled) is not bool:
            raise ValueError("imagination_enabled must be an exact bool")
        # Reject the complete persistent allocation and a conservative
        # three-copy gradient/update envelope before any JAX array allocation.
        replay_slot_bytes = 8 * self.observation_dim + 17
        replay_bytes = self.replay_capacity * replay_slot_bytes + 12
        model_scalars = (
            self.latent_dim * self.observation_dim
            + self.latent_dim
            + self.latent_dim * self.latent_dim
            + self.latent_dim * self.n_actions
            + self.latent_dim
            + self.observation_dim * self.latent_dim
            + self.observation_dim
            + 2 * self.latent_dim
            + 2
        )
        actor_value_scalars = self.n_actions * self.latent_dim + self.n_actions
        actor_value_scalars += self.latent_dim + 1
        model_bytes = 4 * model_scalars
        actor_value_bytes = 4 * actor_value_scalars
        parameter_bytes = model_bytes + actor_value_bytes
        fixed_bytes = 3 * 8 + 9 * 4
        persistent_bytes = parameter_bytes + replay_bytes + fixed_bytes
        sequence_bytes = 4 * (
            (self.sequence_length + 1) * self.observation_dim
            + 4 * self.sequence_length
        ) + 1
        trace_bytes = self.imagination_horizon * (4 * self.latent_dim + 20)
        sequence_training_trace_bytes = 4 * self.sequence_length * (
            self.latent_dim + self.observation_dim + 4
        )
        imagination_gradient_trace_bytes = self.imagination_horizon * (
            4 * self.n_actions + 32
        )
        # Keep this formula identical to ``resource_budget``.  Construction is
        # the preallocation gate, so no configuration may defer discovering an
        # unrepresentable update envelope until after its replay is allocated.
        update_envelope = (
            2 * persistent_bytes
            + model_bytes
            + sequence_bytes
            + sequence_training_trace_bytes
            + trace_bytes
            + imagination_gradient_trace_bytes
            + 2 * actor_value_bytes
        )
        if persistent_bytes > _MAX_STATE_NBYTES:
            raise ValueError("persistent state exceeds the 256 MiB development bound")
        if update_envelope > _INT32_MAX:
            raise ValueError("update working-set byte count must fit signed int32")

    def to_config(self) -> dict[str, object]:
        return {"schema": SCHEMA, **dataclasses.asdict(self)}

    @classmethod
    def from_config(cls, value: object) -> DreamerSequenceConfig:
        values = _copy_mapping(value, name="DreamerSequenceConfig")
        if set(values) != {"schema", *(field.name for field in dataclasses.fields(cls))}:
            raise ValueError("DreamerSequenceConfig fields differ from the schema")
        if values.pop("schema") != SCHEMA:
            raise ValueError("DreamerSequenceConfig schema mismatch")
        return cls(**values)


class DreamerModelParameters(NamedTuple):
    encoder_kernel: Array
    encoder_bias: Array
    recurrent_kernel: Array
    action_kernel: Array
    recurrent_bias: Array
    observation_head: Array
    observation_bias: Array
    reward_head: Array
    reward_bias: Array
    continuation_head: Array
    continuation_bias: Array


class DreamerActorParameters(NamedTuple):
    kernel: Array
    bias: Array


class DreamerValueParameters(NamedTuple):
    kernel: Array
    bias: Array


@chex.dataclass(frozen=True)
class DreamerReplayState:
    observations: Array
    actions: Array
    rewards: Array
    discounts: Array
    next_observations: Array
    terminated: Array
    insertion_ids: Array
    index: Array
    size: Array
    next_insertion_id: Array


@chex.dataclass(frozen=True)
class DreamerSequenceState:
    model_parameters: DreamerModelParameters
    actor_parameters: DreamerActorParameters
    value_parameters: DreamerValueParameters
    replay: DreamerReplayState
    action_key: Array
    replay_key: Array
    imagination_key: Array
    real_transition_count: Array
    replay_insert_count: Array
    sequence_sample_count: Array
    world_model_update_count: Array
    imagination_rollout_count: Array
    imagination_query_count: Array
    lambda_return_target_count: Array
    actor_update_count: Array
    value_update_count: Array


@chex.dataclass(frozen=True)
class DreamerDecision:
    owner_transition_count: Array
    owner_action_key: Array
    observation: Array
    latent: Array
    action: Array
    action_probability: Array
    next_action_key: Array
    valid: Array


@chex.dataclass(frozen=True)
class DreamerTransition:
    observation: Array
    action: Array
    reward: Array
    discount: Array
    next_observation: Array
    terminated: Array


@chex.dataclass(frozen=True)
class DreamerSequenceSample:
    observations: Array
    actions: Array
    rewards: Array
    discounts: Array
    insertion_ids: Array
    valid: Array


@chex.dataclass(frozen=True)
class DreamerLearnResult:
    state: DreamerSequenceState
    model_loss: Array
    actor_loss: Array
    value_loss: Array
    lambda_returns: Array
    sequence_sampled: Array
    imagination_applied: Array
    applied: Array


@dataclasses.dataclass(frozen=True, slots=True)
class DreamerResourceBudget:
    persistent_scalars: int
    persistent_bytes: int
    replay_scalars: int
    replay_bytes: int
    sequence_sample_bytes: int
    imagination_trace_bytes: int
    update_working_set_bytes: int


def _tree_accounting(value: object) -> tuple[int, int]:
    leaves = tuple(jnp.asarray(leaf) for leaf in jax.tree_util.tree_leaves(value))
    return (
        sum(int(leaf.size) for leaf in leaves),
        sum(int(leaf.size * leaf.dtype.itemsize) for leaf in leaves),
    )


def _tree_finite(value: object) -> bool:
    return all(
        not jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.inexact)
        or bool(jnp.all(jnp.isfinite(jnp.asarray(leaf))))
        for leaf in jax.tree_util.tree_leaves(value)
    )


def _tree_equal(left: object, right: object) -> bool:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    return cast(Any, left_tree) == right_tree and all(
        bool(jnp.array_equal(left_leaf, right_leaf))
        for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True)
    )


def _threefry_key(value: object) -> bool:
    try:
        key = jnp.asarray(value)
        return (
            key.shape == ()
            and jr.key_data(key).shape == (2,)
            and str(jr.key_impl(key)) == PRNG_IMPLEMENTATION
        )
    except (TypeError, ValueError):
        return False


def _array_matches(value: object, shape: tuple[int, ...], dtype: Any) -> bool:
    if not isinstance(value, Array):
        return False
    try:
        array = jnp.asarray(value)
    except (TypeError, ValueError):
        return False
    return array.shape == shape and array.dtype == jnp.dtype(dtype)


def _increment(value: Array, maximum: int) -> Array:
    return jnp.minimum(value, jnp.asarray(maximum - 1, dtype=jnp.int32)) + 1


def _clipped_step(parameters: Any, gradients: Any, learning_rate: float, clip: float) -> Any:
    squared = sum(
        jnp.sum(jnp.square(jnp.asarray(leaf, dtype=jnp.float32)))
        for leaf in jax.tree_util.tree_leaves(gradients)
    )
    norm = jnp.sqrt(squared)
    scale = jnp.minimum(1.0, jnp.asarray(clip, dtype=jnp.float32) / jnp.maximum(norm, 1e-12))
    return jax.tree_util.tree_map(
        lambda parameter, gradient: parameter
        - jnp.asarray(learning_rate, dtype=jnp.float32) * scale * gradient,
        parameters,
        gradients,
    )


def lambda_returns(
    rewards: Array,
    discounts: Array,
    next_values: Array,
    bootstrap: Array,
    lambda_: float,
) -> Array:
    """Compute finite-horizon reverse-view lambda returns exactly once."""
    reward_array = jnp.asarray(rewards, dtype=jnp.float32)
    discount_array = jnp.asarray(discounts, dtype=jnp.float32)
    next_value_array = jnp.asarray(next_values, dtype=jnp.float32)
    if (
        reward_array.ndim != 1
        or discount_array.shape != reward_array.shape
        or next_value_array.shape != reward_array.shape
        or jnp.asarray(bootstrap).shape != ()
    ):
        raise ValueError("lambda-return inputs must be matching vectors and a scalar bootstrap")
    coefficient = _float("lambda_", lambda_, minimum=0.0, maximum=1.0)
    if (
        not bool(jnp.all(jnp.isfinite(reward_array)))
        or not bool(jnp.all(jnp.isfinite(discount_array)))
        or not bool(jnp.all(jnp.isfinite(next_value_array)))
        or not bool(jnp.isfinite(jnp.asarray(bootstrap)))
        or not bool(jnp.all((discount_array >= 0.0) & (discount_array <= 1.0)))
    ):
        raise ValueError("lambda-return inputs must be finite with discounts in [0, 1]")

    def step(carry: Array, inputs: tuple[Array, Array, Array]) -> tuple[Array, Array]:
        reward, discount, next_value = inputs
        target = reward + discount * (
            (1.0 - coefficient) * next_value + coefficient * carry
        )
        return target, target

    _, reverse = jax.lax.scan(
        step,
        jnp.asarray(bootstrap, dtype=jnp.float32),
        (reward_array[::-1], discount_array[::-1], next_value_array[::-1]),
    )
    return reverse[::-1]


class DreamerSequenceControl:
    """One immutable recurrent replay/imagination/control aggregate."""

    def __init__(self, config: DreamerSequenceConfig):
        if type(config) is not DreamerSequenceConfig:
            raise TypeError("config must be an exact DreamerSequenceConfig")
        self._config = config

    @property
    def config(self) -> DreamerSequenceConfig:
        return self._config

    def to_config(self) -> dict[str, object]:
        return self._config.to_config()

    @classmethod
    def from_config(cls, value: object) -> DreamerSequenceControl:
        return cls(DreamerSequenceConfig.from_config(value))

    def init(self, key: Array) -> DreamerSequenceState:
        key_array = jnp.asarray(key)
        try:
            key_data = jr.key_data(key_array)
            key_implementation = str(jr.key_impl(key_array))
        except (TypeError, ValueError) as error:
            raise ValueError("key must be an explicit scalar JAX PRNG key") from error
        if (
            key_array.shape != ()
            or key_data.shape != (2,)
            or key_implementation != PRNG_IMPLEMENTATION
        ):
            raise ValueError("key must be an explicit scalar JAX PRNG key")
        cfg = self._config
        keys = jr.split(key_array, 10)

        def normal(current: Array, shape: tuple[int, ...], fan_in: int) -> Array:
            return jr.normal(current, shape, dtype=jnp.float32) / jnp.sqrt(float(fan_in))

        model = DreamerModelParameters(
            encoder_kernel=normal(
                keys[0], (cfg.latent_dim, cfg.observation_dim), cfg.observation_dim
            ),
            encoder_bias=jnp.zeros((cfg.latent_dim,), dtype=jnp.float32),
            recurrent_kernel=normal(keys[1], (cfg.latent_dim, cfg.latent_dim), cfg.latent_dim),
            action_kernel=normal(keys[2], (cfg.latent_dim, cfg.n_actions), cfg.n_actions),
            recurrent_bias=jnp.zeros((cfg.latent_dim,), dtype=jnp.float32),
            observation_head=normal(keys[3], (cfg.observation_dim, cfg.latent_dim), cfg.latent_dim),
            observation_bias=jnp.zeros((cfg.observation_dim,), dtype=jnp.float32),
            reward_head=normal(keys[4], (cfg.latent_dim,), cfg.latent_dim),
            reward_bias=jnp.asarray(0.0, dtype=jnp.float32),
            continuation_head=normal(keys[5], (cfg.latent_dim,), cfg.latent_dim),
            continuation_bias=jnp.asarray(0.0, dtype=jnp.float32),
        )
        actor = DreamerActorParameters(
            kernel=normal(keys[6], (cfg.n_actions, cfg.latent_dim), cfg.latent_dim),
            bias=jnp.zeros((cfg.n_actions,), dtype=jnp.float32),
        )
        value = DreamerValueParameters(
            kernel=normal(keys[7], (cfg.latent_dim,), cfg.latent_dim),
            bias=jnp.asarray(0.0, dtype=jnp.float32),
        )
        replay = DreamerReplayState(
            observations=jnp.zeros(
                (cfg.replay_capacity, cfg.observation_dim), dtype=jnp.float32
            ),
            actions=jnp.zeros((cfg.replay_capacity,), dtype=jnp.int32),
            rewards=jnp.zeros((cfg.replay_capacity,), dtype=jnp.float32),
            discounts=jnp.zeros((cfg.replay_capacity,), dtype=jnp.float32),
            next_observations=jnp.zeros(
                (cfg.replay_capacity, cfg.observation_dim), dtype=jnp.float32
            ),
            terminated=jnp.zeros((cfg.replay_capacity,), dtype=jnp.bool_),
            insertion_ids=jnp.full((cfg.replay_capacity,), -1, dtype=jnp.int32),
            index=jnp.asarray(0, dtype=jnp.int32),
            size=jnp.asarray(0, dtype=jnp.int32),
            next_insertion_id=jnp.asarray(0, dtype=jnp.int32),
        )
        zero = jnp.asarray(0, dtype=jnp.int32)
        return DreamerSequenceState(
            model_parameters=model,
            actor_parameters=actor,
            value_parameters=value,
            replay=replay,
            action_key=keys[8],
            replay_key=jr.fold_in(keys[9], 0),
            imagination_key=jr.fold_in(keys[9], 1),
            real_transition_count=zero,
            replay_insert_count=zero,
            sequence_sample_count=zero,
            world_model_update_count=zero,
            imagination_rollout_count=zero,
            imagination_query_count=zero,
            lambda_return_target_count=zero,
            actor_update_count=zero,
            value_update_count=zero,
        )

    def _encode(self, parameters: DreamerModelParameters, observation: Array) -> Array:
        return jnp.tanh(parameters.encoder_kernel @ observation + parameters.encoder_bias)

    def _model_step(
        self, parameters: DreamerModelParameters, latent: Array, action: Array
    ) -> tuple[Array, Array, Array, Array]:
        one_hot = jax.nn.one_hot(action, self._config.n_actions, dtype=jnp.float32)
        next_latent = jnp.tanh(
            parameters.recurrent_kernel @ latent
            + parameters.action_kernel @ one_hot
            + parameters.recurrent_bias
        )
        observation = parameters.observation_head @ next_latent + parameters.observation_bias
        reward = jnp.dot(parameters.reward_head, next_latent) + parameters.reward_bias
        continuation = jax.nn.sigmoid(
            jnp.dot(parameters.continuation_head, next_latent)
            + parameters.continuation_bias
        )
        return next_latent, observation, reward, continuation

    def _policy(self, parameters: DreamerActorParameters, latent: Array) -> Array:
        return jax.nn.softmax(parameters.kernel @ latent + parameters.bias)

    def _value(self, parameters: DreamerValueParameters, latent: Array) -> Array:
        return jnp.dot(parameters.kernel, latent) + parameters.bias

    def _replay_static_valid(self, replay: object) -> bool:
        if type(replay) is not DreamerReplayState:
            return False
        cfg = self._config
        specifications = (
            (replay.observations, (cfg.replay_capacity, cfg.observation_dim), jnp.float32),
            (replay.actions, (cfg.replay_capacity,), jnp.int32),
            (replay.rewards, (cfg.replay_capacity,), jnp.float32),
            (replay.discounts, (cfg.replay_capacity,), jnp.float32),
            (
                replay.next_observations,
                (cfg.replay_capacity, cfg.observation_dim),
                jnp.float32,
            ),
            (replay.terminated, (cfg.replay_capacity,), jnp.bool_),
            (replay.insertion_ids, (cfg.replay_capacity,), jnp.int32),
            (replay.index, (), jnp.int32),
            (replay.size, (), jnp.int32),
            (replay.next_insertion_id, (), jnp.int32),
        )
        return all(_array_matches(value, shape, dtype) for value, shape, dtype in specifications)

    def _replay_semantic_valid(self, replay: DreamerReplayState) -> bool:
        """Validate the complete ring layout and every occupied transition."""
        cfg = self._config
        index = int(replay.index)
        size = int(replay.size)
        next_id = int(replay.next_insertion_id)
        if (
            not 0 <= index < cfg.replay_capacity
            or not 0 <= size <= cfg.replay_capacity
            or not 0 <= next_id <= cfg.max_transitions
            or index != next_id % cfg.replay_capacity
            or size != min(next_id, cfg.replay_capacity)
        ):
            return False
        slots = jnp.arange(cfg.replay_capacity, dtype=jnp.int32)
        newest = jnp.asarray(next_id - 1, dtype=jnp.int32)
        last_for_slot = newest - jnp.mod(newest - slots, cfg.replay_capacity)
        occupied = (last_for_slot >= next_id - size) & (last_for_slot >= 0)
        expected_ids = jnp.where(occupied, last_for_slot, -1).astype(jnp.int32)
        bounded_observations = jnp.all(
            jnp.abs(replay.observations) <= cfg.max_input_magnitude, axis=1
        ) & jnp.all(jnp.abs(replay.next_observations) <= cfg.max_input_magnitude, axis=1)
        transition_valid = (
            (replay.actions >= 0)
            & (replay.actions < cfg.n_actions)
            & jnp.isfinite(replay.rewards)
            & (jnp.abs(replay.rewards) <= cfg.max_input_magnitude)
            & jnp.isfinite(replay.discounts)
            & (replay.discounts >= 0.0)
            & (replay.discounts <= 1.0)
            & ((replay.discounts == 0.0) == replay.terminated)
            & bounded_observations
        )
        return bool(
            jnp.array_equal(replay.insertion_ids, expected_ids)
            & jnp.all(jnp.isfinite(replay.observations))
            & jnp.all(jnp.isfinite(replay.next_observations))
            & jnp.all(~occupied | transition_valid)
        )

    def _state_static_valid(self, state: object) -> bool:
        if type(state) is not DreamerSequenceState or not self._replay_static_valid(state.replay):
            return False
        cfg = self._config
        model = state.model_parameters
        actor = state.actor_parameters
        value = state.value_parameters
        if (
            type(model) is not DreamerModelParameters
            or type(actor) is not DreamerActorParameters
            or type(value) is not DreamerValueParameters
        ):
            return False
        specifications = (
            (model.encoder_kernel, (cfg.latent_dim, cfg.observation_dim), jnp.float32),
            (model.encoder_bias, (cfg.latent_dim,), jnp.float32),
            (model.recurrent_kernel, (cfg.latent_dim, cfg.latent_dim), jnp.float32),
            (model.action_kernel, (cfg.latent_dim, cfg.n_actions), jnp.float32),
            (model.recurrent_bias, (cfg.latent_dim,), jnp.float32),
            (model.observation_head, (cfg.observation_dim, cfg.latent_dim), jnp.float32),
            (model.observation_bias, (cfg.observation_dim,), jnp.float32),
            (model.reward_head, (cfg.latent_dim,), jnp.float32),
            (model.reward_bias, (), jnp.float32),
            (model.continuation_head, (cfg.latent_dim,), jnp.float32),
            (model.continuation_bias, (), jnp.float32),
            (actor.kernel, (cfg.n_actions, cfg.latent_dim), jnp.float32),
            (actor.bias, (cfg.n_actions,), jnp.float32),
            (value.kernel, (cfg.latent_dim,), jnp.float32),
            (value.bias, (), jnp.float32),
        )
        counters = (
            state.real_transition_count,
            state.replay_insert_count,
            state.sequence_sample_count,
            state.world_model_update_count,
            state.imagination_rollout_count,
            state.imagination_query_count,
            state.lambda_return_target_count,
            state.actor_update_count,
            state.value_update_count,
        )
        static_valid = (
            all(_array_matches(item, shape, dtype) for item, shape, dtype in specifications)
            and all(_array_matches(counter, (), jnp.int32) for counter in counters)
            and _threefry_key(state.action_key)
            and _threefry_key(state.replay_key)
            and _threefry_key(state.imagination_key)
        )
        if not static_valid or not self._replay_semantic_valid(state.replay):
            return False
        counters_host = tuple(int(counter) for counter in counters)
        (
            real,
            inserts,
            samples,
            model_updates,
            rollouts,
            queries,
            targets,
            actor_updates,
            value_updates,
        ) = counters_host
        return (
            all(counter >= 0 for counter in counters_host)
            and real == inserts == int(state.replay.next_insertion_id)
            and real <= cfg.max_transitions
            and samples == model_updates
            and samples <= real
            and rollouts == actor_updates == value_updates
            and rollouts <= samples
            and queries == targets == rollouts * cfg.imagination_horizon
            and (cfg.imagination_enabled or rollouts == 0)
        )

    def policy(self, state: DreamerSequenceState, observation: Array) -> Array:
        """Return the learned categorical policy for one bounded observation."""
        if not self._state_static_valid(state):
            raise ValueError("state differs from the configured static contract")
        obs = jnp.asarray(observation)
        if (
            obs.shape != (self._config.observation_dim,)
            or obs.dtype != jnp.float32
            or not bool(jnp.all(jnp.isfinite(obs)))
            or not bool(jnp.all(jnp.abs(obs) <= self._config.max_input_magnitude))
            or not _tree_finite(state)
        ):
            raise ValueError("policy observation or state is invalid")
        probabilities = self._policy(
            state.actor_parameters, self._encode(state.model_parameters, obs)
        )
        if not bool(jnp.all(jnp.isfinite(probabilities))):
            raise ValueError("policy produced non-finite probabilities")
        return probabilities

    def decide(
        self,
        state: DreamerSequenceState,
        observation: Array,
        *,
        action: Array | None = None,
    ) -> DreamerDecision:
        if not self._state_static_valid(state):
            raise ValueError("state differs from the configured static contract")
        cfg = self._config
        obs = jnp.asarray(observation)
        valid = (
            obs.shape == (cfg.observation_dim,)
            and obs.dtype == jnp.float32
            and bool(jnp.all(jnp.isfinite(obs)))
            and bool(jnp.all(jnp.abs(obs) <= cfg.max_input_magnitude))
            and _tree_finite(state)
        )
        if not valid:
            return DreamerDecision(
                owner_transition_count=jnp.asarray(0, dtype=jnp.int32),
                owner_action_key=state.action_key,
                observation=jnp.zeros((cfg.observation_dim,), dtype=jnp.float32),
                latent=jnp.zeros((cfg.latent_dim,), dtype=jnp.float32),
                action=jnp.asarray(0, dtype=jnp.int32),
                action_probability=jnp.asarray(0.0, dtype=jnp.float32),
                next_action_key=state.action_key,
                valid=jnp.asarray(False, dtype=jnp.bool_),
            )
        latent = self._encode(state.model_parameters, obs)
        probabilities = self._policy(state.actor_parameters, latent)
        next_key, sample_key = jr.split(state.action_key)
        chosen = (
            jr.categorical(sample_key, jnp.log(probabilities)).astype(jnp.int32)
            if action is None
            else jnp.asarray(action)
        )
        action_valid = chosen.shape == () and chosen.dtype == jnp.int32 and bool(
            (chosen >= 0) & (chosen < cfg.n_actions)
        )
        if not action_valid:
            return DreamerDecision(
                owner_transition_count=state.real_transition_count,
                owner_action_key=state.action_key,
                observation=obs,
                latent=latent,
                action=jnp.asarray(0, dtype=jnp.int32),
                action_probability=jnp.asarray(0.0, dtype=jnp.float32),
                next_action_key=state.action_key,
                valid=jnp.asarray(False, dtype=jnp.bool_),
            )
        return DreamerDecision(
            owner_transition_count=state.real_transition_count,
            owner_action_key=state.action_key,
            observation=obs,
            latent=latent,
            action=chosen,
            action_probability=probabilities[chosen],
            next_action_key=next_key,
            valid=jnp.asarray(action_valid, dtype=jnp.bool_),
        )

    def _insert(
        self, replay: DreamerReplayState, transition: DreamerTransition
    ) -> DreamerReplayState:
        index = replay.index
        capacity = self._config.replay_capacity
        return DreamerReplayState(
            observations=replay.observations.at[index].set(transition.observation),
            actions=replay.actions.at[index].set(transition.action),
            rewards=replay.rewards.at[index].set(transition.reward),
            discounts=replay.discounts.at[index].set(transition.discount),
            next_observations=replay.next_observations.at[index].set(
                transition.next_observation
            ),
            terminated=replay.terminated.at[index].set(transition.terminated),
            insertion_ids=replay.insertion_ids.at[index].set(replay.next_insertion_id),
            index=((index + 1) % capacity).astype(jnp.int32),
            size=jnp.minimum(replay.size + 1, capacity).astype(jnp.int32),
            next_insertion_id=_increment(replay.next_insertion_id, self._config.max_transitions),
        )

    def sample_sequence(self, replay: DreamerReplayState, key: Array) -> DreamerSequenceSample:
        cfg = self._config
        if not self._replay_static_valid(replay):
            raise ValueError("replay differs from the configured static contract")
        try:
            key_array = jnp.asarray(key)
            key_valid = (
                key_array.shape == ()
                and jr.key_data(key_array).shape == (2,)
                and str(jr.key_impl(key_array)) == PRNG_IMPLEMENTATION
            )
        except (TypeError, ValueError):
            key_valid = False
        if not key_valid:
            raise ValueError("sequence sampling requires an explicit Threefry key")
        offsets = jnp.arange(cfg.sequence_length, dtype=jnp.int32)
        starts = jnp.arange(cfg.replay_capacity, dtype=jnp.int32)
        indices = (starts[:, None] + offsets[None, :]) % cfg.replay_capacity
        ids = replay.insertion_ids[indices]
        contiguous = jnp.all(ids == ids[:, :1] + offsets[None, :], axis=1)
        boundary_safe = jnp.all(replay.discounts[indices[:, :-1]] > 0.0, axis=1)
        valid_starts = contiguous & boundary_safe & (ids[:, 0] >= 0)
        any_valid = jnp.any(valid_starts)
        logits = jnp.where(valid_starts, 0.0, -jnp.inf)
        safe_logits = jnp.where(any_valid, logits, jnp.zeros_like(logits))
        start = jnp.where(any_valid, jr.categorical(key_array, safe_logits), 0).astype(
            jnp.int32
        )
        selected = (start + offsets) % cfg.replay_capacity
        observations = jnp.concatenate(
            (replay.observations[selected[:1]], replay.next_observations[selected]), axis=0
        )
        return DreamerSequenceSample(
            observations=observations,
            actions=replay.actions[selected],
            rewards=replay.rewards[selected],
            discounts=replay.discounts[selected],
            insertion_ids=replay.insertion_ids[selected],
            valid=any_valid,
        )

    def _sequence_loss(
        self, parameters: DreamerModelParameters, sample: DreamerSequenceSample
    ) -> Array:
        latent = self._encode(parameters, sample.observations[0])

        def step(carry: Array, values: tuple[Array, Array, Array, Array]) -> tuple[Array, Array]:
            action, target_observation, target_reward, target_discount = values
            next_latent, observation, reward, continuation = self._model_step(
                parameters, carry, action
            )
            loss = (
                jnp.mean(jnp.square(observation - target_observation))
                + jnp.square(reward - target_reward)
                + jnp.square(
                    continuation - (target_discount > 0.0).astype(jnp.float32)
                )
            )
            return next_latent, loss

        _, losses = jax.lax.scan(
            step,
            latent,
            (
                sample.actions,
                sample.observations[1:],
                sample.rewards,
                sample.discounts,
            ),
        )
        return jnp.mean(losses)

    def _imagine_and_update(
        self,
        model: DreamerModelParameters,
        actor: DreamerActorParameters,
        value: DreamerValueParameters,
        initial_latent: Array,
        key: Array,
    ) -> tuple[
        DreamerActorParameters,
        DreamerValueParameters,
        Array,
        Array,
        Array,
        bool,
    ]:
        cfg = self._config
        keys = jr.split(key, cfg.imagination_horizon)
        latents: list[Array] = []
        actions: list[Array] = []
        rewards: list[Array] = []
        discounts: list[Array] = []
        next_values: list[Array] = []
        current = jax.lax.stop_gradient(initial_latent)
        for step in range(cfg.imagination_horizon):
            probabilities = self._policy(actor, current)
            action = jr.categorical(keys[step], jnp.log(probabilities)).astype(jnp.int32)
            next_latent, _, reward, continuation = self._model_step(model, current, action)
            latents.append(current)
            actions.append(action)
            rewards.append(reward)
            discounts.append(jnp.asarray(cfg.discount, dtype=jnp.float32) * continuation)
            next_values.append(self._value(value, next_latent))
            current = next_latent
        latent_array = jnp.stack(latents)
        action_array = jnp.stack(actions)
        reward_array = jnp.stack(rewards)
        discount_array = jnp.stack(discounts)
        next_value_array = jnp.stack(next_values)
        targets = lambda_returns(
            reward_array,
            discount_array,
            next_value_array,
            next_value_array[-1],
            cfg.lambda_,
        )
        stopped_targets = jax.lax.stop_gradient(targets)

        def value_loss(parameters: DreamerValueParameters) -> Array:
            predictions = jax.vmap(lambda latent: self._value(parameters, latent))(latent_array)
            return jnp.mean(jnp.square(predictions - stopped_targets))

        critic_loss, critic_gradient = jax.value_and_grad(value_loss)(value)
        current_values = jax.vmap(lambda latent: self._value(value, latent))(latent_array)
        advantages = jax.lax.stop_gradient(targets - current_values)

        def actor_loss(parameters: DreamerActorParameters) -> Array:
            probabilities = jax.vmap(lambda latent: self._policy(parameters, latent))(latent_array)
            selected = jnp.take_along_axis(probabilities, action_array[:, None], axis=1)[:, 0]
            log_probabilities = jnp.log(jnp.maximum(selected, 1e-8))
            entropy = -jnp.sum(probabilities * jnp.log(jnp.maximum(probabilities, 1e-8)), axis=1)
            return -jnp.mean(log_probabilities * advantages + cfg.entropy_scale * entropy)

        policy_loss, policy_gradient = jax.value_and_grad(actor_loss)(actor)
        candidate_actor = cast(
            DreamerActorParameters,
            _clipped_step(actor, policy_gradient, cfg.actor_learning_rate, cfg.gradient_clip_norm),
        )
        candidate_value = cast(
            DreamerValueParameters,
            _clipped_step(value, critic_gradient, cfg.value_learning_rate, cfg.gradient_clip_norm),
        )
        valid = _tree_finite(candidate_actor) and _tree_finite(candidate_value) and bool(
            jnp.all(jnp.isfinite(targets))
            & jnp.isfinite(policy_loss)
            & jnp.isfinite(critic_loss)
        )
        if not valid:
            return (
                actor,
                value,
                jnp.asarray(0.0),
                jnp.asarray(0.0),
                jnp.zeros_like(targets),
                False,
            )
        return candidate_actor, candidate_value, policy_loss, critic_loss, targets, True

    def learn(
        self,
        state: DreamerSequenceState,
        decision: DreamerDecision,
        transition: DreamerTransition,
    ) -> DreamerLearnResult:
        if not self._state_static_valid(state):
            raise ValueError("state differs from the configured static contract")
        if type(decision) is not DreamerDecision or type(transition) is not DreamerTransition:
            raise TypeError("decision and transition must be exact Dreamer values")
        cfg = self._config
        observation = jnp.asarray(transition.observation)
        action = jnp.asarray(transition.action)
        reward = jnp.asarray(transition.reward)
        discount = jnp.asarray(transition.discount)
        next_observation = jnp.asarray(transition.next_observation)
        terminated = jnp.asarray(transition.terminated)
        static_inputs_valid = (
            _array_matches(decision.valid, (), jnp.bool_)
            and _array_matches(decision.owner_transition_count, (), jnp.int32)
            and _threefry_key(decision.owner_action_key)
            and _threefry_key(decision.next_action_key)
            and observation.shape == next_observation.shape == (cfg.observation_dim,)
            and observation.dtype == next_observation.dtype == jnp.float32
            and action.shape == ()
            and action.dtype == jnp.int32
            and reward.shape == discount.shape == terminated.shape == ()
            and reward.dtype == discount.dtype == jnp.float32
            and terminated.dtype == jnp.bool_
            and _array_matches(decision.observation, (cfg.observation_dim,), jnp.float32)
            and _array_matches(decision.action, (), jnp.int32)
            and _array_matches(decision.latent, (cfg.latent_dim,), jnp.float32)
            and _array_matches(decision.action_probability, (), jnp.float32)
        )
        zeros = jnp.zeros((cfg.imagination_horizon,), dtype=jnp.float32)
        if not static_inputs_valid:
            return DreamerLearnResult(
                state=state,
                model_loss=jnp.asarray(0.0, dtype=jnp.float32),
                actor_loss=jnp.asarray(0.0, dtype=jnp.float32),
                value_loss=jnp.asarray(0.0, dtype=jnp.float32),
                lambda_returns=zeros,
                sequence_sampled=jnp.asarray(False),
                imagination_applied=jnp.asarray(False),
                applied=jnp.asarray(False),
            )
        expected_latent = self._encode(state.model_parameters, observation)
        expected_probabilities = self._policy(state.actor_parameters, expected_latent)
        expected_next_action_key, _ = jr.split(state.action_key)
        valid = (
            bool(decision.valid)
            and _tree_finite(state)
            and bool(jnp.all(jnp.isfinite(observation)))
            and bool(jnp.all(jnp.isfinite(next_observation)))
            and bool(jnp.isfinite(reward) & jnp.isfinite(discount))
            and bool(jnp.all(jnp.abs(observation) <= cfg.max_input_magnitude))
            and bool(jnp.all(jnp.abs(next_observation) <= cfg.max_input_magnitude))
            and bool(jnp.abs(reward) <= cfg.max_input_magnitude)
            and bool((action >= 0) & (action < cfg.n_actions))
            and bool((discount >= 0.0) & (discount <= 1.0))
            and bool((discount == 0.0) == terminated)
            and bool(decision.owner_transition_count == state.real_transition_count)
            and bool(jnp.array_equal(decision.owner_action_key, state.action_key))
            and bool(jnp.array_equal(decision.observation, observation))
            and bool(decision.action == action)
            and bool(jnp.array_equal(decision.latent, expected_latent))
            and bool(decision.action_probability == expected_probabilities[action])
            and bool(jnp.array_equal(decision.next_action_key, expected_next_action_key))
            and int(state.real_transition_count) < cfg.max_transitions
        )
        if not valid:
            return DreamerLearnResult(
                state=state,
                model_loss=jnp.asarray(0.0, dtype=jnp.float32),
                actor_loss=jnp.asarray(0.0, dtype=jnp.float32),
                value_loss=jnp.asarray(0.0, dtype=jnp.float32),
                lambda_returns=zeros,
                sequence_sampled=jnp.asarray(False),
                imagination_applied=jnp.asarray(False),
                applied=jnp.asarray(False),
            )
        replay = self._insert(state.replay, transition)
        next_replay_key, sample_key = jr.split(state.replay_key)
        sample = self.sample_sequence(replay, sample_key)
        model = state.model_parameters
        model_loss = jnp.asarray(0.0, dtype=jnp.float32)
        sampled = bool(sample.valid)
        if sampled:
            model_loss, model_gradient = jax.value_and_grad(self._sequence_loss)(model, sample)
            candidate_model = cast(
                DreamerModelParameters,
                _clipped_step(
                    model,
                    model_gradient,
                    cfg.model_learning_rate,
                    cfg.gradient_clip_norm,
                ),
            )
            if not _tree_finite(candidate_model) or not bool(jnp.isfinite(model_loss)):
                return DreamerLearnResult(
                    state=state,
                    model_loss=jnp.asarray(0.0, dtype=jnp.float32),
                    actor_loss=jnp.asarray(0.0, dtype=jnp.float32),
                    value_loss=jnp.asarray(0.0, dtype=jnp.float32),
                    lambda_returns=zeros,
                    sequence_sampled=jnp.asarray(False),
                    imagination_applied=jnp.asarray(False),
                    applied=jnp.asarray(False),
                )
            model = candidate_model
        actor = state.actor_parameters
        value = state.value_parameters
        next_imagination_key = state.imagination_key
        actor_loss = value_loss = jnp.asarray(0.0, dtype=jnp.float32)
        targets = zeros
        imagined = sampled and cfg.imagination_enabled
        if imagined:
            next_imagination_key, imagination_key = jr.split(state.imagination_key)
            actor, value, actor_loss, value_loss, targets, imagined = self._imagine_and_update(
                model,
                actor,
                value,
                self._encode(model, sample.observations[0]),
                imagination_key,
            )
        next_state = DreamerSequenceState(
            model_parameters=model,
            actor_parameters=actor,
            value_parameters=value,
            replay=replay,
            action_key=decision.next_action_key,
            replay_key=next_replay_key,
            imagination_key=next_imagination_key,
            real_transition_count=_increment(state.real_transition_count, cfg.max_transitions),
            replay_insert_count=_increment(state.replay_insert_count, cfg.max_transitions),
            sequence_sample_count=(
                _increment(state.sequence_sample_count, cfg.max_transitions)
                if sampled
                else state.sequence_sample_count
            ),
            world_model_update_count=(
                _increment(state.world_model_update_count, cfg.max_transitions)
                if sampled
                else state.world_model_update_count
            ),
            imagination_rollout_count=(
                _increment(state.imagination_rollout_count, cfg.max_transitions)
                if imagined
                else state.imagination_rollout_count
            ),
            imagination_query_count=(
                state.imagination_query_count + cfg.imagination_horizon
                if imagined
                else state.imagination_query_count
            ).astype(jnp.int32),
            lambda_return_target_count=(
                state.lambda_return_target_count + cfg.imagination_horizon
                if imagined
                else state.lambda_return_target_count
            ).astype(jnp.int32),
            actor_update_count=(
                _increment(state.actor_update_count, cfg.max_transitions)
                if imagined
                else state.actor_update_count
            ),
            value_update_count=(
                _increment(state.value_update_count, cfg.max_transitions)
                if imagined
                else state.value_update_count
            ),
        )
        return DreamerLearnResult(
            state=next_state,
            model_loss=model_loss,
            actor_loss=actor_loss,
            value_loss=value_loss,
            lambda_returns=targets,
            sequence_sampled=jnp.asarray(sampled),
            imagination_applied=jnp.asarray(imagined),
            applied=jnp.asarray(True),
        )

    def resource_budget(self, state: DreamerSequenceState | None = None) -> DreamerResourceBudget:
        measured = self.init(jr.key(0, impl=PRNG_IMPLEMENTATION)) if state is None else state
        if not self._state_static_valid(measured) or not _tree_finite(measured):
            raise ValueError("resource accounting requires an exact finite state")
        persistent_scalars, persistent_bytes = _tree_accounting(measured)
        replay_scalars, replay_bytes = _tree_accounting(measured.replay)
        sample = self.sample_sequence(
            measured.replay, jr.key(0, impl=PRNG_IMPLEMENTATION)
        )
        _, sequence_bytes = _tree_accounting(sample)
        cfg = self._config
        # Time-leading latent/action/reward/discount/value/return arrays plus
        # one actor and one value parameter-gradient candidate.
        imagination_trace_bytes = cfg.imagination_horizon * (4 * cfg.latent_dim + 20)
        actor_value_bytes = _tree_accounting(measured.actor_parameters)[1] + _tree_accounting(
            measured.value_parameters
        )[1]
        model_bytes = _tree_accounting(measured.model_parameters)[1]
        sequence_training_trace_bytes = 4 * cfg.sequence_length * (
            cfg.latent_dim + cfg.observation_dim + 4
        )
        imagination_gradient_trace_bytes = cfg.imagination_horizon * (
            4 * cfg.n_actions + 32
        )
        update_working_set = (
            2 * persistent_bytes
            + model_bytes
            + sequence_bytes
            + sequence_training_trace_bytes
            + imagination_trace_bytes
            + imagination_gradient_trace_bytes
            + 2 * actor_value_bytes
        )
        if update_working_set > _INT32_MAX:
            raise ValueError("update working-set byte count must fit signed int32")
        return DreamerResourceBudget(
            persistent_scalars=persistent_scalars,
            persistent_bytes=persistent_bytes,
            replay_scalars=replay_scalars,
            replay_bytes=replay_bytes,
            sequence_sample_bytes=sequence_bytes,
            imagination_trace_bytes=imagination_trace_bytes,
            update_working_set_bytes=update_working_set,
        )


__all__ = [
    "DreamerActorParameters",
    "DreamerDecision",
    "DreamerLearnResult",
    "DreamerModelParameters",
    "DreamerReplayState",
    "DreamerResourceBudget",
    "DreamerSequenceConfig",
    "DreamerSequenceControl",
    "DreamerSequenceSample",
    "DreamerSequenceState",
    "DreamerTransition",
    "DreamerValueParameters",
    "EVIDENCE_LEVEL",
    "PRNG_IMPLEMENTATION",
    "SCHEMA",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "lambda_returns",
]
