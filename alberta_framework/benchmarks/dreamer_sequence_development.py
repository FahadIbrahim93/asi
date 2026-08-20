# mypy: disable-error-code="call-arg"
"""Matched nonpromoting consumer for recurrent Dreamer-family sequence control.

The two learnable arms consume the exact same predeclared real transitions.
Only the candidate runs latent imagination and actor/value/lambda-return
updates; the mechanism-off arm retains identical replay and model bytes.  A
finite-horizon task-aware policy is a normalization control and is excluded
from candidate comparisons.  A final no-learning control phase consumes each
learned actor in the environment so this is a prediction-to-control slice,
not a model-loss-only diagnostic.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import operator
import sys
import time
from collections.abc import Mapping
from typing import SupportsIndex, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import alberta_framework.core.dreamer_sequence_control as dreamer_module
from alberta_framework.benchmarks.development_provenance import (
    DevelopmentIdentity,
    collect_development_identity,
    require_current_identity,
)
from alberta_framework.core.dreamer_sequence_control import (
    PRNG_IMPLEMENTATION,
    DreamerSequenceConfig,
    DreamerSequenceControl,
    DreamerTransition,
)

SCHEMA = "asi.dreamer_sequence_development.v1"
DREAMERV3_SOURCE = "danijar/dreamerv3@e3f02248693a79dc8b0ebd62c93683888ddaccfe"
WMAR_PAPER = "Liu et al., arXiv:2401.16650"
CONTINUAL_DREAMER_PAPER = "Kessler et al., PMLR 232:184-204; arXiv:2211.15944"
FROZEN_SEEDS = (1760, 1761, 1762, 1763, 1764)
FROZEN_TASK_TARGETS = (0, 1, 0)
ARM_IDS = ("recurrent_sequence_imagination", "imagination_off", "privileged_task_control")
MAX_STEPS_PER_TASK = 16
REPLAY_CAPACITY = 16
SEQUENCE_LENGTH = 2
IMAGINATION_HORIZON = 3
LATENT_DIM = 4
WORKLOAD_REGISTRY = (
    ("arm_ids", ARM_IDS),
    ("frozen_seeds", FROZEN_SEEDS),
    ("frozen_task_targets", FROZEN_TASK_TARGETS),
    ("imagination_horizon", IMAGINATION_HORIZON),
    ("latent_dim", LATENT_DIM),
    ("max_steps_per_task", MAX_STEPS_PER_TASK),
    ("prng_implementation", PRNG_IMPLEMENTATION),
    ("replay_capacity", REPLAY_CAPACITY),
    ("sequence_length", SEQUENCE_LENGTH),
)
PAPER_REGISTRY = (
    ("continual_dreamer", CONTINUAL_DREAMER_PAPER),
    ("dreamerv3_source", DREAMERV3_SOURCE),
    ("wmar", WMAR_PAPER),
)


def _identity() -> DevelopmentIdentity:
    return collect_development_identity(
        lane_module=sys.modules[__name__],
        dependency_modules=(dreamer_module,),
        workload_registry=WORKLOAD_REGISTRY,
        paper_registry=PAPER_REGISTRY,
    )


def _exact_int(value: object, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an exact integer")
    result = operator.index(cast(SupportsIndex, value))
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must lie in [{minimum}, {maximum}]")
    return result


def _digest(value: object) -> str:
    leaves, tree = jax.tree_util.tree_flatten(value)
    digest = hashlib.sha256(str(tree).encode("ascii"))
    for raw_leaf in leaves:
        leaf = jnp.asarray(raw_leaf)
        if str(leaf.dtype).startswith("key<"):
            leaf = jr.key_data(leaf)
        array = np.asarray(jax.device_get(leaf))
        digest.update(str((array.shape, array.dtype.str)).encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _transition(observation: np.ndarray, action: int, target: int) -> tuple[np.ndarray, float]:
    direction = 1.0 if action == 0 else -1.0
    next_observation = np.asarray(
        (0.75 * observation[0] + direction, float(target)), dtype=np.float32
    )
    reward = 1.0 if action == target else -1.0
    return next_observation, reward


@dataclasses.dataclass(frozen=True, slots=True)
class ResourceReceipt:
    environment_steps: int
    replay_inserts: int
    sequence_samples: int
    world_model_updates: int
    imagination_rollouts: int
    imagination_queries: int
    lambda_return_targets: int
    actor_updates: int
    value_updates: int
    persistent_scalars: int
    persistent_bytes: int
    replay_bytes: int
    sequence_sample_bytes: int
    imagination_trace_bytes: int
    update_working_set_bytes: int
    logical_compute_units: int
    elapsed_ns: int
    timing_telemetry_only: bool = True

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            if field.name == "timing_telemetry_only":
                continue
            _exact_int(getattr(self, field.name), field.name, 0, 2**63 - 1)
        if self.environment_steps == 0 or self.persistent_bytes == 0:
            raise ValueError("environment steps and persistent bytes must be positive")
        if type(self.timing_telemetry_only) is not bool or not self.timing_telemetry_only:
            raise ValueError("timing must remain telemetry-only")


@dataclasses.dataclass(frozen=True, slots=True)
class ArmResult:
    arm_id: str
    training_returns: tuple[float, ...]
    evaluation_returns: tuple[float, ...]
    final_action_probabilities: tuple[float, float]
    model_digest: str
    replay_digest: str
    receipt: ResourceReceipt
    candidate_eligible: bool

    def __post_init__(self) -> None:
        if type(self.arm_id) is not str or self.arm_id not in ARM_IDS:
            raise ValueError("unknown arm")
        for name in ("training_returns", "evaluation_returns"):
            values = getattr(self, name)
            if type(values) is not tuple or len(values) != len(FROZEN_TASK_TARGETS):
                raise ValueError(f"{name} must bind the frozen task sequence")
            if any(type(item) is not float or not math.isfinite(item) for item in values):
                raise ValueError(f"{name} values must be finite exact floats")
        if (
            type(self.final_action_probabilities) is not tuple
            or len(self.final_action_probabilities) != 2
            or any(
                type(item) is not float or not math.isfinite(item) or not 0.0 <= item <= 1.0
                for item in self.final_action_probabilities
            )
            or not math.isclose(sum(self.final_action_probabilities), 1.0, abs_tol=1e-6)
        ):
            raise ValueError("final policy must be a finite probability pair")
        for name in ("model_digest", "replay_digest"):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        expected = self.arm_id != "privileged_task_control"
        if type(self.candidate_eligible) is not bool or self.candidate_eligible != expected:
            raise ValueError("privileged normalization control must be excluded")


@dataclasses.dataclass(frozen=True, slots=True)
class DevelopmentResult:
    schema: str
    seed: int
    steps_per_task: int
    replay_capacity: int
    sequence_length: int
    imagination_horizon: int
    arms: tuple[ArmResult, ...]
    identity: DevelopmentIdentity
    development_only: bool = True
    scientific_promotion_allowed: bool = False
    negative_results_must_be_retained: bool = True
    dreamerv3_parity_claimed: bool = False
    wmar_parity_claimed: bool = False
    continual_dreamer_parity_claimed: bool = False

    def __post_init__(self) -> None:
        if self.schema != SCHEMA or self.seed not in FROZEN_SEEDS:
            raise ValueError("schema or frozen seed mismatch")
        _exact_int(self.steps_per_task, "steps_per_task", 1, MAX_STEPS_PER_TASK)
        if (
            self.replay_capacity != REPLAY_CAPACITY
            or self.sequence_length != SEQUENCE_LENGTH
            or self.imagination_horizon != IMAGINATION_HORIZON
        ):
            raise ValueError("fixed mechanism dimensions differ from the workload")
        if type(self.arms) is not tuple or any(type(arm) is not ArmResult for arm in self.arms):
            raise ValueError("arms must contain exact ArmResult values")
        if tuple(arm.arm_id for arm in self.arms) != ARM_IDS:
            raise ValueError("arm roster differs from the workload")
        if type(self.identity) is not DevelopmentIdentity:
            raise ValueError("identity must be exact")
        required = (
            self.development_only,
            not self.scientific_promotion_allowed,
            self.negative_results_must_be_retained,
            not self.dreamerv3_parity_claimed,
            not self.wmar_parity_claimed,
            not self.continual_dreamer_parity_claimed,
        )
        if any(type(flag) is not bool or not flag for flag in required):
            raise ValueError("result must remain retained, nonpromoting, and non-parity")


def _config(enabled: bool) -> DreamerSequenceConfig:
    return DreamerSequenceConfig(
        observation_dim=2,
        n_actions=2,
        latent_dim=LATENT_DIM,
        replay_capacity=REPLAY_CAPACITY,
        sequence_length=SEQUENCE_LENGTH,
        imagination_horizon=IMAGINATION_HORIZON,
        model_learning_rate=0.01,
        actor_learning_rate=0.01,
        value_learning_rate=0.01,
        discount=0.9,
        lambda_=0.8,
        entropy_scale=0.001,
        imagination_enabled=enabled,
    )


def _run_learned(seed: int, steps: int, *, enabled: bool) -> ArmResult:
    agent = DreamerSequenceControl(_config(enabled))
    state = agent.init(jr.key(seed, impl=PRNG_IMPLEMENTATION))
    training_returns: list[float] = []
    start = time.perf_counter_ns()
    transition_index = 0
    for target in FROZEN_TASK_TARGETS:
        observation = np.zeros((2,), dtype=np.float32)
        total = 0.0
        for within_task in range(steps):
            # A fixed action schedule gives the enabled/off pair identical real
            # evidence.  The learned actor is consumed in the separate frozen
            # evaluation phase below.
            action = (seed + transition_index) % 2
            next_observation, reward = _transition(observation, action, target)
            terminal = within_task == steps - 1
            record = DreamerTransition(
                observation=jnp.asarray(observation),
                action=jnp.asarray(action, dtype=jnp.int32),
                reward=jnp.asarray(reward, dtype=jnp.float32),
                discount=jnp.asarray(0.0 if terminal else 0.9, dtype=jnp.float32),
                next_observation=jnp.asarray(next_observation),
                terminated=jnp.asarray(terminal, dtype=jnp.bool_),
            )
            decision = agent.decide(state, record.observation, action=record.action)
            result = agent.learn(state, decision, record)
            if not bool(result.applied):
                raise RuntimeError("bounded Dreamer transition was rejected")
            state = result.state
            observation = next_observation
            total += reward
            transition_index += 1
        training_returns.append(total)

    evaluation_returns: list[float] = []
    for target in FROZEN_TASK_TARGETS:
        observation = np.zeros((2,), dtype=np.float32)
        total = 0.0
        for _ in range(steps):
            probabilities = agent.policy(state, jnp.asarray(observation))
            action = int(jnp.argmax(probabilities))
            observation, reward = _transition(observation, action, target)
            total += reward
        evaluation_returns.append(total)
    elapsed = time.perf_counter_ns() - start
    probabilities = agent.policy(state, jnp.zeros((2,), dtype=jnp.float32))
    budget = agent.resource_budget(state)
    queries = int(state.imagination_query_count)
    logical_compute = (
        2 * transition_index
        + int(state.sequence_sample_count)
        + int(state.world_model_update_count)
        + int(state.imagination_rollout_count)
        + queries
        + int(state.lambda_return_target_count)
        + int(state.actor_update_count)
        + int(state.value_update_count)
    )
    return ArmResult(
        arm_id="recurrent_sequence_imagination" if enabled else "imagination_off",
        training_returns=tuple(float(item) for item in training_returns),
        evaluation_returns=tuple(float(item) for item in evaluation_returns),
        final_action_probabilities=(float(probabilities[0]), float(probabilities[1])),
        model_digest=_digest(state.model_parameters),
        replay_digest=_digest(state.replay),
        receipt=ResourceReceipt(
            environment_steps=2 * transition_index,
            replay_inserts=int(state.replay_insert_count),
            sequence_samples=int(state.sequence_sample_count),
            world_model_updates=int(state.world_model_update_count),
            imagination_rollouts=int(state.imagination_rollout_count),
            imagination_queries=queries,
            lambda_return_targets=int(state.lambda_return_target_count),
            actor_updates=int(state.actor_update_count),
            value_updates=int(state.value_update_count),
            persistent_scalars=budget.persistent_scalars,
            persistent_bytes=budget.persistent_bytes,
            replay_bytes=budget.replay_bytes,
            sequence_sample_bytes=budget.sequence_sample_bytes,
            imagination_trace_bytes=budget.imagination_trace_bytes,
            update_working_set_bytes=budget.update_working_set_bytes,
            logical_compute_units=logical_compute,
            elapsed_ns=elapsed,
        ),
        candidate_eligible=True,
    )


def _run_oracle(steps: int) -> ArmResult:
    training = tuple(float(steps) for _ in FROZEN_TASK_TARGETS)
    task_bytes = int(np.asarray(FROZEN_TASK_TARGETS, dtype=np.int32).nbytes)
    environment_steps = 2 * steps * len(FROZEN_TASK_TARGETS)
    empty = hashlib.sha256(b"privileged-no-model-or-replay").hexdigest()
    return ArmResult(
        arm_id="privileged_task_control",
        training_returns=training,
        evaluation_returns=training,
        final_action_probabilities=(0.5, 0.5),
        model_digest=empty,
        replay_digest=empty,
        receipt=ResourceReceipt(
            environment_steps=environment_steps,
            replay_inserts=0,
            sequence_samples=0,
            world_model_updates=0,
            imagination_rollouts=0,
            imagination_queries=0,
            lambda_return_targets=0,
            actor_updates=0,
            value_updates=0,
            persistent_scalars=len(FROZEN_TASK_TARGETS),
            persistent_bytes=task_bytes,
            replay_bytes=0,
            sequence_sample_bytes=0,
            imagination_trace_bytes=0,
            update_working_set_bytes=task_bytes,
            logical_compute_units=environment_steps,
            elapsed_ns=0,
        ),
        candidate_eligible=False,
    )


def _execute(seed: int, steps: int) -> DevelopmentResult:
    return DevelopmentResult(
        schema=SCHEMA,
        seed=seed,
        steps_per_task=steps,
        replay_capacity=REPLAY_CAPACITY,
        sequence_length=SEQUENCE_LENGTH,
        imagination_horizon=IMAGINATION_HORIZON,
        arms=(
            _run_learned(seed, steps, enabled=True),
            _run_learned(seed, steps, enabled=False),
            _run_oracle(steps),
        ),
        identity=_identity(),
    )


def run_development_lane(*, seed: object, steps_per_task: object = 3) -> DevelopmentResult:
    host_seed = _exact_int(seed, "seed", 0, 2**32 - 1)
    if host_seed not in FROZEN_SEEDS:
        raise ValueError("seed is outside the frozen development schedule")
    steps = _exact_int(steps_per_task, "steps_per_task", 1, MAX_STEPS_PER_TASK)
    return validate_result(_execute(host_seed, steps), replay_execution=False)


def validate_result(
    value: object, *, replay_execution: bool = True
) -> DevelopmentResult:
    if type(replay_execution) is not bool:
        raise ValueError("replay_execution must be an exact bool")
    if type(value) is not DevelopmentResult:
        if not isinstance(value, Mapping) or type(value) is not dict:
            raise ValueError("payload must be an exact result or dict")
        raise ValueError("mapping decoding is unavailable until a campaign schema is frozen")
    DevelopmentResult.__post_init__(value)
    require_current_identity(value.identity, _identity())
    for arm in value.arms:
        ArmResult.__post_init__(arm)
        if type(arm.receipt) is not ResourceReceipt:
            raise ValueError("receipt must be exact")
        ResourceReceipt.__post_init__(arm.receipt)
    candidate, off, oracle = value.arms
    transitions = value.steps_per_task * len(FROZEN_TASK_TARGETS)
    environment_steps = 2 * transitions
    if any(arm.receipt.environment_steps != environment_steps for arm in value.arms):
        raise ValueError("environment-step receipt mismatch")
    expected_samples = 0 if value.steps_per_task < SEQUENCE_LENGTH else transitions - 1
    for arm in (candidate, off):
        if arm.receipt.replay_inserts != transitions:
            raise ValueError("replay insert receipt mismatch")
        if arm.receipt.sequence_samples != expected_samples:
            raise ValueError("sequence sample receipt mismatch")
        if arm.receipt.world_model_updates != expected_samples:
            raise ValueError("world-model update receipt mismatch")
    expected_queries = expected_samples * IMAGINATION_HORIZON
    if (
        candidate.receipt.imagination_rollouts != expected_samples
        or candidate.receipt.imagination_queries != expected_queries
        or candidate.receipt.lambda_return_targets != expected_queries
        or candidate.receipt.actor_updates != expected_samples
        or candidate.receipt.value_updates != expected_samples
    ):
        raise ValueError("imagination/lambda-return/control receipt mismatch")
    if any(
        getattr(off.receipt, field) != 0
        for field in (
            "imagination_rollouts",
            "imagination_queries",
            "lambda_return_targets",
            "actor_updates",
            "value_updates",
        )
    ):
        raise ValueError("mechanism-off arm performed imagination or control updates")
    if candidate.model_digest != off.model_digest or candidate.replay_digest != off.replay_digest:
        raise ValueError("mechanism-off model/replay path is not exact")
    if any(
        getattr(oracle.receipt, field) != 0
        for field in (
            "replay_inserts",
            "sequence_samples",
            "world_model_updates",
            "imagination_rollouts",
            "imagination_queries",
            "lambda_return_targets",
            "actor_updates",
            "value_updates",
        )
    ):
        raise ValueError("privileged control must not own learned mechanisms")
    if replay_execution:
        expected = _execute(value.seed, value.steps_per_task)
        for actual_arm, expected_arm in zip(value.arms, expected.arms, strict=True):
            actual = dataclasses.replace(actual_arm.receipt, elapsed_ns=0)
            replayed = dataclasses.replace(expected_arm.receipt, elapsed_ns=0)
            if dataclasses.replace(actual_arm, receipt=actual) != dataclasses.replace(
                expected_arm, receipt=replayed
            ):
                raise ValueError("result differs from exact current-source replay")
    return value


def main() -> int:
    result = run_development_lane(seed=FROZEN_SEEDS[0])
    payload = dataclasses.asdict(result)
    print(json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":")))
    return 0


__all__ = [
    "ARM_IDS",
    "CONTINUAL_DREAMER_PAPER",
    "DREAMERV3_SOURCE",
    "DevelopmentResult",
    "FROZEN_SEEDS",
    "ResourceReceipt",
    "SCHEMA",
    "WMAR_PAPER",
    "ArmResult",
    "main",
    "run_development_lane",
    "validate_result",
]


if __name__ == "__main__":
    raise SystemExit(main())
