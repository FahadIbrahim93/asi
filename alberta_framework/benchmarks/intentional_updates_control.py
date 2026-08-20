"""Bounded end-to-end Intentional Updates TD/control development lane.

The existing IPMNIST adapter is supervised and deliberately has no temporal
credit assignment.  This module keeps that adapter intact and adds the missing
trajectory consumer: a recurring two-state continuing MDP driving linear
TD(0), TD(lambda), and Q-learning learners.  Every candidate has an exact
fixed-step mechanism-off reduction.

This is paper-informed development infrastructure, not a reproduction result,
scientific evidence, or a promotable artifact.  No result is written unless a
caller explicitly supplies a new output path to the command-line entry point.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Final, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.benchmarks.plasticity_comparators import (
    intentional_td_step_size,
    intentional_trace_step_size,
)

SCHEMA: Final[str] = "asi.intentional_updates.control.development.v1"
PAPER_REVISION: Final[str] = "arXiv:2604.19033v1"
OFFICIAL_CODE_REVISION: Final[str] = (
    "sharifnassab/Intentional_RL@e86e26fd8613ac212e9a52c3fed8a01d0a31f685"
)
ARM_NAMES: Final[tuple[str, ...]] = (
    "fixed_td0",
    "intentional_td0",
    "fixed_trace",
    "intentional_trace",
    "fixed_q_learning",
    "intentional_q_learning",
)
_OFF_ALIASES: Final[dict[str, str]] = {
    "intentional_td0_off": "fixed_td0",
    "intentional_trace_off": "fixed_trace",
    "intentional_q_learning_off": "fixed_q_learning",
}
_MAX_HORIZON: Final[int] = 10_000
_STREAM_ID: Final[str] = "asi.recurring_two_state_continuing_mdp.v1"
_AGENT_RNG_IMPL: Final[str] = "threefry2x32"
_POLICY: Final[dict[str, bool]] = {
    "development_only": True,
    "scientific_promotion_allowed": False,
    "negative_results_retained": True,
    "publication_equivalent": False,
}


def _exact_float(name: str, value: object, *, lower: float, upper: float) -> float:
    if type(value) is not float or not math.isfinite(value) or not lower <= value < upper:
        raise ValueError(f"{name} must be an exact finite float in [{lower}, {upper})")
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class IntentionalUpdatesControlConfig:
    """Small, bounded, immutable protocol configuration."""

    horizon: int = 512
    phase_length: int = 64
    discount: float = 0.95
    trace_decay: float = 0.8
    fixed_step_size: float = 0.05
    intended_fraction: float = 0.1
    diagonal_decay: float = 0.99
    diagonal_epsilon: float = 1e-6
    epsilon_greedy: float = 0.1

    def __post_init__(self) -> None:
        if type(self.horizon) is not int or not 1 <= self.horizon <= _MAX_HORIZON:
            raise ValueError(f"horizon must be an exact integer in [1, {_MAX_HORIZON}]")
        if (
            type(self.phase_length) is not int
            or not 1 <= self.phase_length <= self.horizon
        ):
            raise ValueError("phase_length must be an exact integer in [1, horizon]")
        _exact_float("discount", self.discount, lower=0.0, upper=1.0)
        _exact_float("trace_decay", self.trace_decay, lower=0.0, upper=1.0)
        _exact_float("fixed_step_size", self.fixed_step_size, lower=0.0, upper=1.0)
        if self.fixed_step_size == 0.0:
            raise ValueError("fixed_step_size must be positive")
        _exact_float("intended_fraction", self.intended_fraction, lower=0.0, upper=1.0)
        if self.intended_fraction == 0.0:
            raise ValueError("intended_fraction must be positive")
        _exact_float("diagonal_decay", self.diagonal_decay, lower=0.0, upper=1.0)
        _exact_float("diagonal_epsilon", self.diagonal_epsilon, lower=0.0, upper=1.0)
        if self.diagonal_epsilon == 0.0:
            raise ValueError("diagonal_epsilon must be positive")
        _exact_float("epsilon_greedy", self.epsilon_greedy, lower=0.0, upper=1.0)


def _config_payload(config: IntentionalUpdatesControlConfig) -> dict[str, int | float]:
    if type(config) is not IntentionalUpdatesControlConfig:
        raise ValueError("config must be an exact IntentionalUpdatesControlConfig")
    return {field.name: getattr(config, field.name) for field in dataclasses.fields(config)}


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _current_source_identity() -> dict[str, str]:
    here = Path(__file__).resolve()
    comparator = here.with_name("plasticity_comparators.py")
    return {
        "intentional_updates_control.py": _file_sha256(here),
        "plasticity_comparators.py": _file_sha256(comparator),
    }


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _runtime_identity() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "jax": _version("jax"),
        "jaxlib": _version("jaxlib"),
        "numpy": _version("numpy"),
        "backend": jax.default_backend(),
        "platform": platform.platform(),
    }


def _workload_identity(config: IntentionalUpdatesControlConfig) -> str:
    return _canonical_sha256({
        "stream": _STREAM_ID,
        "stream_semantics": {
            "state_count": 2,
            "action_count": 2,
            "phase": "floor(step / phase_length) modulo 2",
            "transition": "action=0 stays; action=1 toggles",
            "reward": "one iff successor state equals phase goal",
            "prediction_behavior": "(step + seed) modulo 2",
        },
        "config": _config_payload(config),
    })


def _transition(state: int, action: int, step: int, phase_length: int) -> tuple[int, float]:
    next_state = state if action == 0 else 1 - state
    goal = (step // phase_length) % 2
    reward = 1.0 if next_state == goal else 0.0
    return next_state, reward


def _features(state: int) -> np.ndarray:
    result = np.zeros(2, dtype=np.float32)
    result[state] = np.float32(1.0)
    return result


def _diagonal_scale(
    squared_gradient: np.ndarray, gradient: np.ndarray, config: IntentionalUpdatesControlConfig
) -> tuple[np.ndarray, np.ndarray]:
    next_squared = (
        np.float32(config.diagonal_decay) * squared_gradient
        + np.float32(1.0 - config.diagonal_decay) * np.square(gradient)
    ).astype(np.float32)
    scale = np.reciprocal(
        np.sqrt(next_squared + np.float32(config.diagonal_epsilon))
    ).astype(np.float32)
    return next_squared, scale


def _step_size_td(
    *, intentional: bool, gradient: np.ndarray, scale: np.ndarray,
    config: IntentionalUpdatesControlConfig
) -> np.float32:
    if not intentional:
        return np.float32(config.fixed_step_size)
    return np.float32(intentional_td_step_size(
        jnp.asarray(gradient),
        intended_fraction=config.intended_fraction,
        diagonal_scale=jnp.asarray(scale),
        epsilon=config.diagonal_epsilon,
    ))


def _select_action(
    values: np.ndarray, key: jax.Array, epsilon: float
) -> int:
    explore_key, action_key = jr.split(key)
    explore = float(jr.uniform(explore_key, (), dtype=jnp.float32)) < epsilon
    if explore:
        return int(jr.randint(action_key, (), 0, 2, dtype=jnp.int32))
    # Stable, explicit tie breaking makes the consumer reproducible.
    return int(np.argmax(values))


def _run_prediction(
    family: str,
    *,
    intentional: bool,
    seed: int,
    config: IntentionalUpdatesControlConfig,
) -> tuple[dict[str, object], dict[str, object], dict[str, int]]:
    weights = np.zeros(2, dtype=np.float32)
    trace = np.zeros(2, dtype=np.float32)
    squared_gradient = np.zeros(2, dtype=np.float32)
    discounted_energy = np.float32(0.0)
    state = seed % 2
    rewards: list[float] = []
    td_errors: list[float] = []
    states: list[int] = [state]
    actions: list[int] = []

    for step in range(config.horizon):
        action = (step + seed) % 2
        next_state, reward = _transition(state, action, step, config.phase_length)
        gradient = _features(state)
        next_gradient = _features(next_state)
        prediction = np.float32(np.dot(weights, gradient))
        next_prediction = np.float32(np.dot(weights, next_gradient))
        td_error = np.float32(
            reward + config.discount * float(next_prediction) - float(prediction)
        )
        squared_gradient, scale = _diagonal_scale(squared_gradient, gradient, config)
        if family == "td0":
            update_direction = gradient
            step_size = _step_size_td(
                intentional=intentional,
                gradient=gradient,
                scale=scale,
                config=config,
            )
        else:
            trace = (
                np.float32(config.discount * config.trace_decay) * trace + gradient
            ).astype(np.float32)
            discounted_energy = np.float32(
                config.trace_decay * config.discount * float(discounted_energy)
                + float(np.dot(gradient, scale * gradient))
            )
            update_direction = trace
            if intentional:
                step_size = np.float32(intentional_trace_step_size(
                    jnp.asarray(trace),
                    jnp.asarray(scale),
                    intended_fraction=config.intended_fraction,
                    discounted_gradient_energy=jnp.asarray(discounted_energy),
                    epsilon=config.diagonal_epsilon,
                ))
            else:
                step_size = np.float32(config.fixed_step_size)
        weights = (
            weights + step_size * td_error * scale * update_direction
        ).astype(np.float32)
        if not np.isfinite(weights).all():
            raise ValueError("prediction learner produced non-finite weights")
        rewards.append(float(reward))
        td_errors.append(float(td_error))
        actions.append(action)
        states.append(next_state)
        state = next_state

    numeric_bytes = int(
        weights.nbytes + trace.nbytes + squared_gradient.nbytes + discounted_energy.nbytes
    )
    return (
        {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "td_errors": td_errors,
        },
        {
            "weights": weights.tolist(),
            "trace": trace.tolist(),
            "squared_gradient": squared_gradient.tolist(),
            "discounted_gradient_energy": float(discounted_energy),
        },
        {
            "environment_steps": config.horizon,
            "reward_observations": config.horizon,
            "updates": config.horizon,
            "model_queries": 2 * config.horizon,
            "action_queries": 0,
            "rng_splits": 0,
            "rng_fold_ins": 0,
            "trajectory_items": config.horizon,
            "persistent_numeric_bytes": numeric_bytes,
            "diagonal_state_updates": config.horizon,
            "eligibility_trace_updates": config.horizon if family == "trace" else 0,
            "intentional_step_size_solves": config.horizon if intentional else 0,
        },
    )


def _run_control(
    *, intentional: bool, seed: int, config: IntentionalUpdatesControlConfig
) -> tuple[dict[str, object], dict[str, object], dict[str, int]]:
    weights = np.zeros((2, 2), dtype=np.float32)
    trace = np.zeros((2, 2), dtype=np.float32)
    squared_gradient = np.zeros((2, 2), dtype=np.float32)
    state = seed % 2
    root = jr.key(seed, impl=_AGENT_RNG_IMPL)
    states: list[int] = [state]
    actions: list[int] = []
    rewards: list[float] = []
    td_errors: list[float] = []

    for step in range(config.horizon):
        current_values = weights[state].copy()
        action = _select_action(
            current_values, jr.fold_in(root, step), config.epsilon_greedy
        )
        next_state, reward = _transition(state, action, step, config.phase_length)
        next_values = weights[next_state].copy()
        prediction = np.float32(current_values[action])
        next_prediction = np.float32(np.max(next_values))
        td_error = np.float32(
            reward + config.discount * float(next_prediction) - float(prediction)
        )
        gradient = np.zeros((2, 2), dtype=np.float32)
        gradient[state, action] = np.float32(1.0)
        squared_gradient, scale = _diagonal_scale(squared_gradient, gradient, config)
        step_size = _step_size_td(
            intentional=intentional,
            gradient=gradient,
            scale=scale,
            config=config,
        )
        weights = (weights + step_size * td_error * scale * gradient).astype(np.float32)
        if not np.isfinite(weights).all():
            raise ValueError("control learner produced non-finite weights")
        states.append(next_state)
        actions.append(action)
        rewards.append(float(reward))
        td_errors.append(float(td_error))
        state = next_state

    numeric_bytes = int(
        weights.nbytes + trace.nbytes + squared_gradient.nbytes + root.nbytes
    )
    return (
        {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "td_errors": td_errors,
        },
        {
            "weights": weights.tolist(),
            "trace": trace.tolist(),
            "squared_gradient": squared_gradient.tolist(),
            "discounted_gradient_energy": 0.0,
        },
        {
            "environment_steps": config.horizon,
            "reward_observations": config.horizon,
            "updates": config.horizon,
            "model_queries": 2 * config.horizon,
            "action_queries": config.horizon,
            "rng_splits": config.horizon,
            "rng_fold_ins": config.horizon,
            "trajectory_items": config.horizon,
            "persistent_numeric_bytes": numeric_bytes,
            "diagonal_state_updates": config.horizon,
            "eligibility_trace_updates": 0,
            "intentional_step_size_solves": config.horizon if intentional else 0,
        },
    )


def _run_payload(
    execution_arm: str, *, seed: int, config: IntentionalUpdatesControlConfig
) -> tuple[dict[str, object], dict[str, object], dict[str, int]]:
    intentional = execution_arm.startswith("intentional_")
    if execution_arm.endswith("td0"):
        return _run_prediction("td0", intentional=intentional, seed=seed, config=config)
    if execution_arm.endswith("trace"):
        return _run_prediction("trace", intentional=intentional, seed=seed, config=config)
    return _run_control(intentional=intentional, seed=seed, config=config)


def run_intentional_updates_control(
    arm: str, *, seed: int, config: IntentionalUpdatesControlConfig | None = None
) -> dict[str, object]:
    """Execute one bounded fresh in-memory development shard."""
    if type(arm) is not str or (arm not in ARM_NAMES and arm not in _OFF_ALIASES):
        raise ValueError("arm must name a registered Intentional Updates control arm")
    if type(seed) is not int or not 0 <= seed <= (1 << 31) - 1:
        raise ValueError("seed must be an exact nonnegative signed-int32 integer")
    checked_config = IntentionalUpdatesControlConfig() if config is None else config
    config_payload = _config_payload(checked_config)
    execution_arm = _OFF_ALIASES.get(arm, arm)
    trajectory, final_state, resources = _run_payload(
        execution_arm, seed=seed, config=checked_config
    )
    rewards = cast(list[float], trajectory["rewards"])
    td_errors = cast(list[float], trajectory["td_errors"])
    metrics = {
        "mean_reward": float(np.mean(np.asarray(rewards, dtype=np.float64))),
        "tail_mean_reward": float(
            np.mean(np.asarray(rewards[-checked_config.phase_length :], dtype=np.float64))
        ),
        "mean_squared_td_error": float(
            np.mean(np.square(np.asarray(td_errors, dtype=np.float64)))
        ),
    }
    return {
        "schema": SCHEMA,
        "arm": arm,
        "execution_arm": execution_arm,
        "seed": seed,
        "config": config_payload,
        "references": {"paper": PAPER_REVISION, "official_code": OFFICIAL_CODE_REVISION},
        "adaptation": (
            "linear prediction/control on ASI recurring two-state MDP; "
            "not publication-equivalent"
        ),
        "identity": {
            "agent_rng_impl": _AGENT_RNG_IMPL,
            "current_source_sha256": _current_source_identity(),
            "runtime": _runtime_identity(),
            "workload_sha256": _workload_identity(checked_config),
        },
        "policy": dict(_POLICY),
        "resources": resources,
        "trajectory": trajectory,
        "final_state": final_state,
        "metrics": metrics,
    }


def validate_intentional_updates_control_record(value: object) -> dict[str, object]:
    """Re-execute and require exact equality with the current code and runtime."""
    if type(value) is not dict:
        raise ValueError("record must be an exact object")
    record = cast(dict[object, object], value)
    expected_keys = {
        "schema", "arm", "execution_arm", "seed", "config", "references",
        "adaptation", "identity", "policy", "resources", "trajectory",
        "final_state", "metrics",
    }
    if set(record) != expected_keys or not all(type(key) is str for key in record):
        raise ValueError("record keys do not match the protocol")
    config_value = record["config"]
    if type(config_value) is not dict:
        raise ValueError("config must be an exact object")
    config_dict = cast(dict[object, object], config_value)
    config_keys = {field.name for field in dataclasses.fields(IntentionalUpdatesControlConfig)}
    if set(config_dict) != config_keys or not all(type(key) is str for key in config_dict):
        raise ValueError("config keys do not match the protocol")
    try:
        config = IntentionalUpdatesControlConfig(
            horizon=cast(int, config_dict["horizon"]),
            phase_length=cast(int, config_dict["phase_length"]),
            discount=cast(float, config_dict["discount"]),
            trace_decay=cast(float, config_dict["trace_decay"]),
            fixed_step_size=cast(float, config_dict["fixed_step_size"]),
            intended_fraction=cast(float, config_dict["intended_fraction"]),
            diagonal_decay=cast(float, config_dict["diagonal_decay"]),
            diagonal_epsilon=cast(float, config_dict["diagonal_epsilon"]),
            epsilon_greedy=cast(float, config_dict["epsilon_greedy"]),
        )
        expected = run_intentional_updates_control(
            cast(str, record["arm"]), seed=cast(int, record["seed"]), config=config
        )
        actual_json = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        expected_json = json.dumps(
            expected, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("record contains invalid protocol values") from error
    if actual_json != expected_json:
        raise ValueError("record differs from current exact re-execution")
    return cast(dict[str, object], value)


def _write_new(path: Path, payload: dict[str, object]) -> None:
    encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARM_NAMES)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--horizon", type=int, default=512)
    parser.add_argument("--phase-length", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    record = run_intentional_updates_control(
        args.arm,
        seed=args.seed,
        config=IntentionalUpdatesControlConfig(
            horizon=args.horizon, phase_length=args.phase_length
        ),
    )
    validate_intentional_updates_control_record(record)
    if args.output is None:
        json.dump(record, sys.stdout, sort_keys=True, indent=2, allow_nan=False)
        sys.stdout.write("\n")
    else:
        _write_new(args.output, record)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ARM_NAMES",
    "IntentionalUpdatesControlConfig",
    "run_intentional_updates_control",
    "validate_intentional_updates_control_record",
]
