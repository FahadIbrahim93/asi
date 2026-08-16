"""Development-only exact-dispatch controls for the reference-life runner.

These adapters are matched development controls, not ``reference-dev``, a
checkpoint surface, or scientific evidence.  They intentionally support only
the primitive continuing one-hot/discrete slices used by
``SwitchingTwoStateMDP`` and ``RiverSwimMDP``.  Every live state is immutable
and bound to one process-local adapter owner.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.core.average_reward import (
    DifferentialSARSAAgent,
    DifferentialSARSAConfig,
    DifferentialSARSAState,
)
from alberta_framework.core.sarsa import SARSAAgent, SARSAConfig, SARSAState
from alberta_framework.reference_agent import (
    MAX_DECISION_INDEX,
    REFERENCE_AGENT_MANIFEST_SCHEMA,
    AgentCapabilities,
    AgentManifest,
    ArrayValue,
    AuthorizationStatus,
    Decision,
    DecisionOwnershipError,
    DispatchAuthorization,
    DispatchStatus,
    ReferenceAgentUpdate,
    SpaceSpec,
    Transaction,
    canonical_config_sha256,
)
from alberta_framework.streams.closed_loop import (
    RiverSwimConfig,
    RiverSwimMDP,
    SwitchingTwoStateConfig,
    SwitchingTwoStateMDP,
)

# All selected simulators use signed-int32 transition counters.  An accepted
# event at this capacity leaves decision ``int32.max`` armed, but no selected
# life may request another transition.
REFERENCE_CONTROL_MAX_ACCEPTED_EVENTS = int(np.iinfo(np.int32).max)
REFERENCE_CONTROL_CONFIG_SCHEMA = "asi.reference_life_control_config.preview1"
REFERENCE_CONTROL_OBSERVATION_SEMANTIC_ID = (
    "asi.reference_life_control.one_hot_observation.preview1"
)
REFERENCE_CONTROL_ACTION_SEMANTIC_ID = (
    "asi.reference_life_control.primitive_action.preview1"
)

UNIFORM_RANDOM_IMPLEMENTATION_ID = "asi.uniform_random_exact_control.preview1"
UNIFORM_RANDOM_STATE_SCHEMA = "asi.uniform_random_exact_control_state.preview1"
ANALYTIC_ORACLE_IMPLEMENTATION_ID = "asi.analytic_oracle_exact_control.preview1"
ANALYTIC_ORACLE_STATE_SCHEMA = "asi.analytic_oracle_exact_control_state.preview1"
DIFFERENTIAL_SARSA_IMPLEMENTATION_ID = "asi.differential_sarsa_exact_control.preview1"
DIFFERENTIAL_SARSA_STATE_SCHEMA = "asi.differential_sarsa_exact_control_state.preview1"
DISCOUNTED_SARSA_IMPLEMENTATION_ID = "asi.discounted_sarsa_exact_control.preview1"
DISCOUNTED_SARSA_STATE_SCHEMA = "asi.discounted_sarsa_exact_control_state.preview1"

_RIVERSWIM_REFERENCE_MAX_STATES = 12
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._:-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_ID_LENGTH = 256
_RANDOM_DOMAIN = 0x4354524C

EnvironmentKind = Literal["switching_two_state", "riverswim"]


def _require_environment_shape(
    environment_kind: EnvironmentKind,
    observation_dim: int,
    n_actions: int,
) -> None:
    if environment_kind not in ("switching_two_state", "riverswim"):
        raise ValueError("environment_kind must be switching_two_state or riverswim")
    if (
        isinstance(observation_dim, bool)
        or not isinstance(observation_dim, int)
        or observation_dim < 2
    ):
        raise ValueError("observation_dim must be an integer of at least two")
    if environment_kind == "switching_two_state" and observation_dim != 2:
        raise ValueError("SwitchingTwoState controls require observation_dim == 2")
    if environment_kind == "riverswim" and observation_dim > _RIVERSWIM_REFERENCE_MAX_STATES:
        raise ValueError(
            "RiverSwim controls require observation_dim <= "
            f"{_RIVERSWIM_REFERENCE_MAX_STATES}"
        )
    if isinstance(n_actions, bool) or not isinstance(n_actions, int) or n_actions != 2:
        raise ValueError("selected reference controls require exactly two actions")


def _finite_float(value: Any, *, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(converted) or converted < minimum or converted > maximum:
        raise ValueError(f"{name} must be finite and lie in [{minimum}, {maximum}]")
    return converted


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _canonical_float32(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite float32 value")
    try:
        raw = float(value)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            converted = float(np.asarray(raw, dtype=np.float32))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite float32 value") from exc
    if not math.isfinite(raw) or not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite float32 value")
    return converted


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _switching_policy(payoff: np.ndarray[Any, Any]) -> tuple[int, int]:
    gains = (
        float(payoff[0, 0]),
        0.5 * (float(payoff[0, 1]) + float(payoff[1, 0])),
        float(payoff[1, 1]),
    )
    if not all(math.isfinite(gain) for gain in gains):
        raise ValueError("switching oracle cycle gains must be finite")
    winner = max(range(3), key=gains.__getitem__)
    # Direct every state into the selected optimal recurrent cycle.
    return ((0, 0), (1, 0), (1, 1))[winner]


def _canonical_switching_environment(
    config: SwitchingTwoStateConfig,
) -> tuple[dict[str, Any], int, tuple[tuple[int, ...], ...]]:
    if not isinstance(config, SwitchingTwoStateConfig):
        raise ValueError("switching environment config has the wrong type")
    phase_length = config.phase_length
    if (
        isinstance(phase_length, bool)
        or not isinstance(phase_length, int)
        or phase_length <= 0
        or phase_length > REFERENCE_CONTROL_MAX_ACCEPTED_EVENTS
    ):
        raise ValueError("phase_length must lie within signed-int32 capacity")
    try:
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            payoffs_a = np.asarray(config.payoffs_a, dtype=np.float32)
            payoffs_b = np.asarray(config.payoffs_b, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("switching payoffs must be numeric 2x2 matrices") from exc
    if payoffs_a.shape != (2, 2) or payoffs_b.shape != (2, 2):
        raise ValueError("switching payoffs must be 2x2 matrices")
    if not np.all(np.isfinite(payoffs_a)) or not np.all(np.isfinite(payoffs_b)):
        raise ValueError("switching payoffs must be finite float32 values")
    canonical = SwitchingTwoStateConfig(  # type: ignore[call-arg]
        phase_length=phase_length,
        payoffs_a=tuple(tuple(float(value) for value in row) for row in payoffs_a),
        payoffs_b=tuple(tuple(float(value) for value in row) for row in payoffs_b),
    )
    # Retain the kernel validation as part of the same config gate.
    SwitchingTwoStateMDP(canonical)
    payload = {
        "phase_length": phase_length,
        "payoffs_a": [list(row) for row in canonical.payoffs_a],
        "payoffs_b": [list(row) for row in canonical.payoffs_b],
    }
    policies = (_switching_policy(payoffs_a), _switching_policy(payoffs_b))
    return payload, phase_length, policies


def _canonical_riverswim_environment(
    config: RiverSwimConfig,
) -> tuple[dict[str, Any], tuple[tuple[int, ...], ...]]:
    if not isinstance(config, RiverSwimConfig):
        raise ValueError("RiverSwim environment config has the wrong type")
    n_states = config.n_states
    if (
        isinstance(n_states, bool)
        or not isinstance(n_states, int)
        or n_states < 2
        or n_states > _RIVERSWIM_REFERENCE_MAX_STATES
    ):
        raise ValueError(
            "RiverSwim n_states must lie in "
            f"[2, {_RIVERSWIM_REFERENCE_MAX_STATES}]"
        )
    initial_state = config.initial_state
    if (
        isinstance(initial_state, bool)
        or not isinstance(initial_state, int)
        or initial_state < 0
        or initial_state >= n_states
    ):
        raise ValueError("RiverSwim initial_state is outside the configured chain")
    p_right_up = _canonical_float32(config.p_right_up, name="p_right_up")
    p_right_down = _canonical_float32(config.p_right_down, name="p_right_down")
    reward_left = _canonical_float32(config.reward_left, name="reward_left")
    reward_right = _canonical_float32(config.reward_right, name="reward_right")
    if p_right_up <= 0.0 or p_right_down <= 0.0:
        raise ValueError("RiverSwim right-transition probabilities must be positive")
    if p_right_up + p_right_down > 1.0:
        raise ValueError("RiverSwim right-transition probabilities must sum to at most one")
    canonical = RiverSwimConfig(  # type: ignore[call-arg]
        n_states=n_states,
        p_right_up=p_right_up,
        p_right_down=p_right_down,
        reward_left=reward_left,
        reward_right=reward_right,
        initial_state=initial_state,
    )
    kernel = RiverSwimMDP(canonical)
    policy = kernel.optimal_policy()
    payload = {
        "n_states": n_states,
        "p_right_up": p_right_up,
        "p_right_down": p_right_down,
        "reward_left": reward_left,
        "reward_right": reward_right,
        "initial_state": initial_state,
    }
    return payload, (policy,)


@dataclasses.dataclass(frozen=True, slots=True)
class UniformRandomReferenceConfig:
    """Canonical uniform-random control configuration."""

    environment_kind: EnvironmentKind
    observation_dim: int
    n_actions: int = 2

    def __post_init__(self) -> None:
        _require_environment_shape(
            self.environment_kind,
            self.observation_dim,
            self.n_actions,
        )

    @classmethod
    def for_switching(
        cls,
        environment_config: SwitchingTwoStateConfig,
    ) -> UniformRandomReferenceConfig:
        _canonical_switching_environment(environment_config)
        return cls(
            environment_kind="switching_two_state",
            observation_dim=2,
        )

    @classmethod
    def for_riverswim(
        cls,
        environment_config: RiverSwimConfig,
    ) -> UniformRandomReferenceConfig:
        payload, _ = _canonical_riverswim_environment(environment_config)
        return cls(
            environment_kind="riverswim",
            observation_dim=cast(int, payload["n_states"]),
        )

    def to_config(self) -> dict[str, Any]:
        return {
            "schema": REFERENCE_CONTROL_CONFIG_SCHEMA,
            "algorithm": "uniform_random",
            "environment_kind": self.environment_kind,
            "observation_dim": self.observation_dim,
            "n_actions": self.n_actions,
            "action_distribution": "uniform",
            "rng_schedule": "jax_fold_in_uint64_decision_index.preview1",
            "boundary_mode": "continuing_unit_discount_only",
        }


@dataclasses.dataclass(frozen=True, slots=True)
class DifferentialSARSAReferenceConfig:
    """Canonical linear differential-SARSA control configuration."""

    environment_kind: EnvironmentKind
    observation_dim: int
    n_actions: int = 2
    q_step_size: float = 0.1
    average_reward_step_size: float = 0.01
    trace_decay: float = 0.0
    epsilon_start: float = 0.5
    epsilon_end: float = 0.02
    epsilon_decay_steps: int = 2500
    use_bias: bool = True

    def __post_init__(self) -> None:
        _require_environment_shape(
            self.environment_kind,
            self.observation_dim,
            self.n_actions,
        )
        object.__setattr__(
            self,
            "q_step_size",
            _finite_float(
                self.q_step_size,
                name="q_step_size",
                minimum=0.0,
                maximum=float("inf"),
            ),
        )
        object.__setattr__(
            self,
            "average_reward_step_size",
            _finite_float(
                self.average_reward_step_size,
                name="average_reward_step_size",
                minimum=0.0,
                maximum=float("inf"),
            ),
        )
        for name in ("trace_decay", "epsilon_start", "epsilon_end"):
            object.__setattr__(
                self,
                name,
                _finite_float(
                    getattr(self, name),
                    name=name,
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
        object.__setattr__(
            self,
            "epsilon_decay_steps",
            _nonnegative_int(self.epsilon_decay_steps, name="epsilon_decay_steps"),
        )
        if not isinstance(self.use_bias, bool):
            raise ValueError("use_bias must be boolean")

    @classmethod
    def for_switching(
        cls,
        environment_config: SwitchingTwoStateConfig,
        **overrides: Any,
    ) -> DifferentialSARSAReferenceConfig:
        _canonical_switching_environment(environment_config)
        return cls(
            environment_kind="switching_two_state",
            observation_dim=2,
            **overrides,
        )

    @classmethod
    def for_riverswim(
        cls,
        environment_config: RiverSwimConfig,
        **overrides: Any,
    ) -> DifferentialSARSAReferenceConfig:
        payload, _ = _canonical_riverswim_environment(environment_config)
        return cls(
            environment_kind="riverswim",
            observation_dim=cast(int, payload["n_states"]),
            **overrides,
        )

    def core_config(self) -> DifferentialSARSAConfig:
        return DifferentialSARSAConfig(
            n_actions=self.n_actions,
            q_step_size=self.q_step_size,
            average_reward_step_size=self.average_reward_step_size,
            trace_decay=self.trace_decay,
            epsilon_start=self.epsilon_start,
            epsilon_end=self.epsilon_end,
            epsilon_decay_steps=self.epsilon_decay_steps,
            use_bias=self.use_bias,
        )

    def to_config(self) -> dict[str, Any]:
        return {
            "schema": REFERENCE_CONTROL_CONFIG_SCHEMA,
            "algorithm": "differential_sarsa",
            "environment_kind": self.environment_kind,
            "observation_dim": self.observation_dim,
            "n_actions": self.n_actions,
            "q_step_size": self.q_step_size,
            "average_reward_step_size": self.average_reward_step_size,
            "trace_decay": self.trace_decay,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "use_bias": self.use_bias,
            "rng_schedule": "owned_differential_sarsa_key.preview1",
            "boundary_mode": "continuing_unit_discount_only",
        }


@dataclasses.dataclass(frozen=True, slots=True)
class DiscountedSARSAReferenceConfig:
    """Canonical discounted continuing-SARSA control configuration."""

    environment_kind: EnvironmentKind
    observation_dim: int
    n_actions: int = 2
    gamma: float = 0.9
    epsilon_start: float = 0.5
    epsilon_end: float = 0.02
    epsilon_decay_steps: int = 2500
    hidden_sizes: tuple[int, ...] = ()
    step_size: float = 0.05
    sparsity: float = 0.0
    leaky_relu_slope: float = 0.01
    use_layer_norm: bool = False
    lamda: float = 0.0

    def __post_init__(self) -> None:
        _require_environment_shape(
            self.environment_kind,
            self.observation_dim,
            self.n_actions,
        )
        gamma = _finite_float(
            self.gamma,
            name="gamma",
            minimum=0.0,
            maximum=1.0,
        )
        if gamma >= 1.0:
            raise ValueError("discounted SARSA gamma must be strictly less than one")
        object.__setattr__(self, "gamma", gamma)
        for name in ("epsilon_start", "epsilon_end", "sparsity", "lamda"):
            object.__setattr__(
                self,
                name,
                _finite_float(
                    getattr(self, name),
                    name=name,
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
        if self.sparsity >= 1.0:
            raise ValueError("sparsity must be strictly less than one")
        object.__setattr__(
            self,
            "step_size",
            _finite_float(
                self.step_size,
                name="step_size",
                minimum=0.0,
                maximum=float("inf"),
            ),
        )
        object.__setattr__(
            self,
            "leaky_relu_slope",
            _finite_float(
                self.leaky_relu_slope,
                name="leaky_relu_slope",
                minimum=0.0,
                maximum=float("inf"),
            ),
        )
        object.__setattr__(
            self,
            "epsilon_decay_steps",
            _nonnegative_int(self.epsilon_decay_steps, name="epsilon_decay_steps"),
        )
        hidden_sizes = tuple(self.hidden_sizes)
        if any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in hidden_sizes
        ):
            raise ValueError("hidden_sizes must contain only positive integers")
        object.__setattr__(self, "hidden_sizes", hidden_sizes)
        if not isinstance(self.use_layer_norm, bool):
            raise ValueError("use_layer_norm must be boolean")

    @classmethod
    def for_switching(
        cls,
        environment_config: SwitchingTwoStateConfig,
        **overrides: Any,
    ) -> DiscountedSARSAReferenceConfig:
        _canonical_switching_environment(environment_config)
        return cls(
            environment_kind="switching_two_state",
            observation_dim=2,
            **overrides,
        )

    @classmethod
    def for_riverswim(
        cls,
        environment_config: RiverSwimConfig,
        **overrides: Any,
    ) -> DiscountedSARSAReferenceConfig:
        payload, _ = _canonical_riverswim_environment(environment_config)
        return cls(
            environment_kind="riverswim",
            observation_dim=cast(int, payload["n_states"]),
            **overrides,
        )

    def core_config(self) -> SARSAConfig:
        return SARSAConfig(  # type: ignore[call-arg]
            n_actions=self.n_actions,
            gamma=self.gamma,
            epsilon_start=self.epsilon_start,
            epsilon_end=self.epsilon_end,
            epsilon_decay_steps=self.epsilon_decay_steps,
        )

    def to_config(self) -> dict[str, Any]:
        return {
            "schema": REFERENCE_CONTROL_CONFIG_SCHEMA,
            "algorithm": "discounted_sarsa",
            "environment_kind": self.environment_kind,
            "observation_dim": self.observation_dim,
            "n_actions": self.n_actions,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "hidden_sizes": list(self.hidden_sizes),
            "step_size": self.step_size,
            "sparsity": self.sparsity,
            "leaky_relu_slope": self.leaky_relu_slope,
            "use_layer_norm": self.use_layer_norm,
            "lamda": self.lamda,
            "rng_schedule": "owned_sarsa_key.preview1",
            "boundary_mode": "continuing_unit_discount_only",
        }


@dataclasses.dataclass(frozen=True, slots=True)
class AnalyticOracleReferenceConfig:
    """Exact environment-bound privileged analytic policy configuration."""

    environment_kind: EnvironmentKind
    observation_dim: int
    n_actions: int
    phase_length: int | None
    policies: tuple[tuple[int, ...], ...]
    environment_config_json: str = dataclasses.field(repr=False)

    def __post_init__(self) -> None:
        _require_environment_shape(
            self.environment_kind,
            self.observation_dim,
            self.n_actions,
        )
        try:
            decoded = json.loads(self.environment_config_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("oracle environment config must be canonical JSON") from exc
        if (
            not isinstance(decoded, dict)
            or _canonical_json(decoded) != self.environment_config_json
        ):
            raise ValueError("oracle environment config must use canonical JSON")
        if self.environment_kind == "switching_two_state":
            try:
                switching_source = SwitchingTwoStateConfig(  # type: ignore[call-arg]
                    phase_length=decoded["phase_length"],
                    payoffs_a=tuple(tuple(row) for row in decoded["payoffs_a"]),
                    payoffs_b=tuple(tuple(row) for row in decoded["payoffs_b"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("oracle switching config is malformed") from exc
            expected, phase_length, policies = _canonical_switching_environment(
                switching_source
            )
            if set(decoded) != {"phase_length", "payoffs_a", "payoffs_b"}:
                raise ValueError("oracle switching config has unknown fields")
            if decoded != expected or self.phase_length != phase_length:
                raise ValueError("oracle switching config is not canonical")
        else:
            try:
                river_source = RiverSwimConfig(  # type: ignore[call-arg]
                    n_states=decoded["n_states"],
                    p_right_up=decoded["p_right_up"],
                    p_right_down=decoded["p_right_down"],
                    reward_left=decoded["reward_left"],
                    reward_right=decoded["reward_right"],
                    initial_state=decoded["initial_state"],
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("oracle RiverSwim config is malformed") from exc
            expected, policies = _canonical_riverswim_environment(river_source)
            if set(decoded) != {
                "n_states",
                "p_right_up",
                "p_right_down",
                "reward_left",
                "reward_right",
                "initial_state",
            }:
                raise ValueError("oracle RiverSwim config has unknown fields")
            if decoded != expected or self.phase_length is not None:
                raise ValueError("oracle RiverSwim config is not canonical")
        if self.policies != policies:
            raise ValueError("oracle policies do not match the exact environment config")
        if any(
            len(policy) != self.observation_dim
            or any(action < 0 or action >= self.n_actions for action in policy)
            for policy in self.policies
        ):
            raise ValueError("oracle policy has an invalid shape or action")

    @classmethod
    def for_switching(
        cls,
        environment_config: SwitchingTwoStateConfig,
    ) -> AnalyticOracleReferenceConfig:
        payload, phase_length, policies = _canonical_switching_environment(
            environment_config
        )
        return cls(
            environment_kind="switching_two_state",
            observation_dim=2,
            n_actions=2,
            phase_length=phase_length,
            policies=policies,
            environment_config_json=_canonical_json(payload),
        )

    @classmethod
    def for_riverswim(
        cls,
        environment_config: RiverSwimConfig,
    ) -> AnalyticOracleReferenceConfig:
        payload, policies = _canonical_riverswim_environment(environment_config)
        return cls(
            environment_kind="riverswim",
            observation_dim=cast(int, payload["n_states"]),
            n_actions=2,
            phase_length=None,
            policies=policies,
            environment_config_json=_canonical_json(payload),
        )

    @property
    def environment_config(self) -> dict[str, Any]:
        decoded = json.loads(self.environment_config_json)
        assert isinstance(decoded, dict)
        return decoded

    @property
    def environment_config_sha256(self) -> str:
        return canonical_config_sha256(self.environment_config)

    def to_config(self) -> dict[str, Any]:
        return {
            "schema": REFERENCE_CONTROL_CONFIG_SCHEMA,
            "algorithm": "analytic_oracle",
            "environment_kind": self.environment_kind,
            "observation_dim": self.observation_dim,
            "n_actions": self.n_actions,
            "phase_length": self.phase_length,
            "policies": [list(policy) for policy in self.policies],
            "environment_config": self.environment_config,
            "environment_config_sha256": self.environment_config_sha256,
            "privileged": True,
            "privileged_environment_model": True,
            "privileged_inputs": (
                ["environment_dynamics", "phase_schedule"]
                if self.environment_kind == "switching_two_state"
                else ["environment_dynamics"]
            ),
            "tie_break": "lowest_action_index",
            "rng_schedule": "none_deterministic",
            "boundary_mode": "continuing_unit_discount_only",
        }


ControlConfig = (
    UniformRandomReferenceConfig
    | AnalyticOracleReferenceConfig
    | DifferentialSARSAReferenceConfig
    | DiscountedSARSAReferenceConfig
)
ControlAgentState = DifferentialSARSAState | SARSAState | None


@dataclasses.dataclass(frozen=True, slots=True)
class ReferenceLifeControlState:
    """Immutable process-local envelope shared by the four control adapters."""

    schema: str
    manifest_id: str
    config_sha256: str
    lifecycle_id: str
    decision_index: int
    current_observation_id: str | None
    current_observation: ArrayValue | None
    current_action: ArrayValue | None
    random_key: Array | None
    agent_state: ControlAgentState
    _owner_token: object = dataclasses.field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.schema, str) or _SAFE_ID.fullmatch(self.schema) is None:
            raise ValueError("control state schema must be a safe identifier")
        if not isinstance(self.manifest_id, str) or _SHA256.fullmatch(self.manifest_id) is None:
            raise ValueError("control state manifest_id must be a SHA-256 digest")
        if (
            not isinstance(self.config_sha256, str)
            or _SHA256.fullmatch(self.config_sha256) is None
        ):
            raise ValueError("control state config_sha256 must be a SHA-256 digest")
        if (
            not isinstance(self.lifecycle_id, str)
            or len(self.lifecycle_id) > _MAX_ID_LENGTH
            or _SAFE_ID.fullmatch(self.lifecycle_id) is None
        ):
            raise ValueError("control state lifecycle_id must be a safe identifier")
        if (
            isinstance(self.decision_index, bool)
            or not isinstance(self.decision_index, int)
            or self.decision_index < 0
            or self.decision_index > MAX_DECISION_INDEX
        ):
            raise ValueError("control state decision_index must be uint64")
        armed_fields = (
            self.current_observation_id,
            self.current_observation,
            self.current_action,
        )
        if any(value is None for value in armed_fields) and not all(
            value is None for value in armed_fields
        ):
            raise ValueError("control state decision cache must be wholly fresh or armed")
        if self.current_observation_id is not None and (
            not isinstance(self.current_observation_id, str)
            or _SAFE_ID.fullmatch(self.current_observation_id) is None
        ):
            raise ValueError("current_observation_id must be a safe identifier")
        if self.current_observation is not None and not isinstance(
            self.current_observation, ArrayValue
        ):
            raise ValueError("current_observation must be an ArrayValue")
        if self.current_action is not None and not isinstance(self.current_action, ArrayValue):
            raise ValueError("current_action must be an ArrayValue")

    def __reduce__(self) -> Any:
        raise TypeError(
            "reference-life control state is process-local and has no checkpoint codec"
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ReferenceControlResourceUsage:
    """Primitive resource counts for one in-memory control state."""

    array_leaves: int
    array_elements: int
    persistent_bytes: int
    floating_array_leaves: int


def control_state_resource_usage(
    state: ReferenceLifeControlState,
) -> ReferenceControlResourceUsage:
    """Count numeric state leaves without making a checkpoint claim."""

    if not isinstance(state, ReferenceLifeControlState):
        raise ValueError("state must be a ReferenceLifeControlState")
    arrays: list[np.ndarray[Any, Any]] = []
    for encoded in (state.current_observation, state.current_action):
        if encoded is not None:
            arrays.append(encoded.to_numpy())
    roots: tuple[Any, ...] = (state.random_key, state.agent_state)
    for root in roots:
        if root is None:
            continue
        for leaf in jax.tree.leaves(root):
            try:
                dtype = getattr(leaf, "dtype", None)
                value = (
                    np.asarray(jr.key_data(leaf))
                    if dtype is not None and str(dtype).startswith("key<")
                    else np.asarray(leaf)
                )
            except (TypeError, ValueError):
                continue
            if value.dtype.kind in {"b", "f", "i", "u", "c"}:
                arrays.append(value)
    return ReferenceControlResourceUsage(
        array_leaves=len(arrays),
        array_elements=sum(int(value.size) for value in arrays),
        persistent_bytes=sum(int(value.nbytes) for value in arrays),
        floating_array_leaves=sum(value.dtype.kind in {"f", "c"} for value in arrays),
    )


def _observation_spec(observation_dim: int) -> SpaceSpec:
    return SpaceSpec.box(
        shape=(observation_dim,),
        dtype="float32",
        low=(0.0,) * observation_dim,
        high=(1.0,) * observation_dim,
        semantic_id=REFERENCE_CONTROL_OBSERVATION_SEMANTIC_ID,
    )


def _action_spec(n_actions: int) -> SpaceSpec:
    return SpaceSpec.discrete(
        cardinality=n_actions,
        dtype="int32",
        semantic_id=REFERENCE_CONTROL_ACTION_SEMANTIC_ID,
    )


def _require_prng_key(key: Any, *, name: str) -> None:
    try:
        words = np.asarray(jr.key_data(key))
    except (TypeError, ValueError) as exc:
        raise DecisionOwnershipError(f"{name} must be a scalar JAX PRNG key") from exc
    if words.shape != (2,) or words.dtype != np.dtype(np.uint32):
        raise DecisionOwnershipError(f"{name} must be a scalar JAX PRNG key")


def _float32_scalar(value: Any, *, name: str) -> np.float32:
    try:
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            converted = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DecisionOwnershipError(f"{name} must be a finite float32 scalar") from exc
    if converted.shape != () or not bool(np.isfinite(converted)):
        raise DecisionOwnershipError(f"{name} must be a finite float32 scalar")
    return np.float32(converted)


def _tree_exactly_equal(left: Any, right: Any) -> bool:
    left_leaves, left_structure = jax.tree.flatten(left)
    right_leaves, right_structure = jax.tree.flatten(right)
    if cast(Any, left_structure) != right_structure or len(left_leaves) != len(
        right_leaves
    ):
        return False
    return all(
        np.asarray(left_leaf).dtype == np.asarray(right_leaf).dtype
        and np.asarray(left_leaf).shape == np.asarray(right_leaf).shape
        and np.array_equal(np.asarray(left_leaf), np.asarray(right_leaf))
        for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True)
    )


def _tree_finite(tree: Any) -> bool:
    for leaf in jax.tree.leaves(tree):
        try:
            array = np.asarray(leaf)
        except (TypeError, ValueError):
            continue
        if array.dtype.kind in {"f", "c"} and not bool(np.all(np.isfinite(array))):
            return False
    return True


def _counter_words(index: int) -> np.ndarray[Any, np.dtype[np.uint32]]:
    return np.asarray(((index >> 32) & 0xFFFFFFFF, index & 0xFFFFFFFF), dtype=np.uint32)


class _BaseReferenceControlAdapter:
    """Shared exact-dispatch ownership and continuing-outcome machinery."""

    _agent: DifferentialSARSAAgent | SARSAAgent | None
    _config: ControlConfig
    _frozen: bool
    _manifest: AgentManifest
    _owner_token: object
    _state_schema: str

    __slots__ = (
        "_agent",
        "_config",
        "_frozen",
        "_manifest",
        "_owner_token",
        "_state_schema",
    )

    def __init__(
        self,
        config: ControlConfig,
        *,
        implementation_id: str,
        state_schema: str,
        agent: DifferentialSARSAAgent | SARSAAgent | None,
    ) -> None:
        object.__setattr__(self, "_frozen", False)
        object.__setattr__(self, "_config", config)
        object.__setattr__(self, "_state_schema", state_schema)
        object.__setattr__(self, "_agent", agent)
        object.__setattr__(self, "_owner_token", object())
        object.__setattr__(
            self,
            "_manifest",
            AgentManifest.from_config(
                schema=REFERENCE_AGENT_MANIFEST_SCHEMA,
                implementation_id=implementation_id,
                state_schema=state_schema,
                config=config.to_config(),
                observation_spec=_observation_spec(config.observation_dim),
                action_spec=_action_spec(config.n_actions),
                capabilities=AgentCapabilities(dispatch_rebinding=False),
            ),
        )
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("reference-life control adapter is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("reference-life control adapter is immutable")
        object.__delattr__(self, name)

    def __reduce__(self) -> Any:
        raise TypeError("reference-life control adapter is process-local")

    @property
    def manifest(self) -> AgentManifest:
        return self._manifest

    @property
    def config(self) -> ControlConfig:
        return self._config

    @property
    def max_accepted_events(self) -> int:
        return REFERENCE_CONTROL_MAX_ACCEPTED_EVENTS

    def _base_state(
        self,
        *,
        lifecycle_id: str,
        random_key: Array | None,
        agent_state: ControlAgentState,
    ) -> ReferenceLifeControlState:
        return ReferenceLifeControlState(
            schema=self._state_schema,
            manifest_id=self.manifest.manifest_id,
            config_sha256=self.manifest.config_sha256,
            lifecycle_id=lifecycle_id,
            decision_index=0,
            current_observation_id=None,
            current_observation=None,
            current_action=None,
            random_key=random_key,
            agent_state=agent_state,
            _owner_token=self._owner_token,
        )

    def _require_bound_state(self, state: ReferenceLifeControlState) -> None:
        if not isinstance(state, ReferenceLifeControlState):
            raise DecisionOwnershipError("state must be a ReferenceLifeControlState")
        if state._owner_token is not self._owner_token:
            raise DecisionOwnershipError("control state belongs to another adapter owner")
        if state.schema != self._state_schema:
            raise DecisionOwnershipError("control state has another algorithm schema")
        if state.manifest_id != self.manifest.manifest_id:
            raise DecisionOwnershipError("control state belongs to another manifest")
        if state.config_sha256 != self.manifest.config_sha256:
            raise DecisionOwnershipError("control state belongs to another configuration")
        if state.decision_index > self.max_accepted_events:
            raise DecisionOwnershipError("control state exceeds adapter capacity")

    def _one_hot(self, value: Any) -> np.ndarray[Any, np.dtype[np.float32]]:
        try:
            encoded = self.manifest.observation_spec.encode(value)
        except (TypeError, ValueError) as exc:
            raise DecisionOwnershipError(
                "observation does not match the control codec"
            ) from exc
        observation = encoded.to_numpy()
        if (
            observation.dtype != np.dtype(np.float32)
            or observation.shape != (self.config.observation_dim,)
            or not np.all((observation == 0.0) | (observation == 1.0))
            or int(np.count_nonzero(observation)) != 1
        ):
            raise DecisionOwnershipError("control observation must be exact float32 one-hot")
        return observation

    def _action_index(self, value: Any) -> int:
        try:
            encoded = self.manifest.action_spec.encode(value)
        except (TypeError, ValueError) as exc:
            raise DecisionOwnershipError("action does not match the control codec") from exc
        action = encoded.to_numpy()
        if action.shape != () or action.dtype != np.dtype(np.int32):
            raise DecisionOwnershipError("control action must be scalar int32")
        return int(action)

    def _initial_payload(
        self,
        key: Array,
    ) -> tuple[Array | None, ControlAgentState]:
        raise NotImplementedError

    def _start_payload(
        self,
        state: ReferenceLifeControlState,
        observation: np.ndarray[Any, np.dtype[np.float32]],
    ) -> tuple[Array | None, ControlAgentState, int]:
        raise NotImplementedError

    def _validate_payload(
        self,
        state: ReferenceLifeControlState,
        observation: np.ndarray[Any, np.dtype[np.float32]],
        action: int,
    ) -> None:
        raise NotImplementedError

    def _advance_payload(
        self,
        state: ReferenceLifeControlState,
        *,
        reward: np.float32,
        discount: np.float32,
        next_observation: np.ndarray[Any, np.dtype[np.float32]],
        next_index: int,
    ) -> tuple[Array | None, ControlAgentState, int, bool]:
        raise NotImplementedError

    def init(self, key: Array, *, lifecycle_id: str) -> ReferenceLifeControlState:
        _require_prng_key(key, name="initial control key")
        if (
            not isinstance(lifecycle_id, str)
            or len(lifecycle_id) > _MAX_ID_LENGTH
            or _SAFE_ID.fullmatch(lifecycle_id) is None
        ):
            raise ValueError("lifecycle_id must be a safe identifier")
        random_key, agent_state = self._initial_payload(key)
        return self._base_state(
            lifecycle_id=lifecycle_id,
            random_key=random_key,
            agent_state=agent_state,
        )

    def start(
        self,
        state: ReferenceLifeControlState,
        *,
        observation_id: str,
        observation: Any,
    ) -> tuple[ReferenceLifeControlState, Decision]:
        self._require_bound_state(state)
        if (
            state.decision_index != 0
            or state.current_observation_id is not None
            or state.current_observation is not None
            or state.current_action is not None
        ):
            raise DecisionOwnershipError("start requires a fresh control state")
        encoded_observation = self.manifest.observation_spec.encode(observation)
        one_hot = self._one_hot(encoded_observation)
        random_key, agent_state, action = self._start_payload(state, one_hot)
        encoded_action = self.manifest.action_spec.encode(np.asarray(action, dtype=np.int32))
        started = dataclasses.replace(
            state,
            current_observation_id=observation_id,
            current_observation=encoded_observation,
            current_action=encoded_action,
            random_key=random_key,
            agent_state=agent_state,
        )
        decision = self.current_decision(started)
        return started, decision

    def current_decision(self, state: ReferenceLifeControlState) -> Decision:
        self._require_bound_state(state)
        if (
            state.current_observation_id is None
            or state.current_observation is None
            or state.current_action is None
        ):
            raise DecisionOwnershipError("fresh control state has no armed decision")
        observation = self._one_hot(state.current_observation)
        action = self._action_index(state.current_action)
        self._validate_payload(state, observation, action)
        return self.manifest.make_decision(
            lifecycle_id=state.lifecycle_id,
            decision_index=state.decision_index,
            observation_id=state.current_observation_id,
            observation=state.current_observation,
            proposed_action=state.current_action,
            armed=True,
        )

    def validate_state(self, state: ReferenceLifeControlState) -> None:
        self.current_decision(state)

    def settle_dispatch(
        self,
        state: ReferenceLifeControlState,
        authorization: DispatchAuthorization,
    ) -> tuple[ReferenceLifeControlState, Literal[False]]:
        if not isinstance(authorization, DispatchAuthorization):
            raise DecisionOwnershipError("settlement requires a DispatchAuthorization")
        current = self.current_decision(state)
        if authorization.decision != current:
            raise DecisionOwnershipError(
                "authorization decision does not match the current control cache"
            )
        if authorization.status is not AuthorizationStatus.EXACT:
            raise DecisionOwnershipError("control adapters allow exact authorization only")
        if authorization.authorized_action != current.proposed_action:
            raise DecisionOwnershipError("authorized action does not match the control action")
        return state, False

    @staticmethod
    def _rejected(state: Any, reason: str) -> ReferenceAgentUpdate:
        return ReferenceAgentUpdate(
            state=state,
            next_decision=None,
            accepted=False,
            parameters_changed=False,
            rejection_reason=reason,
        )

    def _require_current_transaction(
        self,
        state: ReferenceLifeControlState,
        transaction: Transaction,
    ) -> tuple[np.float32, np.float32, np.ndarray[Any, np.dtype[np.float32]]]:
        if not isinstance(transaction, Transaction):
            raise DecisionOwnershipError("outcome must be a Transaction")
        current = self.current_decision(state)
        if transaction.decision != current:
            raise DecisionOwnershipError("outcome does not own the current control decision")
        authorization = transaction.receipt.dispatch.authorization
        if (
            authorization.status is not AuthorizationStatus.EXACT
            or transaction.receipt.dispatch.status is not DispatchStatus.EXACT
            or transaction.receipt.applied_action != current.proposed_action
        ):
            raise DecisionOwnershipError("outcome does not preserve exact action ownership")
        if transaction.is_boundary or transaction.autoreset:
            raise DecisionOwnershipError("control adapters support continuing outcomes only")
        if transaction.discount != 1.0:
            raise DecisionOwnershipError(
                "selected continuing controls require unit environment discount"
            )
        if (
            transaction.next_decision_observation_id is None
            or transaction.next_decision_observation is None
        ):
            raise DecisionOwnershipError("continuing outcome lacks a next observation")
        if state.decision_index >= self.max_accepted_events:
            raise DecisionOwnershipError("control adapter capacity is exhausted")
        reward = _float32_scalar(transaction.reward, name="reward")
        discount = _float32_scalar(transaction.discount, name="discount")
        next_observation = self._one_hot(transaction.next_decision_observation)
        return reward, discount, next_observation

    def apply_outcome(
        self,
        state: Any,
        transaction: Transaction,
    ) -> ReferenceAgentUpdate:
        """Stage one update, preserving the exact input state on every failure."""

        if not isinstance(state, ReferenceLifeControlState):
            return self._rejected(state, "state must be a ReferenceLifeControlState")
        try:
            reward, discount, next_observation = self._require_current_transaction(
                state,
                transaction,
            )
            assert transaction.next_decision_observation_id is not None
            assert transaction.next_decision_observation is not None
            next_index = state.decision_index + 1
            random_key, agent_state, action, parameters_changed = self._advance_payload(
                state,
                reward=reward,
                discount=discount,
                next_observation=next_observation,
                next_index=next_index,
            )
            candidate = dataclasses.replace(
                state,
                decision_index=next_index,
                current_observation_id=transaction.next_decision_observation_id,
                current_observation=transaction.next_decision_observation,
                current_action=self.manifest.action_spec.encode(
                    np.asarray(action, dtype=np.int32)
                ),
                random_key=random_key,
                agent_state=agent_state,
            )
            decision = self.current_decision(candidate)
        except Exception as exc:
            return self._rejected(state, f"control transition rejected: {exc}")
        return ReferenceAgentUpdate(
            state=candidate,
            next_decision=decision,
            accepted=True,
            parameters_changed=parameters_changed,
            rejection_reason=None,
        )


class UniformRandomReferenceAdapter(_BaseReferenceControlAdapter):
    """Owner-bound uniform random primitive-action control."""

    __slots__ = ()

    def __init__(self, config: UniformRandomReferenceConfig) -> None:
        if not isinstance(config, UniformRandomReferenceConfig):
            raise ValueError("uniform-random adapter config has the wrong type")
        super().__init__(
            config,
            implementation_id=UNIFORM_RANDOM_IMPLEMENTATION_ID,
            state_schema=UNIFORM_RANDOM_STATE_SCHEMA,
            agent=None,
        )

    @property
    def config(self) -> UniformRandomReferenceConfig:
        return cast(UniformRandomReferenceConfig, super().config)

    def _random_action(self, key: Array, decision_index: int) -> int:
        _require_prng_key(key, name="uniform-random root key")
        scheduled = jr.fold_in(key, _RANDOM_DOMAIN)
        scheduled = jr.fold_in(scheduled, (decision_index >> 32) & 0xFFFFFFFF)
        scheduled = jr.fold_in(scheduled, decision_index & 0xFFFFFFFF)
        return int(jr.randint(scheduled, (), 0, self.config.n_actions, dtype=jnp.int32))

    def _initial_payload(
        self,
        key: Array,
    ) -> tuple[Array | None, ControlAgentState]:
        return key, None

    def _start_payload(
        self,
        state: ReferenceLifeControlState,
        observation: np.ndarray[Any, np.dtype[np.float32]],
    ) -> tuple[Array | None, ControlAgentState, int]:
        del observation
        if state.random_key is None or state.agent_state is not None:
            raise DecisionOwnershipError("uniform-random fresh state is inconsistent")
        return state.random_key, None, self._random_action(state.random_key, 0)

    def _validate_payload(
        self,
        state: ReferenceLifeControlState,
        observation: np.ndarray[Any, np.dtype[np.float32]],
        action: int,
    ) -> None:
        del observation
        if state.random_key is None or state.agent_state is not None:
            raise DecisionOwnershipError("uniform-random state payload is inconsistent")
        expected = self._random_action(state.random_key, state.decision_index)
        if action != expected:
            raise DecisionOwnershipError("uniform-random action cache does not match its key")

    def _advance_payload(
        self,
        state: ReferenceLifeControlState,
        *,
        reward: np.float32,
        discount: np.float32,
        next_observation: np.ndarray[Any, np.dtype[np.float32]],
        next_index: int,
    ) -> tuple[Array | None, ControlAgentState, int, bool]:
        del reward, discount, next_observation
        assert state.random_key is not None
        return state.random_key, None, self._random_action(state.random_key, next_index), False


class AnalyticOracleReferenceAdapter(_BaseReferenceControlAdapter):
    """Privileged phase-aware/stationary exact analytic control."""

    __slots__ = ()

    def __init__(self, config: AnalyticOracleReferenceConfig) -> None:
        if not isinstance(config, AnalyticOracleReferenceConfig):
            raise ValueError("analytic-oracle adapter config has the wrong type")
        super().__init__(
            config,
            implementation_id=ANALYTIC_ORACLE_IMPLEMENTATION_ID,
            state_schema=ANALYTIC_ORACLE_STATE_SCHEMA,
            agent=None,
        )

    @property
    def config(self) -> AnalyticOracleReferenceConfig:
        return cast(AnalyticOracleReferenceConfig, super().config)

    def _oracle_action(
        self,
        observation: np.ndarray[Any, np.dtype[np.float32]],
        decision_index: int,
    ) -> int:
        state_index = int(np.argmax(observation))
        if self.config.environment_kind == "switching_two_state":
            assert self.config.phase_length is not None
            phase = (decision_index // self.config.phase_length) % len(
                self.config.policies
            )
        else:
            phase = 0
        return self.config.policies[phase][state_index]

    def _initial_payload(
        self,
        key: Array,
    ) -> tuple[Array | None, ControlAgentState]:
        del key
        return None, None

    def _start_payload(
        self,
        state: ReferenceLifeControlState,
        observation: np.ndarray[Any, np.dtype[np.float32]],
    ) -> tuple[Array | None, ControlAgentState, int]:
        if state.random_key is not None or state.agent_state is not None:
            raise DecisionOwnershipError("analytic-oracle fresh state is inconsistent")
        return None, None, self._oracle_action(observation, 0)

    def _validate_payload(
        self,
        state: ReferenceLifeControlState,
        observation: np.ndarray[Any, np.dtype[np.float32]],
        action: int,
    ) -> None:
        if state.random_key is not None or state.agent_state is not None:
            raise DecisionOwnershipError("analytic-oracle state payload is inconsistent")
        if action != self._oracle_action(observation, state.decision_index):
            raise DecisionOwnershipError("analytic-oracle action cache is inconsistent")

    def _advance_payload(
        self,
        state: ReferenceLifeControlState,
        *,
        reward: np.float32,
        discount: np.float32,
        next_observation: np.ndarray[Any, np.dtype[np.float32]],
        next_index: int,
    ) -> tuple[Array | None, ControlAgentState, int, bool]:
        del state, reward, discount
        return None, None, self._oracle_action(next_observation, next_index), False


class DifferentialSARSAReferenceAdapter(_BaseReferenceControlAdapter):
    """Exact-dispatch wrapper for the linear differential-SARSA control."""

    __slots__ = ()

    def __init__(self, config: DifferentialSARSAReferenceConfig) -> None:
        if not isinstance(config, DifferentialSARSAReferenceConfig):
            raise ValueError("differential-SARSA adapter config has the wrong type")
        super().__init__(
            config,
            implementation_id=DIFFERENTIAL_SARSA_IMPLEMENTATION_ID,
            state_schema=DIFFERENTIAL_SARSA_STATE_SCHEMA,
            agent=DifferentialSARSAAgent(config.core_config()),
        )

    @property
    def config(self) -> DifferentialSARSAReferenceConfig:
        return cast(DifferentialSARSAReferenceConfig, super().config)

    @property
    def _differential_agent(self) -> DifferentialSARSAAgent:
        assert isinstance(self._agent, DifferentialSARSAAgent)
        return self._agent

    def _initial_payload(
        self,
        key: Array,
    ) -> tuple[Array | None, ControlAgentState]:
        return None, self._differential_agent.init(self.config.observation_dim, key)

    def _start_payload(
        self,
        state: ReferenceLifeControlState,
        observation: np.ndarray[Any, np.dtype[np.float32]],
    ) -> tuple[Array | None, ControlAgentState, int]:
        if (
            not isinstance(state.agent_state, DifferentialSARSAState)
            or state.random_key is not None
        ):
            raise DecisionOwnershipError("differential-SARSA fresh state is inconsistent")
        started, action = self._differential_agent.start(
            state.agent_state,
            jnp.asarray(observation, dtype=jnp.float32),
        )
        return None, started, int(action)

    def _validate_payload(
        self,
        state: ReferenceLifeControlState,
        observation: np.ndarray[Any, np.dtype[np.float32]],
        action: int,
    ) -> None:
        learner = state.agent_state
        if not isinstance(learner, DifferentialSARSAState) or state.random_key is not None:
            raise DecisionOwnershipError("differential-SARSA state payload is inconsistent")
        expected_float_shapes = (
            ("q_weights", learner.q_weights, (self.config.n_actions, self.config.observation_dim)),
            ("q_bias", learner.q_bias, (self.config.n_actions,)),
            (
                "q_trace_weights",
                learner.q_trace_weights,
                (self.config.n_actions, self.config.observation_dim),
            ),
            ("q_trace_bias", learner.q_trace_bias, (self.config.n_actions,)),
            ("average_reward", learner.average_reward, ()),
            ("last_observation", learner.last_observation, (self.config.observation_dim,)),
            ("epsilon", learner.epsilon, ()),
        )
        for name, value, shape in expected_float_shapes:
            array = np.asarray(value)
            if array.shape != shape or array.dtype != np.dtype(np.float32):
                raise DecisionOwnershipError(
                    f"differential-SARSA {name} violates shape/dtype contract"
                )
        for name, value in (
            ("last_action", learner.last_action),
            ("step_count", learner.step_count),
        ):
            array = np.asarray(value)
            if array.shape != () or array.dtype != np.dtype(np.int32):
                raise DecisionOwnershipError(
                    f"differential-SARSA {name} must be scalar int32"
                )
        _require_prng_key(learner.rng_key, name="differential-SARSA key")
        if not _tree_finite(learner):
            raise DecisionOwnershipError("differential-SARSA state must be finite")
        if int(learner.step_count) != state.decision_index:
            raise DecisionOwnershipError("differential-SARSA counter does not match decision")
        if not np.array_equal(np.asarray(learner.step_words), _counter_words(state.decision_index)):
            raise DecisionOwnershipError("differential-SARSA exact counter is inconsistent")
        if not np.array_equal(np.asarray(learner.last_observation), observation):
            raise DecisionOwnershipError("differential-SARSA observation cache is inconsistent")
        if int(learner.last_action) != action:
            raise DecisionOwnershipError("differential-SARSA action cache is inconsistent")
        epsilon = float(learner.epsilon)
        if not 0.0 <= epsilon <= 1.0:
            raise DecisionOwnershipError("differential-SARSA epsilon is invalid")

    def _advance_payload(
        self,
        state: ReferenceLifeControlState,
        *,
        reward: np.float32,
        discount: np.float32,
        next_observation: np.ndarray[Any, np.dtype[np.float32]],
        next_index: int,
    ) -> tuple[Array | None, ControlAgentState, int, bool]:
        del next_index
        learner = state.agent_state
        assert isinstance(learner, DifferentialSARSAState)
        result = self._differential_agent.update(
            learner,
            jnp.asarray(reward, dtype=jnp.float32),
            jnp.asarray(next_observation, dtype=jnp.float32),
            discount=jnp.asarray(discount, dtype=jnp.float32),
        )
        if not bool(np.asarray(result.update_applied)):
            raise DecisionOwnershipError("differential-SARSA functional update was unavailable")
        old_parameters = (learner.q_weights, learner.q_bias, learner.average_reward)
        new_parameters = (
            result.state.q_weights,
            result.state.q_bias,
            result.state.average_reward,
        )
        return (
            None,
            result.state,
            int(result.action),
            not _tree_exactly_equal(old_parameters, new_parameters),
        )


class DiscountedSARSAReferenceAdapter(_BaseReferenceControlAdapter):
    """Exact-dispatch wrapper for discounted SARSA on continuing outcomes."""

    __slots__ = ()

    def __init__(self, config: DiscountedSARSAReferenceConfig) -> None:
        if not isinstance(config, DiscountedSARSAReferenceConfig):
            raise ValueError("discounted-SARSA adapter config has the wrong type")
        agent = SARSAAgent(
            sarsa_config=config.core_config(),
            hidden_sizes=config.hidden_sizes,
            step_size=config.step_size,
            sparsity=config.sparsity,
            leaky_relu_slope=config.leaky_relu_slope,
            use_layer_norm=config.use_layer_norm,
            lamda=config.lamda,
        )
        super().__init__(
            config,
            implementation_id=DISCOUNTED_SARSA_IMPLEMENTATION_ID,
            state_schema=DISCOUNTED_SARSA_STATE_SCHEMA,
            agent=agent,
        )

    @property
    def config(self) -> DiscountedSARSAReferenceConfig:
        return cast(DiscountedSARSAReferenceConfig, super().config)

    @property
    def _sarsa_agent(self) -> SARSAAgent:
        assert isinstance(self._agent, SARSAAgent)
        return self._agent

    def _initial_payload(
        self,
        key: Array,
    ) -> tuple[Array | None, ControlAgentState]:
        return None, self._sarsa_agent.init(self.config.observation_dim, key)

    def _start_payload(
        self,
        state: ReferenceLifeControlState,
        observation: np.ndarray[Any, np.dtype[np.float32]],
    ) -> tuple[Array | None, ControlAgentState, int]:
        if not isinstance(state.agent_state, SARSAState) or state.random_key is not None:
            raise DecisionOwnershipError("discounted-SARSA fresh state is inconsistent")
        observation_array = jnp.asarray(observation, dtype=jnp.float32)
        action, key = self._sarsa_agent.select_action(
            state.agent_state,
            observation_array,
        )
        started = state.agent_state.replace(  # type: ignore[attr-defined]
            last_action=action,
            last_observation=observation_array,
            rng_key=key,
        )
        return None, started, int(action)

    def _validate_payload(
        self,
        state: ReferenceLifeControlState,
        observation: np.ndarray[Any, np.dtype[np.float32]],
        action: int,
    ) -> None:
        learner = state.agent_state
        if not isinstance(learner, SARSAState) or state.random_key is not None:
            raise DecisionOwnershipError("discounted-SARSA state payload is inconsistent")
        last_observation = np.asarray(learner.last_observation)
        last_action = np.asarray(learner.last_action)
        step_count = np.asarray(learner.step_count)
        epsilon = np.asarray(learner.epsilon)
        if (
            last_observation.shape != (self.config.observation_dim,)
            or last_observation.dtype != np.dtype(np.float32)
        ):
            raise DecisionOwnershipError("discounted-SARSA observation cache is malformed")
        if last_action.shape != () or last_action.dtype != np.dtype(np.int32):
            raise DecisionOwnershipError("discounted-SARSA action cache is malformed")
        if step_count.shape != () or step_count.dtype != np.dtype(np.int32):
            raise DecisionOwnershipError("discounted-SARSA step_count must be scalar int32")
        if epsilon.shape != () or epsilon.dtype != np.dtype(np.float32):
            raise DecisionOwnershipError("discounted-SARSA epsilon must be scalar float32")
        _require_prng_key(learner.rng_key, name="discounted-SARSA key")
        if not _tree_finite(learner):
            raise DecisionOwnershipError("discounted-SARSA state must be finite")
        if int(step_count) != state.decision_index:
            raise DecisionOwnershipError("discounted-SARSA counter does not match decision")
        inner_count = np.asarray(learner.learner_state.step_count)
        inner_words = np.asarray(learner.learner_state.step_words)
        if (
            inner_count.shape != ()
            or inner_count.dtype != np.dtype(np.int32)
            or int(inner_count) != state.decision_index
            or not np.array_equal(inner_words, _counter_words(state.decision_index))
        ):
            raise DecisionOwnershipError("discounted-SARSA learner counter is inconsistent")
        if not np.array_equal(last_observation, observation):
            raise DecisionOwnershipError("discounted-SARSA observation cache is inconsistent")
        if int(last_action) != action:
            raise DecisionOwnershipError("discounted-SARSA action cache is inconsistent")
        if not 0.0 <= float(epsilon) <= 1.0:
            raise DecisionOwnershipError("discounted-SARSA epsilon is invalid")
        try:
            q_values = np.asarray(
                self._sarsa_agent.horde.predict(
                    learner.learner_state,
                    jnp.asarray(observation, dtype=jnp.float32),
                )[: self.config.n_actions]
            )
        except Exception as exc:
            raise DecisionOwnershipError(
                "discounted-SARSA learner structure is invalid"
            ) from exc
        if q_values.shape != (self.config.n_actions,) or not np.all(np.isfinite(q_values)):
            raise DecisionOwnershipError("discounted-SARSA action values are invalid")

    def _advance_payload(
        self,
        state: ReferenceLifeControlState,
        *,
        reward: np.float32,
        discount: np.float32,
        next_observation: np.ndarray[Any, np.dtype[np.float32]],
        next_index: int,
    ) -> tuple[Array | None, ControlAgentState, int, bool]:
        del discount, next_index
        learner = state.agent_state
        assert isinstance(learner, SARSAState)
        observation_array = jnp.asarray(next_observation, dtype=jnp.float32)
        next_action, key = self._sarsa_agent.select_action(learner, observation_array)
        selected = learner.replace(rng_key=key)  # type: ignore[attr-defined]
        result = self._sarsa_agent.update(
            selected,
            jnp.asarray(reward, dtype=jnp.float32),
            observation_array,
            jnp.asarray(False, dtype=jnp.bool_),
            next_action,
        )
        old_parameters = (
            learner.learner_state.trunk_params,
            learner.learner_state.head_params,
        )
        new_parameters = (
            result.state.learner_state.trunk_params,
            result.state.learner_state.head_params,
        )
        return (
            None,
            result.state,
            int(result.action),
            not _tree_exactly_equal(old_parameters, new_parameters),
        )


__all__ = [
    "ANALYTIC_ORACLE_IMPLEMENTATION_ID",
    "ANALYTIC_ORACLE_STATE_SCHEMA",
    "DIFFERENTIAL_SARSA_IMPLEMENTATION_ID",
    "DIFFERENTIAL_SARSA_STATE_SCHEMA",
    "DISCOUNTED_SARSA_IMPLEMENTATION_ID",
    "DISCOUNTED_SARSA_STATE_SCHEMA",
    "REFERENCE_CONTROL_ACTION_SEMANTIC_ID",
    "REFERENCE_CONTROL_CONFIG_SCHEMA",
    "REFERENCE_CONTROL_MAX_ACCEPTED_EVENTS",
    "REFERENCE_CONTROL_OBSERVATION_SEMANTIC_ID",
    "UNIFORM_RANDOM_IMPLEMENTATION_ID",
    "UNIFORM_RANDOM_STATE_SCHEMA",
    "AnalyticOracleReferenceAdapter",
    "AnalyticOracleReferenceConfig",
    "DifferentialSARSAReferenceAdapter",
    "DifferentialSARSAReferenceConfig",
    "DiscountedSARSAReferenceAdapter",
    "DiscountedSARSAReferenceConfig",
    "ReferenceControlResourceUsage",
    "ReferenceLifeControlState",
    "UniformRandomReferenceAdapter",
    "UniformRandomReferenceConfig",
    "control_state_resource_usage",
]
