"""Experience streams for continual learning."""

from typing import TYPE_CHECKING, Any

from alberta_framework.streams.base import ScanStream
from alberta_framework.streams.closed_loop import (
    LEFT_ACTION,
    PHASE_A,
    PHASE_B,
    RIGHT_ACTION,
    RiverSwimConfig,
    RiverSwimMDP,
    RiverSwimState,
    SwitchingTwoStateConfig,
    SwitchingTwoStateMDP,
    SwitchingTwoStateState,
)
from alberta_framework.streams.gauntlet import (
    NUM_SEGMENTS,
    SEGMENT_NAMES,
    ContextGatedFeatures,
    GauntletConfig,
    GauntletState,
    GauntletStream,
    LifetimeGauntletStream,
    LifetimeState,
    best_fixed_alpha_errors,
    gauntlet_scorecard,
    lifetime_scorecard,
    run_gauntlet,
    run_gauntlet_batched,
)
from alberta_framework.streams.recurring_multiagent import (
    AVOID_CONTEXT,
    AVOID_CONTEXT_INDEX,
    BASE_OBSERVATION_DIM,
    MEET_CONTEXT,
    MEET_CONTEXT_INDEX,
    OTHER_VELOCITY_INDEX,
    OWN_POSITION_INDEX,
    OWN_VELOCITY_INDEX,
    RELATIVE_POSITION_INDEX,
    PartnerPolicy,
    RecurringTwoAgentOracle,
    RecurringTwoAgentState,
    RecurringTwoAgentTransition,
    RecurringTwoAgentWorld,
    scripted_meet_avoid_partner_policy,
)
from alberta_framework.streams.synthetic import (
    AbruptChangeState,
    AbruptChangeStream,
    AbruptChangeTarget,
    CyclicState,
    CyclicStream,
    CyclicTarget,
    DynamicScaleShiftState,
    DynamicScaleShiftStream,
    PeriodicChangeState,
    PeriodicChangeStream,
    PeriodicChangeTarget,
    RandomWalkState,
    RandomWalkStream,
    RandomWalkTarget,
    ScaleDriftState,
    ScaleDriftStream,
    ScaledStreamState,
    ScaledStreamWrapper,
    SuttonExperiment1State,
    SuttonExperiment1Stream,
    make_scale_range,
)

__all__ = [
    # Protocol
    "ScanStream",
    # Stream classes
    "AbruptChangeState",
    "AbruptChangeStream",
    "AbruptChangeTarget",
    "CyclicState",
    "CyclicStream",
    "CyclicTarget",
    "DynamicScaleShiftState",
    "DynamicScaleShiftStream",
    "PeriodicChangeState",
    "PeriodicChangeStream",
    "PeriodicChangeTarget",
    "RandomWalkState",
    "RandomWalkStream",
    "RandomWalkTarget",
    "ScaleDriftState",
    "ScaleDriftStream",
    "ScaledStreamState",
    "ScaledStreamWrapper",
    "SuttonExperiment1State",
    "SuttonExperiment1Stream",
    # Closed-loop micro-MDPs (actions affect observations)
    "LEFT_ACTION",
    "PHASE_A",
    "PHASE_B",
    "RIGHT_ACTION",
    "RiverSwimConfig",
    "RiverSwimMDP",
    "RiverSwimState",
    "SwitchingTwoStateConfig",
    "SwitchingTwoStateMDP",
    "SwitchingTwoStateState",
    # Recurring two-agent continual-control world
    "AVOID_CONTEXT",
    "AVOID_CONTEXT_INDEX",
    "BASE_OBSERVATION_DIM",
    "MEET_CONTEXT",
    "MEET_CONTEXT_INDEX",
    "OTHER_VELOCITY_INDEX",
    "OWN_POSITION_INDEX",
    "OWN_VELOCITY_INDEX",
    "RELATIVE_POSITION_INDEX",
    "PartnerPolicy",
    "RecurringTwoAgentOracle",
    "RecurringTwoAgentState",
    "RecurringTwoAgentTransition",
    "RecurringTwoAgentWorld",
    "scripted_meet_avoid_partner_policy",
    # The Alberta Gauntlet diagnostic stream + harness
    "NUM_SEGMENTS",
    "SEGMENT_NAMES",
    "ContextGatedFeatures",
    "GauntletConfig",
    "GauntletState",
    "GauntletStream",
    "LifetimeGauntletStream",
    "LifetimeState",
    "best_fixed_alpha_errors",
    "gauntlet_scorecard",
    "lifetime_scorecard",
    "run_gauntlet",
    "run_gauntlet_batched",
    # Utilities
    "make_scale_range",
]

_GYMNASIUM_EXPORTS = (
    "GymnasiumStream",
    "PredictionMode",
    "TDStream",
    "collect_trajectory",
    "learn_from_trajectory",
    "learn_from_trajectory_normalized",
    "make_epsilon_greedy_policy",
    "make_gymnasium_stream",
    "make_random_policy",
)
__all__ += list(_GYMNASIUM_EXPORTS)

if TYPE_CHECKING:
    from alberta_framework.streams import gymnasium as _gymnasium_types

    GymnasiumStream = _gymnasium_types.GymnasiumStream
    PredictionMode = _gymnasium_types.PredictionMode
    TDStream = _gymnasium_types.TDStream
    collect_trajectory = _gymnasium_types.collect_trajectory
    learn_from_trajectory = _gymnasium_types.learn_from_trajectory
    learn_from_trajectory_normalized = _gymnasium_types.learn_from_trajectory_normalized
    make_epsilon_greedy_policy = _gymnasium_types.make_epsilon_greedy_policy
    make_gymnasium_stream = _gymnasium_types.make_gymnasium_stream
    make_random_policy = _gymnasium_types.make_random_policy


def __getattr__(name: str) -> Any:
    """Load Gymnasium adapter exports without creating a core/streams cycle."""
    if name not in _GYMNASIUM_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from alberta_framework.streams import gymnasium as gymnasium_streams

    value = getattr(gymnasium_streams, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy adapter names in interactive discovery."""
    return sorted(set(globals()) | set(_GYMNASIUM_EXPORTS))
