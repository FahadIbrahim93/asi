"""Development contracts for exact-dispatch reference-life control adapters."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import pytest

from alberta_framework.core.average_reward import DifferentialSARSAState
from alberta_framework.core.sarsa import SARSAState
from alberta_framework.reference_agent import (
    AuthorizationStatus,
    DecisionOwnershipError,
    DispatchAuthorization,
)
from alberta_framework.reference_life import (
    ExactDispatchConfig,
    LifePhase,
    ReferenceLifeMetricsConfig,
    ReferenceLifeRunner,
    RiverSwimReferenceEnvironment,
    SwitchingTwoStateReferenceEnvironment,
)
from alberta_framework.reference_life_controls import (
    REFERENCE_CONTROL_MAX_ACCEPTED_EVENTS,
    AnalyticOracleReferenceAdapter,
    AnalyticOracleReferenceConfig,
    DifferentialSARSAReferenceAdapter,
    DifferentialSARSAReferenceConfig,
    DiscountedSARSAReferenceAdapter,
    DiscountedSARSAReferenceConfig,
    ReferenceLifeControlState,
    UniformRandomReferenceAdapter,
    UniformRandomReferenceConfig,
    control_state_resource_usage,
)
from alberta_framework.streams.closed_loop import (
    RiverSwimConfig,
    RiverSwimMDP,
    SwitchingTwoStateConfig,
)


def _switching_runner(
    adapter: object,
    environment_config: SwitchingTwoStateConfig,
    *,
    lifecycle_id: str,
    seed: int = 17,
    horizon: int = 4,
) -> ReferenceLifeRunner:
    manifest = adapter.manifest  # type: ignore[attr-defined]
    environment = SwitchingTwoStateReferenceEnvironment(
        environment_config,
        observation_spec=manifest.observation_spec,
        action_spec=manifest.action_spec,
    )
    return ReferenceLifeRunner.create(
        agent_adapter=adapter,  # type: ignore[arg-type]
        environment_adapter=environment,
        lifecycle_id=lifecycle_id,
        seed=seed,
        max_accepted_events=horizon,
        metrics_config=ReferenceLifeMetricsConfig(mode="switching_two_phase"),
    )


def _river_runner(
    adapter: object,
    environment_config: RiverSwimConfig,
    *,
    lifecycle_id: str,
    seed: int = 17,
    horizon: int = 4,
) -> ReferenceLifeRunner:
    manifest = adapter.manifest  # type: ignore[attr-defined]
    dispatch = ExactDispatchConfig(
        executor_id="asi.riverswim.executor",
        executor_epoch="asi.riverswim.executor_epoch.1",
    )
    environment = RiverSwimReferenceEnvironment(
        environment_config,
        observation_spec=manifest.observation_spec,
        action_spec=manifest.action_spec,
        executor_id=dispatch.executor_id,
        executor_epoch=dispatch.executor_epoch,
    )
    return ReferenceLifeRunner.create(
        agent_adapter=adapter,  # type: ignore[arg-type]
        environment_adapter=environment,
        lifecycle_id=lifecycle_id,
        seed=seed,
        max_accepted_events=horizon,
        dispatch_config=dispatch,
        metrics_config=ReferenceLifeMetricsConfig(mode="stationary"),
    )


def _uniform_switching() -> UniformRandomReferenceAdapter:
    environment = SwitchingTwoStateConfig(phase_length=2)  # type: ignore[call-arg]
    return UniformRandomReferenceAdapter(
        UniformRandomReferenceConfig.for_switching(environment)
    )


def _oracle_river() -> AnalyticOracleReferenceAdapter:
    environment = RiverSwimConfig(n_states=3)  # type: ignore[call-arg]
    return AnalyticOracleReferenceAdapter(
        AnalyticOracleReferenceConfig.for_riverswim(environment)
    )


def _differential_switching() -> DifferentialSARSAReferenceAdapter:
    environment = SwitchingTwoStateConfig(phase_length=2)  # type: ignore[call-arg]
    return DifferentialSARSAReferenceAdapter(
        DifferentialSARSAReferenceConfig.for_switching(
            environment,
            q_step_size=0.1,
            average_reward_step_size=0.01,
            epsilon_start=0.25,
            epsilon_end=0.25,
        )
    )


def _discounted_switching() -> DiscountedSARSAReferenceAdapter:
    environment = SwitchingTwoStateConfig(phase_length=2)  # type: ignore[call-arg]
    return DiscountedSARSAReferenceAdapter(
        DiscountedSARSAReferenceConfig.for_switching(
            environment,
            gamma=0.9,
            epsilon_start=0.25,
            epsilon_end=0.25,
            hidden_sizes=(),
            step_size=0.05,
        )
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("adapter_factory", "environment_kind"),
    (
        (_uniform_switching, "switching"),
        (_oracle_river, "river"),
        (_differential_switching, "switching"),
        (_discounted_switching, "switching"),
    ),
)
def test_every_control_adapter_runs_one_complete_reference_life(
    adapter_factory: Callable[[], object],
    environment_kind: str,
) -> None:
    adapter = adapter_factory()
    if environment_kind == "river":
        runner = _river_runner(
            adapter,
            RiverSwimConfig(n_states=3),  # type: ignore[call-arg]
            lifecycle_id=f"control.{adapter.manifest.implementation_id}.life",  # type: ignore[attr-defined]
            horizon=3,
        )
    else:
        runner = _switching_runner(
            adapter,
            SwitchingTwoStateConfig(phase_length=2),  # type: ignore[call-arg]
            lifecycle_id=f"control.{adapter.manifest.implementation_id}.life",  # type: ignore[attr-defined]
            horizon=3,
        )

    run = runner.run_to_completion(runner.init())

    assert run.state.phase is LifePhase.COMPLETED
    assert run.state.accepted_events == 3
    assert len(run.events) == 3
    assert all(event.step_result.transaction_accepted for event in run.events)
    assert all(event.transaction.decision.proposed_action is not None for event in run.events)


def test_uniform_random_schedule_is_seed_deterministic_and_owner_bound() -> None:
    environment = SwitchingTwoStateConfig(phase_length=3)  # type: ignore[call-arg]
    config = UniformRandomReferenceConfig.for_switching(environment)
    first = UniformRandomReferenceAdapter(config)
    second = UniformRandomReferenceAdapter(config)
    first_runner = _switching_runner(
        first,
        environment,
        lifecycle_id="control.random.determinism",
        seed=123,
        horizon=24,
    )
    second_runner = _switching_runner(
        second,
        environment,
        lifecycle_id="control.random.determinism",
        seed=123,
        horizon=24,
    )

    first_run = first_runner.run_to_completion(first_runner.init())
    second_run = second_runner.run_to_completion(second_runner.init())
    first_actions = tuple(
        event.transaction.decision.proposed_action.to_python()  # type: ignore[union-attr]
        for event in first_run.events
    )
    second_actions = tuple(
        event.transaction.decision.proposed_action.to_python()  # type: ignore[union-attr]
        for event in second_run.events
    )

    assert first_actions == second_actions
    assert first_run.state.transcript_sha256 == second_run.state.transcript_sha256
    with pytest.raises(DecisionOwnershipError, match="owner|another adapter"):
        second.validate_state(first_run.state.agent_state)


def test_oracle_uses_privileged_switching_phase_and_stationary_river_policy() -> None:
    switching = SwitchingTwoStateConfig(phase_length=2)  # type: ignore[call-arg]
    switching_config = AnalyticOracleReferenceConfig.for_switching(switching)
    switching_runner = _switching_runner(
        AnalyticOracleReferenceAdapter(switching_config),
        switching,
        lifecycle_id="control.oracle.switching.valid",
        horizon=6,
    )
    valid_switching_run = switching_runner.run_to_completion(switching_runner.init())
    expected_policies = switching_config.policies
    for event in valid_switching_run.events:
        decision = event.transaction.decision
        observation = np.asarray(decision.observation.to_python())
        state_index = int(np.argmax(observation))
        phase = (decision.decision_index // switching.phase_length) % 2
        assert decision.proposed_action is not None
        assert decision.proposed_action.to_python() == expected_policies[phase][state_index]

    river = RiverSwimConfig(n_states=4)  # type: ignore[call-arg]
    river_kernel = RiverSwimMDP(river)
    river_config = AnalyticOracleReferenceConfig.for_riverswim(river)
    river_adapter = AnalyticOracleReferenceAdapter(river_config)
    river_runner = _river_runner(
        river_adapter,
        river,
        lifecycle_id="control.oracle.river",
        horizon=8,
    )
    river_run = river_runner.run_to_completion(river_runner.init())
    policy = river_kernel.optimal_policy()
    for event in river_run.events:
        decision = event.transaction.decision
        state_index = int(np.argmax(np.asarray(decision.observation.to_python())))
        assert decision.proposed_action is not None
        assert decision.proposed_action.to_python() == policy[state_index]

    manifest_config = river_adapter.manifest.config
    assert manifest_config["privileged"] is True
    assert manifest_config["environment_config_sha256"] == river_config.environment_config_sha256


def test_stale_cross_config_and_cached_action_corruption_are_failure_atomic() -> None:
    environment = SwitchingTwoStateConfig(phase_length=2)  # type: ignore[call-arg]
    adapter = UniformRandomReferenceAdapter(
        UniformRandomReferenceConfig.for_switching(environment)
    )
    runner = _switching_runner(
        adapter,
        environment,
        lifecycle_id="control.random.ownership",
        horizon=2,
    )
    initial = runner.init()
    state = initial.agent_state
    assert isinstance(state, ReferenceLifeControlState)
    decision = adapter.current_decision(state)
    assert decision.proposed_action is not None
    proposed_action = decision.proposed_action.to_python()
    assert isinstance(proposed_action, int)
    other_action = adapter.manifest.action_spec.encode(
        np.asarray(1 - proposed_action, dtype=np.int32)
    )
    substituted_decision = dataclasses.replace(decision, proposed_action=other_action)
    substituted_authorization = DispatchAuthorization(
        decision=substituted_decision,
        status=AuthorizationStatus.EXACT,
        authorized_action=other_action,
        authority_id="control.test.authority",
        policy_version="control.test.policy.1",
        authorization_id=f"{substituted_decision.decision_id}:authorization",
    )
    with pytest.raises(DecisionOwnershipError, match="decision|action|cache"):
        adapter.settle_dispatch(state, substituted_authorization)

    corrupted = dataclasses.replace(state, current_action=other_action)
    with pytest.raises(DecisionOwnershipError, match="action|random"):
        adapter.validate_state(corrupted)

    different_config = UniformRandomReferenceAdapter(
        UniformRandomReferenceConfig.for_riverswim(
            RiverSwimConfig(n_states=3)  # type: ignore[call-arg]
        )
    )
    with pytest.raises(DecisionOwnershipError, match="owner|manifest|config"):
        different_config.validate_state(state)
    relabeled_config = dataclasses.replace(
        state,
        config_sha256=different_config.manifest.config_sha256,
    )
    with pytest.raises(DecisionOwnershipError, match="configuration"):
        adapter.validate_state(relabeled_config)

    first_step = runner.step(initial)
    assert first_step.event is not None
    stale_state = first_step.state.agent_state
    rejected = adapter.apply_outcome(stale_state, first_step.event.transaction)
    assert not rejected.accepted
    assert rejected.state is stale_state
    assert rejected.next_decision is None
    assert rejected.parameters_changed is False


def test_differential_sarsa_applies_hand_computed_first_update() -> None:
    rewarding = SwitchingTwoStateConfig(  # type: ignore[call-arg]
        phase_length=10,
        payoffs_a=((1.0, 1.0), (1.0, 1.0)),
        payoffs_b=((1.0, 1.0), (1.0, 1.0)),
    )
    adapter = DifferentialSARSAReferenceAdapter(
        DifferentialSARSAReferenceConfig.for_switching(
            rewarding,
            q_step_size=0.5,
            average_reward_step_size=0.25,
            epsilon_start=0.0,
            epsilon_end=0.0,
            use_bias=False,
        )
    )
    runner = _switching_runner(
        adapter,
        rewarding,
        lifecycle_id="control.differential.update",
        horizon=1,
    )
    initial = runner.init()
    initial_control = initial.agent_state
    assert isinstance(initial_control, ReferenceLifeControlState)
    initial_learning = initial_control.agent_state
    assert isinstance(initial_learning, DifferentialSARSAState)
    initial_decision = adapter.current_decision(initial_control)
    initial_state_index = int(np.argmax(initial_decision.observation.to_numpy()))
    assert initial_decision.proposed_action is not None
    initial_action = initial_decision.proposed_action.to_python()
    assert isinstance(initial_action, int)

    step = runner.step(initial)

    assert step.accepted
    final_control = step.state.agent_state
    assert isinstance(final_control, ReferenceLifeControlState)
    final_learning = final_control.agent_state
    assert isinstance(final_learning, DifferentialSARSAState)
    expected_weights = np.zeros((2, 2), dtype=np.float32)
    expected_weights[initial_action, initial_state_index] = 0.5
    np.testing.assert_array_equal(np.asarray(final_learning.q_weights), expected_weights)
    np.testing.assert_array_equal(np.asarray(final_learning.q_bias), np.zeros(2, dtype=np.float32))
    assert float(final_learning.average_reward) == pytest.approx(0.25)
    assert int(final_learning.step_count) == 1
    assert step.event is not None
    assert step.event.step_result.parameters_changed


def test_discounted_sarsa_updates_on_continuing_unit_discount_outcome() -> None:
    rewarding = SwitchingTwoStateConfig(  # type: ignore[call-arg]
        phase_length=10,
        payoffs_a=((1.0, 1.0), (1.0, 1.0)),
        payoffs_b=((1.0, 1.0), (1.0, 1.0)),
    )
    adapter = DiscountedSARSAReferenceAdapter(
        DiscountedSARSAReferenceConfig.for_switching(
            rewarding,
            gamma=0.5,
            epsilon_start=0.0,
            epsilon_end=0.0,
            hidden_sizes=(),
            step_size=0.25,
            sparsity=0.0,
            use_layer_norm=False,
        )
    )
    runner = _switching_runner(
        adapter,
        rewarding,
        lifecycle_id="control.discounted.update",
        horizon=2,
    )
    initial = runner.init()
    run = runner.run_to_completion(initial)

    assert run.state.phase is LifePhase.COMPLETED
    final_control = run.state.agent_state
    assert isinstance(final_control, ReferenceLifeControlState)
    final_learning = final_control.agent_state
    assert isinstance(final_learning, SARSAState)
    assert int(final_learning.step_count) == 2
    assert int(final_learning.learner_state.step_count) == 2
    assert all(not event.transaction.is_boundary for event in run.events)
    assert all(event.transaction.discount == 1.0 for event in run.events)
    assert any(event.step_result.parameters_changed for event in run.events)
    assert adapter.manifest.config["gamma"] == 0.5

    resources = control_state_resource_usage(final_control)
    assert resources.array_leaves > 0
    assert resources.array_elements > 0
    assert resources.persistent_bytes > 0
    assert resources.floating_array_leaves > 0


def test_all_control_adapters_declare_the_exact_signed_int32_capacity() -> None:
    adapters = (
        _uniform_switching(),
        AnalyticOracleReferenceAdapter(
            AnalyticOracleReferenceConfig.for_switching(
                SwitchingTwoStateConfig(phase_length=2)  # type: ignore[call-arg]
            )
        ),
        _differential_switching(),
        _discounted_switching(),
    )
    assert REFERENCE_CONTROL_MAX_ACCEPTED_EVENTS == int(np.iinfo(np.int32).max)
    assert all(
        adapter.max_accepted_events == REFERENCE_CONTROL_MAX_ACCEPTED_EVENTS
        for adapter in adapters
    )


def test_control_configs_and_states_are_immutable() -> None:
    config = UniformRandomReferenceConfig.for_switching(
        SwitchingTwoStateConfig(phase_length=2)  # type: ignore[call-arg]
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.observation_dim = 3  # type: ignore[misc]

    adapter = UniformRandomReferenceAdapter(config)
    runner = _switching_runner(
        adapter,
        SwitchingTwoStateConfig(phase_length=2),  # type: ignore[call-arg]
        lifecycle_id="control.random.immutable",
        horizon=1,
    )
    state = runner.init().agent_state
    assert isinstance(state, ReferenceLifeControlState)
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.decision_index = 3  # type: ignore[misc]
