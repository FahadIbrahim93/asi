"""Regression coverage for late-wave fail-closed numerical transactions."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.dreaming import DreamSelectionConfig, score_dream_candidates
from alberta_framework.core.latent_world_model import (
    LatentWorldModel,
    LatentWorldModelConfig,
    run_latent_world_model_learning_loop,
)
from alberta_framework.core.prototype_memory import (
    PrototypeMemoryConfig,
    PrototypeMemoryLearner,
    run_prototype_memory_arrays,
)
from alberta_framework.core.resource_manager import (
    GeneratorMetaResourceManager,
    LearnedResourceManager,
)
from alberta_framework.core.state_builder import (
    OnlineGatedStateBuilder,
    OnlineGatedStateBuilderConfig,
)
from alberta_framework.core.upgd_memory import (
    UPGDMemoryConfig,
    UPGDMemoryLearner,
    run_upgd_memory_arrays,
)
from alberta_framework.core.working_memory import (
    WorkingMemoryConfig,
    WorkingMemoryFeaturizer,
)
from alberta_framework.core.world_model import (
    ActionConditionedWorldModel,
    ActionConditionedWorldModelConfig,
    OneStepWorldModel,
    WorldModelConfig,
    run_action_conditioned_world_model_learning_loop,
)


def test_working_memory_rejects_nonfinite_event_atomically_without_hiding_output() -> None:
    memory = WorkingMemoryFeaturizer(
        WorkingMemoryConfig(
            observation_dim=1,
            action_dim=0,
            reward_dim=0,
            observation_decay_rates=(0.5,),
            include_current_action=False,
            include_current_reward=False,
            include_innovations=True,
            gated_update=True,
        )
    )
    state = memory.update(
        memory.init(), jnp.asarray([2.0]), memory.zero_action(), memory.zero_reward()
    )
    rejected, features = memory.step(
        state,
        jnp.asarray([jnp.inf]),
        memory.zero_action(),
        memory.zero_reward(),
        external_gate=0.0,
    )

    chex.assert_trees_all_equal(rejected, state)
    assert not bool(jnp.all(jnp.isfinite(features)))
    recovered = memory.update(
        rejected, jnp.asarray([0.0]), memory.zero_action(), memory.zero_reward()
    )
    chex.assert_tree_all_finite(recovered)


def test_prototype_memory_rejection_is_explicit_atomic_and_scan_visible() -> None:
    learner = PrototypeMemoryLearner(
        PrototypeMemoryConfig(feature_dim=2, n_classes=2, slots_per_class=2)
    )
    target = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    state = learner.update(
        learner.init(), jnp.asarray([0.25, 0.75], dtype=jnp.float32), target
    ).state
    result = learner.update(state, jnp.asarray([jnp.inf, 0.0]), target)

    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, state)
    chex.assert_trees_all_equal(result.predictions, jnp.zeros((2,), dtype=jnp.float32))
    chex.assert_trees_all_equal(result.errors, jnp.zeros((2,), dtype=jnp.float32))
    chex.assert_trees_all_equal(result.metrics, jnp.zeros((6,), dtype=jnp.float32))
    assert not bool(jnp.all(jnp.isfinite(learner.predict(state, jnp.asarray([jnp.inf, 0.0])))))

    scanned = run_prototype_memory_arrays(
        learner,
        jnp.asarray([[jnp.inf, 0.0], [0.25, 0.75]], dtype=jnp.float32),
        jnp.stack([target, target]),
        state=state,
    )
    chex.assert_trees_all_equal(scanned.updates_applied, jnp.asarray([False, True]))
    assert int(scanned.state.step_count) == int(state.step_count) + 1


def test_action_world_model_rolls_back_entire_transition_and_reports_scan_verdict() -> None:
    model = ActionConditionedWorldModel(
        ActionConditionedWorldModelConfig(
            observation_dim=2,
            n_actions=2,
            hidden_sizes=(),
            sparsity=0.0,
            error_decay=0.5,
        )
    )
    observation = jnp.asarray([0.2, -0.1], dtype=jnp.float32)
    next_observation = jnp.asarray([0.3, 0.1], dtype=jnp.float32)
    state = model.update(
        model.init(jr.key(0)), observation, jnp.asarray(1), 0.5, 0.95, next_observation
    ).state
    result = model.update(
        state,
        observation,
        jnp.asarray(1),
        0.5,
        0.95,
        jnp.asarray([jnp.inf, 0.1]),
    )

    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, state)
    chex.assert_trees_all_equal(result.learner_result.state, state.learner_state)
    chex.assert_tree_all_finite(result.prediction_error)
    chex.assert_trees_all_equal(result.prediction_error, jnp.asarray(0.0))

    scanned = run_action_conditioned_world_model_learning_loop(
        model,
        state,
        jnp.stack([observation, observation]),
        jnp.asarray([1, 1]),
        jnp.asarray([0.5, 0.5]),
        jnp.stack([jnp.asarray([jnp.inf, 0.1]), next_observation]),
        jnp.asarray([0.95, 0.95]),
    )
    chex.assert_trees_all_equal(scanned.updates_applied, jnp.asarray([False, True]))
    assert int(scanned.state.step_count) == int(state.step_count) + 1


def test_online_gated_builder_holds_full_state_and_keeps_invalid_raw_output_visible() -> None:
    builder = OnlineGatedStateBuilder(
        OnlineGatedStateBuilderConfig(observation_dim=2, hidden_dim=3, step_size=0.05)
    )
    state, _ = builder.start(builder.init(jr.key(7)), jnp.asarray([1.0, 1.0]))
    rejected, representation = builder.update(
        state, jnp.asarray([jnp.inf, 0.0]), -1, 0.0, 1.0
    )

    chex.assert_trees_all_equal(rejected, state)
    assert not bool(jnp.all(jnp.isfinite(representation)))
    recovered, _ = builder.update(rejected, jnp.asarray([0.0, 0.0]), -1, 0.0, 1.0)
    chex.assert_tree_all_finite(recovered)


@pytest.mark.parametrize("invalid_action", [jnp.nan, jnp.inf, -2.0, 0.5, 2.0])
def test_discrete_action_casts_cannot_launder_invalid_builder_events(
    invalid_action: float,
) -> None:
    builder = OnlineGatedStateBuilder(
        OnlineGatedStateBuilderConfig(
            observation_dim=2,
            n_actions=2,
            hidden_dim=3,
            step_size=0.05,
        )
    )
    state, _ = builder.start(builder.init(jr.key(70)), jnp.asarray([1.0, 1.0]))

    rejected, representation = jax.jit(builder.update)(
        state,
        jnp.asarray([0.0, 0.0]),
        jnp.asarray(invalid_action),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
    )

    chex.assert_trees_all_equal(rejected, state)
    assert not bool(jnp.all(jnp.isfinite(representation)))
    recovered, recovered_representation = builder.update(
        rejected,
        jnp.asarray([0.0, 0.0]),
        jnp.asarray(1),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
    )
    chex.assert_tree_all_finite(recovered)
    chex.assert_tree_all_finite(recovered_representation)


@pytest.mark.parametrize("invalid_action", [jnp.nan, jnp.inf, -1.0, 0.5, 2.0])
def test_discrete_world_models_reject_actions_before_integer_cast(
    invalid_action: float,
) -> None:
    observation = jnp.asarray([0.2, -0.1], dtype=jnp.float32)
    next_observation = jnp.asarray([0.3, 0.1], dtype=jnp.float32)
    action = jnp.asarray(invalid_action)

    one_step = OneStepWorldModel(
        WorldModelConfig(
            observation_dim=2,
            n_actions=2,
            hidden_sizes=(),
            sparsity=0.0,
        )
    )
    one_step_state = one_step.init(jr.key(71))
    one_step_prediction = one_step.predict(one_step_state, observation, action)
    assert not bool(jnp.all(jnp.isfinite(one_step_prediction.raw_predictions)))
    one_step_result = jax.jit(one_step.update)(
        one_step_state,
        observation,
        action,
        jnp.asarray(0.5),
        next_observation,
    )
    assert not bool(one_step_result.update_applied)
    chex.assert_trees_all_equal(
        one_step_result.state,
        jax.jit(lambda value: value)(one_step_state),
    )

    action_model = ActionConditionedWorldModel(
        ActionConditionedWorldModelConfig(
            observation_dim=2,
            n_actions=2,
            hidden_sizes=(),
            sparsity=0.0,
        )
    )
    action_state = action_model.init(jr.key(72))
    action_prediction = action_model.predict(action_state, observation, action)
    assert not bool(jnp.all(jnp.isfinite(action_prediction.raw_predictions)))
    action_result = jax.jit(action_model.update)(
        action_state,
        observation,
        action,
        jnp.asarray(0.5),
        jnp.asarray(0.95),
        next_observation,
    )
    assert not bool(action_result.update_applied)
    chex.assert_trees_all_equal(
        action_result.state,
        jax.jit(lambda value: value)(action_state),
    )

    latent_model = LatentWorldModel(
        LatentWorldModelConfig(
            observation_dim=2,
            n_actions=2,
            latent_dim=3,
            hidden_sizes=(),
            sparsity=0.0,
        )
    )
    latent_state = latent_model.init(jr.key(73))
    latent_prediction = latent_model.predict(latent_state, observation, action)
    assert not bool(jnp.all(jnp.isfinite(latent_prediction.raw_predictions)))
    latent_result = jax.jit(latent_model.update)(
        latent_state,
        observation,
        action,
        jnp.asarray(0.5),
        jnp.asarray(0.95),
        next_observation,
    )
    assert not bool(latent_result.update_applied)
    chex.assert_trees_all_equal(
        latent_result.state,
        jax.jit(lambda value: value)(latent_state),
    )


def test_dream_selection_rejects_nonfinite_unused_channels() -> None:
    result = score_dream_candidates(
        surprises=jnp.asarray([1.0, 0.5]),
        utilities=jnp.asarray([1.0, 0.5]),
        confidences=jnp.asarray([jnp.inf, 1.0]),
        model_errors=jnp.asarray([0.0, 0.0]),
        config=DreamSelectionConfig(max_items=1, confidence_weight=0.0),
    )

    assert not bool(result.accepted[0])
    assert bool(jnp.isneginf(result.scores[0]))
    assert not bool(result.selected_mask[0])
    assert int(result.selected_indices[0]) == 1


def test_upgd_memory_rejection_is_atomic_neutral_and_scan_visible() -> None:
    learner = UPGDMemoryLearner(
        UPGDMemoryConfig(
            feature_dim=2,
            n_heads=2,
            hidden_sizes=(4,),
            slots_per_class=2,
            target_trace_blend_scale=0.0,
        )
    )
    target = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    state = learner.update(
        learner.init(jr.key(1)), jnp.asarray([1.0, -1.0]), target
    ).state
    result = learner.update(state, jnp.asarray([jnp.inf, -1.0]), target)

    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, state)
    chex.assert_trees_all_equal(result.predictions, jnp.zeros((2,), dtype=jnp.float32))
    chex.assert_trees_all_equal(result.metrics, jnp.zeros((10,), dtype=jnp.float32))

    scanned = run_upgd_memory_arrays(
        learner,
        state,
        jnp.asarray([[jnp.inf, -1.0], [1.0, -1.0]], dtype=jnp.float32),
        jnp.stack([target, target]),
    )
    chex.assert_trees_all_equal(scanned.updates_applied, jnp.asarray([False, True]))
    assert int(scanned.state.step_count) == int(state.step_count) + 1


def _generator_manager(cost_weight: float) -> GeneratorMetaResourceManager:
    return GeneratorMetaResourceManager(
        policy_names=("a", "b"),
        op_ids=(0, 1),
        parent_modes=(0, 1),
        replacement_multipliers=(1.0, 1.0),
        promotion_margin_multipliers=(1.0, 1.0),
        candidate_min_age_multipliers=(1.0, 1.0),
        imprint_scales=(0.0, 0.0),
        cost_weight=cost_weight,
    )


@pytest.mark.parametrize("weight", [jnp.nan, jnp.inf, 1.0e100, 1.0e-50])
def test_resource_cost_weight_must_be_finite_float32_representable(weight: float) -> None:
    with pytest.raises(ValueError, match="cost_weight"):
        LearnedResourceManager(n_actions=2, cost_weight=weight)
    with pytest.raises(ValueError, match="cost_weight"):
        _generator_manager(weight)


def test_resource_cost_product_validity_is_explicit_and_zero_weight_is_exact() -> None:
    losses = jnp.asarray([0.1, 1.0], dtype=jnp.float32)
    costs = jnp.asarray([jnp.inf, 0.0], dtype=jnp.float32)
    zero_manager = LearnedResourceManager(n_actions=2, cost_weight=0.0)
    zero_result = zero_manager.update(zero_manager.init(), losses, resource_costs=costs)
    chex.assert_trees_all_equal(zero_result.valid_actions, jnp.asarray([True, True]))
    chex.assert_trees_all_close(zero_result.adjusted_losses, losses)
    assert bool(zero_result.update_applied)

    weighted = LearnedResourceManager(n_actions=2, cost_weight=2.0)
    overflow_costs = jnp.asarray([jnp.finfo(jnp.float32).max, 0.0], dtype=jnp.float32)
    weighted_result = weighted.update(
        weighted.init(), losses, resource_costs=overflow_costs
    )
    chex.assert_trees_all_equal(weighted_result.valid_actions, jnp.asarray([False, True]))
    chex.assert_trees_all_equal(
        weighted_result.state.action_counts[0], jnp.asarray([0.0, 1.0])
    )
    chex.assert_tree_all_finite(weighted_result.state)
    chex.assert_tree_all_finite(weighted_result.adjusted_losses)

    zero_generator = _generator_manager(0.0)
    zero_generator_result = zero_generator.update(
        zero_generator.init(), losses, resource_costs=costs
    )
    chex.assert_trees_all_equal(
        zero_generator_result.valid_actions, jnp.asarray([True, True])
    )
    chex.assert_trees_all_close(zero_generator_result.adjusted_rewards, losses)

    weighted_generator = _generator_manager(2.0)
    weighted_generator_result = weighted_generator.update(
        weighted_generator.init(), losses, resource_costs=overflow_costs
    )
    chex.assert_trees_all_equal(
        weighted_generator_result.valid_actions, jnp.asarray([False, True])
    )
    chex.assert_trees_all_equal(
        weighted_generator_result.state.action_counts[0], jnp.asarray([0.0, 1.0])
    )
    chex.assert_tree_all_finite(weighted_generator_result.state)


def test_negative_resource_costs_are_never_treated_as_benefits() -> None:
    values = jnp.asarray([0.25, 0.5], dtype=jnp.float32)
    costs = jnp.asarray([-1.0, 0.25], dtype=jnp.float32)

    learned = LearnedResourceManager(n_actions=2, cost_weight=1.0)
    learned_result = learned.update(learned.init(), values, resource_costs=costs)
    chex.assert_trees_all_equal(
        learned_result.valid_actions,
        jnp.asarray([False, True]),
    )
    assert float(learned_result.adjusted_losses[0]) == 0.0

    generator = _generator_manager(1.0)
    generator_result = generator.update(
        generator.init(),
        values,
        resource_costs=costs,
    )
    chex.assert_trees_all_equal(
        generator_result.valid_actions,
        jnp.asarray([False, True]),
    )
    assert float(generator_result.adjusted_rewards[0]) == 0.0


@pytest.mark.parametrize(
    "selected_probability",
    [jnp.nan, jnp.inf, -1.0, 0.0, 1.01],
)
def test_exp3_rejects_invalid_selected_probability_atomically(
    selected_probability: float,
) -> None:
    manager = GeneratorMetaResourceManager(
        policy_names=("a", "b"),
        op_ids=(0, 1),
        parent_modes=(0, 1),
        replacement_multipliers=(1.0, 1.0),
        promotion_margin_multipliers=(1.0, 1.0),
        candidate_min_age_multipliers=(1.0, 1.0),
        imprint_scales=(0.0, 0.0),
        update_rule="exp3",
    )
    state = manager.init()
    result = manager.update(
        state,
        jnp.asarray([0.5, 1.0], dtype=jnp.float32),
        selected_action=jnp.asarray(1),
        selected_probability=jnp.asarray(selected_probability),
    )

    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, state)
    chex.assert_trees_all_equal(result.advantages, jnp.zeros((2,), dtype=jnp.float32))


@pytest.mark.parametrize("selected_action", [jnp.nan, jnp.inf, -1.0, 0.5, 2.0])
def test_exp3_rejects_invalid_selected_action_before_integer_cast(
    selected_action: float,
) -> None:
    manager = GeneratorMetaResourceManager(
        policy_names=("a", "b"),
        op_ids=(0, 1),
        parent_modes=(0, 1),
        replacement_multipliers=(1.0, 1.0),
        promotion_margin_multipliers=(1.0, 1.0),
        candidate_min_age_multipliers=(1.0, 1.0),
        imprint_scales=(0.0, 0.0),
        update_rule="exp3",
    )
    state = manager.init()
    result = manager.update(
        state,
        jnp.asarray([0.5, 1.0], dtype=jnp.float32),
        selected_action=jnp.asarray(selected_action),
        selected_probability=jnp.asarray(0.5),
    )

    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, state)


@pytest.mark.parametrize("context_id", [jnp.nan, jnp.inf, -1.0, 0.5, 2.0])
def test_resource_updates_reject_invalid_context_before_integer_cast(
    context_id: float,
) -> None:
    learned = LearnedResourceManager(n_actions=2, n_contexts=2)
    learned_state = learned.init()
    learned_weights = learned.weights(
        learned_state,
        context_id=jnp.asarray(context_id),
    )
    assert not bool(jnp.all(jnp.isfinite(learned_weights)))
    learned_result = learned.update(
        learned_state,
        jnp.asarray([0.25, 0.5], dtype=jnp.float32),
        context_id=jnp.asarray(context_id),
    )
    assert not bool(learned_result.update_applied)
    chex.assert_trees_all_equal(learned_result.state, learned_state)

    generator = GeneratorMetaResourceManager(
        policy_names=("a", "b"),
        op_ids=(0, 1),
        parent_modes=(0, 1),
        replacement_multipliers=(1.0, 1.0),
        promotion_margin_multipliers=(1.0, 1.0),
        candidate_min_age_multipliers=(1.0, 1.0),
        imprint_scales=(0.0, 0.0),
        n_contexts=2,
    )
    generator_state = generator.init()
    generator_weights = generator.weights(
        generator_state,
        context_id=jnp.asarray(context_id),
    )
    assert not bool(jnp.all(jnp.isfinite(generator_weights)))
    decision = generator.select(
        generator_state,
        jr.key(74),
        context_id=jnp.asarray(context_id),
    )
    assert not bool(decision.valid)
    assert int(decision.action) == -1
    assert not bool(jnp.all(jnp.isfinite(decision.weights)))
    generator_result = generator.update(
        generator_state,
        jnp.asarray([0.25, 0.5], dtype=jnp.float32),
        context_id=jnp.asarray(context_id),
    )
    assert not bool(generator_result.update_applied)
    chex.assert_trees_all_equal(generator_result.state, generator_state)


def test_latent_world_model_rolls_back_learner_encoder_and_scan_state() -> None:
    model = LatentWorldModel(
        LatentWorldModelConfig(
            observation_dim=3,
            n_actions=2,
            latent_dim=4,
            hidden_sizes=(),
            sparsity=0.0,
            encoder_learning=True,
            encoder_collapse_gate_threshold=1.0,
        )
    )
    observation = jnp.asarray([0.2, -0.1, 0.3], dtype=jnp.float32)
    next_observation = jnp.asarray([0.3, 0.0, 0.2], dtype=jnp.float32)
    state = model.update(
        model.init(jr.key(9)), observation, jnp.asarray(1), 0.5, 0.95, next_observation
    ).state
    result = model.update(
        state, observation, jnp.asarray(1), jnp.inf, 0.95, next_observation
    )

    assert not bool(result.update_applied)
    assert not bool(result.encoder_update_applied)
    chex.assert_trees_all_equal(result.state, state)
    chex.assert_trees_all_equal(result.learner_result.state, state.learner_state)
    chex.assert_trees_all_equal(result.prediction_error, jnp.asarray(0.0))

    scanned = run_latent_world_model_learning_loop(
        model,
        state,
        jnp.stack([observation, observation]),
        jnp.asarray([1, 1]),
        jnp.asarray([jnp.inf, 0.5]),
        jnp.stack([next_observation, next_observation]),
        jnp.asarray([0.95, 0.95]),
    )
    chex.assert_trees_all_equal(scanned.updates_applied, jnp.asarray([False, True]))
    assert int(scanned.state.step_count) == int(state.step_count) + 1
