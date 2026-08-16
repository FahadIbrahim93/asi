# mypy: disable-error-code="attr-defined"
"""Tests for the Step 2 fixed-budget associative memory."""

import math

import chex
import jax.numpy as jnp
import pytest

from alberta_framework.core.associative_memory import (
    AssociativeMemoryConfig,
    AssociativeMemoryLearner,
    run_associative_memory_arrays,
)
from alberta_framework.steps.step2 import (
    Step2AssociativeConfig,
    make_step2_associative_learner,
    run_step2_associative_smoke,
)


def test_associative_config_roundtrip() -> None:
    config = AssociativeMemoryConfig(
        vocab_size=7,
        block_size=5,
        suffix_length=3,
        feature_family="token_suffix_pair",
        max_features=31,
    )

    restored = AssociativeMemoryConfig.from_config(config.to_config())

    assert restored == config
    learner = AssociativeMemoryLearner(restored)
    assert learner.max_active_features == 8
    chex.assert_shape(learner.init().keys, (31, 5))


def test_config_rejects_infinite_write_lr() -> None:
    with pytest.raises(ValueError, match="write_lr"):
        AssociativeMemoryLearner(
            AssociativeMemoryConfig(
                vocab_size=4,
                block_size=3,
                suffix_length=2,
                write_lr=float("inf"),
            )
        )


@pytest.mark.parametrize(
    ("field", "invalid_values"),
    [
        (
            "scope_lr",
            (float("nan"), float("inf"), float("-inf"), -0.01, True, "0.1"),
        ),
        (
            "budget_lr",
            (float("nan"), float("inf"), float("-inf"), -0.01, True, "0.1"),
        ),
        (
            "initial_budget_fraction",
            (float("nan"), float("inf"), float("-inf"), 0.0, 1.01, True, "0.5"),
        ),
        (
            "scope_logit_clip",
            (float("nan"), float("inf"), float("-inf"), 0.0, True, "8.0"),
        ),
        ("min_effective_budget", (float("nan"), 1.5, True, "1")),
        ("adaptive_feature_family", (0, 1, "yes")),
        ("adaptive_window", (0, 1, "yes")),
        ("adaptive_budget", (0, 1, "yes")),
    ],
)
def test_config_rejects_invalid_adaptive_scalars(
    field: str,
    invalid_values: tuple[object, ...],
) -> None:
    base = AssociativeMemoryConfig(vocab_size=4, block_size=3, suffix_length=2)

    for invalid in invalid_values:
        payload = base.to_config()
        payload[field] = invalid
        with pytest.raises(ValueError, match=field):
            AssociativeMemoryLearner(AssociativeMemoryConfig.from_config(payload))


def test_silent_feature_does_not_turn_inf_value_into_nan() -> None:
    """Weight 0 times an inf stored row is 0*inf = NaN in the evidence sum."""
    learner = AssociativeMemoryLearner(
        AssociativeMemoryConfig(vocab_size=4, block_size=3, suffix_length=2, max_features=8)
    )
    state = learner.init()
    poisoned = state.replace(
        values=state.values.at[0].set(jnp.full((4,), jnp.inf, dtype=jnp.float32))
    )
    context = jnp.asarray([1, 2, 3], dtype=jnp.int32)
    raw = jnp.array([0.0], dtype=jnp.float32)[:, None] * poisoned.values[0][None, :]
    assert not bool(jnp.all(jnp.isfinite(raw)))

    prediction = learner.predict(poisoned, context)
    chex.assert_tree_all_finite(prediction.logits)
    chex.assert_tree_all_finite(prediction.probabilities)
    assert float(jnp.sum(prediction.probabilities)) == pytest.approx(1.0)

    result = learner.update(poisoned, context, jnp.asarray(1, dtype=jnp.int32))
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, poisoned)
    chex.assert_trees_all_equal(result.predictions, jnp.zeros_like(result.predictions))
    chex.assert_trees_all_equal(result.logits, jnp.zeros_like(result.logits))
    chex.assert_trees_all_equal(result.metrics, jnp.zeros_like(result.metrics))


def test_corrupted_active_family_logits_remain_fail_visible() -> None:
    """An invalid active gate must not masquerade as a uniform valid gate."""
    learner = AssociativeMemoryLearner(
        AssociativeMemoryConfig(
            vocab_size=4,
            block_size=3,
            suffix_length=2,
            max_features=8,
            adaptive_feature_family=True,
        )
    )
    state = learner.init().replace(family_logits=jnp.full((2,), jnp.inf, dtype=jnp.float32))
    context = jnp.asarray([1, 2, 3], dtype=jnp.int32)
    prediction = learner.predict(state, context)

    assert not bool(jnp.all(jnp.isfinite(prediction.family_probs)))
    result = learner.update(state, context, jnp.asarray(1, dtype=jnp.int32))
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, state)


@pytest.mark.parametrize(
    "leaf",
    [
        "values",
        "utility",
        "counts",
        "prior",
        "family_logits",
        "window_logits",
        "budget_logit",
    ],
)
def test_update_rejects_nonfinite_source_state_leaf(leaf: str) -> None:
    learner = AssociativeMemoryLearner(
        AssociativeMemoryConfig(
            vocab_size=4,
            block_size=3,
            suffix_length=2,
            max_features=8,
            adaptive_feature_family=True,
            adaptive_window=True,
            adaptive_budget=True,
        )
    )
    state = learner.init()
    poison = jnp.asarray(jnp.nan, dtype=jnp.float32)
    if leaf == "values":
        state = state.replace(values=state.values.at[0, 0].set(poison))
    elif leaf == "utility":
        state = state.replace(utility=state.utility.at[0].set(poison))
    elif leaf == "counts":
        state = state.replace(counts=state.counts.at[0].set(poison))
    elif leaf == "prior":
        state = state.replace(prior=state.prior.at[0].set(poison))
    elif leaf == "family_logits":
        state = state.replace(family_logits=state.family_logits.at[0].set(poison))
    elif leaf == "window_logits":
        state = state.replace(window_logits=state.window_logits.at[0].set(poison))
    else:
        state = state.replace(budget_logit=poison)

    result = learner.update(
        state,
        jnp.asarray([1, 2, 3], dtype=jnp.int32),
        jnp.asarray(1, dtype=jnp.int32),
    )

    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, state)
    chex.assert_tree_all_finite((result.predictions, result.logits, result.metrics))
    chex.assert_trees_all_equal(result.predictions, jnp.zeros_like(result.predictions))
    chex.assert_trees_all_equal(result.logits, jnp.zeros_like(result.logits))
    chex.assert_trees_all_equal(result.metrics, jnp.zeros_like(result.metrics))


def test_array_runner_exposes_rejected_update_mask() -> None:
    learner = AssociativeMemoryLearner(
        AssociativeMemoryConfig(vocab_size=4, block_size=3, suffix_length=2, max_features=8)
    )
    state = learner.init().replace(
        counts=learner.init().counts.at[0].set(jnp.asarray(jnp.inf, dtype=jnp.float32))
    )
    contexts = jnp.asarray([[1, 2, 3], [2, 3, 0]], dtype=jnp.int32)
    labels = jnp.asarray([1, 2], dtype=jnp.int32)

    result = run_associative_memory_arrays(learner, state, contexts, labels)

    chex.assert_trees_all_equal(
        result.updates_applied,
        jnp.zeros((contexts.shape[0],), dtype=jnp.bool_),
    )
    chex.assert_trees_all_equal(result.state, state)
    chex.assert_trees_all_equal(result.predictions, jnp.zeros_like(result.predictions))
    chex.assert_trees_all_equal(result.metrics, jnp.zeros_like(result.metrics))


def test_associative_prediction_is_before_write() -> None:
    learner = AssociativeMemoryLearner(
        AssociativeMemoryConfig(vocab_size=5, block_size=4, suffix_length=3)
    )
    state = learner.init()
    context = jnp.asarray([1, 2, 3, 4], dtype=jnp.int32)

    before = learner.predict(state, context)
    result = learner.update(state, context, jnp.asarray(2, dtype=jnp.int32))

    assert bool(result.update_applied)
    chex.assert_trees_all_close(before.probabilities, jnp.full((5,), 0.2))
    chex.assert_trees_all_close(result.predictions, before.probabilities)
    assert int(result.state.step_count) == 1
    assert float(result.metrics[0]) == pytest.approx(math.log(5), abs=1e-5)


def test_associative_scope_controls_are_disabled_by_default() -> None:
    config = AssociativeMemoryConfig(
        vocab_size=5,
        block_size=4,
        suffix_length=3,
        max_features=32,
    )
    learner = AssociativeMemoryLearner(config)
    state = learner.init()
    context = jnp.asarray([1, 2, 3, 4], dtype=jnp.int32)

    result = learner.update(state, context, jnp.asarray(2, dtype=jnp.int32))
    prediction = learner.predict(result.state, context)

    assert not config.adaptive_feature_family
    assert not config.adaptive_window
    assert not config.adaptive_budget
    chex.assert_trees_all_close(result.state.family_logits, state.family_logits)
    chex.assert_trees_all_close(result.state.window_logits, state.window_logits)
    chex.assert_trees_all_close(result.state.budget_logit, state.budget_logit)
    chex.assert_trees_all_close(
        prediction.scope_weights,
        prediction.feature_mask.astype(jnp.float32),
    )
    assert float(prediction.effective_budget) == pytest.approx(config.max_features)


def test_associative_memory_learns_repeated_binding() -> None:
    learner = AssociativeMemoryLearner(
        AssociativeMemoryConfig(
            vocab_size=6,
            block_size=4,
            suffix_length=3,
            max_features=64,
        )
    )
    context = jnp.asarray([1, 2, 3, 4], dtype=jnp.int32)
    contexts = jnp.tile(context[None, :], (32, 1))
    labels = jnp.full((32,), 5, dtype=jnp.int32)

    result = run_associative_memory_arrays(learner, learner.init(), contexts, labels)
    chex.assert_tree_all_finite((result.predictions, result.metrics))
    chex.assert_trees_all_equal(
        result.updates_applied,
        jnp.ones((contexts.shape[0],), dtype=jnp.bool_),
    )

    initial_nll = float(jnp.mean(result.metrics[:4, 0]))
    final_nll = float(jnp.mean(result.metrics[-4:, 0]))
    final_accuracy = float(jnp.mean(result.metrics[-4:, 1]))

    assert final_nll < initial_nll * 0.5
    assert final_accuracy == 1.0


def test_associative_memory_respects_fixed_budget() -> None:
    learner = AssociativeMemoryLearner(
        AssociativeMemoryConfig(
            vocab_size=9,
            block_size=5,
            suffix_length=4,
            max_features=4,
        )
    )
    contexts = jnp.asarray(
        [
            [0, 1, 2, 3, 4],
            [4, 3, 2, 1, 0],
            [1, 3, 5, 7, 8],
        ],
        dtype=jnp.int32,
    )
    labels = jnp.asarray([1, 2, 3], dtype=jnp.int32)

    result = run_associative_memory_arrays(learner, learner.init(), contexts, labels)
    occupied = int(jnp.sum(result.state.counts > 0.0))

    assert occupied <= 4
    assert int(result.state.replacements) > 0


def test_associative_adaptive_family_scope_prefers_useful_pairs() -> None:
    learner = AssociativeMemoryLearner(
        AssociativeMemoryConfig(
            vocab_size=4,
            block_size=4,
            suffix_length=3,
            max_features=128,
            adaptive_feature_family=True,
            scope_lr=0.2,
        )
    )
    base_contexts = jnp.asarray(
        [
            [0, 0, 1, 2],
            [0, 0, 1, 3],
            [0, 0, 2, 2],
            [0, 0, 2, 3],
        ],
        dtype=jnp.int32,
    )
    base_labels = jnp.asarray([0, 1, 1, 0], dtype=jnp.int32)
    pattern_ids = jnp.arange(240, dtype=jnp.int32) % base_contexts.shape[0]
    contexts = base_contexts[pattern_ids]
    labels = base_labels[pattern_ids]

    result = run_associative_memory_arrays(learner, learner.init(), contexts, labels)
    prediction = learner.predict(result.state, contexts[-1])

    assert float(result.state.family_logits[1]) > float(result.state.family_logits[0])
    assert float(prediction.family_probs[1]) > 0.80


def test_associative_adaptive_window_scope_prefers_useful_long_window() -> None:
    learner = AssociativeMemoryLearner(
        AssociativeMemoryConfig(
            vocab_size=4,
            block_size=4,
            suffix_length=4,
            feature_family="suffix_pair",
            max_features=512,
            adaptive_window=True,
            scope_lr=0.2,
        )
    )
    contexts_list: list[list[int]] = []
    labels_list: list[int] = []
    for _ in range(3):
        for old_token in (1, 2):
            for middle_a in range(4):
                for middle_b in range(4):
                    for recent_token in range(4):
                        contexts_list.append(
                            [old_token, middle_a, middle_b, recent_token]
                        )
                        labels_list.append(old_token - 1)
    contexts = jnp.asarray(contexts_list, dtype=jnp.int32)
    labels = jnp.asarray(labels_list, dtype=jnp.int32)

    result = run_associative_memory_arrays(learner, learner.init(), contexts, labels)
    prediction = learner.predict(result.state, contexts[-1])

    assert float(result.state.window_logits[-1]) > float(result.state.window_logits[0])
    assert float(prediction.window_probs[-1]) > 0.80


def test_associative_adaptive_budget_expands_under_replacement_pressure() -> None:
    learner = AssociativeMemoryLearner(
        AssociativeMemoryConfig(
            vocab_size=13,
            block_size=4,
            suffix_length=3,
            max_features=64,
            adaptive_budget=True,
            initial_budget_fraction=0.10,
            budget_lr=0.5,
        )
    )
    contexts = (
        jnp.arange(80 * 4, dtype=jnp.int32).reshape(80, 4)
        * jnp.asarray([1, 2, 3, 4], dtype=jnp.int32)
    ) % 13
    labels = (contexts[:, 0] + 2 * contexts[:, 1] + 3 * contexts[:, 2]) % 13
    state = learner.init()
    initial_budget = learner.predict(state, contexts[0]).effective_budget

    result = run_associative_memory_arrays(learner, state, contexts, labels)
    final_budget = learner.predict(result.state, contexts[-1]).effective_budget

    assert int(result.state.replacements) > 0
    assert float(result.state.budget_logit) > float(state.budget_logit)
    assert float(final_budget) > float(initial_budget) + 10.0


def test_step2_associative_facade_smoke_and_roundtrip() -> None:
    config = Step2AssociativeConfig(
        vocab_size=8,
        block_size=5,
        suffix_length=3,
        max_features=128,
        adaptive_feature_family=True,
        adaptive_window=True,
        adaptive_budget=True,
        initial_budget_fraction=0.25,
    )
    restored = Step2AssociativeConfig.from_dict(config.to_dict())
    learner = make_step2_associative_learner(restored)

    assert learner.config == config.to_core_config()
    assert learner.config.adaptive_feature_family
    assert learner.config.adaptive_window
    assert learner.config.adaptive_budget
    assert learner.config.initial_budget_fraction == pytest.approx(0.25)

    result = run_step2_associative_smoke(config, steps=64, seed=0, window=16)
    assert result.finite
    assert result.metrics_shape == (64, 8)
    assert result.final_window_nll < result.initial_window_nll
