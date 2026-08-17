"""Strict transaction tests for the keyboard-chord learner."""

from __future__ import annotations

from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.oak import (
    KeyboardChordLearnerConfig,
    KeyboardChordLearnerState,
    init_keyboard_chord_learner,
    update_keyboard_chord_learner,
)


@pytest.mark.parametrize("disable_jit", [True, False])
def test_saturation_continues_learning_atomically(disable_jit: bool) -> None:
    config = KeyboardChordLearnerConfig(
        n_options=2,
        step_size=0.5,
        baseline_decay=0.5,
    )
    state = init_keyboard_chord_learner(config).replace(
        step_count=jnp.asarray(np.iinfo(np.int32).max - 1, dtype=jnp.int32)
    )
    chord = jnp.asarray([1.0, 0.0], dtype=jnp.float32)

    with jax.disable_jit(disable_jit):
        at_max = update_keyboard_chord_learner(
            config,
            state,
            chord,
            jnp.asarray(1.0, dtype=jnp.float32),
        )
        after_max = update_keyboard_chord_learner(
            config,
            at_max,
            chord,
            jnp.asarray(2.0, dtype=jnp.float32),
        )
        rejected = update_keyboard_chord_learner(
            config,
            after_max,
            chord,
            jnp.asarray(jnp.inf, dtype=jnp.float32),
        )

    assert int(at_max.step_count) == np.iinfo(np.int32).max
    assert int(after_max.step_count) == np.iinfo(np.int32).max
    assert not bool(jnp.array_equal(at_max.chord_vector, after_max.chord_vector))
    assert float(after_max.reward_baseline) > float(at_max.reward_baseline)
    chex.assert_trees_all_equal(rejected, after_max)


def test_scan_saturates_and_rejects_invalid_rows_atomically() -> None:
    config = KeyboardChordLearnerConfig(
        n_options=2,
        step_size=0.5,
        baseline_decay=0.5,
    )
    initial = init_keyboard_chord_learner(config).replace(
        step_count=jnp.asarray(np.iinfo(np.int32).max - 1, dtype=jnp.int32)
    )
    chords = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=jnp.float32)
    rewards = jnp.asarray([1.0, jnp.nan, 2.0], dtype=jnp.float32)

    @jax.jit
    def run(
        state: KeyboardChordLearnerState,
        chord_rows: jax.Array,
        reward_rows: jax.Array,
    ) -> KeyboardChordLearnerState:
        def update(
            carry: KeyboardChordLearnerState,
            row: tuple[jax.Array, jax.Array],
        ) -> tuple[KeyboardChordLearnerState, None]:
            next_state = update_keyboard_chord_learner(config, carry, row[0], row[1])
            return next_state, None

        return jax.lax.scan(update, state, (chord_rows, reward_rows))[0]

    final = run(initial, chords, rewards)
    expected = update_keyboard_chord_learner(
        config,
        update_keyboard_chord_learner(config, initial, chords[0], rewards[0]),
        chords[2],
        rewards[2],
    )

    chex.assert_trees_all_equal(final, expected)
    assert int(final.step_count) == np.iinfo(np.int32).max


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("chord_vector", jnp.zeros((2,), dtype=jnp.int32)),
        ("chord_vector", jnp.zeros((1,), dtype=jnp.float32)),
        ("reward_baseline", jnp.zeros((1,), dtype=jnp.float32)),
        ("reward_baseline", jnp.asarray(0, dtype=jnp.int32)),
        ("step_count", jnp.zeros((1,), dtype=jnp.int32)),
        ("step_count", jnp.asarray(0.0, dtype=jnp.float32)),
        ("step_count", jnp.asarray(False, dtype=jnp.bool_)),
    ],
)
def test_state_metadata_is_exact(field: str, value: jax.Array) -> None:
    config = KeyboardChordLearnerConfig(n_options=2)
    state = init_keyboard_chord_learner(config).replace(**{field: value})

    with pytest.raises(ValueError, match=f"state.{field}"):
        update_keyboard_chord_learner(
            config,
            state,
            jnp.ones((2,), dtype=jnp.float32),
            jnp.asarray(1.0, dtype=jnp.float32),
        )


@pytest.mark.parametrize(
    ("chord", "reward", "match"),
    [
        (jnp.ones((2,), dtype=jnp.int32), jnp.asarray(1.0, dtype=jnp.float32), "selected_chord"),
        (
            jnp.ones((1, 2), dtype=jnp.float32),
            jnp.asarray(1.0, dtype=jnp.float32),
            "selected_chord",
        ),
        (jnp.ones((2,), dtype=jnp.float32), jnp.asarray(1, dtype=jnp.int32), "reward"),
        (jnp.ones((2,), dtype=jnp.float32), jnp.ones((1,), dtype=jnp.float32), "reward"),
    ],
)
def test_inputs_are_not_reshaped_or_narrowed(
    chord: jax.Array,
    reward: jax.Array,
    match: str,
) -> None:
    config = KeyboardChordLearnerConfig(n_options=2)
    with pytest.raises(ValueError, match=match):
        update_keyboard_chord_learner(
            config,
            init_keyboard_chord_learner(config),
            chord,
            reward,
        )


def test_hostile_objects_are_rejected_without_hooks() -> None:
    class Hostile:
        calls = 0

        def __getattribute__(self, name: str) -> Any:
            if name == "calls":
                return object.__getattribute__(self, name)
            type(self).calls += 1
            raise AssertionError("hostile attribute hook executed")

        def __jax_array__(self) -> jax.Array:
            type(self).calls += 1
            raise AssertionError("hostile JAX hook executed")

    config = KeyboardChordLearnerConfig(n_options=2)
    state = init_keyboard_chord_learner(config)
    hostile = Hostile()

    with pytest.raises(ValueError, match="state must be"):
        update_keyboard_chord_learner(  # type: ignore[arg-type]
            config, hostile, jnp.ones((2,), dtype=jnp.float32), jnp.asarray(1.0)
        )
    with pytest.raises(ValueError, match="selected_chord"):
        update_keyboard_chord_learner(  # type: ignore[arg-type]
            config, state, hostile, jnp.asarray(1.0, dtype=jnp.float32)
        )
    assert Hostile.calls == 0


def test_invalid_source_values_hold_the_complete_state() -> None:
    config = KeyboardChordLearnerConfig(n_options=2)
    base = init_keyboard_chord_learner(config)
    invalid_states = (
        base.replace(step_count=jnp.asarray(-1, dtype=jnp.int32)),
        base.replace(chord_vector=jnp.asarray([jnp.inf, 0.0], dtype=jnp.float32)),
        base.replace(reward_baseline=jnp.asarray(jnp.nan, dtype=jnp.float32)),
    )
    for state in invalid_states:
        result = update_keyboard_chord_learner(
            config,
            state,
            jnp.ones((2,), dtype=jnp.float32),
            jnp.asarray(1.0, dtype=jnp.float32),
        )
        chex.assert_trees_all_equal(result, state)


def test_wide_finite_chord_preserves_direction_and_bounds_update() -> None:
    config = KeyboardChordLearnerConfig(
        n_options=2,
        step_size=1.0,
        baseline_decay=0.0,
        max_norm=0.75,
    )
    state = init_keyboard_chord_learner(config)
    largest = jnp.finfo(jnp.float32).max
    result = update_keyboard_chord_learner(
        config,
        state,
        jnp.asarray([largest, largest], dtype=jnp.float32),
        jnp.asarray(largest, dtype=jnp.float32),
    )

    chex.assert_tree_all_finite(result)
    assert float(jnp.linalg.norm(result.chord_vector)) == pytest.approx(0.75)
    assert bool(jnp.all(result.chord_vector > 0.0))
    assert int(result.step_count) == 1
