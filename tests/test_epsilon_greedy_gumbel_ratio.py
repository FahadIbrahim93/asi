"""Regression: epsilon-greedy importance ratios must use the exact Gumbel-max
greedy distribution, not a uniform tie over values within atol=1e-6.

_select_action_epsilon_greedy selects argmax(q + 1e-6 * Gumbel(0,1)), whose
greedy-component probabilities are softmax(q / 1e-6). The probability helper
instead treated every value within absolute tolerance 1e-6 of the maximum as
an exactly uniform tie, so near-tied Q values produced ratio 1.0 where the
true target/behavior ratio differs. Issue #2136.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from alberta_framework.core.options import (
    _clipped_epsilon_greedy_importance_ratio,
    _epsilon_greedy_action_probabilities,
)

pytestmark = pytest.mark.unit


def test_near_tied_q_values_use_gumbel_max_probabilities() -> None:
    q = jnp.array([0.0, 5e-7], dtype=jnp.float32)
    greedy = jax.nn.softmax(q / 1e-6)

    behavior = _epsilon_greedy_action_probabilities(q, jnp.asarray(0.2))
    target = _epsilon_greedy_action_probabilities(q, jnp.asarray(0.0))

    expected_behavior = 0.2 / 2 + (1 - 0.2) * greedy
    expected_target = 0.0 / 2 + (1 - 0.0) * greedy
    assert jnp.allclose(behavior, expected_behavior, atol=1e-5)
    assert jnp.allclose(target, expected_target, atol=1e-5)


def test_near_tied_importance_ratio_matches_true_ratio() -> None:
    # q_weights @ observation reproduces q = [0, 5e-7].
    q_weights = jnp.array([[0.0, 0.0], [0.0, 5e-7]], dtype=jnp.float32)
    observation = jnp.array([1.0, 1.0], dtype=jnp.float32)

    greedy = jax.nn.softmax(jnp.array([0.0, 5e-7]) / 1e-6)
    behavior = 0.2 / 2 + (1 - 0.2) * greedy
    target = 0.0 / 2 + (1 - 0.0) * greedy
    expected = target / jnp.maximum(behavior, 1e-6)

    for action in (0, 1):
        ratio = _clipped_epsilon_greedy_importance_ratio(
            q_weights,
            observation,
            jnp.asarray(action, dtype=jnp.int32),
            behavior_epsilon=0.2,
            target_epsilon=0.0,
            clip=10.0,
        )
        assert float(ratio) == pytest.approx(float(expected[action]), abs=1e-5)


def test_exact_tie_stays_uniform() -> None:
    q = jnp.array([1.0, 1.0], dtype=jnp.float32)
    probs = _epsilon_greedy_action_probabilities(q, jnp.asarray(0.2))
    assert jnp.allclose(probs, jnp.array([0.5, 0.5]), atol=1e-6)


def test_clear_winner_stays_greedy() -> None:
    q = jnp.array([0.0, 1.0], dtype=jnp.float32)
    probs = _epsilon_greedy_action_probabilities(q, jnp.asarray(0.0))
    # softmax([0,1]/1e-6) is effectively [0, 1]
    assert float(probs[1]) == pytest.approx(1.0, abs=1e-4)
    assert float(probs[0]) == pytest.approx(0.0, abs=1e-4)
