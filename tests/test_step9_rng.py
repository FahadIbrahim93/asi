"""Tests for RNG threading through Step 9 dreaming.

Regression coverage for a defect where ``dream_step`` seeded the rollout
behavior chain with the same post-split ``key`` it returned in the scan
carry: ``jr.split`` is deterministic, so the next dream's
``(key, candidate_key)`` pair was bit-identical to the ``(stored, sample)``
pair the rollout's first ``sample_action`` had already produced from the
same parent — consecutive dreams drew correlated randomness (issue #160).
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from alberta_framework.steps.step9 import (
    Step9DreamingConfig,
    init_step9_state,
    make_step9_components,
    step9_update,
)


def _accepted_dream_result(planning_budget: int, seed: int):
    cfg = Step9DreamingConfig(
        observation_dim=2,
        n_actions=2,
        model_hidden_sizes=(),
        model_sparsity=0.0,
        planning_budget=planning_budget,
        dream_rollout_horizon=1,
        dream_candidate_count=1,
        dreaming_warmup_steps=0,
        dreaming_max_model_error=1e30,
    )
    agent, model, buffer = make_step9_components(cfg)
    state = init_step9_state(
        agent, model, buffer,
        key=jr.key(seed),
        initial_observation=jnp.zeros(2),
    )
    return step9_update(
        cfg, agent, model, buffer,
        state,
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array([0.1, 0.2], dtype=jnp.float32),
    )


class TestStep9DreamRng:
    def test_rollout_seed_is_not_the_carried_dream_key(self) -> None:
        """The rollout behavior chain must not be seeded with the carried key.

        On the defective path the returned behavior rng after one accepted
        horizon-1 dream equals ``split(split(scan_key)[0])[0]`` — the exact
        chain the next dream would re-derive by re-splitting the carry.
        """
        result = _accepted_dream_result(planning_budget=1, seed=7)
        assert bool(result.dream_accepted[0])
        scan_key = result.real_control_result.state.rng_key
        collided_chain = jr.split(jr.split(scan_key)[0])[0]
        final_behavior = result.state.behavior_model_state.rng_key
        assert not jnp.array_equal(
            jr.key_data(final_behavior), jr.key_data(collided_chain)
        ), (
            "dream rollout was seeded with the carried scan key: the next "
            "dream re-splits that key into the exact (stored, sample) pair "
            "the rollout's first sample_action already consumed"
        )

    def test_dream_key_streams_are_pairwise_distinct(self) -> None:
        """The carry, candidate, and rollout streams must never share a key."""
        result = _accepted_dream_result(planning_budget=2, seed=11)
        scan_key = result.real_control_result.state.rng_key
        carry_1, candidate_1, rollout_1 = jr.split(scan_key, 3)
        carry_2, candidate_2, rollout_2 = jr.split(carry_1, 3)
        derived = [carry_1, candidate_1, rollout_1, carry_2, candidate_2, rollout_2]
        raw = [jr.key_data(k) for k in derived]
        for i in range(len(raw)):
            for j in range(i + 1, len(raw)):
                assert not jnp.array_equal(raw[i], raw[j]), (
                    f"derived dream key streams {i} and {j} collide"
                )
