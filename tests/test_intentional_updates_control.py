"""End-to-end, permanently nonpromoting Intentional Updates control lane."""

from __future__ import annotations

import copy

import jax.random as jr
import numpy as np
import pytest

import alberta_framework.benchmarks.intentional_updates_control as control_lane
from alberta_framework.benchmarks.intentional_updates_control import (
    ARM_NAMES,
    IntentionalUpdatesControlConfig,
    run_intentional_updates_control,
    validate_intentional_updates_control_record,
)

SMALL = IntentionalUpdatesControlConfig(horizon=48, phase_length=12)


def test_arm_family_is_three_exact_matched_pairs() -> None:
    assert ARM_NAMES == (
        "fixed_td0",
        "intentional_td0",
        "fixed_trace",
        "intentional_trace",
        "fixed_q_learning",
        "intentional_q_learning",
    )


@pytest.mark.parametrize(
    ("fixed", "off"),
    [
        ("fixed_td0", "intentional_td0_off"),
        ("fixed_trace", "intentional_trace_off"),
        ("fixed_q_learning", "intentional_q_learning_off"),
    ],
)
def test_mechanism_off_reduces_bit_exactly_to_fixed_step_control(
    fixed: str, off: str
) -> None:
    expected = run_intentional_updates_control(fixed, seed=7, config=SMALL)
    actual = run_intentional_updates_control(off, seed=7, config=SMALL)
    assert actual["arm"] == off
    for key in ("trajectory", "final_state", "metrics", "resources"):
        assert actual[key] == expected[key]


@pytest.mark.parametrize("arm", ARM_NAMES)
def test_each_arm_runs_through_the_continuing_stream_and_consumer(arm: str) -> None:
    record = run_intentional_updates_control(arm, seed=19, config=SMALL)
    assert validate_intentional_updates_control_record(record) == record
    assert record["policy"] == {
        "development_only": True,
        "scientific_promotion_allowed": False,
        "negative_results_retained": True,
        "publication_equivalent": False,
    }
    assert record["resources"]["environment_steps"] == SMALL.horizon
    assert record["resources"]["updates"] == SMALL.horizon
    assert record["resources"]["reward_observations"] == SMALL.horizon
    assert record["resources"]["trajectory_items"] == SMALL.horizon
    assert len(record["trajectory"]["rewards"]) == SMALL.horizon
    assert np.isfinite(record["metrics"]["mean_reward"])
    assert np.isfinite(record["metrics"]["mean_squared_td_error"])


def test_record_binds_current_source_runtime_workload_and_exact_receipts() -> None:
    record = run_intentional_updates_control("intentional_trace", seed=3, config=SMALL)
    assert set(record["identity"]) == {
        "agent_rng_impl",
        "current_source_sha256",
        "runtime",
        "workload_sha256",
    }
    assert record["identity"]["agent_rng_impl"] == "threefry2x32"
    assert set(record["identity"]["current_source_sha256"]) == {
        "intentional_updates_control.py",
        "plasticity_comparators.py",
    }
    assert set(record["identity"]["runtime"]) == {
        "python",
        "jax",
        "jaxlib",
        "numpy",
        "backend",
        "platform",
    }
    assert record["resources"]["model_queries"] == 2 * SMALL.horizon
    assert record["resources"]["action_queries"] == 0
    assert record["resources"]["eligibility_trace_updates"] == SMALL.horizon
    assert record["resources"]["intentional_step_size_solves"] == SMALL.horizon


def test_control_receipt_counts_actual_action_and_model_queries() -> None:
    record = run_intentional_updates_control("intentional_q_learning", seed=3, config=SMALL)
    assert record["resources"]["model_queries"] == 2 * SMALL.horizon
    assert record["resources"]["action_queries"] == SMALL.horizon
    assert record["resources"]["rng_splits"] == SMALL.horizon
    assert record["resources"]["rng_fold_ins"] == SMALL.horizon
    # 2x2 weights, trace, and diagonal state plus one two-word Threefry key.
    assert record["resources"]["persistent_numeric_bytes"] == 56


def test_control_rng_root_requests_threefry_explicitly(monkeypatch) -> None:
    calls: list[object] = []
    real_key = jr.key

    def recording_key(seed: int, *, impl: object = None):
        calls.append(impl)
        return real_key(seed, impl=impl)

    monkeypatch.setattr(control_lane.jr, "key", recording_key)
    run_intentional_updates_control("fixed_q_learning", seed=3, config=SMALL)
    assert calls == ["threefry2x32"]


def test_trace_energy_uses_paper_lambda_gamma_discount() -> None:
    config = IntentionalUpdatesControlConfig(horizon=2, phase_length=1)
    record = run_intentional_updates_control("intentional_trace", seed=0, config=config)
    rho_first = 1.0 / np.sqrt((1.0 - config.diagonal_decay) + config.diagonal_epsilon)
    second_moment = config.diagonal_decay * (1.0 - config.diagonal_decay) + (
        1.0 - config.diagonal_decay
    )
    rho_second = 1.0 / np.sqrt(second_moment + config.diagonal_epsilon)
    expected = config.trace_decay * config.discount * rho_first + rho_second
    assert record["final_state"]["discounted_gradient_energy"] == pytest.approx(
        expected, rel=2e-6
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["resources"].__setitem__("updates", 47),
        lambda value: value["identity"]["runtime"].__setitem__("jax", "forged"),
        lambda value: value["identity"]["current_source_sha256"].__setitem__(
            "intentional_updates_control.py", "0" * 64
        ),
        lambda value: value["trajectory"]["rewards"].__setitem__(0, 999.0),
        lambda value: value["policy"].__setitem__("scientific_promotion_allowed", True),
    ],
)
def test_validator_rejects_receipt_identity_result_and_policy_forgery(mutation) -> None:
    record = run_intentional_updates_control("intentional_td0", seed=5, config=SMALL)
    hostile = copy.deepcopy(record)
    mutation(hostile)
    with pytest.raises(ValueError):
        validate_intentional_updates_control_record(hostile)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizon": 0},
        {"horizon": 10_001},
        {"phase_length": 0},
        {"discount": 1.0},
        {"trace_decay": 1.0},
        {"epsilon_greedy": -0.1},
    ],
)
def test_config_fails_closed_outside_bounded_domain(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        IntentionalUpdatesControlConfig(**kwargs)


def test_unknown_arm_and_non_exact_seed_fail_before_execution() -> None:
    with pytest.raises(ValueError, match="arm"):
        run_intentional_updates_control("invented", seed=0, config=SMALL)
    with pytest.raises(ValueError, match="seed"):
        run_intentional_updates_control("fixed_td0", seed=True, config=SMALL)
