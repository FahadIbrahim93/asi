"""Cheap contracts for the permanently nonpromoting reference-life scorecard."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import pytest

from alberta_framework.benchmarks import reference_life_scorecard as scorecard
from alberta_framework.benchmarks.reference_life_scorecard import (
    ARM_ROSTER,
    ENVIRONMENT_ROSTER,
    SEED_ROSTER,
    ReferenceLifeDevelopmentPlan,
    StreamingRunSummary,
    build_development_plan,
    canonical_json_bytes,
    estimate_jax_resources,
    parameter_change_check,
    summarize_run_records,
    write_new_json,
)

pytestmark = pytest.mark.unit


def test_fixed_plan_is_immutable_canonical_and_explicit() -> None:
    plan = build_development_plan()
    payload = plan.to_payload()

    assert dataclasses.is_dataclass(plan)
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.plan_sha256 = "0" * 64  # type: ignore[misc]
    assert plan.seeds == tuple(range(70_000, 70_012)) == SEED_ROSTER
    assert plan.arms == ARM_ROSTER
    assert plan.environments == ENVIRONMENT_ROSTER
    assert payload["evidence_policy"]["permanently_nonpromoting"] is True
    assert payload["evidence_policy"]["scientific_promotion_allowed"] is False
    assert payload["protocols"]["switching_two_state"]["horizon"] == 4_000
    assert payload["protocols"]["switching_two_state"]["phase_length"] == 250
    assert payload["protocols"]["switching_two_state"]["post_switch_window"] == 50
    assert payload["protocols"]["riverswim"]["horizon"] == 20_000
    assert payload["protocols"]["riverswim"]["n_states"] == 6
    assert payload["protocols"]["riverswim"]["early_window"] == 2_000
    assert payload["protocols"]["riverswim"]["late_window"] == 2_000
    assert canonical_json_bytes(payload) == canonical_json_bytes(
        ReferenceLifeDevelopmentPlan.from_payload(payload).to_payload()
    )

    payload["seed_roster"][0] = 1
    assert plan.seeds == SEED_ROSTER, "returned JSON must not alias immutable plan state"


def test_plan_rejects_tampering_even_if_digest_is_left_unchanged() -> None:
    payload = build_development_plan().to_payload()
    payload["protocols"]["riverswim"]["horizon"] -= 1
    with pytest.raises(ValueError, match="canonical fixed development plan"):
        ReferenceLifeDevelopmentPlan.from_payload(payload)


def test_cyclic_order_is_explicit_and_balanced() -> None:
    plan = build_development_plan()
    observed = [plan.arm_order(seed) for seed in SEED_ROSTER]
    assert observed[0] == ARM_ROSTER
    assert observed[1] == ARM_ROSTER[1:] + ARM_ROSTER[:1]
    assert observed[6] == ARM_ROSTER
    for position in range(len(ARM_ROSTER)):
        assert sorted(order[position] for order in observed) == sorted(list(ARM_ROSTER) * 2)


def test_streaming_switching_summary_keeps_only_fixed_aba_windows() -> None:
    summary = StreamingRunSummary.for_switching(
        horizon=8,
        phase_length=2,
        post_switch_window=2,
    )
    for index in range(8):
        summary.observe(
            reward=float(index),
            oracle_reward=10.0,
            regime_id=(index // 2) % 2,
            parameters_changed=index == 3,
            next_state_index=index % 2,
        )

    result = summary.finalize()
    assert result["accepted_events"] == 8
    assert result["parameter_change_events"] == 1
    assert result["windows"] == {
        "initial_a": {
            "event_count": 2,
            "reward_sum": 1.0,
            "mean_reward": 0.5,
            "mean_oracle_regret": 9.5,
        },
        "first_b": {
            "event_count": 2,
            "reward_sum": 5.0,
            "mean_reward": 2.5,
            "mean_oracle_regret": 7.5,
        },
        "return_a": {
            "event_count": 2,
            "reward_sum": 9.0,
            "mean_reward": 4.5,
            "mean_oracle_regret": 5.5,
        },
    }
    assert not hasattr(summary, "events")
    with pytest.raises(ValueError, match="horizon"):
        summary.observe(
            reward=0.0,
            oracle_reward=1.0,
            regime_id=0,
            parameters_changed=False,
            next_state_index=0,
        )


def test_streaming_river_summary_tracks_early_late_and_high_end_visits() -> None:
    summary = StreamingRunSummary.for_riverswim(
        horizon=6,
        early_window=2,
        late_window=2,
        n_states=3,
    )
    for index, state_index in enumerate((0, 1, 2, 2, 1, 2)):
        summary.observe(
            reward=float(index + 1),
            oracle_reward=7.0,
            regime_id=0,
            parameters_changed=False,
            next_state_index=state_index,
        )
    result = summary.finalize()
    assert result["windows"]["early"]["reward_sum"] == 3.0
    assert result["windows"]["late"]["reward_sum"] == 11.0
    assert result["high_end_visit_count"] == 3
    assert result["high_end_visit_rate"] == 0.5


@dataclasses.dataclass(frozen=True)
class _TinyState:
    weights: Any
    counter: Any


def test_resource_estimate_is_explicitly_a_pytree_estimate() -> None:
    estimate = estimate_jax_resources(
        _TinyState(
            weights=jnp.zeros((2, 3), dtype=jnp.float32),
            counter=jnp.asarray(0, dtype=jnp.int32),
        )
    )
    assert estimate["persistent_jax_array_bytes"] == 28
    assert estimate["persistent_jax_array_scalar_count"] == 7
    assert estimate["trainable_scalar_count_estimate"] == 6
    assert estimate["trainable_scalar_count_method"] == (
        "floating_jax_pytree_leaves_upper_bound"
    )


@pytest.mark.parametrize(
    ("arm", "changes", "passed"),
    [
        ("prototype", 2, True),
        ("prototype", 0, False),
        ("prototype_frozen", 0, True),
        ("prototype_frozen", 1, False),
        ("random", 0, True),
        ("privileged_oracle", 0, True),
    ],
)
def test_parameter_change_checks_are_fail_closed(
    arm: str, changes: int, passed: bool
) -> None:
    assert parameter_change_check(arm, changes)["passed"] is passed


def _summary_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    offsets = {
        "random": 0.0,
        "privileged_oracle": 100.0,
        "prototype": 50.0,
        "prototype_frozen": 0.0,
        "differential_sarsa": 40.0,
        "sarsa": 30.0,
    }
    for environment in ENVIRONMENT_ROSTER:
        for arm in ARM_ROSTER:
            for seed in SEED_ROSTER:
                records.append(
                    {
                        "environment_kind": environment,
                        "arm": arm,
                        "seed": seed,
                        "status": "completed",
                        "outcome": {"reward_sum": offsets[arm]},
                    }
                )
    return records


def test_summary_normalizes_within_environment_and_never_pools() -> None:
    records = _summary_records()
    summary = summarize_run_records(build_development_plan(), list(reversed(records)))
    assert summary["status"] == "development_scorecard_complete"
    assert summary["cross_environment_pooled_score"] is None
    assert summary["cross_environment_pooling_forbidden"] is True
    for environment in ENVIRONMENT_ROSTER:
        environment_summary = summary["environments"][environment]
        assert environment_summary["normalization"]["scale"] == 100.0
        assert environment_summary["arms"]["random"]["normalized_score_mean"] == 0.0
        assert environment_summary["arms"]["privileged_oracle"][
            "normalized_score_mean"
        ] == 1.0
        assert environment_summary["arms"]["differential_sarsa"][
            "paired_t_lcb_95"
        ] > 0.10
    assert canonical_json_bytes(summary) == canonical_json_bytes(
        summarize_run_records(build_development_plan(), records)
    )


def test_summary_retains_failures_and_reports_valid_baseline_failure() -> None:
    records = _summary_records()
    for record in records:
        if record["arm"] in ("differential_sarsa", "sarsa"):
            record["outcome"]["reward_sum"] = 5.0
    records[0] = {
        **records[0],
        "status": "failed",
        "outcome": None,
        "failure": {"stage": "step", "type": "RuntimeError", "message": "boom"},
    }
    summary = summarize_run_records(build_development_plan(), records)
    assert summary["failure_count"] == 1
    assert summary["failures"][0]["failure"]["message"] == "boom"
    assert summary["status"] == "valid_execution_failure"

    complete = [record for record in records if record["status"] == "completed"]
    complete.append(
        {
            **records[0],
            "status": "completed",
            "outcome": {"reward_sum": 0.0},
        }
    )
    summary = summarize_run_records(build_development_plan(), complete)
    assert summary["status"] == "valid_baseline_failure"


def test_new_json_publication_is_canonical_and_refuses_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "plan.json"
    payload = build_development_plan().to_payload()
    write_new_json(destination, payload)
    assert destination.read_bytes() == json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_new_json(destination, {"different": math.pi})


def test_failed_shard_is_retained_and_digest_tampering_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = {"schema": "test.identity.v1", "value": "fixed"}
    monkeypatch.setattr(scorecard, "_checkpoint_source_identity", lambda: identity)
    monkeypatch.setattr(scorecard, "_checkpoint_runtime_identity", lambda: identity)
    monkeypatch.setattr(scorecard, "_checkpoint_dependency_identity", lambda: identity)

    def fail_build(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError("intentional cheap build failure")

    monkeypatch.setattr(scorecard, "build_scorecard_runner", fail_build)
    plan = build_development_plan()
    spec = scorecard.iter_run_specs(plan)[0]
    record = scorecard.run_scorecard_shard(plan, spec)
    assert record["status"] == "failed"
    assert record["failure"] == {
        "stage": "build",
        "type": "RuntimeError",
        "message": "intentional cheap build failure",
        "accepted_events": 0,
    }
    assert record["partial_outcome"]["summary_mode"] == (
        "streaming_o1_no_retained_events"
    )
    assert scorecard.validate_scorecard_run_record(record)["valid"] is True

    tampered = json.loads(json.dumps(record))
    tampered["failure"]["message"] = "forged"
    with pytest.raises(ValueError, match="content digest mismatch"):
        scorecard.validate_scorecard_run_record(tampered)
