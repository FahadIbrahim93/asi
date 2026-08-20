"""Contracts for the permanently nonpromoting #1575 matched campaign."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, cast

import jax
import pytest

import alberta_framework.evaluation.action_conditioned_latent_campaign as campaign
from alberta_framework.benchmarks.action_conditioned_latent import (
    FROZEN_ARM_IDS,
    ActionLatentProtocol,
    run_action_conditioned_latent_lane,
)


@pytest.fixture(scope="module")
def protocol() -> ActionLatentProtocol:
    return ActionLatentProtocol(steps=8, phase_length=4, warmup_steps=2, exploration_period=2)


@pytest.fixture(scope="module")
def lane_result(protocol: ActionLatentProtocol):
    return run_action_conditioned_latent_lane(protocol)


def _resign(value: dict[str, Any]) -> None:
    unsigned = dict(value)
    unsigned.pop("result_sha256", None)
    value["result_sha256"] = hashlib.sha256(campaign._canonical(unsigned)).hexdigest()


def test_campaign_has_exact_five_seed_arm_roster_and_replays(
    monkeypatch: pytest.MonkeyPatch,
    protocol: ActionLatentProtocol,
    lane_result: object,
) -> None:
    calls = 0

    def fake(actual: ActionLatentProtocol):
        nonlocal calls
        calls += 1
        assert actual == protocol
        return lane_result

    monkeypatch.setattr(campaign, "run_action_conditioned_latent_lane", fake)
    result = cast(dict[str, Any], campaign.run_action_latent_campaign(protocol))
    campaign.validate_action_latent_campaign(result, protocol=protocol)
    assert calls == 2
    assert len(campaign.CAMPAIGN_SEEDS) == 5
    assert result["roster"] == [
        [seed, arm] for seed in campaign.CAMPAIGN_SEEDS for arm in FROZEN_ARM_IDS
    ]
    assert result["decision"] == "inconclusive"
    assert result["identity"]["prng_implementation"] == "threefry2x32"
    assert all(len(item["initial_state_sha256"]) == 64 for item in result["executions"])
    assert all(len(item["schedule_sha256"]) == 64 for item in result["executions"])


def test_replay_rejects_self_consistent_metric_and_identity_forgery(
    monkeypatch: pytest.MonkeyPatch,
    protocol: ActionLatentProtocol,
    lane_result: object,
) -> None:
    monkeypatch.setattr(campaign, "run_action_conditioned_latent_lane", lambda actual: lane_result)
    result = cast(dict[str, Any], campaign.run_action_latent_campaign(protocol))
    forged = copy.deepcopy(result)
    forged["lane_result"]["arms"][0]["return_sum"] += 1.0
    forged["aggregate"] = campaign._aggregate(forged["lane_result"]["arms"])
    _resign(forged)
    with pytest.raises(ValueError, match="reexecution"):
        campaign.validate_action_latent_campaign(forged, protocol=protocol)

    forged = copy.deepcopy(result)
    forged["executions"][0]["initial_state_sha256"] = "0" * 64
    _resign(forged)
    with pytest.raises(ValueError, match="execution identity"):
        campaign.validate_action_latent_campaign(forged, protocol=protocol)


def test_campaign_is_permanently_inconclusive_and_type_strict(
    monkeypatch: pytest.MonkeyPatch,
    protocol: ActionLatentProtocol,
    lane_result: object,
) -> None:
    monkeypatch.setattr(campaign, "run_action_conditioned_latent_lane", lambda actual: lane_result)
    result = cast(dict[str, Any], campaign.run_action_latent_campaign(protocol))
    decided = copy.deepcopy(result)
    decided["decision"] = "supported"
    _resign(decided)
    with pytest.raises(ValueError, match="inconclusive"):
        campaign.validate_action_latent_campaign(decided, protocol=protocol)

    mistyped = copy.deepcopy(result)
    mistyped["policy"]["development_only"] = 1
    _resign(mistyped)
    with pytest.raises(ValueError, match="nonpromoting"):
        campaign.validate_action_latent_campaign(mistyped, protocol=protocol)


def test_writer_replays_and_never_replaces(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    protocol: ActionLatentProtocol,
    lane_result: object,
) -> None:
    monkeypatch.setattr(campaign, "run_action_conditioned_latent_lane", lambda actual: lane_result)
    result = campaign.run_action_latent_campaign(protocol)
    destination = tmp_path / "action-latent-campaign.json"
    campaign.write_action_latent_campaign(destination, result, protocol=protocol)
    retained = campaign.load_action_latent_campaign(destination)
    campaign.validate_action_latent_campaign(retained, protocol=protocol)
    with pytest.raises(FileExistsError):
        campaign.write_action_latent_campaign(destination, result, protocol=protocol)


def test_source_identity_binds_complete_package_python_tree() -> None:
    identity = campaign._source_identity()
    assert set(identity) == {"package_python_tree_sha256", "file_count"}
    assert type(identity["file_count"]) is int and identity["file_count"] > 100
    assert len(cast(str, identity["package_python_tree_sha256"])) == 64


def test_execution_roots_ignore_ambient_rbg_default(
    protocol: ActionLatentProtocol,
) -> None:
    baseline = campaign._executions(protocol)
    with jax.default_prng_impl("rbg"):
        under_rbg = campaign._executions(protocol)
    assert under_rbg == baseline


def test_execution_identities_bind_both_requested_control_states(
    protocol: ActionLatentProtocol,
) -> None:
    executions = campaign._executions(protocol)
    by_arm = {item["arm_id"]: item for item in executions[: len(FROZEN_ARM_IDS)]}
    assert set(by_arm) == set(FROZEN_ARM_IDS)
    for arm_id in ("reconstruction_control", "one_step_ftl_control"):
        assert len(cast(str, by_arm[arm_id]["initial_state_sha256"])) == 64
        assert by_arm[arm_id]["prng_implementation"] == "threefry2x32"


def test_campaign_aggregate_includes_sarsa_agent_updates(
    monkeypatch: pytest.MonkeyPatch,
    protocol: ActionLatentProtocol,
    lane_result: object,
) -> None:
    monkeypatch.setattr(campaign, "run_action_conditioned_latent_lane", lambda actual: lane_result)
    result = cast(dict[str, Any], campaign.run_action_latent_campaign(protocol))
    resources = result["aggregate"]["arms"]["sarsa_control"]["total_additive_resources"]
    assert resources["model_updates"] == 0
    assert resources["agent_updates"] == len(campaign.CAMPAIGN_SEEDS) * protocol.steps
    assert resources["training_queries"] == len(campaign.CAMPAIGN_SEEDS) * protocol.steps
