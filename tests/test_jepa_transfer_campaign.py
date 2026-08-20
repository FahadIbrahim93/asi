"""Contracts for the permanently nonpromoting #1577 matched campaign."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
from pathlib import Path
from typing import Any, cast

import jax
import pytest

import alberta_framework.evaluation.jepa_transfer_campaign as campaign
from alberta_framework.benchmarks import jepa_transfer_feasibility as lane


@pytest.fixture(scope="module")
def protocol() -> lane.JEPATransferProtocol:
    return lane.JEPATransferProtocol(
        steps=8,
        phase_length=4,
        pretraining_steps=4,
        warmup_steps=2,
        exploration_period=2,
    )


@pytest.fixture(scope="module")
def lane_result(protocol: lane.JEPATransferProtocol) -> lane.JEPATransferResult:
    return lane.run_jepa_transfer_feasibility(protocol)


def _resign(value: dict[str, Any]) -> None:
    unsigned = dict(value)
    unsigned.pop("result_sha256", None)
    value["result_sha256"] = hashlib.sha256(campaign._canonical(unsigned)).hexdigest()


def _retimed(result: lane.JEPATransferResult) -> lane.JEPATransferResult:
    arms = tuple(
        dataclasses.replace(
            arm,
            timing=dataclasses.replace(
                arm.timing,
                environment_ns=arm.timing.environment_ns + 1,
                online_update_ns=arm.timing.online_update_ns + 1,
            ),
        )
        for arm in result.arms
    )
    return dataclasses.replace(result, arms=arms)


def test_campaign_has_exact_five_seed_roster_and_replays_timing_neutrally(
    monkeypatch: pytest.MonkeyPatch,
    protocol: lane.JEPATransferProtocol,
    lane_result: lane.JEPATransferResult,
) -> None:
    monkeypatch.setattr(campaign, "run_jepa_transfer_feasibility", lambda actual: lane_result)
    result = cast(dict[str, Any], campaign.run_jepa_transfer_campaign(protocol))
    monkeypatch.setattr(
        campaign, "run_jepa_transfer_feasibility", lambda actual: _retimed(lane_result)
    )
    campaign.validate_jepa_transfer_campaign(result, protocol=protocol)
    assert len(campaign.CAMPAIGN_SEEDS) == 5
    assert result["roster"] == [
        [seed, arm] for seed in campaign.CAMPAIGN_SEEDS for arm in lane.FROZEN_ARM_IDS
    ]
    assert result["decision"] == "inconclusive"
    assert result["identity"]["prng_implementation"] == "threefry2x32"
    assert all(len(item["initial_state_sha256"]) == 64 for item in result["executions"])
    assert all(len(item["schedule_sha256"]) == 64 for item in result["executions"])
    assert all(len(item["replay_sha256"]) == 64 for item in result["executions"])


def test_replay_rejects_self_consistent_result_and_execution_forgery(
    monkeypatch: pytest.MonkeyPatch,
    protocol: lane.JEPATransferProtocol,
    lane_result: lane.JEPATransferResult,
) -> None:
    monkeypatch.setattr(campaign, "run_jepa_transfer_feasibility", lambda actual: lane_result)
    result = cast(dict[str, Any], campaign.run_jepa_transfer_campaign(protocol))
    forged = copy.deepcopy(result)
    forged["lane_result"]["arms"][0]["return_sum"] += 1.0
    forged["aggregate"] = campaign._aggregate(forged["lane_result"]["arms"])
    _resign(forged)
    with pytest.raises(ValueError, match="reexecution"):
        campaign.validate_jepa_transfer_campaign(forged, protocol=protocol)

    forged = copy.deepcopy(result)
    forged["executions"][0]["replay_sha256"] = "0" * 64
    _resign(forged)
    with pytest.raises(ValueError, match="execution identity"):
        campaign.validate_jepa_transfer_campaign(forged, protocol=protocol)


def test_validator_is_pure_and_campaign_is_inconclusive_only(
    monkeypatch: pytest.MonkeyPatch,
    protocol: lane.JEPATransferProtocol,
    lane_result: lane.JEPATransferResult,
) -> None:
    monkeypatch.setattr(campaign, "run_jepa_transfer_feasibility", lambda actual: lane_result)
    result = cast(dict[str, Any], campaign.run_jepa_transfer_campaign(protocol))
    before = copy.deepcopy(result)
    campaign.validate_jepa_transfer_campaign(result, protocol=protocol)
    assert result == before

    decided = copy.deepcopy(result)
    decided["decision"] = "supported"
    _resign(decided)
    before_failure = copy.deepcopy(decided)
    with pytest.raises(ValueError, match="inconclusive"):
        campaign.validate_jepa_transfer_campaign(decided, protocol=protocol)
    assert decided == before_failure

    mistyped = copy.deepcopy(result)
    mistyped["policy"]["visual_robotics_parity_claimed"] = 0
    _resign(mistyped)
    with pytest.raises(ValueError, match="nonpromoting"):
        campaign.validate_jepa_transfer_campaign(mistyped, protocol=protocol)


def test_writer_replays_and_never_replaces(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    protocol: lane.JEPATransferProtocol,
    lane_result: lane.JEPATransferResult,
) -> None:
    monkeypatch.setattr(campaign, "run_jepa_transfer_feasibility", lambda actual: lane_result)
    result = campaign.run_jepa_transfer_campaign(protocol)
    destination = tmp_path / "jepa-transfer-campaign.json"
    campaign.write_jepa_transfer_campaign(destination, result, protocol=protocol)
    retained = campaign.load_jepa_transfer_campaign(destination)
    campaign.validate_jepa_transfer_campaign(retained, protocol=protocol)
    with pytest.raises(FileExistsError):
        campaign.write_jepa_transfer_campaign(destination, result, protocol=protocol)


def test_source_and_rng_identities_are_complete_and_ambient_independent(
    protocol: lane.JEPATransferProtocol,
) -> None:
    source = campaign._source_identity()
    assert set(source) == {"package_python_tree_sha256", "file_count"}
    assert type(source["file_count"]) is int and source["file_count"] > 100
    baseline = campaign._executions(protocol)
    with jax.default_prng_impl("rbg"):
        under_rbg = campaign._executions(protocol)
    assert under_rbg == baseline
