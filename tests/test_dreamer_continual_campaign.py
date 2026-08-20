"""Strict five-seed campaign for the bounded native Dreamer family."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import jax
import pytest

from alberta_framework.evaluation import dreamer_continual_campaign as campaign

pytestmark = pytest.mark.integration


def test_plan_is_exact_five_seed_three_arm_and_inconclusive_only() -> None:
    assert campaign.SEEDS == (1760, 1761, 1762, 1763, 1764)
    assert campaign.ARM_IDS == (
        "guarded_imagination",
        "imagination_off",
        "privileged_task_control",
    )
    assert campaign.FROZEN_PLAN == {
        "steps_per_task": 4,
        "replay_capacity": 8,
        "imaginations_per_step": 2,
    }
    assert campaign.POLICY["scientific_promotion_allowed"] is False


def test_campaign_runs_exact_roster_and_strictly_replays_all_five_runs() -> None:
    report = campaign.run_dreamer_continual_campaign()
    assert campaign.validate_dreamer_continual_campaign(report) == report
    assert [(row["seed"], row["arm"]) for row in report["records"]] == [
        (seed, arm) for seed in campaign.SEEDS for arm in campaign.ARM_IDS
    ]
    assert len(report["seed_identities"]) == 5
    assert report["decision"] == {
        "status": "inconclusive",
        "reason": "no_registered_selection_rule",
        "candidate_selected": None,
    }
    assert report["resources"]["runs"] == 5
    assert report["resources"]["arm_cells"] == 15
    assert report["resources"]["total_environment_steps"] == 180
    assert report["resources"]["physical_peak_rss_claimed"] is False


def test_runner_and_replay_identities_are_ambient_prng_invariant() -> None:
    with jax.default_prng_impl("threefry2x32"):
        first = campaign.run_dreamer_continual_campaign()
    with jax.default_prng_impl("rbg"):
        second = campaign.run_dreamer_continual_campaign()
    assert first["seed_identities"] == second["seed_identities"]
    assert first["records"] == second["records"]


def test_hostile_self_consistent_forgery_is_rejected() -> None:
    report = campaign.run_dreamer_continual_campaign()
    mutations: list[tuple[dict[str, object], str]] = []
    metric = copy.deepcopy(report)
    metric["records"][0]["task_returns"][0] += 2.0
    campaign._resign_for_test(metric)
    mutations.append((metric, "replay|record"))
    replay = copy.deepcopy(report)
    replay["seed_identities"][0]["replay_schedule_sha256"] = "sha256:" + "0" * 64
    campaign._resign_for_test(replay)
    mutations.append((replay, "replay|identity"))
    resource = copy.deepcopy(report)
    resource["resources"]["total_environment_steps"] -= 1
    campaign._resign_for_test(resource)
    mutations.append((resource, "resource|replay"))
    source = copy.deepcopy(report)
    source["identity"]["sources"][0]["sha256"] = "0" * 64
    campaign._resign_for_test(source)
    mutations.append((source, "source"))
    runtime = copy.deepcopy(report)
    runtime["identity"]["runtime"]["python_version"] = "forged"
    campaign._resign_for_test(runtime)
    mutations.append((runtime, "runtime"))
    policy = copy.deepcopy(report)
    policy["policy"]["scientific_promotion_allowed"] = True
    campaign._resign_for_test(policy)
    mutations.append((policy, "policy|nonpromoting"))
    decision = copy.deepcopy(report)
    decision["decision"]["status"] = "selected"
    campaign._resign_for_test(decision)
    mutations.append((decision, "decision|inconclusive"))
    for forged, message in mutations:
        with pytest.raises(ValueError, match=message):
            campaign.validate_dreamer_continual_campaign(forged)


def test_validator_rejects_nonexact_and_cyclic_json() -> None:
    report = campaign.run_dreamer_continual_campaign()

    class DictSubclass(dict[str, object]):
        pass

    with pytest.raises(ValueError, match="exact JSON|exact object"):
        campaign.validate_dreamer_continual_campaign(DictSubclass(report))
    cyclic: dict[str, object] = {}
    cyclic["cycle"] = cyclic
    with pytest.raises(ValueError, match="alias|cycle|exact JSON"):
        campaign.validate_dreamer_continual_campaign(cyclic)


def test_create_only_writer_validates_and_never_overwrites(tmp_path: Path) -> None:
    report = campaign.run_dreamer_continual_campaign()
    path = tmp_path / "campaign.json"
    assert campaign.write_dreamer_continual_campaign_new(path, report) == path
    assert os.stat(path).st_mode & 0o222 == 0
    assert campaign.load_dreamer_continual_campaign(path) == report
    with pytest.raises(FileExistsError, match="overwrite|exists"):
        campaign.write_dreamer_continual_campaign_new(path, report)
    target = tmp_path / "target"
    target.write_text("preserve", encoding="utf-8")
    symlink = tmp_path / "link.json"
    symlink.symlink_to(target)
    with pytest.raises(FileExistsError, match="overwrite|exists"):
        campaign.write_dreamer_continual_campaign_new(symlink, report)
    assert target.read_text(encoding="utf-8") == "preserve"
