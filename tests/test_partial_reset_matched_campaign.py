"""Matched five-seed development campaign for calibrated partial resets."""

from __future__ import annotations

import copy
import dataclasses
import os
from pathlib import Path

import jax
import numpy as np
import pytest

from alberta_framework.benchmarks import partial_reset_matched_campaign as campaign
from alberta_framework.benchmarks.partial_reset_matched_campaign import (
    ARM_IDS,
    FROZEN_PLAN,
    SEEDS,
    build_partial_reset_campaign,
    validate_partial_reset_campaign,
    write_partial_reset_campaign_new,
)

pytestmark = pytest.mark.integration


def _data() -> tuple[np.ndarray, np.ndarray]:
    labels = np.arange(12, dtype=np.int32) % 3
    features = np.linspace(-1.0, 1.0, 72, dtype=np.float32).reshape(12, 6)
    return features, labels


def _tiny_plan() -> campaign.PartialResetCampaignPlan:
    return campaign._test_plan(
        config=campaign.IPMNISTConfig(
            n_tasks=2,
            task_length=4,
            input_dim=6,
            hidden1=5,
            hidden2=4,
            n_classes=3,
        ),
        data_x=_data()[0],
        data_y=_data()[1],
    )


@pytest.fixture
def tiny_plan(monkeypatch: pytest.MonkeyPatch) -> campaign.PartialResetCampaignPlan:
    plan = _tiny_plan()
    monkeypatch.setattr(campaign, "FROZEN_PLAN", plan)
    return plan


def _run_tiny() -> dict[str, object]:
    return build_partial_reset_campaign(*_data())


def test_frozen_plan_is_full_ipmnist_five_seed_five_arm_nonpromoting() -> None:
    assert FROZEN_PLAN.seeds == SEEDS == (156301, 156302, 156303, 156304, 156305)
    assert (
        FROZEN_PLAN.arm_ids
        == ARM_IDS
        == (
            "cpr_ipmnist",
            "cpr_hard_reset",
            "cpr_l2_init",
            "cpr_utility_free",
            "cpr_off",
        )
    )
    assert FROZEN_PLAN.config == campaign.IPMNISTConfig()
    assert FROZEN_PLAN.canonical_dataset_required is True
    payload = campaign.plan_payload(FROZEN_PLAN)
    assert payload["cells"] == 25
    assert payload["observations_per_cell"] == 1_000_000
    assert payload["total_observations"] == 25_000_000
    assert payload["rng_contract"] == campaign.RNG_CONTRACT
    assert payload["policy"] == {
        "development_only": True,
        "scientific_promotion_allowed": False,
        "publication_equivalent": False,
        "outcome_retention_required": True,
    }


def test_campaign_runs_complete_roster_and_strictly_replays_identities(
    tiny_plan: campaign.PartialResetCampaignPlan,
) -> None:
    report = _run_tiny()
    assert validate_partial_reset_campaign(report, *_data()) == report
    records = report["records"]
    assert isinstance(records, list)
    assert len(records) == 25
    assert [(record["seed"], record["arm"]) for record in records] == [
        (seed, arm) for seed in tiny_plan.seeds for arm in tiny_plan.arm_ids
    ]
    assert report["validation_scope"] == (
        "strict_receipt_schedule_initial_state_and_arithmetic_replay_without_learner_reexecution"
    )
    for seed in tiny_plan.seeds:
        matched = [record for record in records if record["seed"] == seed]
        assert len({record["rng_root_sha256"] for record in matched}) == 1
        assert len({record["schedule_sha256"] for record in matched}) == 1
        assert len({record["initial_parameters_sha256"] for record in matched}) == 1
        assert len({record["initial_learner_state_sha256"] for record in matched}) == 1
        assert len({record["resources_sha256"] for record in matched}) == 1
    assert set(report["paired_comparisons"]) == set(ARM_IDS[:-1])
    assert report["resources"]["cells"] == 25
    assert report["resources"]["total_observations"] == 200
    assert report["resources"]["total_model_queries"] == 400
    assert report["resources"]["physical_peak_rss_claimed"] is False
    assert report["decision"] == {
        "status": "inconclusive",
        "reason": "no_registered_selection_rule",
        "candidate_selected": None,
    }


def test_explicit_threefry_is_invariant_to_ambient_default(
    tiny_plan: campaign.PartialResetCampaignPlan,
) -> None:
    baseline = campaign._seed_execution_identity(
        tiny_plan.seeds[0], tiny_plan, n_train=_data()[0].shape[0]
    )
    with jax.default_prng_impl("rbg"):
        ambient = campaign._seed_execution_identity(
            tiny_plan.seeds[0], tiny_plan, n_train=_data()[0].shape[0]
        )
    assert baseline == ambient

    run_config = campaign.IPMNISTConfig(
        n_tasks=1,
        task_length=4,
        input_dim=6,
        hidden1=5,
        hidden2=4,
        n_classes=3,
    )
    spec = campaign.screening_spec("cpr_off")
    expected = campaign.run_screening_config(*_data(), spec, tiny_plan.seeds[0], run_config)
    with jax.default_prng_impl("rbg"):
        actual = campaign.run_screening_config(*_data(), spec, tiny_plan.seeds[0], run_config)
    np.testing.assert_array_equal(actual.per_task_accuracy, expected.per_task_accuracy)
    np.testing.assert_array_equal(actual.per_task_loss, expected.per_task_loss)
    np.testing.assert_array_equal(actual.per_task_plasticity, expected.per_task_plasticity)


def test_validator_rejects_roster_identity_resource_arithmetic_and_data_forgery(
    tiny_plan: campaign.PartialResetCampaignPlan,
) -> None:
    report = _run_tiny()
    mutations: list[tuple[dict[str, object], str]] = []

    missing = copy.deepcopy(report)
    missing["records"].pop()
    mutations.append((missing, "records|roster"))

    schedule = copy.deepcopy(report)
    schedule["records"][0]["schedule_sha256"] = "0" * 64
    mutations.append((schedule, "schedule"))

    initial = copy.deepcopy(report)
    initial["records"][0]["initial_parameters_sha256"] = "0" * 64
    mutations.append((initial, "initial"))

    resource = copy.deepcopy(report)
    resource["records"][0]["development_record"]["resources"]["updates"] -= 1
    mutations.append((resource, "resource|record"))

    paired = copy.deepcopy(report)
    paired["paired_comparisons"]["cpr_ipmnist"]["mean_delta"] += 0.25
    mutations.append((paired, "paired|mean"))

    policy = copy.deepcopy(report)
    policy["policy"]["scientific_promotion_allowed"] = True
    mutations.append((policy, "nonpromoting|policy"))

    decision = copy.deepcopy(report)
    decision["decision"]["status"] = "selected"
    mutations.append((decision, "decision|inconclusive"))

    for forged, message in mutations:
        with pytest.raises(ValueError, match=message):
            validate_partial_reset_campaign(forged, *_data())

    changed_x, changed_y = _data()
    changed_x = changed_x.copy()
    changed_x[0, 0] += np.float32(0.125)
    with pytest.raises(ValueError, match="dataset"):
        validate_partial_reset_campaign(report, changed_x, changed_y)


def test_validator_rejects_source_runtime_and_hostile_json(
    tiny_plan: campaign.PartialResetCampaignPlan,
) -> None:
    report = _run_tiny()
    source = copy.deepcopy(report)
    source["source_identity"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source"):
        validate_partial_reset_campaign(source, *_data())

    runtime = copy.deepcopy(report)
    runtime["runtime_identity"]["python"]["version"] = "forged"
    with pytest.raises(ValueError, match="runtime"):
        validate_partial_reset_campaign(runtime, *_data())

    class DictSubclass(dict[str, object]):
        pass

    with pytest.raises(ValueError, match="exact JSON|exact object"):
        validate_partial_reset_campaign(DictSubclass(report), *_data())

    cyclic: dict[str, object] = {}
    cyclic["cycle"] = cyclic
    with pytest.raises(ValueError, match="alias|cycle|exact JSON"):
        validate_partial_reset_campaign(cyclic, *_data())

    oversized = copy.deepcopy(report)
    oversized["records"] = [None] * (campaign.MAX_RECORDS + 1)
    with pytest.raises(ValueError, match="bound|records|node"):
        validate_partial_reset_campaign(oversized, *_data())


def test_create_only_writer_validates_before_publish_and_never_overwrites(
    tiny_plan: campaign.PartialResetCampaignPlan,
    tmp_path: Path,
) -> None:
    report = _run_tiny()
    destination = tmp_path / "partial-reset-report.json"
    assert write_partial_reset_campaign_new(destination, report, *_data()) == destination
    assert destination.exists()
    assert os.stat(destination).st_mode & 0o222 == 0
    loaded = campaign.load_partial_reset_campaign(destination, *_data())
    assert loaded == report
    with pytest.raises(FileExistsError, match="overwrite|exists"):
        write_partial_reset_campaign_new(destination, report, *_data())

    forged = copy.deepcopy(report)
    forged["policy"]["scientific_promotion_allowed"] = True
    absent = tmp_path / "must-not-exist.json"
    with pytest.raises(ValueError, match="nonpromoting|policy"):
        write_partial_reset_campaign_new(absent, forged, *_data())
    assert not absent.exists()

    symlink = tmp_path / "symlink.json"
    target = tmp_path / "target.json"
    target.write_text("preserve", encoding="utf-8")
    symlink.symlink_to(target)
    with pytest.raises(FileExistsError, match="overwrite|exists"):
        write_partial_reset_campaign_new(symlink, report, *_data())
    assert target.read_text(encoding="utf-8") == "preserve"


def test_plan_and_result_dataclasses_reject_nonexact_types() -> None:
    plan = _tiny_plan()
    with pytest.raises(ValueError, match="seeds"):
        dataclasses.replace(plan, seeds=tuple(np.int64(seed) for seed in plan.seeds))
    with pytest.raises(ValueError, match="arm"):
        dataclasses.replace(plan, arm_ids=list(plan.arm_ids))
    with pytest.raises(ValueError, match="canonical"):
        dataclasses.replace(plan, canonical_dataset_required=1)
