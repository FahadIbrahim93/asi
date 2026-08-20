"""Five-seed nonpromoting campaign for the additive gradual-transition family."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import jax
import numpy as np
import pytest

from alberta_framework.evaluation import gradual_micro_phase_campaign as campaign

pytestmark = pytest.mark.integration


def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray([[-1.0, 1.0], [1.0, -1.0]], dtype=np.float32)
    return x, np.asarray([0, 1], np.int32), x.copy(), np.asarray([1, 0], np.int32)


@pytest.fixture(autouse=True)
def _tiny_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        campaign,
        "FROZEN_CONFIG",
        campaign.GradualMicroPhaseConfig(
            transition_intervals=2,
            phase_examples=2,
            input_dim=2,
            hidden1=2,
            hidden2=2,
            n_classes=2,
        ),
    )


def test_frozen_plan_is_five_seed_three_arm_and_permanently_nonpromoting() -> None:
    assert campaign.SEEDS == (156901, 156902, 156903, 156904, 156905)
    assert campaign.ARM_IDS == ("abrupt", "output_interpolation", "task_sampling")
    assert campaign.DEFAULT_FROZEN_CONFIG == campaign.GradualMicroPhaseConfig(
        transition_intervals=10,
        phase_examples=5000,
        input_dim=784,
        hidden1=300,
        hidden2=150,
        n_classes=10,
    )
    assert campaign.POLICY["scientific_promotion_allowed"] is False


def test_campaign_executes_exact_roster_and_strictly_replays_all_five_runs() -> None:
    report = campaign.run_gradual_micro_phase_campaign(*_data())
    assert campaign.validate_gradual_micro_phase_campaign(report, *_data()) == report
    assert [(r["seed"], r["arm"]) for r in report["records"]] == [
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
    assert report["resources"]["total_updates"] == 90
    assert report["resources"]["total_model_queries"] == 270
    assert report["resources"]["physical_peak_rss_claimed"] is False


def test_seed_identity_and_execution_are_ambient_prng_invariant() -> None:
    with jax.default_prng_impl("threefry2x32"):
        first = campaign.run_gradual_micro_phase_campaign(*_data())
    with jax.default_prng_impl("rbg"):
        second = campaign.run_gradual_micro_phase_campaign(*_data())
    assert first["seed_identities"] == second["seed_identities"]
    assert first["records"] == second["records"]


def test_validator_rejects_hostile_self_consistent_forgery_before_acceptance() -> None:
    report = campaign.run_gradual_micro_phase_campaign(*_data())
    mutations: list[tuple[dict[str, object], str]] = []
    metric = copy.deepcopy(report)
    metric["records"][0]["training_loss_sums"][0] += 0.25
    campaign._resign_for_test(metric)
    mutations.append((metric, "replay|record"))
    schedule = copy.deepcopy(report)
    schedule["seed_identities"][0]["schedule_sha256"] = "sha256:" + "0" * 64
    campaign._resign_for_test(schedule)
    mutations.append((schedule, "identity|schedule"))
    resource = copy.deepcopy(report)
    resource["resources"]["total_updates"] -= 1
    campaign._resign_for_test(resource)
    mutations.append((resource, "resource|replay"))
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
            campaign.validate_gradual_micro_phase_campaign(forged, *_data())

    changed = list(_data())
    changed[2] = changed[2].copy()
    changed[2][0, 0] += np.float32(0.25)
    with pytest.raises(ValueError, match="row-aligned|dataset"):
        campaign.validate_gradual_micro_phase_campaign(report, *changed)


def test_validator_rejects_source_runtime_and_nonexact_json() -> None:
    report = campaign.run_gradual_micro_phase_campaign(*_data())
    source = copy.deepcopy(report)
    source["identity"]["sources"][0]["sha256"] = "0" * 64
    campaign._resign_for_test(source)
    with pytest.raises(ValueError, match="source"):
        campaign.validate_gradual_micro_phase_campaign(source, *_data())
    runtime = copy.deepcopy(report)
    runtime["identity"]["runtime"]["python_version"] = "forged"
    campaign._resign_for_test(runtime)
    with pytest.raises(ValueError, match="runtime"):
        campaign.validate_gradual_micro_phase_campaign(runtime, *_data())

    class DictSubclass(dict[str, object]):
        pass

    with pytest.raises(ValueError, match="exact JSON|exact object"):
        campaign.validate_gradual_micro_phase_campaign(DictSubclass(report), *_data())
    cyclic: dict[str, object] = {}
    cyclic["cycle"] = cyclic
    with pytest.raises(ValueError, match="alias|cycle|exact JSON"):
        campaign.validate_gradual_micro_phase_campaign(cyclic, *_data())

    old_x, old_y, new_x, new_y = _data()
    with pytest.raises(ValueError, match="exactly one row-aligned phase"):
        campaign.run_gradual_micro_phase_campaign(
            np.concatenate((old_x, old_x)),
            np.concatenate((old_y, old_y)),
            np.concatenate((new_x, new_x)),
            np.concatenate((new_y, new_y)),
        )
    out_of_domain = old_x.copy()
    out_of_domain[0, 0] = np.float32(-1.25)
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        campaign.run_gradual_micro_phase_campaign(
            out_of_domain, old_y, out_of_domain.copy(), new_y
        )


def test_create_only_writer_validates_and_never_overwrites(tmp_path: Path) -> None:
    report = campaign.run_gradual_micro_phase_campaign(*_data())
    path = tmp_path / "campaign.json"
    assert campaign.write_gradual_micro_phase_campaign_new(path, report, *_data()) == path
    assert os.stat(path).st_mode & 0o222 == 0
    assert campaign.load_gradual_micro_phase_campaign(path, *_data()) == report
    with pytest.raises(FileExistsError, match="overwrite|exists"):
        campaign.write_gradual_micro_phase_campaign_new(path, report, *_data())
    target = tmp_path / "target"
    target.write_text("preserve", encoding="utf-8")
    symlink = tmp_path / "link.json"
    symlink.symlink_to(target)
    with pytest.raises(FileExistsError, match="overwrite|exists"):
        campaign.write_gradual_micro_phase_campaign_new(symlink, report, *_data())
    assert target.read_text(encoding="utf-8") == "preserve"
