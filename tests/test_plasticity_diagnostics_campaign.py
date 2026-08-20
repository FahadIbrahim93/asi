"""Strict matched campaign contracts for bounded plasticity diagnostics."""

from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from alberta_framework.evaluation import plasticity_diagnostics_campaign as campaign

pytestmark = pytest.mark.integration


def _dataset() -> tuple[np.ndarray, np.ndarray]:
    rows = 64
    values = np.arange(rows * 784, dtype=np.float32).reshape(rows, 784)
    return (values % 256) / 255.0, np.arange(rows, dtype=np.int32) % 10


def _source() -> dict[str, str]:
    return {
        "kind": "caller-supplied-array-materialization",
        "dataset_name": "test-mnist",
        "dataset_version": "fixture-v1",
        "split": "train",
        "acquisition": "test fixture; not canonical MNIST",
        "artifact_sha256": hashlib.sha256(b"test-fixture-v1").hexdigest(),
    }


@pytest.fixture(scope="module")
def report() -> dict[str, object]:
    return campaign.run_plasticity_diagnostics_campaign(*_dataset(), source=_source())


def test_plan_is_exact_five_seed_three_arm_and_inconclusive_only(
    report: dict[str, object],
) -> None:
    assert campaign.SEEDS == (15830, 15831, 15832, 15833, 15834)
    assert report["plan"] == {
        "seeds": list(campaign.SEEDS),
        "arms": ["sgd_control", "cbp_mechanism_off", "cbp_bounded"],
        "profile_id": "bounded-development",
        "run_order": "seed_major_arm_minor",
        "rng_impl": "threefry2x32",
    }
    assert report["decision"] == {
        "status": "inconclusive",
        "reason": "no_registered_selection_rule",
        "candidate_selected": None,
    }
    assert report["policy"]["scientific_promotion_allowed"] is False
    assert report["scope_gaps"] == [
        "no_800_task_mnist_protocol",
        "no_three_hidden_layer_width_2000_network",
        "no_official_continual_backprop_trace_or_code_parity",
        "no_continual_imagenet_protocol_or_accelerator_budget",
        "no_continual_rl_mujoco_protocol_or_environment_budget",
        "no_scientific_retention_or_fresh_seed_evaluation",
    ]


def test_strict_validator_replays_exact_five_seed_roster(report: dict[str, object]) -> None:
    assert campaign.validate_plasticity_diagnostics_campaign(
        report, *_dataset(), source=_source()
    ) == report
    assert [(row["seed"], row["arm"]) for row in report["records"]] == [
        (seed, arm) for seed in campaign.SEEDS for arm in campaign.ARM_IDS
    ]
    assert report["resources"]["arm_cells"] == 15
    assert report["resources"]["total_data_steps"] == 5 * 3 * 8 * 64
    assert report["resources"]["total_parameter_updates"] == 5 * 3 * 8 * 64
    assert report["resources"]["total_model_queries"] == 5 * 3 * 8 * 64 * 3
    initial_bytes = {
        identity["initial_state_numeric_bytes"] for identity in report["seed_identities"]
    }
    assert initial_bytes == {report["resources"]["max_cell_persistent_bytes"]}
    assert {
        row["receipt"]["persistent_bytes"] for row in report["records"]
    } == initial_bytes


def test_hostile_self_consistent_forgery_is_rejected(report: dict[str, object]) -> None:
    mutations: list[tuple[dict[str, object], str]] = []
    metric = copy.deepcopy(report)
    metric["records"][0]["task_accuracy"][0] = 1.0 - metric["records"][0]["task_accuracy"][0]
    campaign._resign_for_test(metric)
    mutations.append((metric, "replay|record"))
    schedule = copy.deepcopy(report)
    schedule["seed_identities"][0]["schedule_sha256"] = "sha256:" + "0" * 64
    campaign._resign_for_test(schedule)
    mutations.append((schedule, "schedule|identity"))
    dataset = copy.deepcopy(report)
    dataset["identity"]["dataset"]["dataset_sha256"] = "0" * 64
    campaign._resign_for_test(dataset)
    mutations.append((dataset, "dataset"))
    source = copy.deepcopy(report)
    source["identity"]["dataset"]["source"]["dataset_version"] = "forged"
    campaign._resign_for_test(source)
    mutations.append((source, "source|dataset"))
    resource = copy.deepcopy(report)
    resource["resources"]["total_parameter_updates"] -= 1
    campaign._resign_for_test(resource)
    mutations.append((resource, "resource|replay"))
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
            campaign.validate_plasticity_diagnostics_campaign(
                forged, *_dataset(), source=_source()
            )


def test_validator_rejects_nonexact_cyclic_and_wrong_supplied_data(
    report: dict[str, object],
) -> None:
    class DictSubclass(dict[str, object]):
        pass

    with pytest.raises(ValueError, match="exact JSON|exact object"):
        campaign.validate_plasticity_diagnostics_campaign(
            DictSubclass(report), *_dataset(), source=_source()
        )
    cyclic: dict[str, object] = {}
    cyclic["cycle"] = cyclic
    with pytest.raises(ValueError, match="alias|cycle|exact JSON"):
        campaign.validate_plasticity_diagnostics_campaign(
            cyclic, *_dataset(), source=_source()
        )
    images, labels = _dataset()
    images[0, 0] = 0.5
    with pytest.raises(ValueError, match="dataset"):
        campaign.validate_plasticity_diagnostics_campaign(
            report, images, labels, source=_source()
        )


def test_create_only_writer_validates_and_never_overwrites(
    tmp_path: Path, report: dict[str, object]
) -> None:
    path = tmp_path / "campaign.json"
    assert campaign.write_plasticity_diagnostics_campaign_new(
        path, report, *_dataset(), source=_source()
    ) == path
    assert os.stat(path).st_mode & 0o222 == 0
    assert campaign.load_plasticity_diagnostics_campaign(
        path, *_dataset(), source=_source()
    ) == report
    with pytest.raises(FileExistsError, match="overwrite|exists"):
        campaign.write_plasticity_diagnostics_campaign_new(
            path, report, *_dataset(), source=_source()
        )
