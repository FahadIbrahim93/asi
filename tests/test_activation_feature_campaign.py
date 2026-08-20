from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import alberta_framework.benchmarks.activation_feature_ipmnist as activation
import alberta_framework.evaluation.activation_feature_campaign as campaign
from alberta_framework.benchmarks.ipmnist_screening import ScreeningRunResult
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig


@pytest.fixture(scope="module")
def data() -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros((5_000, 784), dtype=np.float32)
    y = np.arange(5_000, dtype=np.int32) % 10
    return x, y


@pytest.fixture(scope="module")
def cheap_plan(data: tuple[np.ndarray, np.ndarray]) -> dict[str, object]:
    return campaign.build_plan("cheap_screen", *data)


def _metric(arm: str, seed: int) -> float:
    seed_index = (
        campaign.CHEAP_SCREEN_SEEDS.index(seed)
        if seed in campaign.CHEAP_SCREEN_SEEDS
        else campaign.FULL_CONFIRMATION_SEEDS.index(seed)
    )
    base = 0.50 + seed_index / 1_000.0
    if arm == "smooth_leaky":
        return base + 0.10
    if arm == "aid":
        return base - 0.10
    return base


def _receipt(
    plan: dict[str, object], data: tuple[np.ndarray, np.ndarray], arm: str, seed: int
) -> dict[str, object]:
    config_values = cast(dict[str, int], plan["config"])
    config = IPMNISTConfig(**config_values)
    value = _metric(arm, seed)
    spec = activation.ACTIVATION_FEATURE_SPECS[arm]
    screening = ScreeningRunResult(
        config_name=arm,
        base_learner="upgd_w",
        hyperparameters=dict(spec.hyperparameters),
        seed=seed,
        config=config,
        per_task_accuracy=np.full(config.n_tasks, value, dtype=np.float64),
        per_task_loss=np.full(config.n_tasks, 1.0 - value, dtype=np.float64),
        per_task_plasticity=np.full(config.n_tasks, value, dtype=np.float64),
        wall_clock_seconds=1.0,
    )
    stage = cast(str, plan["stage"])
    execution_identity = campaign._expected_execution_identity(stage, seed, data[0].shape[0])
    result = activation.ActivationFeatureRunResult(
        screening=screening,
        dataset_sha256=activation._array_bundle_sha256(*data),
        schedule_sha256=execution_identity["schedule_sha256"],
        source_identity=activation._current_source_identity(),
        runtime_identity=activation._runtime_identity(),
        n_train=data[0].shape[0],
        peak_schedule_working_bytes=activation._preflight_activation_feature_resources(
            config, n_train=data[0].shape[0]
        ),
    )
    return (
        activation.activation_feature_result_payload(result, outcome="inconclusive")
        if stage == "cheap_screen"
        else activation.activation_feature_campaign_result_payload(
            result,
            outcome="inconclusive",
            development_seeds=campaign.FULL_CONFIRMATION_SEEDS,
        )
    )


def _shard(
    plan: dict[str, object],
    data: tuple[np.ndarray, np.ndarray],
    arm: str,
    seed: int,
    prerequisite: object | None = None,
) -> dict[str, object]:
    value = campaign._unsigned_shard(
        plan,
        arm=arm,
        seed=seed,
        authorization=campaign._execution_authorization(plan, prerequisite),
        execution_identity=campaign._expected_execution_identity(
            cast(str, plan["stage"]), seed, data[0].shape[0]
        ),
        result=_receipt(plan, data, arm, seed),
    )
    value["shard_sha256"] = campaign._digest(value)
    return campaign.validate_shard(value, plan, prerequisite=prerequisite)


def _matrix(
    plan: dict[str, object],
    data: tuple[np.ndarray, np.ndarray],
    prerequisite: object | None = None,
) -> list[dict[str, object]]:
    stage = cast(str, plan["stage"])
    seeds = (
        campaign.CHEAP_SCREEN_SEEDS
        if stage == "cheap_screen"
        else campaign.FULL_CONFIRMATION_SEEDS
    )
    authorization = campaign._execution_authorization(plan, prerequisite)
    shards: list[dict[str, object]] = []
    for seed in seeds:
        for arm in campaign.ARM_ROSTER:
            value = campaign._unsigned_shard(
                plan,
                arm=arm,
                seed=seed,
                authorization=authorization,
                execution_identity=campaign._expected_execution_identity(
                    stage, seed, data[0].shape[0]
                ),
                result=_receipt(plan, data, arm, seed),
            )
            value["shard_sha256"] = campaign._digest(value)
            shards.append(campaign._validate_shard_against_plan(value, plan, authorization))
    return shards


def _resign_shard(value: dict[str, Any]) -> None:
    unsigned = dict(value)
    unsigned.pop("shard_sha256", None)
    value["shard_sha256"] = campaign._digest(unsigned)


def _resign_aggregate(value: dict[str, Any]) -> None:
    unsigned = dict(value)
    unsigned.pop("aggregate_sha256", None)
    value["aggregate_sha256"] = campaign._digest(unsigned)


def test_both_plans_freeze_full_11_by_5_matrix_without_horizon_shrink(
    data: tuple[np.ndarray, np.ndarray],
) -> None:
    cheap = campaign.build_plan("cheap_screen", *data)
    full = campaign.build_plan("full_confirmation", *data)
    assert cheap["matrix"] == {
        "arms": list(campaign.ARM_ROSTER),
        "seeds": [0, 1, 2, 3, 4],
        "shard_count": 55,
        "ordering": "seed_major_then_arm_roster",
        "execution": "one_shard_per_fresh_python_process",
    }
    full_matrix = cast(dict[str, object], full["matrix"])
    assert full_matrix["arms"] == list(campaign.ARM_ROSTER)
    assert full_matrix["seeds"] == list(campaign.FULL_CONFIRMATION_SEEDS)
    assert full_matrix["shard_count"] == 55
    assert set(campaign.CHEAP_SCREEN_SEEDS).isdisjoint(campaign.FULL_CONFIRMATION_SEEDS)
    assert cheap["config"] == IPMNISTConfig(n_tasks=2, task_length=500).to_config()
    assert full["config"] == IPMNISTConfig(n_tasks=200, task_length=5_000).to_config()
    full_policy = cast(dict[str, object], full["policy"])
    assert full_policy["cross_stage_independent_confirmation_claimed"] is False
    full_gate = cast(dict[str, object], full["execution_gate"])
    assert full_gate["mode"] == "conditional_on_retained_cheap_screen"
    assert full_gate["complete_matrix_if_authorized"] is True
    assert campaign.validate_plan(copy.deepcopy(cheap), data_x=data[0], data_y=data[1]) == cheap


def test_plan_binds_exact_source_runtime_dataset_resources_and_paper_limits(
    cheap_plan: dict[str, object], data: tuple[np.ndarray, np.ndarray]
) -> None:
    identity = cast(dict[str, Any], cheap_plan["identity"])
    assert identity["dataset"]["sha256"] == activation._array_bundle_sha256(*data)
    assert set(identity["source_sha256"]) == {
        "alberta_framework/benchmarks/activation_feature_ipmnist.py",
        "alberta_framework/benchmarks/ipmnist_screening.py",
        "alberta_framework/benchmarks/plasticity_comparators.py",
        "alberta_framework/benchmarks/upgd_ipmnist.py",
        "alberta_framework/evaluation/activation_feature_campaign.py",
    }
    assert identity["runtime"]["packages"]["jax"]
    resources = cast(dict[str, Any], cheap_plan["resources"])
    assert resources["per_shard"]["data_steps"] == 1_000
    assert resources["matrix_totals"]["data_steps"] == 55_000
    parity = cast(dict[str, Any], cheap_plan["paper_parity"])
    assert parity["paper_protocol_parity_claimed"] is False
    assert parity["paper_result_reproduction_claimed"] is False


def test_plan_rejects_resigned_horizon_dataset_and_runtime_forgery(
    cheap_plan: dict[str, object], data: tuple[np.ndarray, np.ndarray]
) -> None:
    mutations: tuple[tuple[Any, str], ...] = (
        (lambda value: value["config"].__setitem__("task_length", 1), "digest|literal"),
        (
            lambda value: value["identity"]["dataset"].__setitem__("sha256", "0" * 64),
            "dataset|digest|literal",
        ),
        (
            lambda value: value["identity"]["runtime"].__setitem__("backend", "forged"),
            "digest|literal",
        ),
    )
    for mutate, match in mutations:
        forged = copy.deepcopy(cheap_plan)
        mutate(forged)
        unsigned = dict(forged)
        unsigned.pop("plan_sha256")
        forged["plan_sha256"] = campaign._digest(unsigned)
        with pytest.raises(ValueError, match=match):
            campaign.validate_plan(forged, data_x=data[0], data_y=data[1])


def test_shard_preserves_v1_and_cross_validates_wrapper_plan_and_resources(
    cheap_plan: dict[str, object], data: tuple[np.ndarray, np.ndarray]
) -> None:
    shard = _shard(cheap_plan, data, "aid", 2)
    receipt = cast(dict[str, object], shard["result"])
    assert receipt["schema"] == activation.RESULT_SCHEMA
    assert activation.validate_activation_feature_result(copy.deepcopy(receipt)) == receipt

    wrong_arm = copy.deepcopy(shard)
    wrong_arm["arm"] = "smooth_leaky"
    _resign_shard(cast(dict[str, Any], wrong_arm))
    with pytest.raises(ValueError, match="wrapper"):
        campaign.validate_shard(wrong_arm, cheap_plan)

    self_decided = copy.deepcopy(shard)
    cast(dict[str, object], self_decided["result"])["outcome"] = "supported"
    _resign_shard(cast(dict[str, Any], self_decided))
    with pytest.raises(ValueError, match="cannot self-assign"):
        campaign.validate_shard(self_decided, cheap_plan)

    shrunk = copy.deepcopy(shard)
    shrunk_receipt = cast(dict[str, object], shrunk["result"])
    cast(dict[str, object], shrunk_receipt["config"])["task_length"] = 1
    _resign_shard(cast(dict[str, Any], shrunk))
    with pytest.raises(ValueError):
        campaign.validate_shard(shrunk, cheap_plan)

    forged_schedule = copy.deepcopy(shard)
    cast(dict[str, object], forged_schedule["execution_identity"])[
        "schedule_sha256"
    ] = "0" * 64
    _resign_shard(cast(dict[str, Any], forged_schedule))
    with pytest.raises(ValueError, match="schedule, initialization, or PRNG"):
        campaign.validate_shard(forged_schedule, cheap_plan)


def test_complete_aggregate_uses_paired_student_t_and_multiplicity_rule(
    cheap_plan: dict[str, object], data: tuple[np.ndarray, np.ndarray]
) -> None:
    aggregate = campaign.build_aggregate(cheap_plan, list(reversed(_matrix(cheap_plan, data))))
    assert aggregate["status"] == "complete_with_supported_candidates"
    summary = cast(dict[str, Any], aggregate["summary"])
    comparisons = {
        row["candidate"]: row for row in summary["paired_comparisons"]
    }
    assert comparisons["smooth_leaky"]["outcome"] == "supported"
    assert comparisons["aid"]["outcome"] == "rejected"
    assert comparisons["deep_fourier"]["outcome"] == "inconclusive"
    assert len(summary["paired_comparisons"]) == 8
    statistics = cast(dict[str, Any], cheap_plan["statistics"])
    assert statistics["per_comparison_alpha"] == 0.05 / 8
    assert statistics["critical_value"] == 5.261057575065803
    assert summary["resources"]["totals"]["data_steps"] == 55_000
    assert campaign.validate_aggregate(copy.deepcopy(aggregate)) == aggregate


def test_full_confirmation_uses_fresh_v2_seeds_and_fails_closed_without_cheap_win(
    cheap_plan: dict[str, object], data: tuple[np.ndarray, np.ndarray]
) -> None:
    cheap = campaign.build_aggregate(cheap_plan, _matrix(cheap_plan, data))
    full_plan = campaign.build_plan("full_confirmation", *data)

    with pytest.raises(ValueError, match="requires the retained cheap-screen"):
        campaign.validate_shard(
            _shard(
                full_plan,
                data,
                "smooth_leaky",
                campaign.FULL_CONFIRMATION_SEEDS[0],
                cheap,
            ),
            full_plan,
        )

    full_shard = _shard(
        full_plan,
        data,
        "smooth_leaky",
        campaign.FULL_CONFIRMATION_SEEDS[0],
        cheap,
    )
    receipt = cast(dict[str, object], full_shard["result"])
    assert receipt["schema"] == activation.CAMPAIGN_RESULT_SCHEMA
    assert activation.validate_activation_feature_campaign_result(
        copy.deepcopy(receipt), development_seeds=campaign.FULL_CONFIRMATION_SEEDS
    ) == receipt
    with pytest.raises(ValueError, match="unsupported result identity"):
        activation.validate_activation_feature_result(receipt)

    full = campaign.build_aggregate(
        full_plan,
        _matrix(full_plan, data, cheap),
        prerequisite=cheap,
    )
    assert full["prerequisite"] == cheap
    assert campaign.validate_aggregate(copy.deepcopy(full)) == full


def test_full_confirmation_gate_rejects_complete_cheap_screen_without_primary_support(
    cheap_plan: dict[str, object], data: tuple[np.ndarray, np.ndarray]
) -> None:
    shards = _matrix(cheap_plan, data)
    for shard in shards:
        if shard["arm"] != "smooth_leaky":
            continue
        receipt = cast(dict[str, object], shard["result"])
        metrics = cast(dict[str, object], receipt["metrics"])
        metrics["asi_whole_stream_mean_accuracy"] = 0.50 + cast(int, shard["seed"]) / 1_000.0
        _resign_shard(cast(dict[str, Any], shard))
    cheap_without_support = campaign.build_aggregate(cheap_plan, shards)
    full_plan = campaign.build_plan("full_confirmation", *data)
    with pytest.raises(ValueError, match="did not authorize"):
        campaign._execution_authorization(full_plan, cheap_without_support)


def test_full_gate_runs_before_any_arm_execution(
    data: tuple[np.ndarray, np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    full_plan = campaign.build_plan("full_confirmation", *data)
    calls = 0

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("arm execution must remain gated")

    monkeypatch.setattr(campaign, "run_activation_feature_arm", forbidden)
    with pytest.raises(ValueError, match="requires the retained cheap-screen"):
        campaign.build_shard(
            full_plan,
            *data,
            arm="smooth_leaky",
            seed=campaign.FULL_CONFIRMATION_SEEDS[0],
        )
    assert calls == 0


def test_aggregate_rejects_missing_duplicate_and_self_consistent_statistic_forgery(
    cheap_plan: dict[str, object], data: tuple[np.ndarray, np.ndarray]
) -> None:
    shards = _matrix(cheap_plan, data)
    with pytest.raises(ValueError, match="complete"):
        campaign.build_aggregate(cheap_plan, shards[:-1])
    duplicate = shards[:-1] + [copy.deepcopy(shards[0])]
    with pytest.raises(ValueError, match="duplicate|incomplete"):
        campaign.build_aggregate(cheap_plan, duplicate)

    aggregate = cast(dict[str, Any], campaign.build_aggregate(cheap_plan, shards))
    aggregate["summary"]["paired_comparisons"][0]["mean_delta"] = 999.0
    _resign_aggregate(aggregate)
    with pytest.raises(ValueError, match="statistics|drifted"):
        campaign.validate_aggregate(aggregate)


@pytest.mark.skipif(
    not all(hasattr(os, name) for name in ("O_NOFOLLOW", "O_TMPFILE")),
    reason="strict loader/publication requires Linux descriptor support",
)
def test_strict_file_admission_and_append_only_writer(
    cheap_plan: dict[str, object], tmp_path: Path
) -> None:
    destination = tmp_path / "plan.json"
    campaign.write_new_json(destination, cheap_plan)
    assert campaign.load_json_strict(destination, max_bytes=campaign._MAX_SHARD_BYTES) == cheap_plan
    with pytest.raises(FileExistsError):
        campaign.write_new_json(destination, cheap_plan)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"x","schema":"y"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        campaign.load_json_strict(duplicate, max_bytes=1_024)

    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(destination)
    with pytest.raises(ValueError, match="JSON|regular"):
        campaign.load_json_strict(symlink, max_bytes=campaign._MAX_SHARD_BYTES)


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="strict loader requires O_NOFOLLOW")
def test_summarizer_rejects_two_paths_to_one_inode(
    cheap_plan: dict[str, object], data: tuple[np.ndarray, np.ndarray], tmp_path: Path
) -> None:
    shards = _matrix(cheap_plan, data)
    paths: list[Path] = []
    for index, shard in enumerate(shards):
        path = tmp_path / f"{index}.json"
        path.write_text(json.dumps(shard), encoding="utf-8")
        paths.append(path)
    alias = tmp_path / "alias.json"
    os.link(paths[0], alias)
    paths[-1] = alias
    with pytest.raises(ValueError, match="unique regular files"):
        campaign.summarize_shard_files(cheap_plan, paths)


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="strict loader requires O_NOFOLLOW")
def test_summarizer_uses_metadata_from_the_descriptor_that_supplied_bytes(
    cheap_plan: dict[str, object],
    data: tuple[np.ndarray, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shards = _matrix(cheap_plan, data)
    paths: list[Path] = []
    for index, shard in enumerate(shards):
        path = tmp_path / f"{index}.json"
        path.write_text(json.dumps(shard), encoding="utf-8")
        paths.append(path)
    incoming = tmp_path / "incoming.json"
    incoming.write_text(json.dumps(shards[1]), encoding="utf-8")
    alias = tmp_path / "incoming-alias.json"
    os.link(incoming, alias)
    paths[1] = alias

    real_open = os.open
    swapped = False

    def swapping_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        nonlocal swapped
        if not swapped and Path(path) == paths[0]:
            os.replace(incoming, paths[0])
            swapped = True
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", swapping_open)
    with pytest.raises(ValueError, match="unique regular files"):
        campaign.summarize_shard_files(cheap_plan, paths)
    assert swapped is True


@pytest.mark.skipif(
    not all(hasattr(os, name) for name in ("O_NOFOLLOW", "O_TMPFILE")),
    reason="descriptor-pinned publication requires Linux",
)
def test_reserved_publication_resists_parent_swap_and_occupied_race(tmp_path: Path) -> None:
    requested_parent = tmp_path / "requested"
    requested_parent.mkdir()
    destination = requested_parent / "result.json"
    moved_parent = tmp_path / "moved"
    with campaign._reserved_new_output(destination) as target:
        requested_parent.rename(moved_parent)
        requested_parent.mkdir()
        campaign._publish_reserved_json(target, {"value": 1})
    assert json.loads((moved_parent / "result.json").read_text()) == {"value": 1}
    assert not destination.exists()

    occupied = tmp_path / "occupied.json"
    with campaign._reserved_new_output(occupied) as target:
        occupied.write_text("do not replace", encoding="utf-8")
        with pytest.raises(FileExistsError, match="refusing to replace"):
            campaign._publish_reserved_json(target, {"value": 2})
    assert occupied.read_text(encoding="utf-8") == "do not replace"
