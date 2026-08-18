from __future__ import annotations

from copy import deepcopy
from typing import Never

import numpy as np
import pytest

from alberta_framework.benchmarks import l2er_matched_development as matched
from alberta_framework.benchmarks.ipmnist_screening import ScreeningRunResult, screening_spec


def _results() -> list[ScreeningRunResult]:
    offsets = {
        "l2er_mechanism_off": 0.0,
        "l2er_l2_only": 0.02,
        "l2er_er_only": -0.02,
        "l2er_combined": 0.03,
    }
    results = []
    for seed in matched.SEEDS:
        for arm in matched.ARMS:
            spec = screening_spec(arm)
            accuracy = 0.5 + offsets[arm]
            results.append(
                ScreeningRunResult(
                    config_name=arm,
                    base_learner=spec.base_learner,
                    hyperparameters=dict(spec.hyperparameters),
                    seed=seed,
                    config=matched.CONFIG,
                    per_task_accuracy=np.asarray([accuracy, accuracy], dtype=np.float64),
                    per_task_loss=np.asarray([0.8, 0.7], dtype=np.float64),
                    per_task_plasticity=np.asarray([0.1, 0.2], dtype=np.float64),
                    wall_clock_seconds=1.0,
                )
            )
    return results


def test_matched_report_is_complete_paired_and_permanently_nonpromoting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(matched, "_validated_source_provenance", lambda value, **_: value)
    monkeypatch.setattr(matched, "_validated_dataset_provenance", lambda value, **_: value)
    monkeypatch.setattr(matched, "_validated_runtime_environment", lambda value, **_: value)
    report = matched.build_report(
        _results(),
        source_provenance={},
        dataset_provenance={},
        environment={},
    )
    assert len(report["records"]) == len(matched.SEEDS) * len(matched.ARMS)
    paired = report["paired_comparisons"]
    assert paired["l2er_l2_only"]["outcome"] == "supported"
    assert paired["l2er_er_only"]["outcome"] == "rejected"
    assert report["policy"]["scientific_promotion_allowed"] is False
    assert matched.validate_report(report, require_current_source=False) == report


def test_validator_rejects_hostile_plan_container_without_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(matched, "_validated_source_provenance", lambda value, **_: value)
    monkeypatch.setattr(matched, "_validated_dataset_provenance", lambda value, **_: value)
    monkeypatch.setattr(matched, "_validated_runtime_environment", lambda value, **_: value)
    report = matched.build_report(
        _results(), source_provenance={}, dataset_provenance={}, environment={}
    )

    class HostileList(list[object]):
        calls = 0

        def __iter__(self) -> Never:
            type(self).calls += 1
            raise AssertionError("hostile iteration")

        def __eq__(self, other: object) -> Never:
            type(self).calls += 1
            raise AssertionError("hostile equality")

    hostile = deepcopy(report)
    hostile["plan"]["arms"] = HostileList()
    with pytest.raises(ValueError, match="exact string list"):
        matched.validate_report(hostile, require_current_source=False)
    assert HostileList.calls == 0

    hostile_runtime = deepcopy(report)
    hostile_runtime["environment"] = {"devices": HostileList()}
    with pytest.raises(ValueError, match="finite exact JSON"):
        matched.validate_report(hostile_runtime, require_current_source=False)
    assert HostileList.calls == 0

    shared_bomb: object = "leaf"
    for _ in range(3):
        shared_bomb = [shared_bomb] * 64
    hostile_runtime = deepcopy(report)
    hostile_runtime["environment"] = {"bomb": shared_bomb}
    with pytest.raises(ValueError, match="aggregate JSON node limit"):
        matched.validate_report(hostile_runtime, require_current_source=False)


def test_builder_revalidates_forged_result_before_field_dispatch() -> None:
    class HostileInt(int):
        calls = 0

        def __mul__(self, other: object) -> Never:
            type(self).calls += 1
            raise AssertionError("hostile multiply")

        def __eq__(self, other: object) -> Never:
            type(self).calls += 1
            raise AssertionError("hostile equality")

        def __hash__(self) -> Never:
            type(self).calls += 1
            raise AssertionError("hostile hash")

    results = _results()
    forged_config = matched.IPMNISTConfig(**matched.CONFIG.to_config())
    object.__setattr__(forged_config, "n_tasks", HostileInt(2))
    object.__setattr__(results[0], "config", forged_config)
    with pytest.raises(ValueError, match="n_tasks"):
        matched.build_report(
            results, source_provenance={}, dataset_provenance={}, environment={}
        )
    assert HostileInt.calls == 0


def test_validator_recomputes_paired_arithmetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(matched, "_validated_source_provenance", lambda value, **_: value)
    monkeypatch.setattr(matched, "_validated_dataset_provenance", lambda value, **_: value)
    monkeypatch.setattr(matched, "_validated_runtime_environment", lambda value, **_: value)
    report = matched.build_report(
        _results(), source_provenance={}, dataset_provenance={}, environment={}
    )
    hostile = deepcopy(report)
    hostile["paired_comparisons"]["l2er_combined"]["mean_delta"] = 0.0
    with pytest.raises(ValueError, match="mean_delta is inconsistent"):
        matched.validate_report(hostile, require_current_source=False)

    reordered = deepcopy(report)
    reordered["records"][0], reordered["records"][1] = (
        reordered["records"][1],
        reordered["records"][0],
    )
    with pytest.raises(ValueError, match="deterministic frozen"):
        matched.validate_report(reordered, require_current_source=False)


def test_output_namespace_is_one_new_development_path() -> None:
    assert str(matched.OUTPUT_PATH) == "outputs/l2er_matched_development/report.v1.json"
    assert matched.frozen_plan()["development_only"] is True
    assert matched.frozen_plan()["scientific_promotion_allowed"] is False
