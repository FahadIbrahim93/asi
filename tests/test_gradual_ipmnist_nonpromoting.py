from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, cast

import jax
import numpy as np
import pytest

import alberta_framework.evaluation.gradual_ipmnist_nonpromoting as gradual_report
from alberta_framework.benchmarks.ipmnist_gradual import run_gradual_input_pair
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig
from alberta_framework.evaluation.gradual_ipmnist_nonpromoting import (
    GradualInputDevelopmentPlan,
    build_gradual_input_development_report,
    derive_gradual_input_resource_correction,
    retain_frozen_gradual_input_development_report,
    validate_gradual_input_development_report,
)


def _tiny() -> tuple[np.ndarray, np.ndarray, GradualInputDevelopmentPlan]:
    x = np.asarray([[-1.0, -0.5, 0.5, 1.0], [1.0, 0.5, -0.5, -1.0]], dtype=np.float32)
    y = np.asarray([0, 1], dtype=np.int32)
    return (
        x,
        y,
        GradualInputDevelopmentPlan(
            seeds=(19,),
            config=IPMNISTConfig(
                n_tasks=2, task_length=2, input_dim=4, hidden1=3, hidden2=2, n_classes=2
            ),
            transition_steps=1,
        ),
    )


def test_gradual_report_is_strict_derived_and_nonpromoting() -> None:
    x, y, plan = _tiny()
    run = run_gradual_input_pair(
        x,
        y,
        learner_name="adamw_control",
        seed=19,
        config=plan.config,
        transition_steps=1,
    )
    report = build_gradual_input_development_report(plan, (run,), x, y)
    validate_gradual_input_development_report(report, x, y)
    payload = cast(dict[str, Any], report)

    assert payload["policy"]["scientific_promotion_allowed"] is False
    assert payload["records"][0]["arms"][0]["metrics"]["correct"] == int(
        run.correct_counts[0].sum()
    )
    forged = copy.deepcopy(payload)
    forged["records"][0]["arms"][0]["metrics"]["online_accuracy"] = 1.0
    with pytest.raises(ValueError, match="derived"):
        validate_gradual_input_development_report(forged, x, y)


def test_gradual_report_revalidates_mutated_run_receipts() -> None:
    x, y, plan = _tiny()
    run = run_gradual_input_pair(
        x,
        y,
        learner_name="adamw_control",
        seed=19,
        config=plan.config,
        transition_steps=1,
    )
    object.__setattr__(run, "updates_per_arm", 999)
    with pytest.raises(ValueError, match="pair contract"):
        build_gradual_input_development_report(plan, (run,), x, y)


def test_gradual_report_rejects_hostile_json_keys_without_hooks() -> None:
    class Hostile(str):
        calls = 0

        def __hash__(self) -> int:
            self.calls += 1
            return super().__hash__()

        def __eq__(self, other: object) -> bool:
            self.calls += 1
            raise AssertionError("must not compare")

    key = Hostile("schema")
    payload = {key: "x"}
    key.calls = 0
    with pytest.raises(ValueError, match="exact JSON"):
        validate_gradual_input_development_report(payload, np.zeros((2, 2)), np.zeros(2))
    assert key.calls == 0


def test_gradual_report_rejects_hostile_json_metaclass_without_hooks() -> None:
    class HostileType(type):
        calls = 0

        def __eq__(cls, other: object) -> bool:
            cls.calls += 1
            raise AssertionError("must not compare runtime types")

    class Hostile(metaclass=HostileType):
        pass

    HostileType.calls = 0
    with pytest.raises(ValueError, match="exact JSON tree"):
        gradual_report._json_preflight({"value": Hostile()})
    assert HostileType.calls == 0


def test_gradual_report_reconstructs_exact_persistent_resource_formula() -> None:
    x, y, plan = _tiny()
    run = run_gradual_input_pair(
        x,
        y,
        learner_name="adamw_control",
        seed=19,
        config=plan.config,
        transition_steps=1,
    )
    report = build_gradual_input_development_report(plan, (run,), x, y)
    forged = copy.deepcopy(report)
    forged["records"][0]["arms"][0]["resources"]["persistent_numeric_bytes"] = 1
    with pytest.raises(ValueError, match="resource receipt"):
        validate_gradual_input_development_report(forged, x, y)


def test_gradual_report_retention_is_exclusive_and_reload_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    x, y, plan = _tiny()
    run = run_gradual_input_pair(
        x,
        y,
        learner_name="adamw_control",
        seed=19,
        config=plan.config,
        transition_steps=1,
    )
    report = build_gradual_input_development_report(plan, (run,), x, y)
    monkeypatch.setattr(gradual_report, "FROZEN_GRADUAL_INPUT_PLAN", plan)
    destination = retain_frozen_gradual_input_development_report(
        report, x, y, repository_root=tmp_path
    )
    assert destination.read_bytes() == gradual_report.canonical_gradual_input_development_bytes(
        report, x, y
    )
    with pytest.raises(FileExistsError):
        retain_frozen_gradual_input_development_report(report, x, y, repository_root=tmp_path)


def test_resource_correction_preserves_v1_and_rejects_invalid_v2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    parent = json.loads(
        (root / "outputs/ipmnist_gradual/development.v1/"
         "result.7b2bf6c0f73b9fae20fcde53445f5f81656976ea4d042641772501e0108c6561.json")
        .read_text(encoding="utf-8")
    )
    invalid_v2 = json.loads(
        (root / "outputs/ipmnist_gradual/development.v2/"
         "result.9ff58ac51163004208b94e325f9539037cb8cfb3540024da95c847f50154b483.json")
        .read_text(encoding="utf-8")
    )
    retained_v3 = json.loads(
        next((root / "outputs/ipmnist_gradual/development.v3").glob("*.json")).read_bytes()
    )
    monkeypatch.setattr(
        gradual_report, "_dataset_identity", lambda *_: copy.deepcopy(parent["dataset"])
    )
    monkeypatch.setattr(
        gradual_report,
        "_runtime_identity",
        lambda: copy.deepcopy(retained_v3["identity"]["derivation_runtime"]),
    )
    monkeypatch.setattr(
        gradual_report,
        "_source_identity",
        lambda: copy.deepcopy(retained_v3["identity"]["derivation_source_sha256"]),
    )

    validate_gradual_input_development_report(parent, object(), object())
    with pytest.raises(ValueError, match="v2 resource correction"):
        validate_gradual_input_development_report(invalid_v2, object(), object())

    with jax.default_prng_impl("rbg"):
        corrected = derive_gradual_input_resource_correction(parent, object(), object())
    assert corrected == retained_v3
    identity = corrected["identity"]
    assert identity["parent_artifact_sha256"] == gradual_report._PARENT_ARTIFACT_SHA256
    assert identity["invalid_intermediate_artifact_sha256"] == (
        gradual_report._INVALID_V2_ARTIFACT_SHA256
    )
    assert corrected["aggregate"]["outcome"] == parent["aggregate"]["outcome"]
    parameter_bank_bytes = gradual_report.FROZEN_GRADUAL_INPUT_PLAN.config.parameter_count * 4
    for old_record, new_record in zip(parent["records"], corrected["records"], strict=True):
        assert new_record["comparison"] == old_record["comparison"]
        for old_arm, new_arm in zip(old_record["arms"], new_record["arms"], strict=True):
            assert new_arm["metrics"] == old_arm["metrics"]
            assert new_arm["resources"]["timing_ns"] == old_arm["resources"]["timing_ns"]
            assert new_arm["resources"]["persistent_numeric_bytes"] == (
                old_arm["resources"]["persistent_numeric_bytes"] + parameter_bank_bytes
            )
