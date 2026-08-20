"""Prospectively frozen matched-development campaign for issue #1560."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Never

import numpy as np
import pytest

from alberta_framework.benchmarks import adamo_matched_development as matched
from alberta_framework.benchmarks.adamo_diagnostic import run_adamo_diagnostic

pytestmark = pytest.mark.integration


def _patch_identities(
    monkeypatch: pytest.MonkeyPatch, receipts: list[dict[str, object]]
) -> None:
    monkeypatch.setattr(
        matched,
        "_current_source_provenance",
        lambda: {"git_commit": "c" * 40, "relevant_source_sha256": "a" * 64},
    )
    monkeypatch.setattr(
        matched,
        "_current_runtime_environment",
        lambda: {"schema": "test-runtime", "backend": "cpu"},
    )
    monkeypatch.setattr(matched, "validate_adamo_diagnostic", lambda value: value)
    semantic = receipts[0]["dataset"]["sha256"]
    monkeypatch.setattr(matched, "DATASET_FILE_SHA256", "b" * 64)
    monkeypatch.setattr(matched, "DATASET_SEMANTIC_SHA256", semantic)


@pytest.fixture(scope="module")
def receipts() -> list[dict[str, object]]:
    inputs = np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(8, 4)
    labels = np.arange(8, dtype=np.int32) % 2
    first = run_adamo_diagnostic(
        inputs,
        labels,
        profile="contract-smoke",
        seed=matched.SEEDS[0],
    )
    result = []
    for seed in matched.SEEDS:
        receipt = copy.deepcopy(first)
        receipt["seed"] = seed
        receipt["profile"] = matched.PROFILE
        result.append(receipt)
    return result


def test_plan_is_prospective_exact_and_permanently_nonpromoting() -> None:
    plan = matched.frozen_plan()
    assert plan["seeds"] == list(matched.SEEDS)
    assert plan["profile"] == "bounded-development"
    assert plan["execution_authorized"] is False
    assert plan["scientific_promotion_allowed"] is False
    assert plan["outcome_retention_required"] is True
    assert plan["consumed_qualification_seeds"] == [15600, 15601, 15602, 15603]


def test_report_recomputes_paired_statistics_and_retains_every_outcome(
    receipts: list[dict[str, object]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_identities(monkeypatch, receipts)
    report = matched.build_report(
        receipts,
        dataset_file_sha256="b" * 64,
        execution_source_commit="c" * 40,
    )
    assert len(report["runs"]) == len(matched.SEEDS)
    assert set(report["paired_comparisons"]) == {
        "adamo_l1e3",
        "adam_iso_joint_l1e3",
    }
    assert all(
        comparison["outcome"] in {"supported", "rejected", "inconclusive"}
        for comparison in report["paired_comparisons"].values()
    )
    assert report["policy"] == {
        "development_only": True,
        "scientific_promotion_allowed": False,
        "outcome_retained": True,
        "timing_is_telemetry_only": True,
    }
    assert matched.validate_report(report, require_current_source=True) == report


def test_validator_rejects_missing_seed_tampering_and_promotion(
    receipts: list[dict[str, object]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_identities(monkeypatch, receipts)
    report = matched.build_report(
        receipts,
        dataset_file_sha256="b" * 64,
        execution_source_commit="c" * 40,
    )

    missing = copy.deepcopy(report)
    missing["runs"].pop()
    with pytest.raises(ValueError, match="complete frozen seed schedule"):
        matched.validate_report(missing, require_current_source=True)

    arithmetic = copy.deepcopy(report)
    arithmetic["paired_comparisons"]["adamo_l1e3"]["mean_accuracy_delta"] = 1.0
    with pytest.raises(ValueError, match="paired arithmetic"):
        matched.validate_report(arithmetic, require_current_source=True)

    promoting = copy.deepcopy(report)
    promoting["policy"]["scientific_promotion_allowed"] = True
    with pytest.raises(ValueError, match="permanently nonpromoting"):
        matched.validate_report(promoting, require_current_source=True)


def test_atomic_publication_refuses_overwrite(
    tmp_path: Path,
    receipts: list[dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_identities(monkeypatch, receipts)
    report = matched.build_report(
        receipts,
        dataset_file_sha256="b" * 64,
        execution_source_commit="c" * 40,
    )
    destination = tmp_path / "report.json"
    monkeypatch.setattr(matched, "OUTPUT_PATH", destination)
    matched.publish_report(destination, report)
    with pytest.raises(FileExistsError):
        matched.publish_report(destination, report)


def test_execution_gate_is_closed_until_plan_review() -> None:
    with pytest.raises(RuntimeError, match="not authorized"):
        matched.run_campaign(Path("unused.npz"), Path("unused.json"))


def test_student_t_df3_constant_is_exact() -> None:
    assert matched.T95_DF3 == 3.1824463052837078
    assert matched.T95_DF3.hex() == "0x1.975a66893c1a7p+1"


def test_validator_preflights_hostile_nested_plan_without_dispatch(
    receipts: list[dict[str, object]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_identities(monkeypatch, receipts)
    report = matched.build_report(
        receipts,
        dataset_file_sha256="b" * 64,
        execution_source_commit="c" * 40,
    )

    class HostileList(list[object]):
        calls = 0

        def __iter__(self) -> Never:
            type(self).calls += 1
            raise AssertionError("hostile iteration")

        def __eq__(self, other: object) -> Never:
            type(self).calls += 1
            raise AssertionError("hostile equality")

    hostile = copy.deepcopy(report)
    hostile["plan"]["arms"] = HostileList(hostile["plan"]["arms"])
    with pytest.raises(ValueError, match="exact JSON"):
        matched.validate_report(hostile, require_current_source=True)
    assert HostileList.calls == 0


def test_validator_binds_execution_commit_to_source_provenance(
    receipts: list[dict[str, object]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_identities(monkeypatch, receipts)
    report = matched.build_report(
        receipts,
        dataset_file_sha256="b" * 64,
        execution_source_commit="c" * 40,
    )
    hostile = copy.deepcopy(report)
    hostile["execution_source_commit"] = "d" * 40
    with pytest.raises(ValueError, match="does not match source provenance"):
        matched.validate_report(hostile, require_current_source=False)
