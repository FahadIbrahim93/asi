"""Contracts for the bounded CLEAR qualification lane."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from alberta_framework.benchmarks.clear_qualification import (
    AVALANCHE_COMMIT,
    BUCKETS,
    CURATION_COMMIT,
    DEV_SEEDS,
    REFERENCE_COMMIT,
    SCHEMA,
    ClearQualificationError,
    _metric_values,
    execution_config,
    load_dataset_manifest,
    main,
    qualification_plan,
    validate_result,
    verify_dataset_manifest,
)

pytestmark = pytest.mark.unit


def _manifest(root: Path) -> bytes:
    archive = root / "clear100-local.zip"
    archive.write_bytes(b"small CLEAR fixture")
    payload = {
        "schema_version": SCHEMA,
        "dataset": "clear100",
        "protocol": "streaming-near-future",
        "buckets": list(BUCKETS),
        "years": list(range(2005, 2015)),
        "samples_per_bucket": [index + 1 for index in range(10)],
        "archives": [
            {
                "role": "locally-acquired-clear100",
                "path": archive.name,
                "size_bytes": archive.stat().st_size,
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            }
        ],
        "provider_archive_checksums_published": False,
    }
    return json.dumps(payload).encode()


def test_manifest_verification_and_plan_are_exact_and_nonpromoting(tmp_path: Path) -> None:
    receipt = verify_dataset_manifest(_manifest(tmp_path), root=tmp_path)
    plan = qualification_plan(receipt)
    assert receipt.sample_count == 55
    assert plan["promotion_authorized"] is False
    assert plan["execution_authorized"] is False
    assert plan["negative_retention_required"] is True
    assert plan["control_config"] == plan["mechanism_off_config"]
    assert plan["source_revisions"] == {
        "curation": CURATION_COMMIT,
        "reference_runner": REFERENCE_COMMIT,
        "avalanche": AVALANCHE_COMMIT,
    }
    assert plan["axes"] == [
        {"seed": seed, "arm": arm}
        for seed in DEV_SEEDS
        for arm in ("control", "mechanism-off")
    ]
    resources = plan["resource_budget_per_axis"]
    assert isinstance(resources, dict)
    assert resources["training_observations"] == 5_500
    assert resources["data_samples_read"] == 6_050
    assert resources["optimizer_updates"] == 1_000
    assert resources["model_queries"] == 550
    assert resources["environment_steps"] == 0
    assert resources["timing"] == "telemetry-only"


def test_cli_verifies_local_data_and_emits_only_a_nonexecuting_plan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(_manifest(tmp_path))
    assert main((str(manifest), "--dataset-root", str(tmp_path))) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["classification"] == "development-only-permanently-nonpromoting"
    assert payload["execution_authorized"] is False
    assert payload["promotion_authorized"] is False


def test_manifest_path_read_is_metadata_gated_bounded_and_does_not_use_read_bytes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(_manifest(tmp_path))
    monkeypatch.setattr(Path, "read_bytes", lambda _self: pytest.fail("unbounded read_bytes"))
    assert main((str(manifest), "--dataset-root", str(tmp_path))) == 0
    assert json.loads(capsys.readouterr().out)["execution_authorized"] is False


def test_manifest_path_rejects_oversize_before_open_symlink_and_fifo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * ((1 << 20) + 1))
    monkeypatch.setattr(os, "open", lambda *_args, **_kwargs: pytest.fail("file was opened"))
    with pytest.raises(ClearQualificationError, match="byte limit"):
        load_dataset_manifest(oversized)
    monkeypatch.undo()

    target = tmp_path / "target.json"
    target.write_bytes(b"{}")
    link = tmp_path / "manifest-link.json"
    link.symlink_to(target)
    with pytest.raises(ClearQualificationError, match="non-symlink"):
        load_dataset_manifest(link)

    fifo = tmp_path / "manifest.fifo"
    os.mkfifo(fifo)
    with pytest.raises(ClearQualificationError, match="regular"):
        load_dataset_manifest(fifo)


def test_manifest_path_accepts_exact_byte_cap_without_overread(tmp_path: Path) -> None:
    manifest = tmp_path / "exact-limit.json"
    payload = b"{}" + b" " * ((1 << 20) - 2)
    manifest.write_bytes(payload)
    assert load_dataset_manifest(manifest) == payload


def test_candidate_is_not_silently_present_in_mechanism_off() -> None:
    assert execution_config(mechanism_enabled=False) == execution_config(
        mechanism_enabled=False
    )
    assert execution_config(mechanism_enabled=True) != execution_config(
        mechanism_enabled=False
    )


def test_metric_reduction_matches_official_matrix_definitions() -> None:
    matrix = [[float((row * 10 + column) / 100) for column in range(10)] for row in range(10)]
    metrics = _metric_values(matrix)
    assert metrics["in_domain"] == pytest.approx(sum(matrix[i][i] for i in range(10)) / 10)
    assert metrics["next_domain"] == pytest.approx(
        sum(matrix[i][i + 1] for i in range(9)) / 9
    )
    assert metrics["accuracy"] == pytest.approx(
        sum(matrix[i][j] for i in range(10) for j in range(i + 1)) / 55
    )
    assert metrics["forward_transfer"] == pytest.approx(
        sum(matrix[i][j] for i in range(10) for j in range(i + 1, 10)) / 45
    )
    assert metrics["backward_transfer"] == pytest.approx(
        sum(matrix[i][j] for i in range(10) for j in range(i)) / 45
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.update(protocol="iid"), "streaming"),
        (lambda value: value.update(buckets=list(range(10))), "temporal"),
        (lambda value: value.update(provider_archive_checksums_published=True), "invented"),
        (lambda value: value.update(samples_per_bucket=[1] * 9), "every labeled"),
    ],
)
def test_manifest_rejects_protocol_and_shape_drift(
    tmp_path: Path, mutation: object, match: str
) -> None:
    payload = json.loads(_manifest(tmp_path))
    assert callable(mutation)
    mutation(payload)
    with pytest.raises(ClearQualificationError, match=match):
        verify_dataset_manifest(json.dumps(payload).encode(), root=tmp_path)


def test_manifest_rejects_hash_size_path_and_symlink_attacks(tmp_path: Path) -> None:
    payload = json.loads(_manifest(tmp_path))
    payload["archives"][0]["sha256"] = "0" * 64
    with pytest.raises(ClearQualificationError, match="SHA-256 does not match"):
        verify_dataset_manifest(json.dumps(payload).encode(), root=tmp_path)

    payload = json.loads(_manifest(tmp_path))
    payload["archives"][0]["size_bytes"] += 1
    with pytest.raises(ClearQualificationError, match="size"):
        verify_dataset_manifest(json.dumps(payload).encode(), root=tmp_path)

    payload = json.loads(_manifest(tmp_path))
    payload["archives"][0]["path"] = "../escape.zip"
    with pytest.raises(ClearQualificationError, match="canonical and relative"):
        verify_dataset_manifest(json.dumps(payload).encode(), root=tmp_path)

    target = tmp_path / "target.zip"
    target.write_bytes(b"target")
    link = tmp_path / "link.zip"
    link.symlink_to(target)
    payload = json.loads(_manifest(tmp_path))
    payload["archives"][0] = {
        "role": "archive",
        "path": link.name,
        "size_bytes": target.stat().st_size,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    with pytest.raises(ClearQualificationError, match="regular file"):
        verify_dataset_manifest(json.dumps(payload).encode(), root=tmp_path)


def test_manifest_rejects_duplicate_paths_and_unknown_fields(tmp_path: Path) -> None:
    payload = json.loads(_manifest(tmp_path))
    payload["archives"].append({**payload["archives"][0], "role": "duplicate"})
    with pytest.raises(ClearQualificationError, match="paths must be unique"):
        verify_dataset_manifest(json.dumps(payload).encode(), root=tmp_path)
    payload = json.loads(_manifest(tmp_path))
    payload["authority"] = True
    with pytest.raises(ClearQualificationError, match="fields"):
        verify_dataset_manifest(json.dumps(payload).encode(), root=tmp_path)


def test_result_validator_enforces_receipts_nonpromotion_and_negative_retention() -> None:
    plan_sha = "a" * 64
    matrix = [[0.5 for _ in range(10)] for _ in range(10)]
    result = {
        "schema_version": SCHEMA,
        "plan_sha256": plan_sha,
        "status": "negative-development",
        "promotion_authorized": False,
        "negative_retained": True,
        "accuracy_matrix": matrix,
        "metrics": {
            "accuracy": 0.5,
            "in_domain": 0.5,
            "next_domain": 0.5,
            "forward_transfer": 0.5,
            "backward_transfer": 0.5,
        },
        "resource_receipts": {
            "persistent_bytes": 1,
            "archive_bytes": 2,
            "training_observations": 3,
            "data_samples_read": 8,
            "optimizer_updates": 4,
            "model_queries": 5,
            "environment_steps": 0,
            "wall_seconds_telemetry": 6,
        },
    }
    assert validate_result(json.dumps(result).encode(), expected_plan_sha256=plan_sha) == result
    expected_resources = {
        key: value
        for key, value in result["resource_receipts"].items()
        if key not in ("persistent_bytes", "wall_seconds_telemetry")
    }
    assert validate_result(
        json.dumps(result).encode(),
        expected_plan_sha256=plan_sha,
        expected_resource_budget=expected_resources,
    ) == result
    mismatched_resources = {**expected_resources, "model_queries": 6}
    with pytest.raises(ClearQualificationError, match="model_queries"):
        validate_result(
            json.dumps(result).encode(),
            expected_plan_sha256=plan_sha,
            expected_resource_budget=mismatched_resources,
        )
    hostile_metric = {**result, "metrics": {**result["metrics"], "accuracy": True}}
    with pytest.raises(ClearQualificationError, match="finite exact floats"):
        validate_result(json.dumps(hostile_metric).encode(), expected_plan_sha256=plan_sha)
    for field, value, match in (
        ("promotion_authorized", True, "nonpromotion"),
        ("negative_retained", False, "negative retention"),
        ("status", "promoted", "status"),
        ("plan_sha256", "b" * 64, "provenance"),
    ):
        hostile = {**result, field: value}
        with pytest.raises(ClearQualificationError, match=match):
            validate_result(json.dumps(hostile).encode(), expected_plan_sha256=plan_sha)


def test_result_rejects_scalar_alias_and_unbounded_payload() -> None:
    result = {
        "schema_version": SCHEMA,
        "plan_sha256": "a" * 64,
        "status": "completed-development",
        "promotion_authorized": False,
        "negative_retained": True,
        "accuracy_matrix": [[0.5 for _ in range(10)] for _ in range(10)],
        "metrics": {
            "accuracy": 0.5,
            "in_domain": 0.5,
            "next_domain": 0.5,
            "forward_transfer": 0.5,
            "backward_transfer": 0.5,
        },
        "resource_receipts": {
            "persistent_bytes": True,
            "archive_bytes": 0,
            "training_observations": 0,
            "data_samples_read": 0,
            "optimizer_updates": 0,
            "model_queries": 0,
            "environment_steps": 0,
            "wall_seconds_telemetry": 0,
        },
    }
    with pytest.raises(ClearQualificationError, match="exact integer"):
        validate_result(json.dumps(result).encode(), expected_plan_sha256="a" * 64)
    with pytest.raises(ClearQualificationError, match="byte limit"):
        validate_result(b" " * ((1 << 20) + 1), expected_plan_sha256="a" * 64)
