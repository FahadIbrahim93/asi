"""Contracts for the bounded CLEAR qualification lane."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

import alberta_framework.benchmarks.clear_qualification as clear_module
from alberta_framework.benchmarks.clear_qualification import (
    ACQUISITION_REVIEW_SCHEMA,
    AVALANCHE_COMMIT,
    BUCKETS,
    CURATION_COMMIT,
    DEV_SEEDS,
    REFERENCE_COMMIT,
    RIGHTS_REVIEW_SCHEMA,
    RUNNER_INPUT_SCHEMA,
    SCHEMA,
    SPLIT_REVIEW_SCHEMA,
    ArchiveIdentity,
    ClearDatasetReceipt,
    ClearQualificationError,
    _metric_values,
    current_source_identity,
    execution_config,
    load_dataset_manifest,
    main,
    qualification_plan,
    runtime_identity,
    validate_result,
    verify_dataset_manifest,
    verify_runner_input_manifest,
)

pytestmark = pytest.mark.unit


def _result_plan() -> dict[str, object]:
    archive = ArchiveIdentity("fixture", "clear.zip", 2, "d" * 64)
    samples = (1,) * 10
    identity = {
        "dataset": "clear100",
        "protocol": "streaming-near-future",
        "buckets": BUCKETS,
        "years": tuple(range(2005, 2015)),
        "samples_per_bucket": samples,
        "archives": [
            {
                "role": archive.role,
                "path": archive.path,
                "size_bytes": archive.size_bytes,
                "sha256": archive.sha256,
            }
        ],
    }
    receipt = ClearDatasetReceipt(
        archives=(archive,),
        samples_per_bucket=samples,
        archive_bytes=2,
        sample_count=10,
        dataset_sha256=hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    )
    return dict(qualification_plan(receipt))


def _plan_sha256(plan: dict[str, object]) -> str:
    raw = json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(raw).hexdigest()


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


def _runner_dataset_manifest(root: Path) -> bytes:
    payload = json.loads(_manifest(root))
    payload["samples_per_bucket"] = [100] * 10
    return json.dumps(payload).encode()


def _identity(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _runner_input_manifest(
    root: Path, receipt: ClearDatasetReceipt
) -> tuple[bytes, dict[str, object]]:
    acquisition: dict[str, object] = {
        "mode": "independently-reviewed-acquisition",
        "provider_locator": "https://example.invalid/reviewed-clear-snapshot",
        "provider_snapshot_sha256": "a" * 64,
        "provider_checksums_published": False,
        "archives": [
            {
                "role": archive.role,
                "path": archive.path,
                "size_bytes": archive.size_bytes,
                "sha256": archive.sha256,
            }
            for archive in receipt.archives
        ],
    }
    reviews = root / "reviews"
    reviews.mkdir(exist_ok=True)
    acquisition_review = reviews / "acquisition.txt"
    acquisition_review.write_bytes(
        json.dumps(
            {
                "schema_version": ACQUISITION_REVIEW_SCHEMA,
                "decision": "accepted-for-local-development",
                "reviewer": "fixture-reviewer",
                "provider_locator": acquisition["provider_locator"],
                "provider_snapshot_sha256": acquisition["provider_snapshot_sha256"],
                "archives": acquisition["archives"],
                "authentication": "external-review-record-not-authenticated-by-asi",
            }
        ).encode()
    )
    rights_review = reviews / "rights-storage.txt"
    rights_review.write_bytes(
        json.dumps(
            {
                "schema_version": RIGHTS_REVIEW_SCHEMA,
                "decision": "approved-local-development-only",
                "reviewer": "fixture-reviewer",
                "reviewed_scopes": [
                    "yfcc-terms",
                    "flickr-asset-terms",
                    "takedown-process",
                    "approved-storage",
                ],
                "storage_approval_id": "fixture-storage-approval",
                "authentication": "external-review-record-not-authenticated-by-asi",
            }
        ).encode()
    )

    split_values: list[dict[str, object]] = []
    for bucket, year in zip(BUCKETS, range(2005, 2015), strict=True):
        indexes: dict[str, dict[str, object]] = {}
        for split in ("train", "evaluation"):
            records = []
            count = 100 if split == "train" else 101
            for sample_index in range(count):
                class_index = sample_index % 100
                sample = root / "images" / str(bucket) / split / f"{sample_index}.jpg"
                sample.parent.mkdir(parents=True, exist_ok=True)
                if sample.exists() or sample.is_symlink():
                    sample.unlink()
                sample.write_bytes(f"{bucket}:{split}:{class_index}:{sample_index}".encode())
                records.append(
                    {
                        "sample_id": f"b{bucket}-{split}-{sample_index}",
                        "path": sample.relative_to(root).as_posix(),
                        "size_bytes": sample.stat().st_size,
                        "sha256": hashlib.sha256(sample.read_bytes()).hexdigest(),
                        "class_index": class_index,
                    }
                )
            index = root / "indexes" / f"bucket-{bucket}-{split}.jsonl"
            index.parent.mkdir(exist_ok=True)
            index.write_bytes(b"".join(_canonical_line(record) for record in records))
            indexes[split] = {**_identity(index, root), "sample_count": len(records)}
        split_values.append(
            {
                "bucket": bucket,
                "year": year,
                "train_index": indexes["train"],
                "evaluation_index": indexes["evaluation"],
            }
        )
    split_review = reviews / "prepared-splits.json"
    split_review.write_bytes(
        json.dumps(
            {
                "schema_version": SPLIT_REVIEW_SCHEMA,
                "decision": "accepted-prepared-streaming-splits",
                "reviewer": "fixture-reviewer",
                "dataset_sha256": receipt.dataset_sha256,
                "protocol": "streaming-near-future",
                "buckets": list(BUCKETS),
                "years": list(range(2005, 2015)),
                "splits": split_values,
                "authentication": "external-review-record-not-authenticated-by-asi",
            }
        ).encode()
    )

    payload: dict[str, object] = {
        "schema_version": RUNNER_INPUT_SCHEMA,
        "dataset_sha256": receipt.dataset_sha256,
        "protocol": "streaming-near-future",
        "acquisition": acquisition,
        "review_documents": [
            {"role": "acquisition-review", **_identity(acquisition_review, root)},
            {"role": "rights-storage-review", **_identity(rights_review, root)},
            {"role": "split-review", **_identity(split_review, root)},
        ],
        "rights_and_storage": {
            "decision": "approved-local-development-only",
            "yfcc_terms_reviewed": True,
            "flickr_asset_terms_reviewed": True,
            "takedown_process_documented": True,
            "storage_approved": True,
            "authentication": "external-review-record-not-authenticated-by-asi",
        },
        "splits": split_values,
    }
    return json.dumps(payload).encode(), payload


def _canonical_line(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


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


def test_archive_hash_rejects_path_swap_between_metadata_and_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = json.loads(_manifest(tmp_path))
    archive = tmp_path / payload["archives"][0]["path"]
    original_open = os.open
    swapped = False

    def swap_then_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        nonlocal swapped
        if path == archive.name and not swapped:
            swapped = True
            content = archive.read_bytes()
            archive.unlink()
            archive.write_bytes(content)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", swap_then_open)
    with pytest.raises(ClearQualificationError, match="changed before"):
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


def test_dataset_manifest_and_archive_reject_unreferenced_hard_links(tmp_path: Path) -> None:
    root = tmp_path / "archive-case"
    root.mkdir()
    raw = _manifest(root)
    archive = root / "clear100-local.zip"
    os.link(archive, root / "unreferenced-archive-alias")
    with pytest.raises(ClearQualificationError, match="hard-link aliases"):
        verify_dataset_manifest(raw, root=root)

    manifest = tmp_path / "dataset-manifest.json"
    manifest.write_bytes(raw)
    os.link(manifest, tmp_path / "unreferenced-manifest-alias")
    with pytest.raises(ClearQualificationError, match="hard-link alias"):
        load_dataset_manifest(manifest)


def test_runner_input_preflight_binds_reviews_splits_and_exact_samples(tmp_path: Path) -> None:
    receipt = verify_dataset_manifest(_runner_dataset_manifest(tmp_path), root=tmp_path)
    raw, _ = _runner_input_manifest(tmp_path, receipt)
    runner = verify_runner_input_manifest(raw, root=tmp_path, dataset_receipt=receipt)

    assert runner.dataset_sha256 == receipt.dataset_sha256
    assert runner.train_samples_per_bucket == (100,) * 10
    assert runner.evaluation_samples_per_bucket == (101,) * 10
    assert runner.train_sample_count == 1_000
    assert runner.evaluation_sample_count == 1_010
    assert runner.class_count == 100
    assert runner.execution_authorized is False
    assert runner.external_reviews_authenticated is False
    assert runner.provider_snapshot_bytes_verified is False
    assert runner.redistribution_authorized is False
    assert runner.current_source_sha256 == tuple(sorted(current_source_identity().items()))
    assert runner.runtime_identity == tuple(sorted(runtime_identity().items()))
    assert len(runner.runner_input_sha256) == 64
    assert runner.sample_bytes > 0
    assert runner.index_bytes > 0
    assert runner.review_bytes > 0
    assert runner.training_observations == 100_000
    assert runner.optimizer_updates == 1_000
    assert runner.model_queries == 10_100
    assert runner.data_samples_read == 110_100

    with pytest.raises(ClearQualificationError, match="resource|accounting"):
        dataclasses.replace(runner, model_queries=runner.model_queries + 1)


def test_runner_input_rejects_duplicate_manifest_and_review_json_keys(tmp_path: Path) -> None:
    receipt = verify_dataset_manifest(_runner_dataset_manifest(tmp_path), root=tmp_path)
    raw, payload = _runner_input_manifest(tmp_path, receipt)
    duplicated = raw.replace(
        b'"protocol": "streaming-near-future"',
        b'"protocol":"forged","protocol": "streaming-near-future"',
        1,
    )
    with pytest.raises(ClearQualificationError, match="duplicate JSON"):
        verify_runner_input_manifest(duplicated, root=tmp_path, dataset_receipt=receipt)

    documents = payload["review_documents"]
    assert isinstance(documents, list) and isinstance(documents[1], dict)
    rights_path = tmp_path / str(documents[1]["path"])
    review = rights_path.read_bytes().replace(
        b'"decision": "approved-local-development-only"',
        b'"decision":"forged","decision": "approved-local-development-only"',
        1,
    )
    rights_path.write_bytes(review)
    documents[1].update(_identity(rights_path, tmp_path))
    with pytest.raises(ClearQualificationError, match="duplicate JSON"):
        verify_runner_input_manifest(
            json.dumps(payload).encode(), root=tmp_path, dataset_receipt=receipt
        )


@pytest.mark.parametrize("target", ("sample", "index", "review"))
def test_runner_input_rejects_single_referenced_hard_link(
    tmp_path: Path, target: str
) -> None:
    receipt = verify_dataset_manifest(_runner_dataset_manifest(tmp_path), root=tmp_path)
    raw, payload = _runner_input_manifest(tmp_path, receipt)
    splits = payload["splits"]
    documents = payload["review_documents"]
    assert isinstance(splits, list) and isinstance(splits[0], dict)
    assert isinstance(documents, list) and isinstance(documents[0], dict)
    if target == "sample":
        path = tmp_path / "images/1/train/0.jpg"
    elif target == "index":
        index = splits[0]["train_index"]
        assert isinstance(index, dict)
        path = tmp_path / str(index["path"])
    else:
        path = tmp_path / str(documents[0]["path"])
    os.link(path, tmp_path / f"unreferenced-{target}-alias")
    with pytest.raises(ClearQualificationError, match="hard-link"):
        verify_runner_input_manifest(raw, root=tmp_path, dataset_receipt=receipt)


def test_runner_input_rejects_symlink_dataset_root(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    receipt = verify_dataset_manifest(_runner_dataset_manifest(real_root), root=real_root)
    raw, _ = _runner_input_manifest(real_root, receipt)
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(ClearQualificationError, match="real directory"):
        verify_runner_input_manifest(raw, root=linked_root, dataset_receipt=receipt)


def test_runner_input_rejects_sample_entry_swap_after_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = verify_dataset_manifest(_runner_dataset_manifest(tmp_path), root=tmp_path)
    raw, _ = _runner_input_manifest(tmp_path, receipt)
    train_directory = tmp_path / "images/1/train"
    original_open = os.open
    original_read = os.read
    target_descriptor = -1
    swapped = False

    def track_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        nonlocal target_descriptor
        descriptor = original_open(path, flags, *args, **kwargs)
        if path == "0.jpg" and target_descriptor < 0:
            target_descriptor = descriptor
        return descriptor

    def swap_then_read(descriptor: int, size: int) -> bytes:
        nonlocal swapped
        if descriptor == target_descriptor and not swapped:
            swapped = True
            original_directory = tmp_path / "opened-original-train"
            train_directory.rename(original_directory)
            shutil.copytree(original_directory, train_directory)
        return original_read(descriptor, size)

    monkeypatch.setattr(os, "open", track_open)
    monkeypatch.setattr(os, "read", swap_then_read)
    with pytest.raises(ClearQualificationError, match="changed during"):
        verify_runner_input_manifest(raw, root=tmp_path, dataset_receipt=receipt)


def test_runner_input_rejects_cross_split_content_duplicate(tmp_path: Path) -> None:
    receipt = verify_dataset_manifest(_runner_dataset_manifest(tmp_path), root=tmp_path)
    _, payload = _runner_input_manifest(tmp_path, receipt)
    splits = payload["splits"]
    documents = payload["review_documents"]
    assert isinstance(splits, list) and isinstance(splits[0], dict)
    assert isinstance(documents, list) and isinstance(documents[2], dict)
    evaluation_identity = splits[0]["evaluation_index"]
    assert isinstance(evaluation_identity, dict)
    evaluation_index = tmp_path / str(evaluation_identity["path"])
    records = [json.loads(line) for line in evaluation_index.read_bytes().splitlines()]
    train_sample = tmp_path / "images/1/train/0.jpg"
    evaluation_sample = tmp_path / str(records[0]["path"])
    evaluation_sample.write_bytes(train_sample.read_bytes())
    records[0].update(
        size_bytes=evaluation_sample.stat().st_size,
        sha256=hashlib.sha256(evaluation_sample.read_bytes()).hexdigest(),
    )
    evaluation_index.write_bytes(b"".join(_canonical_line(record) for record in records))
    evaluation_identity.update({**_identity(evaluation_index, tmp_path), "sample_count": 101})
    split_review = tmp_path / str(documents[2]["path"])
    review = json.loads(split_review.read_bytes())
    review["splits"] = splits
    split_review.write_bytes(json.dumps(review).encode())
    documents[2].update(_identity(split_review, tmp_path))
    with pytest.raises(ClearQualificationError, match="content.*globally unique"):
        verify_runner_input_manifest(
            json.dumps(payload).encode(), root=tmp_path, dataset_receipt=receipt
        )


def test_runner_input_enforces_aggregate_index_byte_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = verify_dataset_manifest(_runner_dataset_manifest(tmp_path), root=tmp_path)
    raw, payload = _runner_input_manifest(tmp_path, receipt)
    splits = payload["splits"]
    assert isinstance(splits, list)
    sizes = []
    for split in splits:
        assert isinstance(split, dict)
        for name in ("train_index", "evaluation_index"):
            identity = split[name]
            assert isinstance(identity, dict) and isinstance(identity["size_bytes"], int)
            sizes.append(identity["size_bytes"])
    monkeypatch.setattr(clear_module, "MAX_INDEX_BYTES", max(sizes))
    with pytest.raises(ClearQualificationError, match="aggregate index"):
        verify_runner_input_manifest(raw, root=tmp_path, dataset_receipt=receipt)


@pytest.mark.parametrize(
    ("kind", "match"),
    (
        ("duplicate", "duplicate JSON"),
        ("malformed", "valid JSON"),
        ("oversized", "oversized or partial"),
    ),
)
def test_runner_input_rejects_duplicate_malformed_and_oversized_jsonl(
    tmp_path: Path, kind: str, match: str
) -> None:
    receipt = verify_dataset_manifest(_runner_dataset_manifest(tmp_path), root=tmp_path)
    _, payload = _runner_input_manifest(tmp_path, receipt)
    splits = payload["splits"]
    assert isinstance(splits, list) and isinstance(splits[0], dict)
    identity = splits[0]["train_index"]
    assert isinstance(identity, dict)
    index = tmp_path / str(identity["path"])
    lines = index.read_bytes().splitlines(keepends=True)
    if kind == "duplicate":
        lines[0] = lines[0].replace(
            b'"sample_id":', b'"sample_id":"forged","sample_id":', 1
        )
    elif kind == "malformed":
        lines[0] = b'{"sample_id":\n'
    else:
        lines[0] = b'{"sample_id":"' + b"x" * 4096 + b"\n"
    index.write_bytes(b"".join(lines))
    identity.update({**_identity(index, tmp_path), "sample_count": 100})
    with pytest.raises(ClearQualificationError, match=match):
        verify_runner_input_manifest(
            json.dumps(payload).encode(), root=tmp_path, dataset_receipt=receipt
        )


def test_cli_can_emit_bound_nonauthorizing_runner_input_preflight(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset_manifest = tmp_path / "dataset-manifest.json"
    dataset_manifest.write_bytes(_runner_dataset_manifest(tmp_path))
    receipt = verify_dataset_manifest(dataset_manifest.read_bytes(), root=tmp_path)
    runner_manifest = tmp_path / "runner-input-manifest.json"
    runner_manifest.write_bytes(_runner_input_manifest(tmp_path, receipt)[0])
    assert (
        main(
            (
                str(dataset_manifest),
                "--dataset-root",
                str(tmp_path),
                "--runner-input-manifest",
                str(runner_manifest),
            )
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["qualification_plan"]["execution_authorized"] is False
    assert payload["runner_input_receipt"]["execution_authorized"] is False
    assert payload["runner_input_receipt"]["class_count"] == 100


@pytest.mark.parametrize(
    ("mutate", "match"),
    (
        (
            lambda value: value["rights_and_storage"].update(storage_approved=False),
            "rights and storage",
        ),
        (
            lambda value: value["acquisition"].update(mode="mutable-provider-main"),
            "acquisition mode",
        ),
        (
            lambda value: value["acquisition"].update(provider_checksums_published=True),
            "provider checksum",
        ),
        (lambda value: value.update(dataset_sha256="0" * 64), "dataset identity"),
        (lambda value: value["splits"].pop(), "every temporal bucket"),
    ),
)
def test_runner_input_preflight_rejects_unreviewed_or_drifting_authority(
    tmp_path: Path, mutate: object, match: str
) -> None:
    receipt = verify_dataset_manifest(_runner_dataset_manifest(tmp_path), root=tmp_path)
    _, payload = _runner_input_manifest(tmp_path, receipt)
    assert callable(mutate)
    mutate(payload)
    with pytest.raises(ClearQualificationError, match=match):
        verify_runner_input_manifest(
            json.dumps(payload).encode(), root=tmp_path, dataset_receipt=receipt
        )


def test_runner_input_rejects_credentialed_provider_locator(tmp_path: Path) -> None:
    receipt = verify_dataset_manifest(_runner_dataset_manifest(tmp_path), root=tmp_path)
    _, payload = _runner_input_manifest(tmp_path, receipt)
    acquisition = payload["acquisition"]
    assert isinstance(acquisition, dict)
    acquisition["provider_locator"] = "https://secret@example.invalid/clear"
    with pytest.raises(ClearQualificationError, match="HTTPS locator"):
        verify_runner_input_manifest(
            json.dumps(payload).encode(), root=tmp_path, dataset_receipt=receipt
        )


def test_runner_input_preflight_revalidates_the_supplied_dataset_receipt(tmp_path: Path) -> None:
    receipt = verify_dataset_manifest(_runner_dataset_manifest(tmp_path), root=tmp_path)
    raw, _ = _runner_input_manifest(tmp_path, receipt)
    forged = ClearDatasetReceipt(
        archives=receipt.archives,
        samples_per_bucket=receipt.samples_per_bucket,
        archive_bytes=True,
        sample_count=receipt.sample_count,
        dataset_sha256=receipt.dataset_sha256,
    )
    with pytest.raises(ClearQualificationError, match="accounting"):
        verify_runner_input_manifest(raw, root=tmp_path, dataset_receipt=forged)

    with pytest.raises(ClearQualificationError, match="accounting"):
        qualification_plan(forged)

def test_runner_input_preflight_rejects_split_alias_hash_drift_and_class_gaps(
    tmp_path: Path,
) -> None:
    receipt = verify_dataset_manifest(_runner_dataset_manifest(tmp_path), root=tmp_path)
    _, payload = _runner_input_manifest(tmp_path, receipt)
    splits = payload["splits"]
    assert isinstance(splits, list)
    first = splits[0]
    assert isinstance(first, dict)

    first["evaluation_index"] = first["train_index"]
    with pytest.raises(ClearQualificationError, match="index paths must be unique"):
        verify_runner_input_manifest(
            json.dumps(payload).encode(), root=tmp_path, dataset_receipt=receipt
        )

    _, payload = _runner_input_manifest(tmp_path, receipt)
    splits = payload["splits"]
    assert isinstance(splits, list) and isinstance(splits[0], dict)
    index_identity = splits[0]["train_index"]
    assert isinstance(index_identity, dict)
    index_identity["sha256"] = "0" * 64
    with pytest.raises(ClearQualificationError, match="index SHA-256"):
        verify_runner_input_manifest(
            json.dumps(payload).encode(), root=tmp_path, dataset_receipt=receipt
        )

    _, payload = _runner_input_manifest(tmp_path, receipt)
    splits = payload["splits"]
    assert isinstance(splits, list) and isinstance(splits[0], dict)
    index_identity = splits[0]["train_index"]
    assert isinstance(index_identity, dict)
    index = tmp_path / str(index_identity["path"])
    records = [json.loads(line) for line in index.read_bytes().splitlines()]
    records[-1]["class_index"] = 98
    index.write_bytes(b"".join(_canonical_line(record) for record in records))
    index_identity.update(_identity(index, tmp_path))
    with pytest.raises(ClearQualificationError, match="all 100 classes"):
        verify_runner_input_manifest(
            json.dumps(payload).encode(), root=tmp_path, dataset_receipt=receipt
        )


def test_runner_input_preflight_rejects_sample_alias_and_content_drift(tmp_path: Path) -> None:
    receipt = verify_dataset_manifest(_runner_dataset_manifest(tmp_path), root=tmp_path)
    _, payload = _runner_input_manifest(tmp_path, receipt)
    splits = payload["splits"]
    assert isinstance(splits, list) and isinstance(splits[0], dict)
    train_identity = splits[0]["train_index"]
    evaluation_identity = splits[0]["evaluation_index"]
    assert isinstance(train_identity, dict) and isinstance(evaluation_identity, dict)
    train_index = tmp_path / str(train_identity["path"])
    evaluation_index = tmp_path / str(evaluation_identity["path"])
    train_first = json.loads(train_index.read_bytes().splitlines()[0])
    evaluation_records = [json.loads(line) for line in evaluation_index.read_bytes().splitlines()]
    evaluation_records[0] = train_first
    evaluation_index.write_bytes(b"".join(_canonical_line(record) for record in evaluation_records))
    evaluation_identity.update(_identity(evaluation_index, tmp_path))
    with pytest.raises(
        ClearQualificationError, match="sample paths and IDs must be globally unique"
    ):
        verify_runner_input_manifest(
            json.dumps(payload).encode(), root=tmp_path, dataset_receipt=receipt
        )

    _, payload = _runner_input_manifest(tmp_path, receipt)
    splits = payload["splits"]
    assert isinstance(splits, list) and isinstance(splits[0], dict)
    evaluation_identity = splits[0]["evaluation_index"]
    assert isinstance(evaluation_identity, dict)
    evaluation_index = tmp_path / str(evaluation_identity["path"])
    evaluation_records = [json.loads(line) for line in evaluation_index.read_bytes().splitlines()]
    train_sample = tmp_path / "images/1/train/0.jpg"
    evaluation_sample = tmp_path / "images/1/evaluation/0.jpg"
    evaluation_sample.unlink()
    os.link(train_sample, evaluation_sample)
    evaluation_records[0].update(
        size_bytes=train_sample.stat().st_size,
        sha256=hashlib.sha256(train_sample.read_bytes()).hexdigest(),
    )
    evaluation_index.write_bytes(b"".join(_canonical_line(record) for record in evaluation_records))
    evaluation_identity.update(_identity(evaluation_index, tmp_path))
    with pytest.raises(ClearQualificationError, match="hard-link aliases"):
        verify_runner_input_manifest(
            json.dumps(payload).encode(), root=tmp_path, dataset_receipt=receipt
        )

    _, payload = _runner_input_manifest(tmp_path, receipt)
    sample = tmp_path / "images/1/train/0.jpg"
    sample.write_bytes(b"x" * sample.stat().st_size)
    with pytest.raises(ClearQualificationError, match="sample SHA-256"):
        verify_runner_input_manifest(
            json.dumps(payload).encode(), root=tmp_path, dataset_receipt=receipt
        )


def test_runner_input_preflight_does_not_follow_sample_parent_symlinks(tmp_path: Path) -> None:
    receipt = verify_dataset_manifest(_runner_dataset_manifest(tmp_path), root=tmp_path)
    _, payload = _runner_input_manifest(tmp_path, receipt)
    real_images = tmp_path / "images-real"
    (tmp_path / "images").rename(real_images)
    (tmp_path / "images").symlink_to(real_images, target_is_directory=True)
    with pytest.raises(ClearQualificationError, match="unavailable below the dataset root"):
        verify_runner_input_manifest(
            json.dumps(payload).encode(), root=tmp_path, dataset_receipt=receipt
        )


def test_runner_input_preflight_parses_review_semantics_not_only_document_hashes(
    tmp_path: Path,
) -> None:
    receipt = verify_dataset_manifest(_runner_dataset_manifest(tmp_path), root=tmp_path)
    _, payload = _runner_input_manifest(tmp_path, receipt)
    documents = payload["review_documents"]
    assert isinstance(documents, list) and isinstance(documents[1], dict)
    rights_document = documents[1]
    path = tmp_path / str(rights_document["path"])
    review = json.loads(path.read_bytes())
    review["reviewed_scopes"].remove("takedown-process")
    path.write_bytes(json.dumps(review).encode())
    rights_document.update(_identity(path, tmp_path))
    with pytest.raises(ClearQualificationError, match="review semantics"):
        verify_runner_input_manifest(
            json.dumps(payload).encode(), root=tmp_path, dataset_receipt=receipt
        )


def test_result_validator_enforces_receipts_nonpromotion_and_negative_retention() -> None:
    plan = _result_plan()
    plan_sha = _plan_sha256(plan)
    budget = plan["resource_budget_per_axis"]
    assert isinstance(budget, dict)
    matrix = [[0.5 for _ in range(10)] for _ in range(10)]
    result: dict[str, Any] = {
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
            **{
                name: budget[name]
                for name in (
                    "archive_bytes",
                    "training_observations",
                    "data_samples_read",
                    "optimizer_updates",
                    "model_queries",
                    "environment_steps",
                )
            },
            "wall_seconds_telemetry": 6,
        },
    }
    assert validate_result(json.dumps(result).encode(), expected_plan=plan) == result
    with pytest.raises(TypeError, match="expected_plan"):
        validate_result(json.dumps(result).encode())  # type: ignore[call-arg]
    mismatched = json.loads(json.dumps(result))
    mismatched["resource_receipts"]["model_queries"] += 1
    with pytest.raises(ClearQualificationError, match="model_queries"):
        validate_result(json.dumps(mismatched).encode(), expected_plan=plan)
    hostile_metric = {**result, "metrics": {**result["metrics"], "accuracy": True}}
    with pytest.raises(ClearQualificationError, match="finite exact floats"):
        validate_result(json.dumps(hostile_metric).encode(), expected_plan=plan)
    for field, value, match in (
        ("promotion_authorized", True, "nonpromotion"),
        ("negative_retained", False, "negative retention"),
        ("status", "promoted", "status"),
        ("plan_sha256", "b" * 64, "provenance"),
    ):
        hostile = {**result, field: value}
        with pytest.raises(ClearQualificationError, match=match):
            validate_result(json.dumps(hostile).encode(), expected_plan=plan)


def test_result_rejects_scalar_alias_and_unbounded_payload() -> None:
    plan = _result_plan()
    result = {
        "schema_version": SCHEMA,
        "plan_sha256": _plan_sha256(plan),
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
        validate_result(json.dumps(result).encode(), expected_plan=plan)
    with pytest.raises(ClearQualificationError, match="byte limit"):
        validate_result(b" " * ((1 << 20) + 1), expected_plan=plan)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda plan: plan.update(paper_revision="unreviewed"),
        lambda plan: plan.update(runtime={"python": "forged"}),
        lambda plan: plan.update(current_source_sha256={}),
        lambda plan: plan.update(axes=[]),
        lambda plan: plan.update(control_config={}),
        lambda plan: plan["resource_budget_per_axis"].update(model_queries=1),
    ),
)
def test_result_rejects_forged_expected_plan_authority(mutate: object) -> None:
    plan = _result_plan()
    assert callable(mutate)
    mutate(plan)
    with pytest.raises(ClearQualificationError, match="frozen|derivable"):
        validate_result(b"{}", expected_plan=plan)
