from __future__ import annotations

import dataclasses
import gzip
import hashlib
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from alberta_framework.benchmarks import native_supervised_canonical as canonical

pytestmark = pytest.mark.integration


def _idx_images(values: np.ndarray) -> bytes:
    return (
        struct.pack(">IIII", 2051, values.shape[0], values.shape[1], values.shape[2])
        + values.tobytes()
    )


def _idx_labels(values: np.ndarray) -> bytes:
    return struct.pack(">II", 2049, values.shape[0]) + values.tobytes()


def _fixture_definition(root: Path) -> canonical.CanonicalDefinition:
    train_labels = np.repeat(np.arange(10, dtype=np.uint8), 3)
    test_labels = np.repeat(np.arange(10, dtype=np.uint8), 2)
    train_images = np.arange(train_labels.size * 16, dtype=np.uint8).reshape((-1, 4, 4))
    test_images = (np.arange(test_labels.size * 16, dtype=np.uint8) + 7).reshape((-1, 4, 4))
    payloads = {
        "train-images-idx3-ubyte.gz": gzip.compress(_idx_images(train_images), mtime=0),
        "train-labels-idx1-ubyte.gz": gzip.compress(_idx_labels(train_labels), mtime=0),
        "t10k-images-idx3-ubyte.gz": gzip.compress(_idx_images(test_images), mtime=0),
        "t10k-labels-idx1-ubyte.gz": gzip.compress(_idx_labels(test_labels), mtime=0),
    }
    files = []
    for name, raw in payloads.items():
        (root / name).write_bytes(raw)
        files.append(
            canonical.AssetSpec(
                name=name,
                size_bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                md5=hashlib.md5(raw, usedforsecurity=False).hexdigest(),  # noqa: S324
            )
        )
    return canonical.CanonicalDefinition(
        dataset_id="mnist",
        authority_uri="https://example.invalid/mnist-fixture",
        files=tuple(files),
        train_examples=30,
        test_examples=20,
        image_shape=(4, 4),
        n_classes=10,
        train_class_histogram=(3,) * 10,
        test_class_histogram=(2,) * 10,
        loader_contract="test-only exact IDX fixture",
    )


def test_official_registry_pins_exact_assets_and_open_parity_gates() -> None:
    blockers = canonical.qualification_blocker_manifest()
    assert blockers["schema"] == "asi.native_supervised_cl_canonical.blockers.v1"
    assert blockers["avalanche_revision"].endswith("eb075be393e1f458b2c352514ff6c17b5a2c0f4e")
    assert blockers["official_assets_verified_by_runner"] is True
    assert blockers["external_transform_parity"] is False
    assert blockers["external_metric_implementation_parity"] is False
    assert blockers["scientific_promotion_allowed"] is False
    mnist = canonical.canonical_definition("split_mnist")
    assert mnist.train_examples == 60_000
    assert mnist.test_examples == 10_000
    assert len(mnist.files) == 4
    assert all(len(asset.sha256) == 64 and len(asset.md5) == 32 for asset in mnist.files)
    cifar = canonical.canonical_definition("split_cifar100")
    assert (
        cifar.files[0].sha256 == "85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7"
    )
    assert cifar.train_class_histogram == (500,) * 100
    assert cifar.test_class_histogram == (100,) * 100


def test_idx_decoder_is_exact_and_bounded() -> None:
    images = np.arange(32, dtype=np.uint8).reshape((2, 4, 4))
    labels = np.array([1, 9], dtype=np.uint8)
    assert np.array_equal(canonical._decode_idx_images(_idx_images(images), 2, (4, 4)), images)
    assert np.array_equal(canonical._decode_idx_labels(_idx_labels(labels), 2), labels)
    with pytest.raises(canonical.CanonicalQualificationError, match="magic"):
        canonical._decode_idx_images(b"\0" * 48, 2, (4, 4))
    with pytest.raises(canonical.CanonicalQualificationError, match="length"):
        canonical._decode_idx_labels(_idx_labels(labels) + b"x", 2)
    with pytest.raises(canonical.CanonicalQualificationError, match="gzip|decoded"):
        canonical._decompress_exact(gzip.compress(b"x" * 33), 32, "bomb")


def test_asset_reader_rejects_symlinks_hardlinks_and_digest_drift(tmp_path: Path) -> None:
    root = tmp_path / "assets"
    root.mkdir()
    raw = b"canonical"
    target = root / "asset.bin"
    target.write_bytes(raw)
    spec = canonical.AssetSpec(
        "asset.bin",
        len(raw),
        hashlib.sha256(raw).hexdigest(),
        hashlib.md5(raw, usedforsecurity=False).hexdigest(),  # noqa: S324
    )
    receipt, observed = canonical._read_asset(root, spec)
    assert observed == raw
    assert receipt.sha256 == spec.sha256
    root_alias = tmp_path / "root-alias"
    root_alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(canonical.CanonicalQualificationError, match="symlink|ancestor|open"):
        canonical._read_asset(root_alias, spec)
    target.unlink()
    outside = tmp_path / "outside"
    outside.write_bytes(raw)
    target.symlink_to(outside)
    with pytest.raises(canonical.CanonicalQualificationError, match="symlink|regular|open"):
        canonical._read_asset(root, spec)
    target.unlink()
    os.link(outside, target)
    with pytest.raises(canonical.CanonicalQualificationError, match="link"):
        canonical._read_asset(root, spec)
    target.unlink()
    target.write_bytes(b"forged")
    with pytest.raises(canonical.CanonicalQualificationError, match="size|digest"):
        canonical._read_asset(root, spec)


def test_avalanche_matrix_metrics_retain_distinct_definitions() -> None:
    matrix = (
        (0.1, 0.2, 0.3),
        (0.8, 0.2, 0.3),
        (0.7, 0.9, 0.3),
        (0.6, 0.85, 0.95),
    )
    metrics = canonical.avalanche_matrix_metrics(matrix)
    assert metrics.final_stream_accuracy == pytest.approx(0.8)
    assert metrics.first_post_training_forgetting == pytest.approx((0.2 + 0.05) / 2)
    assert metrics.backward_transfer == pytest.approx((-0.2 - 0.05) / 2)
    assert metrics.forward_transfer == pytest.approx(0.0)
    assert metrics.peak_to_final_forgetting == pytest.approx((0.2 + 0.05 + 0.0) / 3)


def test_canonical_receipt_rejects_noncanonical_static_and_resource_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    definition = _fixture_definition(tmp_path)
    monkeypatch.setattr(canonical, "canonical_definition", lambda _: definition)
    result = canonical.run_canonical_asset_qualification(
        "split_mnist",
        tmp_path,
        seed=15780,
        train_examples_per_task=2,
        test_examples_per_task=1,
        replay_capacity=2,
    )
    forged_binding = dataclasses.replace(
        result.asset_binding,
        authority_uri="https://example.invalid/forged-authority",
    )
    with pytest.raises(ValueError, match="asset binding|canonical definition"):
        canonical.CanonicalQualification.__post_init__(
            dataclasses.replace(result, asset_binding=forged_binding)
        )
    forged_resources = dataclasses.replace(
        result.resources,
        asset_files_opened=result.resources.asset_files_opened + 1,
    )
    with pytest.raises(ValueError, match="resource"):
        canonical.CanonicalQualification.__post_init__(
            dataclasses.replace(result, resources=forged_resources)
        )


def test_canonical_adapter_loads_verified_assets_then_replays_v2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    definition = _fixture_definition(tmp_path)
    monkeypatch.setattr(canonical, "canonical_definition", lambda _: definition)
    result = canonical.run_canonical_asset_qualification(
        "split_mnist",
        tmp_path,
        seed=15780,
        train_examples_per_task=2,
        test_examples_per_task=1,
        replay_capacity=2,
    )
    assert result.schema == canonical.SCHEMA
    assert result.asset_binding.official_asset_digests_verified
    assert result.asset_binding.canonical_loader_invariants_verified
    assert result.asset_binding.train_class_histogram == (3,) * 10
    assert result.asset_binding.test_class_histogram == (2,) * 10
    assert result.qualification.benchmark_id == "split_mnist"
    assert len(result.avalanche_metrics) == 4
    assert result.resources.asset_bytes_hashed == sum(
        asset.size_bytes for asset in definition.files
    )
    assert result.resources.decoded_payload_bytes == (
        16
        + definition.train_examples * 16
        + 8
        + definition.train_examples
        + 16
        + definition.test_examples * 16
        + 8
        + definition.test_examples
    )
    assert result.resources.adapter_slice_bytes > 0
    assert result.resources.peak_loader_payload_bytes == max(
        result.resources.asset_bytes_hashed
        + result.resources.decoded_payload_bytes
        + result.resources.canonical_array_bytes,
        result.resources.canonical_array_bytes + result.resources.adapter_slice_bytes,
    )
    assert not result.external_transform_parity
    assert not result.external_metric_implementation_parity
    assert (
        canonical.validate_canonical_asset_qualification(
            result,
            tmp_path,
        )
        == result
    )
    with pytest.raises(ValueError, match="resource|replay"):
        forged = dataclasses.replace(
            result,
            resources=dataclasses.replace(
                result.resources,
                asset_bytes_hashed=result.resources.asset_bytes_hashed + 1,
                peak_loader_payload_bytes=result.resources.peak_loader_payload_bytes + 1,
            ),
        )
        canonical.validate_canonical_asset_qualification(forged, tmp_path)


def test_canonical_adapter_rejects_wrong_histogram_before_learning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    definition = _fixture_definition(tmp_path)
    bad = dataclasses.replace(definition, train_class_histogram=(4, 2) + (3,) * 8)
    monkeypatch.setattr(canonical, "canonical_definition", lambda _: bad)
    with pytest.raises(canonical.CanonicalQualificationError, match="histogram"):
        canonical.run_canonical_asset_qualification(
            "split_mnist",
            tmp_path,
            seed=15780,
            train_examples_per_task=1,
            test_examples_per_task=1,
            replay_capacity=1,
        )
