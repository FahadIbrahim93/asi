from __future__ import annotations

import copy
import dataclasses
import hashlib
import os
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import alberta_framework.evaluation.bimu_matched_campaign as campaign
import alberta_framework.evaluation.bimu_matched_nonpromoting as plan_module
from alberta_framework.benchmarks.bimu import _dataset_sha256
from alberta_framework.evaluation.bimu_matched_nonpromoting import _test_plan


def _full_data(input_dim: int = 4) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(60_000 * input_dim, dtype=np.float32).reshape(60_000, input_dim)
    x /= np.float32(60_000 * input_dim)
    y = np.arange(60_000, dtype=np.int32) % 2
    return x, y


def _slice_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.arange(8 * 4, dtype=np.float32).reshape(8, 4) / np.float32(32.0)
    y = np.arange(8, dtype=np.int32) % 2
    return x[:4], y[:4], x[4:], y[4:]


def _resign(payload: dict[str, Any], field: str = "shard_sha256") -> None:
    payload[field] = campaign.digest_without(payload, field)


@pytest.fixture
def tiny_campaign(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    plan = _test_plan(input_dim=4, n_classes=2, examples=4)
    monkeypatch.setattr(plan_module, "FROZEN_BIMU_MATCHED_PLAN", plan)
    monkeypatch.setattr(plan_module, "EXECUTION_AUTHORIZED", True)
    monkeypatch.setattr(
        plan_module,
        "FROZEN_PLAN_SHA256",
        campaign._sha256(plan_module._plan_payload(plan)),
    )
    monkeypatch.setattr(campaign, "FROZEN_BIMU_MATCHED_PLAN", plan)
    monkeypatch.setattr(campaign, "EXECUTION_AUTHORIZED", True)
    monkeypatch.setattr(campaign, "REGISTERED_OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(campaign, "load_frozen_bimu_dataset", lambda _home=None: _slice_data())
    process_index = 0

    def distinct_process_identity() -> dict[str, object]:
        nonlocal process_index
        process_index += 1
        identity: dict[str, object] = {
            "schema": campaign.PROCESS_SCHEMA,
            "pid": 10_000 + process_index,
            "proc_start_ticks": 20_000 + process_index,
            "boot_id_sha256": "a" * 64,
            "invocation_nonce": f"{process_index:032x}",
            "fresh_process_required": True,
            "identity_is_not_attestation": True,
        }
        identity["execution_instance_id"] = campaign.digest_without(
            identity, "execution_instance_id"
        )
        return identity

    monkeypatch.setattr(campaign, "_process_identity", distinct_process_identity)
    return plan


def test_frozen_dataset_loader_uses_exact_canonical_60k_slice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, y = _full_data()
    monkeypatch.setattr(campaign, "load_mnist_train", lambda _home=None: (x, y))
    plan = _test_plan(input_dim=4, n_classes=2, examples=4)
    plan = dataclasses.replace(
        plan,
        dataset_sha256=_dataset_sha256(x[:4], y[:4], x[-4:], y[-4:]),
    )

    train_x, train_y, test_x, test_y = campaign.load_frozen_bimu_dataset(
        Path("/cache"), plan=plan
    )

    np.testing.assert_array_equal(train_x, x[:4])
    np.testing.assert_array_equal(train_y, y[:4])
    np.testing.assert_array_equal(test_x, x[-4:])
    np.testing.assert_array_equal(test_y, y[-4:])
    assert not np.shares_memory(train_x, x)
    assert not np.shares_memory(test_x, x)


def test_run_shard_refuses_before_loading_data_when_unauthorized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called = False

    def fail(_home: Path | None = None) -> Any:
        nonlocal called
        called = True
        raise AssertionError("dataset must not be loaded")

    monkeypatch.setattr(campaign, "load_frozen_bimu_dataset", fail)
    with pytest.raises(PermissionError, match="not authorized"):
        campaign.run_bimu_shard("memory_off", 157001, data_home=tmp_path)
    assert called is False


@pytest.mark.skipif(sys.platform != "linux", reason="campaign execution requires Linux /proc")
def test_linux_process_identity_is_current_and_self_consistent() -> None:
    identity = campaign._process_identity()
    assert campaign._validate_process(identity) == identity
    assert identity["pid"] > 0
    assert identity["boot_id_sha256"] != hashlib.sha256(b"").hexdigest()


@pytest.mark.skipif(sys.platform != "linux", reason="campaign publication is Linux-only")
def test_cli_run_shard_refuses_before_loading_data_when_unauthorized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(campaign, "REGISTERED_OUTPUT_ROOT", tmp_path)
    campaign.publish_json(
        campaign.campaign_path(tmp_path, "plan"),
        campaign.build_plan_document(),
        root=tmp_path,
    )
    called = False

    def fail(_home: Path | None = None) -> Any:
        nonlocal called
        called = True
        raise AssertionError("dataset must not be loaded")

    monkeypatch.setattr(campaign, "load_frozen_bimu_dataset", fail)
    assert campaign.main(
        [
            "run-shard",
            "--root",
            str(tmp_path),
            "--arm",
            "memory_off",
            "--seed",
            "157001",
        ]
    ) == 2
    assert called is False


def test_one_shard_roundtrip_binds_all_execution_axes(tiny_campaign: Any) -> None:
    shard = campaign.run_bimu_shard("memory_off", 157001)
    validated = campaign.validate_bimu_shard(shard)

    assert validated == shard
    assert shard["spec"] == {"arm": "memory_off", "seed": 157001}
    assert shard["identity"]["source_sha256"] == plan_module._source_identity()
    assert shard["identity"]["runtime"] == plan_module._runtime_identity()
    assert shard["identity"]["dependencies"] == plan_module._dependency_identity()
    assert shard["resources"]["dataset_numeric_bytes"] == 160
    assert shard["timing"]["qualified"] is False
    assert shard["timing"]["used_for_outcome"] is False


def test_shard_validator_rejects_forged_nested_receipts_and_hostile_trees(
    tiny_campaign: Any,
) -> None:
    shard = campaign.run_bimu_shard("bimu", 157002)

    forged = copy.deepcopy(shard)
    forged["result"]["counters"]["optimizer_updates"] -= 1
    _resign(forged)
    with pytest.raises(ValueError, match="counter|optimizer"):
        campaign.validate_bimu_shard(forged)

    forged = copy.deepcopy(shard)
    forged["identity"]["dependencies"]["uv_lock_sha256"] = "0" * 64
    _resign(forged)
    with pytest.raises(ValueError, match="identity"):
        campaign.validate_bimu_shard(forged)

    hostile: object = "leaf"
    for _ in range(campaign.MAX_JSON_DEPTH + 1):
        hostile = [hostile]
    with pytest.raises(ValueError, match="nesting"):
        campaign.validate_bimu_shard(hostile)

    plan = campaign.build_plan_document()
    plan["policy"]["execution_authorized"] = 1
    plan["document_sha256"] = campaign.digest_without(plan, "document_sha256")
    with pytest.raises(ValueError, match="policy"):
        campaign.validate_plan_document(plan)


def test_strict_shard_validator_reexecutes_final_state(tiny_campaign: Any) -> None:
    arrays = _slice_data()
    shard = campaign.run_bimu_shard("bimu", 157002)
    assert campaign.validate_bimu_shard_by_reexecution(shard, *arrays) == shard

    forged = copy.deepcopy(shard)
    forged["result"]["final_state_sha256"] = "0" * 64
    forged["result"]["resources"]["state_changed"] = True
    _resign(forged)
    assert campaign.validate_bimu_shard(forged) == forged
    with pytest.raises(ValueError, match="reexecution"):
        campaign.validate_bimu_shard_by_reexecution(forged, *arrays)


def _six_shards() -> list[dict[str, Any]]:
    return [
        campaign.run_bimu_shard(arm, seed)
        for seed in (157001, 157002, 157003)
        for arm in ("memory_off", "bimu")
    ]


def test_aggregate_requires_six_unique_shards_and_recomputes_paired_rule(
    tiny_campaign: Any,
) -> None:
    shards = _six_shards()
    aggregate = campaign.summarize_bimu_shards(shards)
    campaign.validate_bimu_aggregate(aggregate)

    deltas = aggregate["paired_metrics"]["primary_deltas"]
    expected = (
        "supported"
        if all(delta > 0.0 for delta in deltas)
        else "rejected"
        if all(delta <= 0.0 for delta in deltas)
        else "inconclusive"
    )
    assert aggregate["outcome"]["classification"] == expected
    assert aggregate["outcome"]["scientific_evidence"] is False

    with pytest.raises(ValueError, match="roster"):
        campaign.summarize_bimu_shards(shards[:-1])
    with pytest.raises(ValueError, match="roster|duplicate"):
        campaign.summarize_bimu_shards([*shards[:-1], shards[0]])

    same_process = copy.deepcopy(shards)
    same_process[1]["identity"]["process"] = copy.deepcopy(
        same_process[0]["identity"]["process"]
    )
    _resign(same_process[1])
    with pytest.raises(ValueError, match="process"):
        campaign.summarize_bimu_shards(same_process)


def test_aggregate_rejects_cross_pair_schedule_or_resource_drift(tiny_campaign: Any) -> None:
    shards = _six_shards()
    forged = copy.deepcopy(shards)
    forged[1]["result"]["schedule_sha256"] = "0" * 64
    _resign(forged[1])
    with pytest.raises(ValueError, match="schedule"):
        campaign.summarize_bimu_shards(forged)

    forged = copy.deepcopy(shards)
    forged[1]["result"]["resources"]["parameter_numeric_bytes"] += 4
    _resign(forged[1])
    with pytest.raises(ValueError, match="resource|accounting"):
        campaign.summarize_bimu_shards(forged)


@pytest.mark.skipif(sys.platform != "linux", reason="campaign publication is Linux-only")
def test_fixed_namespace_publication_is_append_only(
    tmp_path: Path, tiny_campaign: Any
) -> None:
    plan_path = campaign.campaign_path(tmp_path, "plan")
    campaign.publish_json(plan_path, campaign.build_plan_document(), root=tmp_path)
    with pytest.raises(FileExistsError):
        campaign.publish_json(plan_path, campaign.build_plan_document(), root=tmp_path)

    outside = tmp_path / "outside.json"
    with pytest.raises(ValueError, match="namespace"):
        campaign.publish_json(outside, {}, root=tmp_path)

    invalid_shard_path = campaign.campaign_path(
        tmp_path, "shard", arm="memory_off", seed=157001
    )
    with pytest.raises(ValueError, match="shard"):
        campaign.publish_json(invalid_shard_path, {}, root=tmp_path)

    (plan_path.parent / "unexpected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected"):
        campaign.main(["validate", "--root", str(tmp_path)])


@pytest.mark.skipif(sys.platform != "linux", reason="campaign validation is Linux-only")
def test_incomplete_namespace_rejects_broken_expected_shard_symlink(
    tmp_path: Path, tiny_campaign: Any
) -> None:
    campaign.publish_json(
        campaign.campaign_path(tmp_path, "plan"),
        campaign.build_plan_document(),
        root=tmp_path,
    )
    shard_path = campaign.campaign_path(
        tmp_path, "shard", arm="memory_off", seed=157001
    )
    shard_path.symlink_to(tmp_path / "absent.json")

    with pytest.raises(ValueError, match="regular non-symlink"):
        campaign.main(["validate", "--root", str(tmp_path)])


@pytest.mark.skipif(sys.platform != "linux", reason="campaign publication is Linux-only")
def test_publication_parent_swap_never_writes_through_replacement(
    tmp_path: Path, tiny_campaign: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = campaign.campaign_path(tmp_path, "plan")
    retired = tmp_path / "retired"
    replacement_bytes = b"replacement-directory"
    original_link = campaign._link_unnamed_file

    def swap_parent(file_descriptor: int, parent_descriptor: int, name: str) -> None:
        path.parent.rename(retired)
        path.parent.mkdir()
        path.write_bytes(replacement_bytes)
        original_link(file_descriptor, parent_descriptor, name)

    monkeypatch.setattr(campaign, "_link_unnamed_file", swap_parent)
    with pytest.raises(RuntimeError, match="parent changed"):
        campaign.publish_json(path, campaign.build_plan_document(), root=tmp_path)

    assert path.read_bytes() == replacement_bytes
    assert (retired / path.name).is_file()


@pytest.mark.skipif(sys.platform != "linux", reason="campaign publication is Linux-only")
def test_publication_race_retains_concurrent_destination(
    tmp_path: Path, tiny_campaign: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = campaign.campaign_path(tmp_path, "plan")
    competitor = b"concurrent-winner"
    original_link = campaign._link_unnamed_file

    def occupy_destination(file_descriptor: int, parent_descriptor: int, name: str) -> None:
        competitor_descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o444,
            dir_fd=parent_descriptor,
        )
        try:
            os.write(competitor_descriptor, competitor)
            os.fsync(competitor_descriptor)
        finally:
            os.close(competitor_descriptor)
        original_link(file_descriptor, parent_descriptor, name)

    monkeypatch.setattr(campaign, "_link_unnamed_file", occupy_destination)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        campaign.publish_json(path, campaign.build_plan_document(), root=tmp_path)

    assert path.read_bytes() == competitor


@pytest.mark.skipif(sys.platform != "linux", reason="campaign publication is Linux-only")
def test_publication_revalidates_readback_without_unlinking_visible_inode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(campaign, "REGISTERED_OUTPUT_ROOT", tmp_path)
    plan_path = campaign.campaign_path(tmp_path, "plan")
    document = campaign.build_plan_document()
    original = campaign.validate_plan_document
    calls = 0

    def fail_readback(value: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("simulated readback rejection")
        return original(value)

    monkeypatch.setattr(campaign, "validate_plan_document", fail_readback)
    with pytest.raises(ValueError, match="readback rejection"):
        campaign.publish_json(plan_path, document, root=tmp_path)
    assert plan_path.is_file()
    assert plan_path.stat().st_mode & 0o777 == 0o444


@pytest.mark.skipif(sys.platform != "linux", reason="campaign publication is Linux-only")
def test_cli_reserves_exact_shard_before_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tiny_campaign: Any
) -> None:
    campaign.publish_json(
        campaign.campaign_path(tmp_path, "plan"),
        campaign.build_plan_document(),
        root=tmp_path,
    )
    destination = campaign.campaign_path(
        tmp_path, "shard", arm="memory_off", seed=157001
    )

    def fail_after_reservation(*args: object, **kwargs: object) -> dict[str, object]:
        assert destination.is_file()
        assert destination.stat().st_size == 0
        raise RuntimeError("simulated execution failure")

    monkeypatch.setattr(campaign, "run_bimu_shard", fail_after_reservation)
    with pytest.raises(RuntimeError, match="execution failure"):
        campaign.main(
            [
                "run-shard",
                "--root",
                str(tmp_path),
                "--arm",
                "memory_off",
                "--seed",
                "157001",
            ]
        )
    assert destination.is_file()
    assert destination.stat().st_size == 0


@pytest.mark.skipif(sys.platform != "linux", reason="campaign publication is Linux-only")
def test_publication_rejects_zero_progress_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tiny_campaign: Any
) -> None:
    document = campaign.build_plan_document()
    monkeypatch.setattr(campaign.os, "write", lambda *args: 0)
    with pytest.raises(OSError, match="no progress"):
        campaign.publish_json(
            campaign.campaign_path(tmp_path, "plan"), document, root=tmp_path
        )


@pytest.mark.skipif(sys.platform != "linux", reason="campaign file validation is Linux-only")
def test_strict_loader_rejects_duplicate_keys_and_oversized_input(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        campaign.load_json_strict(duplicate, byte_ceiling=1024)

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * 1025)
    with pytest.raises(ValueError, match="byte ceiling"):
        campaign.load_json_strict(oversized, byte_ceiling=1024)

    linked = tmp_path / "linked.json"
    linked.write_text("{}", encoding="utf-8")
    linked_alias = tmp_path / "linked-alias.json"
    linked_alias.hardlink_to(linked)
    with pytest.raises(ValueError, match="hard-link"):
        campaign.load_json_strict(linked, byte_ceiling=1024)

    long_key = tmp_path / "long-key.json"
    long_key.write_text('{"' + "x" * (campaign.MAX_TEXT_BYTES + 1) + '":0}', encoding="utf-8")
    with pytest.raises(ValueError, match="keys|text"):
        campaign.load_json_strict(long_key, byte_ceiling=campaign.MAX_TEXT_BYTES + 32)

    recursive = tmp_path / "recursive.json"
    recursive.write_text("[" * 2000 + "0" + "]" * 2000, encoding="ascii")
    with pytest.raises(ValueError, match="strict document|nesting"):
        campaign.load_json_strict(recursive, byte_ceiling=5000)


def test_json_tree_rejects_hostile_exact_type_subclasses_without_hooks() -> None:
    class Hostile(str):
        calls = 0

        def encode(self, *args: object, **kwargs: object) -> bytes:
            self.calls += 1
            raise AssertionError("subclass hooks must not run")

    key = Hostile("schema")
    with pytest.raises(ValueError, match="keys"):
        campaign._validate_json_tree({key: "value"})
    assert key.calls == 0


def test_cli_has_only_plan_run_shard_summarize_and_validate() -> None:
    parser = campaign._parser()
    for command in ("plan", "run-shard", "summarize", "validate"):
        assert parser.parse_args([command, "--root", "/tmp/root", *(
            ["--arm", "bimu", "--seed", "157001"] if command == "run-shard" else []
        )]).command == command


def test_aggregate_validator_rejects_resigned_outcome_reclassification(
    tiny_campaign: Any,
) -> None:
    aggregate = cast(dict[str, Any], campaign.summarize_bimu_shards(_six_shards()))
    aggregate["outcome"]["classification"] = "supported"
    aggregate["aggregate_sha256"] = campaign.digest_without(aggregate, "aggregate_sha256")
    with pytest.raises(ValueError, match="outcome"):
        campaign.validate_bimu_aggregate(aggregate)
