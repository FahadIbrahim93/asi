from __future__ import annotations

import copy
import dataclasses
import json
from typing import cast

import numpy as np
import pytest

import alberta_framework.benchmarks.cora_external_qualification as cora_external_module
from alberta_framework.benchmarks.cora_external_qualification import (
    CORA_COMMIT,
    PROCGEN_TASKS,
    SMOKE_LEVEL_SEEDS,
    AssetIdentity,
    CORAProcgenSmokePlan,
    CORAProcgenSmokeReceipt,
    IsolatedRuntimeIdentity,
    SourceIdentity,
    build_smoke_receipt,
    main,
    qualification_blocker_manifest,
    validate_smoke_payload,
)

pytestmark = pytest.mark.unit


def _sha(character: str) -> str:
    return character * 64


@pytest.fixture
def source() -> SourceIdentity:
    return SourceIdentity(
        repository="https://github.com/AGI-Labs/continual_rl.git",
        commit=CORA_COMMIT,
        git_tree="1" * 40,
        source_archive_sha256=_sha("2"),
        install_tree_sha256=_sha("3"),
        required_file_sha256=(
            ("LICENSE", _sha("4")),
            ("README.md", _sha("5")),
            ("continual_rl/experiment_specs.py", _sha("6")),
            ("continual_rl/utils/metrics.py", _sha("7")),
            ("environment.yml", _sha("8")),
            ("main.py", _sha("9")),
            ("setup.py", _sha("a")),
        ),
        clean_checkout=True,
        commit_verified=True,
        license="MIT",
        authenticated_attestation=False,
    )


@pytest.fixture
def runtime() -> IsolatedRuntimeIdentity:
    return IsolatedRuntimeIdentity(
        image_digest="sha256:" + _sha("b"),
        lock_sha256=_sha("c"),
        python_version="3.8.18",
        torch_version="1.10.2",
        torchvision_version="0.11.3",
        gym_version="0.21.0",
        procgen_version="0.10.7",
        numpy_version="1.21.6",
        platform="linux-x86_64",
        accelerator="cpu",
        network_disabled=True,
        root_filesystem_read_only=True,
    )


@pytest.fixture
def assets() -> AssetIdentity:
    return AssetIdentity(
        procgen_distribution_archive_sha256=_sha("d"),
        procgen_install_tree_sha256=_sha("e"),
        procgen_compiled_data_sha256=_sha("f"),
        procgen_license_sha256=_sha("0"),
        package_version="0.10.7",
        asset_rights_reviewed=True,
    )


@pytest.fixture
def plan(
    source: SourceIdentity,
    runtime: IsolatedRuntimeIdentity,
    assets: AssetIdentity,
) -> CORAProcgenSmokePlan:
    return CORAProcgenSmokePlan(source=source, runtime=runtime, assets=assets)


def _trace(plan: CORAProcgenSmokePlan) -> CORAProcgenSmokeReceipt:
    horizon = len(PROCGEN_TASKS) * 2
    return build_smoke_receipt(
        plan,
        task_indices=np.repeat(np.arange(len(PROCGEN_TASKS), dtype=np.int32), 2),
        evaluation_split=np.tile(np.asarray([False, True], dtype=np.bool_), len(PROCGEN_TASKS)),
        level_seeds=np.asarray(SMOKE_LEVEL_SEEDS, dtype=np.int32),
        actions=np.zeros((horizon,), dtype=np.int32),
        observations=np.zeros((horizon, 64, 64, 3), dtype=np.uint8),
        rewards=np.zeros((horizon,), dtype=np.float32),
        terminated=np.zeros((horizon,), dtype=np.bool_),
        truncated=np.zeros((horizon,), dtype=np.bool_),
        persistent_environment_numeric_bytes=4096,
        timing_ns=12,
        outcome="inconclusive",
    )


def test_plan_binds_official_procgen_source_runtime_assets_and_information_contract(
    plan: CORAProcgenSmokePlan,
) -> None:
    payload = plan.payload()
    source_payload = cast(dict[str, object], payload["source"])
    assert source_payload["commit"] == CORA_COMMIT
    assert source_payload["authenticated_attestation"] is False
    assert payload["tasks"] == list(PROCGEN_TASKS)
    assert payload["paper_training_cycles"] == 5
    assert payload["paper_train_steps_per_task"] == 5_000_000
    assert payload["paper_train_levels"] == 200
    assert payload["paper_evaluation_level_distribution"] == "full"
    assert payload["smoke_level_seeds"] == list(SMOKE_LEVEL_SEEDS)
    assert payload["observation_shape"] == [64, 64, 3]
    assert payload["action_space_n"] == 15
    assert payload["learner_task_information"] == []
    assert payload["learner_boundary_information"] == []
    assert payload["scientific_promotion_allowed"] is False


def test_external_provider_trace_round_trips_with_exact_receipts(
    plan: CORAProcgenSmokePlan,
) -> None:
    receipt = _trace(plan)
    checked = validate_smoke_payload(receipt.payload())
    assert checked == receipt
    assert checked.environment_steps == 12
    assert checked.learner_updates == checked.model_queries == 0
    assert checked.observation_bytes == 12 * 64 * 64 * 3
    assert checked.persistent_environment_numeric_bytes_is_provider_reported_unattested is True
    assert (
        checked.persistent_environment_byte_scope
        == "provider_reported_unattested_external_runtime_numeric_arrays_nbytes_sum"
    )
    assert checked.action_bytes == 12 * np.dtype(np.int32).itemsize
    assert checked.level_seed_bytes == 12 * np.dtype(np.int32).itemsize
    assert checked.mechanism_off is True
    assert checked.external_runtime_executed is True
    assert checked.cora_parity_claimed is False
    assert len(checked.identity.lane_source_sha256) == 64


def test_builder_snapshots_and_rejects_noncanonical_schedule_or_action(
    plan: CORAProcgenSmokePlan,
) -> None:
    horizon = 12
    observations = np.zeros((horizon, 64, 64, 3), dtype=np.uint8)
    receipt = _trace(plan)
    observations[0, 0, 0, 0] = 1
    checked = validate_smoke_payload(receipt.payload())
    assert checked.observation_sha256 == receipt.observation_sha256

    bad_actions = np.zeros((horizon,), dtype=np.int32)
    bad_actions[0] = 1
    with pytest.raises(ValueError, match="fixed-action"):
        build_smoke_receipt(
            plan,
            task_indices=np.repeat(np.arange(6, dtype=np.int32), 2),
            evaluation_split=np.tile(np.asarray([False, True], dtype=np.bool_), 6),
            level_seeds=np.asarray(SMOKE_LEVEL_SEEDS, dtype=np.int32),
            actions=bad_actions,
            observations=np.zeros((horizon, 64, 64, 3), dtype=np.uint8),
            rewards=np.zeros(horizon, dtype=np.float32),
            terminated=np.zeros(horizon, dtype=np.bool_),
            truncated=np.zeros(horizon, dtype=np.bool_),
            persistent_environment_numeric_bytes=1,
            timing_ns=0,
            outcome="rejected",
        )

    bad_tasks = np.repeat(np.arange(6, dtype=np.int32), 2)
    bad_tasks[0] = 1
    with pytest.raises(ValueError, match="task schedule"):
        build_smoke_receipt(
            plan,
            task_indices=bad_tasks,
            evaluation_split=np.tile(np.asarray([False, True], dtype=np.bool_), 6),
            level_seeds=np.asarray(SMOKE_LEVEL_SEEDS, dtype=np.int32),
            actions=np.zeros(horizon, dtype=np.int32),
            observations=np.zeros((horizon, 64, 64, 3), dtype=np.uint8),
            rewards=np.zeros(horizon, dtype=np.float32),
            terminated=np.zeros(horizon, dtype=np.bool_),
            truncated=np.zeros(horizon, dtype=np.bool_),
            persistent_environment_numeric_bytes=1,
            timing_ns=0,
            outcome="rejected",
        )

    bad_levels = np.asarray(SMOKE_LEVEL_SEEDS, dtype=np.int32)
    bad_levels[1] = bad_levels[0]
    with pytest.raises(ValueError, match="level seeds"):
        build_smoke_receipt(
            plan,
            task_indices=np.repeat(np.arange(6, dtype=np.int32), 2),
            evaluation_split=np.tile(np.asarray([False, True], dtype=np.bool_), 6),
            level_seeds=bad_levels,
            actions=np.zeros(horizon, dtype=np.int32),
            observations=np.zeros((horizon, 64, 64, 3), dtype=np.uint8),
            rewards=np.zeros(horizon, dtype=np.float32),
            terminated=np.zeros(horizon, dtype=np.bool_),
            truncated=np.zeros(horizon, dtype=np.bool_),
            persistent_environment_numeric_bytes=1,
            timing_ns=0,
            outcome="rejected",
        )


def test_hostile_payloads_and_nested_postconstruction_mutation_fail_closed(
    plan: CORAProcgenSmokePlan,
) -> None:
    receipt = _trace(plan)
    forged_scope = copy.deepcopy(receipt.payload())
    forged_scope["persistent_environment_numeric_bytes_is_provider_reported_unattested"] = False
    with pytest.raises(ValueError, match="provider-reported"):
        validate_smoke_payload(forged_scope)
    extra = receipt.payload()
    extra["extra"] = True
    with pytest.raises(ValueError, match="fields differ"):
        validate_smoke_payload(extra)

    forged = copy.deepcopy(receipt.payload())
    forged["observation_bytes"] = cast(int, forged["observation_bytes"]) - 1
    with pytest.raises(ValueError, match="trace byte receipt"):
        validate_smoke_payload(forged)

    forged_action = copy.deepcopy(receipt.payload())
    forged_action["action_sha256"] = _sha("f")
    with pytest.raises(ValueError, match="fixed-action schedule"):
        validate_smoke_payload(forged_action)

    promoted = copy.deepcopy(receipt.payload())
    promoted["scientific_promotion_allowed"] = True
    with pytest.raises(ValueError, match="nonpromoting"):
        validate_smoke_payload(promoted)

    identity = copy.deepcopy(receipt.payload())
    identity_payload = cast(dict[str, object], identity["identity"])
    identity_payload["lane_source_sha256"] = _sha("0")
    with pytest.raises(ValueError, match="current tree/runtime"):
        validate_smoke_payload(identity)

    mutated_source = dataclasses.replace(plan.source)
    object.__setattr__(mutated_source, "commit", "0" * 40)
    mutated_plan = dataclasses.replace(plan)
    object.__setattr__(mutated_plan, "source", mutated_source)
    with pytest.raises(ValueError, match="source authority"):
        dataclasses.replace(receipt, plan=mutated_plan)


def test_source_runtime_and_assets_reject_self_consistent_loosening(
    source: SourceIdentity,
    runtime: IsolatedRuntimeIdentity,
    assets: AssetIdentity,
) -> None:
    with pytest.raises(ValueError, match="required file manifest"):
        dataclasses.replace(source, required_file_sha256=source.required_file_sha256[:-1])
    with pytest.raises(ValueError, match="required file manifest"):
        dataclasses.replace(source, required_file_sha256=(("LICENSE",),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="clean exact checkout"):
        dataclasses.replace(source, clean_checkout=False)
    with pytest.raises(ValueError, match="isolated runtime"):
        dataclasses.replace(runtime, network_disabled=False)
    with pytest.raises(ValueError, match="asset rights"):
        dataclasses.replace(assets, asset_rights_reviewed=False)
    with pytest.raises(ValueError, match="Procgen versions differ"):
        CORAProcgenSmokePlan(
            source=source,
            runtime=runtime,
            assets=dataclasses.replace(assets, package_version="0.9.0"),
        )


def test_blocker_manifest_stays_fail_closed_and_cli_emits_only_metadata(
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = qualification_blocker_manifest()
    assert manifest["ready"] is False
    assert manifest["external_execution_authorized"] is False
    assert manifest["native_analogue_is_cora"] is False
    blockers = manifest["blockers"]
    assert type(blockers) is list
    assert "official_source_checkout_verified" in blockers
    assert "paper_scale_execution_completed" in blockers
    assert main(["--blockers"]) == 0
    assert json.loads(capsys.readouterr().out) == manifest
    with pytest.raises(SystemExit):
        main([])


def test_payload_preflight_rejects_aliases_and_oversize_before_reconstruction(
    plan: CORAProcgenSmokePlan,
) -> None:
    receipt = _trace(plan)
    payload = receipt.payload()
    shared: list[object] = []
    payload["aliased"] = shared
    payload["also_aliased"] = shared
    with pytest.raises(ValueError, match="fields differ|aliases"):
        validate_smoke_payload(payload)

    oversized = receipt.payload()
    identity_payload = cast(dict[str, object], oversized["identity"])
    runtime_identity = cast(list[object], identity_payload["runtime_identity"])
    runtime_identity.append(["x", "y" * (1 << 20)])
    with pytest.raises(ValueError, match="limit|ceiling"):
        validate_smoke_payload(oversized)


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("PROCGEN_TASKS", ("forged_game",)),
        ("OUTCOMES", ("paper_parity_supported",)),
        ("PAPER_SEEDS", 1),
        ("_WORKLOAD_REGISTRY", (("tasks", ("forged_game",)),)),
    ),
)
def test_runtime_reassignment_cannot_change_the_frozen_contract(
    monkeypatch: pytest.MonkeyPatch,
    plan: CORAProcgenSmokePlan,
    name: str,
    value: object,
) -> None:
    monkeypatch.setattr(cora_external_module, name, value)
    with pytest.raises(ValueError, match="literal frozen CORA contract"):
        plan.payload()
