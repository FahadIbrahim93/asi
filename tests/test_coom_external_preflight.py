"""Hostile contracts for the nonexecuting COOM external preflight."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from alberta_framework.benchmarks.coom_external_preflight import (
    COOM_EXTERNAL_PREFLIGHT_SCHEMA,
    COOMExternalPreflightError,
    load_preflight_manifest,
    main,
    verify_external_preflight,
    workload_identity_sha256,
)
from alberta_framework.benchmarks.coom_qualification import CO8_TASKS, COOM_COMMIT
from alberta_framework.benchmarks.external_qualification import qualification_plan

pytestmark = pytest.mark.unit


def _file(root: Path, relative: str, content: bytes) -> dict[str, object]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "path": relative,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _manifest(tmp_path: Path) -> tuple[dict[str, Any], Path, Path, Path]:
    source = tmp_path / "source"
    assets = tmp_path / "assets"
    runtime = tmp_path / "runtime"
    source.mkdir(parents=True)
    assets.mkdir()
    runtime.mkdir()
    source_files = []
    for role, relative in (
        ("source-archive", "COOM.git-archive.tar"),
        ("license", "LICENSE"),
        ("package-metadata", "setup.py"),
        ("environment-registration", "COOM/envs/__init__.py"),
        ("sequence-definition", "COOM/utils/config.py"),
    ):
        source_files.append({"role": role, **_file(source, relative, role.encode())})
    asset_files = []
    for task in CO8_TASKS:
        safe = task.removesuffix("-v0")
        for role, suffix in (("scenario-config", "cfg"), ("scenario-asset", "wad")):
            relative = f"{safe}/{role}.{suffix}"
            asset_files.append(
                {
                    "role": role,
                    "task_id": task,
                    "license_spdx": "MIT",
                    "redistribution_reviewed": False,
                    **_file(assets, relative, f"{task}:{role}".encode()),
                }
            )
    engine = _file(runtime, "bin/vizdoom", b"bounded fake engine fixture")
    runtime_value = {
        "system": "Linux",
        "machine": "x86_64",
        "python": "3.10.13",
        "implementation": "CPython",
        "container_image_digest": "sha256:" + "a" * 64,
        "lock": _file(runtime, "runtime.lock", b"exact runtime lock fixture"),
        "network_disabled": True,
        "packages": {
            "vizdoom": "1.2.4",
            "opencv-python": "4.10.0.84",
            "scipy": "1.11.4",
            "gymnasium": "0.28.1",
        },
        "engine": {"name": "ViZDoom", "version": "1.2.4", **engine},
    }
    config = {
        "sequence": "CO8",
        "tasks": list(CO8_TASKS),
        "seed": 0,
        "steps_per_task": 200_000,
        "replay_capacity": 50_000,
        "update_warmup_steps": 5_000,
        "batch_size": 128,
        "frame_skip": 4,
        "frame_stack": 4,
        "frame_height": 84,
        "frame_width": 84,
        "test_episodes": 3,
        "task_boundaries_available": True,
        "task_id_visible": True,
        "previous_environment_access_during_training": False,
        "reset_replay_at_task_boundary": True,
        "reset_optimizer_at_task_boundary": True,
        "reset_critic_at_task_boundary": False,
        "action_space_sha256_by_task": {
            task: hashlib.sha256(f"actions:{task}".encode()).hexdigest() for task in CO8_TASKS
        },
    }
    source_archive_sha256 = str(source_files[0]["sha256"])
    runtime_identity_sha256 = hashlib.sha256(_canonical(runtime_value)).hexdigest()
    assets_identity_sha256 = hashlib.sha256(_canonical(asset_files)).hexdigest()
    config_sha256 = hashlib.sha256(_canonical(config)).hexdigest()
    trace = {
        "seed": 0,
        "task_ids": list(CO8_TASKS),
        "environment_resets": 8,
        "environment_steps": 8,
        "policy_queries": 8,
        "observation_bytes": 2 * 8 * 84 * 84 * 4,
        "action_bytes": 8 * 4,
        "reward_bytes": 8 * 4,
        "terminal_bytes": 8,
        "truncation_bytes": 8,
        "task_id_bytes": 8 * 4,
        "persistent_environment_bytes": 4096,
        "reset_sha256": "1" * 64,
        "observation_sha256": "2" * 64,
        "action_sha256": "3" * 64,
        "reward_sha256": "4" * 64,
        "terminal_sha256": "5" * 64,
        "truncation_sha256": "6" * 64,
        "task_id_sha256": "7" * 64,
        "workload_identity_sha256": workload_identity_sha256(
            source_archive_sha256=source_archive_sha256,
            runtime_identity_sha256=runtime_identity_sha256,
            assets_identity_sha256=assets_identity_sha256,
            config_sha256=config_sha256,
        ),
    }
    manifest: dict[str, Any] = {
        "schema_version": COOM_EXTERNAL_PREFLIGHT_SCHEMA,
        "classification": "local-readiness-only-nonpromoting",
        "source": {
            "repository": "https://github.com/TTomilin/COOM.git",
            "commit": COOM_COMMIT,
            "git_tree_oid": "8" * 40,
            "source_archive_sha256": source_files[0]["sha256"],
            "identity_authenticated": False,
            "files": source_files,
        },
        "license": {
            "spdx": "MIT",
            "reviewed": True,
            "review_authenticated": False,
            "redistribution_authorized": False,
            "file_sha256": source_files[1]["sha256"],
        },
        "runtime": runtime_value,
        "assets": asset_files,
        "config": config,
        "trace_repetitions": [{"repetition": 0, **trace}, {"repetition": 1, **trace}],
        "external_runtime_executed_by_caller": True,
        "negative_outcome_retained": True,
        "execution_authorized_by_asi": False,
        "promotion_authorized": False,
        "benchmark_result_claimed": False,
    }
    return manifest, source, assets, runtime


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def test_exact_local_external_preflight_is_bound_but_nonclaiming(tmp_path: Path) -> None:
    manifest, source, assets, runtime = _manifest(tmp_path)
    receipt = verify_external_preflight(
        json.dumps(manifest).encode(),
        source_root=source,
        asset_root=assets,
        runtime_root=runtime,
    )
    assert receipt.schema_version == COOM_EXTERNAL_PREFLIGHT_SCHEMA
    assert receipt.source_repository == "https://github.com/TTomilin/COOM.git"
    assert receipt.source_commit == COOM_COMMIT
    assert receipt.source_git_tree_oid == "8" * 40
    assert receipt.sequence == "CO8"
    assert receipt.local_files_verified == 5 + 2 * len(CO8_TASKS) + 2
    assert receipt.deterministic_trace_pair_verified is True
    assert len(receipt.workload_identity_sha256) == 64
    assert receipt.source_identity_authenticated is False
    assert receipt.license_review_authenticated is False
    assert receipt.runtime_identity_authenticated is False
    assert receipt.asset_semantics_authenticated is False
    assert receipt.trace_execution_authenticated is False
    assert receipt.execution_authorized is False
    assert receipt.promotion_authorized is False
    assert receipt.benchmark_result_claimed is False
    assert receipt.remaining_qualification_gates == qualification_plan(1582).required_gates


@pytest.mark.parametrize(
    ("section", "field", "value", "match"),
    [
        ("source", "commit", "0" * 40, "official pin"),
        ("source", "identity_authenticated", True, "authentication"),
        ("license", "spdx", "GPL-3.0", "license"),
        ("license", "review_authenticated", True, "authentication"),
        ("license", "redistribution_authorized", True, "redistribution"),
        ("runtime", "system", "Darwin", "Linux"),
        ("runtime", "network_disabled", False, "network"),
        ("config", "task_id_visible", False, "configuration"),
        ("config", "steps_per_task", 2, "configuration"),
    ],
)
def test_preflight_rejects_source_license_runtime_and_config_drift(
    tmp_path: Path, section: str, field: str, value: object, match: str
) -> None:
    manifest, source, assets, runtime = _manifest(tmp_path)
    nested = manifest[section]
    assert isinstance(nested, dict)
    nested[field] = value
    with pytest.raises(COOMExternalPreflightError, match=match):
        verify_external_preflight(
            json.dumps(manifest).encode(),
            source_root=source,
            asset_root=assets,
            runtime_root=runtime,
        )


def test_preflight_rejects_tampered_and_aliased_files(tmp_path: Path) -> None:
    manifest, source, assets, runtime = _manifest(tmp_path)
    (source / "setup.py").write_bytes(b"x" * len(b"package-metadata"))
    with pytest.raises(COOMExternalPreflightError, match="SHA-256"):
        verify_external_preflight(
            json.dumps(manifest).encode(), source_root=source, asset_root=assets,
            runtime_root=runtime,
        )

    manifest, source, assets, runtime = _manifest(tmp_path / "archive-claim")
    manifest["source"]["source_archive_sha256"] = "9" * 64
    with pytest.raises(COOMExternalPreflightError, match="bound to the local archive"):
        verify_external_preflight(
            json.dumps(manifest).encode(), source_root=source, asset_root=assets,
            runtime_root=runtime,
        )

    manifest, source, assets, runtime = _manifest(tmp_path / "runtime-lock")
    lock = runtime / str(manifest["runtime"]["lock"]["path"])
    lock.write_bytes(b"x" * lock.stat().st_size)
    with pytest.raises(COOMExternalPreflightError, match="runtime lock SHA-256"):
        verify_external_preflight(
            json.dumps(manifest).encode(), source_root=source, asset_root=assets,
            runtime_root=runtime,
        )

    manifest, source, assets, runtime = _manifest(tmp_path / "second")
    asset = assets / str(manifest["assets"][0]["path"])
    alias = assets / "alias"
    os.link(asset, alias)
    with pytest.raises(COOMExternalPreflightError, match="hard-link"):
        verify_external_preflight(
            json.dumps(manifest).encode(), source_root=source, asset_root=assets,
            runtime_root=runtime,
        )


def test_preflight_rejects_path_escape_duplicate_and_unknown_fields(tmp_path: Path) -> None:
    manifest, source, assets, runtime = _manifest(tmp_path)
    manifest["extra"] = 1
    with pytest.raises(COOMExternalPreflightError, match="fields"):
        verify_external_preflight(
            json.dumps(manifest).encode(), source_root=source, asset_root=assets,
            runtime_root=runtime,
        )
    manifest, source, assets, runtime = _manifest(tmp_path / "second")
    manifest["assets"][0]["path"] = "../escape.wad"
    with pytest.raises(COOMExternalPreflightError, match="relative"):
        verify_external_preflight(
            json.dumps(manifest).encode(), source_root=source, asset_root=assets,
            runtime_root=runtime,
        )
    manifest, source, assets, runtime = _manifest(tmp_path / "empty-path")
    manifest["assets"][0]["path"] = "."
    with pytest.raises(COOMExternalPreflightError, match="relative"):
        verify_external_preflight(
            json.dumps(manifest).encode(), source_root=source, asset_root=assets,
            runtime_root=runtime,
        )
    manifest, source, assets, runtime = _manifest(tmp_path / "third")
    for field in ("path", "size_bytes", "sha256"):
        manifest["assets"][1][field] = manifest["assets"][0][field]
    with pytest.raises(COOMExternalPreflightError, match="unique"):
        verify_external_preflight(
            json.dumps(manifest).encode(), source_root=source, asset_root=assets,
            runtime_root=runtime,
        )


def test_preflight_rejects_intermediate_link_escape_and_duplicate_json_keys(
    tmp_path: Path,
) -> None:
    manifest, source, assets, runtime = _manifest(tmp_path)
    first = manifest["assets"][0]
    original_parent = assets / Path(str(first["path"])).parent
    escaped = tmp_path / "escaped"
    original_parent.rename(escaped)
    original_parent.symlink_to(escaped, target_is_directory=True)
    with pytest.raises(COOMExternalPreflightError, match="traverse a link"):
        verify_external_preflight(
            json.dumps(manifest).encode(), source_root=source, asset_root=assets,
            runtime_root=runtime,
        )

    duplicate = b'{"schema_version":1,"schema_version":2}'
    with pytest.raises(COOMExternalPreflightError, match="duplicate JSON"):
        verify_external_preflight(
            duplicate, source_root=source, asset_root=assets, runtime_root=runtime
        )


def test_preflight_rejects_trace_drift_type_aliases_and_authority(tmp_path: Path) -> None:
    mutations: tuple[tuple[Callable[[dict[str, Any]], object], str], ...] = (
        (
            lambda value: value["trace_repetitions"][1].update(action_sha256="f" * 64),
            "deterministic trace",
        ),
        (lambda value: value["trace_repetitions"][0].update(environment_steps=True), "integer"),
        (lambda value: value.update(execution_authorized_by_asi=True), "authority"),
        (lambda value: value.update(benchmark_result_claimed=True), "claim"),
    )
    for mutate, match in mutations:
        case_root = tmp_path / hashlib.sha256(match.encode()).hexdigest()
        manifest, source, assets, runtime = _manifest(case_root)
        mutate(manifest)
        with pytest.raises(COOMExternalPreflightError, match=match):
            verify_external_preflight(
                json.dumps(manifest).encode(), source_root=source, asset_root=assets,
                runtime_root=runtime,
            )


def test_trace_is_bound_to_workload_and_exact_reset_plus_step_observation_bytes(
    tmp_path: Path,
) -> None:
    manifest, source, assets, runtime = _manifest(tmp_path)
    manifest["trace_repetitions"][0]["workload_identity_sha256"] = "0" * 64
    manifest["trace_repetitions"][1]["workload_identity_sha256"] = "0" * 64
    with pytest.raises(COOMExternalPreflightError, match="workload identity"):
        verify_external_preflight(
            json.dumps(manifest).encode(), source_root=source, asset_root=assets,
            runtime_root=runtime,
        )

    manifest, source, assets, runtime = _manifest(tmp_path / "observation-bytes")
    manifest["trace_repetitions"][0]["observation_bytes"] -= 1
    manifest["trace_repetitions"][1]["observation_bytes"] -= 1
    with pytest.raises(COOMExternalPreflightError, match="observation_bytes"):
        verify_external_preflight(
            json.dumps(manifest).encode(), source_root=source, asset_root=assets,
            runtime_root=runtime,
        )


def test_preflight_rejects_symlinked_root_ancestors_and_empty_required_files(
    tmp_path: Path,
) -> None:
    manifest, source, assets, runtime = _manifest(tmp_path / "tree")
    real_tree = tmp_path / "real-tree"
    (tmp_path / "tree").rename(real_tree)
    linked_tree = tmp_path / "linked-tree"
    linked_tree.symlink_to(real_tree, target_is_directory=True)
    with pytest.raises(COOMExternalPreflightError, match="traverse a link"):
        verify_external_preflight(
            json.dumps(manifest).encode(),
            source_root=linked_tree / source.name,
            asset_root=linked_tree / assets.name,
            runtime_root=linked_tree / runtime.name,
        )

    manifest, source, assets, runtime = _manifest(tmp_path / "empty")
    package_file = source / "setup.py"
    package_file.write_bytes(b"")
    source_record = manifest["source"]["files"][2]
    source_record.update(size_bytes=0, sha256=hashlib.sha256(b"").hexdigest())
    with pytest.raises(COOMExternalPreflightError, match="size"):
        verify_external_preflight(
            json.dumps(manifest).encode(), source_root=source, asset_root=assets,
            runtime_root=runtime,
        )


def test_manifest_loader_is_bounded_regular_and_cli_writes_no_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest, source, assets, runtime = _manifest(tmp_path)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert load_preflight_manifest(path) == path.read_bytes()
    assert main(
        (
            str(path), "--source-root", str(source), "--asset-root", str(assets),
            "--runtime-root", str(runtime),
        )
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["execution_authorized"] is False
    assert payload["benchmark_result_claimed"] is False
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * ((1 << 20) + 1))
    with pytest.raises(COOMExternalPreflightError, match="byte limit"):
        load_preflight_manifest(oversized)
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(COOMExternalPreflightError, match="regular non-symlink"):
        load_preflight_manifest(link)

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    nested = real_parent / "nested.json"
    nested.write_bytes(b"{}")
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(COOMExternalPreflightError, match="traverse a link"):
        load_preflight_manifest(linked_parent / nested.name)


def test_manifest_loader_handles_short_regular_file_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "manifest.json"
    raw = b'{"bounded":"manifest"}'
    path.write_bytes(raw)
    original_read = os.read

    def short_read(descriptor: int, count: int) -> bytes:
        return original_read(descriptor, min(count, 3))

    monkeypatch.setattr(os, "read", short_read)
    assert load_preflight_manifest(path) == raw
