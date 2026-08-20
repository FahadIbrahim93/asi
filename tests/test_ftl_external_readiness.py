from __future__ import annotations

import dataclasses
import json
import os
import subprocess
from pathlib import Path

import pytest

import alberta_framework.benchmarks.ftl_external_readiness as readiness

pytestmark = pytest.mark.integration


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _fake_checkout(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "ContinualBench"
    required = {
        "README.md": "# Continual Bench\n",
        "LICENSE.txt": "MIT fixture\n",
        "pyproject.toml": (
            '[project]\nname = "continual-bench"\n'
            'dependencies = ["glfw==2.5.0"]\n'
        ),
        "continual_bench/__init__.py": "",
        "continual_bench/envs/__init__.py": "class ContinualBenchEnv: ...\n",
        "continual_bench/envs/reward_fns.py": "",
        "continual_bench/envs/mujoco/mujoco_env.py": "",
        "continual_bench/envs/mujoco/sawyer_bench.py": "",
        "continual_bench/envs/mujoco/sawyer_xyz_env.py": "",
        "continual_bench/envs/assets/model.xml": "<mujoco/>\n",
    }
    for relative, contents in required.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "fixture@example.invalid")
    _git(root, "config", "user.name", "fixture")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "fixture")
    return root, _git(root, "rev-parse", "HEAD")


def test_missing_external_assets_produce_precise_nonclaiming_manifest(tmp_path: Path) -> None:
    report = readiness.inspect_external_readiness(tmp_path / "absent")

    assert report.schema == readiness.SCHEMA
    assert report.source is None
    assert not report.source_checkout_qualified
    assert not report.execution_ready
    assert not report.reproduction_claim_allowed
    assert report.external_results_present is False
    assert "official_checkout_missing" in report.blockers
    assert "oa_reference_implementation_unpublished" in report.blockers
    assert "paper_protocol_lock_missing" in report.blockers
    assert "scientific_protocol_unfrozen" in report.blockers
    readiness.validate_report(report)


def test_exact_clean_checkout_receipt_binds_tree_assets_and_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout, revision = _fake_checkout(tmp_path)
    monkeypatch.setattr(readiness, "OFFICIAL_REVISION", revision)

    report = readiness.inspect_external_readiness(checkout)

    assert report.source is not None, report.blockers
    assert report.source_checkout_qualified
    assert report.source.revision == revision
    assert report.source.head_commit_sha1 == revision
    assert len(report.source.head_tree_sha1) == 40
    assert report.source.file_count == 10
    assert report.source.asset_file_count == 1
    assert report.source.total_tracked_bytes > 0
    assert len(report.source.tracked_files_sha256) == 64
    assert len(report.source.assets_sha256) == 64
    assert report.source.direct_dependencies == ("glfw==2.5.0",)
    assert report.source.repository_origin_attested is False
    assert report.runtime.python_version
    assert report.runtime.python_executable_sha256
    assert report.runtime.distributions
    assert report.protocol.episodes == 600
    assert report.protocol.observation_dim == 26
    assert report.protocol.task_order == readiness.PAPER_TASK_ORDER
    assert not report.execution_ready
    assert "official_checkout_missing" not in report.blockers
    assert "oa_reference_implementation_unpublished" in report.blockers
    readiness.validate_report(report)

    payload = report.to_payload()
    assert readiness.validate_report(payload) == report


@pytest.mark.parametrize(
    "mutation",
    ["dirty", "untracked", "ignored", "missing", "symlink", "dependency", "hidden_mode"],
)
def test_checkout_qualification_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    checkout, revision = _fake_checkout(tmp_path)
    monkeypatch.setattr(readiness, "OFFICIAL_REVISION", revision)
    if mutation == "dirty":
        (checkout / "README.md").write_text("changed\n", encoding="utf-8")
    elif mutation == "untracked":
        (checkout / "untracked.txt").write_text("no\n", encoding="utf-8")
    elif mutation == "ignored":
        (checkout / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
        _git(checkout, "add", ".gitignore")
        _git(checkout, "commit", "-qm", "ignore fixture")
        (checkout / "ignored.txt").write_text("no\n", encoding="utf-8")
        monkeypatch.setattr(readiness, "OFFICIAL_REVISION", _git(checkout, "rev-parse", "HEAD"))
    elif mutation == "missing":
        (checkout / "continual_bench/envs/reward_fns.py").unlink()
        _git(checkout, "add", "-u")
        _git(checkout, "commit", "-qm", "remove required source")
        monkeypatch.setattr(readiness, "OFFICIAL_REVISION", _git(checkout, "rev-parse", "HEAD"))
    elif mutation == "symlink":
        target = checkout / "continual_bench/envs/assets/model.xml"
        target.unlink()
        target.symlink_to("../../../../README.md")
        _git(checkout, "add", "-A")
        _git(checkout, "commit", "-qm", "symlink")
        monkeypatch.setattr(readiness, "OFFICIAL_REVISION", _git(checkout, "rev-parse", "HEAD"))
    elif mutation == "dependency":
        pyproject = checkout / "pyproject.toml"
        pyproject.write_text(
            '[project]\nname = "continual-bench"\ndependencies = ["glfw>=2"]\n',
            encoding="utf-8",
        )
        _git(checkout, "add", "pyproject.toml")
        _git(checkout, "commit", "-qm", "loosen dependency")
        monkeypatch.setattr(readiness, "OFFICIAL_REVISION", _git(checkout, "rev-parse", "HEAD"))
    else:
        _git(checkout, "config", "core.fileMode", "false")
        (checkout / "README.md").chmod(0o755)

    report = readiness.inspect_external_readiness(checkout)
    assert not report.source_checkout_qualified
    assert report.source is None
    assert any(blocker.startswith("official_checkout_invalid:") for blocker in report.blockers)
    assert not report.execution_ready


def test_validator_rejects_forged_readiness_and_extra_payload_fields(tmp_path: Path) -> None:
    report = readiness.inspect_external_readiness(tmp_path / "absent")
    with pytest.raises(ValueError, match="cannot be execution-ready"):
        readiness.validate_report(dataclasses.replace(report, execution_ready=True))

    payload = report.to_payload()
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="exact fields"):
        readiness.validate_report(payload)

    with pytest.raises(ValueError, match="exact integer"):
        dataclasses.replace(report.protocol, reported_accelerators=True)

    forged_runtime = dataclasses.replace(report.runtime, python_executable_sha256="0" * 64)
    with pytest.raises(ValueError, match="current runtime"):
        readiness.validate_report(dataclasses.replace(report, runtime=forged_runtime))

    with pytest.raises(ValueError, match="checkout blocker"):
        dataclasses.replace(
            report,
            blockers=tuple(
                blocker
                for blocker in report.blockers
                if blocker != "official_checkout_missing"
            ),
        )


def test_cli_prints_canonical_manifest_without_writing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checkout = tmp_path / "absent"
    before = tuple(tmp_path.iterdir())

    assert readiness.main(["--checkout", str(checkout)]) == 2
    payload = json.loads(capsys.readouterr().out)

    assert payload["schema"] == readiness.SCHEMA
    assert payload["execution_ready"] is False
    assert tuple(tmp_path.iterdir()) == before


def test_checkout_qualification_never_executes_checkout_configured_fsmonitor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout, revision = _fake_checkout(tmp_path)
    monkeypatch.setattr(readiness, "OFFICIAL_REVISION", revision)
    marker = tmp_path / "fsmonitor-executed"
    hook = tmp_path / "hostile-fsmonitor"
    hook.write_text(
        "#!/bin/sh\nprintf executed > \"$READINESS_FSMONITOR_MARKER\"\nexit 0\n",
        encoding="utf-8",
    )
    hook.chmod(0o755)
    _git(checkout, "config", "core.fsmonitor", str(hook))
    monkeypatch.setenv("READINESS_FSMONITOR_MARKER", os.fspath(marker))

    report = readiness.inspect_external_readiness(checkout)

    assert report.source_checkout_qualified
    assert not marker.exists(), "qualification executed code configured by the checkout"


def test_checkout_qualification_never_imports_tracked_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout, _revision = _fake_checkout(tmp_path)
    marker = tmp_path / "checkout-python-imported"
    (checkout / "continual_bench/__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )
    _git(checkout, "add", "continual_bench/__init__.py")
    _git(checkout, "commit", "-qm", "hostile import fixture")
    monkeypatch.setattr(readiness, "OFFICIAL_REVISION", _git(checkout, "rev-parse", "HEAD"))

    report = readiness.inspect_external_readiness(checkout)

    assert report.source_checkout_qualified
    assert not marker.exists()


def test_tree_enumeration_uses_captured_commit_not_mutable_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout, revision = _fake_checkout(tmp_path)
    monkeypatch.setattr(readiness, "OFFICIAL_REVISION", revision)
    original_git = readiness._git

    def guarded_git(root: Path, *arguments: str) -> bytes:
        if arguments[:5] == ("ls-tree", "-r", "-z", "--full-tree", "HEAD"):
            raise AssertionError("mutable HEAD used after the pinned commit was captured")
        return original_git(root, *arguments)

    monkeypatch.setattr(readiness, "_git", guarded_git)
    report = readiness.inspect_external_readiness(checkout)

    assert report.source_checkout_qualified


def test_checkout_replace_refs_cannot_impersonate_the_pinned_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout, official_revision = _fake_checkout(tmp_path)
    (checkout / "README.md").write_text("substituted tree\n", encoding="utf-8")
    _git(checkout, "add", "README.md")
    _git(checkout, "commit", "-qm", "substitute tree")
    substitute_revision = _git(checkout, "rev-parse", "HEAD")
    _git(checkout, "replace", official_revision, substitute_revision)
    _git(checkout, "update-ref", "HEAD", official_revision)
    monkeypatch.setattr(readiness, "OFFICIAL_REVISION", official_revision)

    report = readiness.inspect_external_readiness(checkout)

    assert report.source is None
    assert not report.source_checkout_qualified


def test_report_validation_rejects_oversized_and_aliased_json(tmp_path: Path) -> None:
    payload = readiness.inspect_external_readiness(tmp_path / "absent").to_payload()
    oversized = dict(payload)
    oversized["blockers"] = [*payload["blockers"], "x" * 20_000]
    with pytest.raises(ValueError, match="bound|oversized"):
        readiness.validate_report(oversized)

    shared = ["current_goal"]
    aliased = dict(payload)
    aliased["protocol"] = dict(payload["protocol"])
    aliased["protocol"]["task_order"] = shared
    aliased["blockers"] = shared
    with pytest.raises(ValueError, match="aliased|cyclic"):
        readiness.validate_report(aliased)


def test_qualification_never_follows_a_raced_intermediate_directory_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout, revision = _fake_checkout(tmp_path)
    monkeypatch.setattr(readiness, "OFFICIAL_REVISION", revision)
    original_git = readiness._git
    swapped = False

    def racing_git(root: Path, *arguments: str) -> bytes:
        nonlocal swapped
        if swapped:
            raise AssertionError("qualification reached a post-read check after following symlink")
        result = original_git(root, *arguments)
        if arguments[:4] == ("ls-tree", "-r", "-z", "--full-tree"):
            asset_directory = checkout / "continual_bench/envs/assets"
            outside = tmp_path / "outside-assets"
            asset_directory.rename(outside)
            asset_directory.symlink_to(outside, target_is_directory=True)
            swapped = True
        return result

    monkeypatch.setattr(readiness, "_git", racing_git)
    report = readiness.inspect_external_readiness(checkout)

    assert report.source is None
    assert any("symlink" in blocker or "path" in blocker for blocker in report.blockers)
