from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path("external_runtimes/coom")


def _smoke_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("coom_external_smoke_test", ROOT / "smoke.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_coom_runtime_is_source_dependency_and_base_image_pinned() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements.lock").read_text(encoding="utf-8")
    patch = ROOT / "coom-gymnasium.patch"
    manifest = json.loads((ROOT / "qualification-manifest.json").read_bytes())

    assert "python:3.12.12-slim-bookworm@sha256:" in dockerfile
    assert "7929801176c6e2e036c7c1c7dd6ce9b84a9d1f3e" in dockerfile
    assert "a4736e9916468482d75831d53a12a8601c4da91cd40b9b24d313522034a15661" in dockerfile
    assert hashlib.sha256(patch.read_bytes()).hexdigest() in dockerfile
    assert "--require-hashes" in dockerfile
    assert "apt-get" not in dockerfile
    assert manifest["base_image_digest"] in dockerfile
    assert manifest["dockerfile_sha256"] == hashlib.sha256(
        (ROOT / "Dockerfile").read_bytes()
    ).hexdigest()
    assert manifest["requirements_lock_sha256"] == hashlib.sha256(
        (ROOT / "requirements.lock").read_bytes()
    ).hexdigest()
    assert manifest["smoke_sha256"] == hashlib.sha256(
        (ROOT / "smoke.py").read_bytes()
    ).hexdigest()
    assert manifest["patch_sha256"] == hashlib.sha256(patch.read_bytes()).hexdigest()
    for requirement in (
        "gymnasium==0.28.1",
        "numpy==1.26.4",
        "opencv-python-headless==4.11.0.86",
        "scipy==1.11.4",
        "vizdoom==1.3.0",
    ):
        assert requirement in requirements


def test_coom_runtime_smoke_is_bounded_external_and_nonpromoting() -> None:
    source = (ROOT / "smoke.py").read_text(encoding="utf-8")
    ast.parse(source)

    assert "Sequence.CO8" in source
    assert "STEPS_PER_TASK = 2" in source
    assert '"action": 0' in source
    assert '"external_runtime_executed": True' in source
    assert '"execution_attested": False' in source
    assert '"performance_metrics_computed": False' in source
    assert '"paper_parity_claimed": False' in source
    assert '"scientific_promotion_allowed": False' in source
    assert '"negative_outcome_retained": False' in source
    assert "elapsed_ns_telemetry_only" in source
    assert '_file_sha256(root / "LICENSE.txt")' in source
    assert '_file_sha256(root / "COOM/wrappers/reward.py")' in source
    assert "_source_tree_sha1(root) != SOURCE_TREE" in source
    assert "_validate_receipt(receipt)" in source
    assert "EXPECTED_TRACE_SHA256" in source
    assert "type(reset_info) is not dict or reset_info" in source
    assert "type(info) is not dict or info" in source


def test_coom_receipt_validator_rejects_hostile_provider_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke = _smoke_module()
    manifest = json.loads((ROOT / "qualification-manifest.json").read_bytes())
    monkeypatch.setattr(smoke, "_load_qualification_manifest", lambda: manifest)
    records = []
    for task_index, task_name in enumerate(smoke.TASK_NAMES):
        step = {
            "action": 0,
            "info": {},
            "observation_dtype": "<f8",
            "observation_sha256": "1" * 64,
            "observation_shape": [84, 84, 3],
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
        }
        records.append(
            {
                "task_index": task_index,
                "name": task_name,
                "reset_info": {},
                "reset_observation_dtype": "<f8",
                "reset_observation_sha256": "2" * 64,
                "reset_observation_shape": [84, 84, 3],
                "steps": [copy.deepcopy(step), copy.deepcopy(step)],
            }
        )
    trace = {
        "seed": 1_582_000,
        "sequence": "CO8",
        "steps_per_task": 2,
        "fixed_action": 0,
        "frame_skip": 4,
        "resize": [84, 84],
        "records": records,
    }
    trace_sha256 = hashlib.sha256(
        json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    monkeypatch.setattr(smoke, "EXPECTED_TRACE_SHA256", trace_sha256)
    receipt = {
        "schema": smoke.SCHEMA,
        "qualification_inputs": manifest,
        "source": {
            "repository": "https://github.com/TTomilin/COOM.git",
            "commit": smoke.SOURCE_COMMIT,
            "git_tree": smoke.SOURCE_TREE,
            "archive_sha256": smoke.SOURCE_ARCHIVE_SHA256,
            "license": "MIT",
            "license_sha256": smoke.SOURCE_LICENSE_SHA256,
            "asset_count": 33,
            "asset_bytes": 4_153_440,
            "asset_manifest_sha256": smoke.SOURCE_ASSET_MANIFEST_SHA256,
            "qualification_patch_sha256": smoke.PATCH_SHA256,
            "qualification_patch_scope": "gym RewardWrapper import only",
            "patched_reward_wrapper_sha256": smoke.PATCHED_REWARD_WRAPPER_SHA256,
        },
        "runtime": {
            "python": "3.12.12",
            "python_implementation": "CPython",
            "platform": "linux-test",
            "numpy": "1.26.4",
            "scipy": "1.11.4",
            "gymnasium": "0.28.1",
            "vizdoom": "1.3.0",
            "opencv_python_headless": "4.11.0.86",
        },
        "trace": trace,
        "trace_sha256": trace_sha256,
        "resource_receipt": {
            "task_resets": 8,
            "environment_steps": 16,
            "environment_step_queries": 16,
            "policy_queries": 0,
            "learner_updates": 0,
            "model_queries": 0,
            "elapsed_ns_telemetry_only": 1,
        },
        "claims": {
            "external_runtime_executed": True,
            "execution_attested": False,
            "mechanism_off": True,
            "performance_metrics_computed": False,
            "paper_parity_claimed": False,
            "scientific_promotion_allowed": False,
            "negative_outcome_retained": False,
        },
    }
    smoke._validate_receipt(receipt)

    hostile = copy.deepcopy(receipt)
    hostile["trace"]["records"][0]["steps"][0]["info"] = {"object": object()}
    with pytest.raises(ValueError, match="step payload"):
        smoke._validate_receipt(hostile)
    hostile = copy.deepcopy(receipt)
    hostile["claims"]["execution_attested"] = True
    with pytest.raises(ValueError, match="claims exceed"):
        smoke._validate_receipt(hostile)
    hostile = copy.deepcopy(receipt)
    hostile["resource_receipt"]["environment_steps"] = True
    with pytest.raises(ValueError, match="resource receipt"):
        smoke._validate_receipt(hostile)


def test_coom_retained_receipt_loader_is_bounded_and_fail_closed(tmp_path: Path) -> None:
    smoke = _smoke_module()
    receipt = tmp_path / "receipt.json"
    receipt.write_text('{"schema":"first","schema":"second"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        smoke._load_receipt(receipt)

    receipt.write_text("NaN", encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        smoke._load_receipt(receipt)

    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    receipt.unlink()
    receipt.symlink_to(target)
    with pytest.raises(ValueError, match="regular file"):
        smoke._load_receipt(receipt)
