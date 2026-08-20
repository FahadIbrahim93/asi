from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

from alberta_framework.benchmarks.external_qualification import qualification_plan
from alberta_framework.benchmarks.plasticity_diagnostics import OFFICIAL_CODE_COMMIT

pytestmark = pytest.mark.unit

ROOT = Path("external_runtimes/loss_of_plasticity_mnist")


class _HookMapping(dict[object, object]):
    calls = 0

    def keys(self):  # type: ignore[no-untyped-def]
        type(self).calls += 1
        raise AssertionError("hostile mapping hook ran")


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "lop_runtime_verifier_test", ROOT / "verify_runtime.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _plan() -> dict[str, object]:
    value = json.loads((ROOT / "qualification-plan.json").read_bytes())
    assert type(value) is dict
    return value


def test_plan_binds_official_source_and_every_runtime_input() -> None:
    plan = _plan()
    authority = plan["authority"]
    inputs = plan["qualification_inputs"]
    assert type(authority) is dict and type(inputs) is dict
    assert authority["commit"] == OFFICIAL_CODE_COMMIT
    assert authority["commit"] == qualification_plan(1583).code_revisions[0].commit
    assert authority["repository"] == qualification_plan(1583).code_revisions[0].repository
    files = {
        "dockerfile_sha256": "Dockerfile",
        "requirements_in_sha256": "requirements.in",
        "requirements_lock_sha256": "requirements.lock",
        "fetch_source_sha256": "fetch_source.py",
        "verify_runtime_sha256": "verify_runtime.py",
    }
    for field, relative in files.items():
        assert inputs[field] == hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_runtime_is_hash_locked_cpu_only_and_never_executes_a_workload() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    lock = (ROOT / "requirements.lock").read_text(encoding="utf-8")
    plan = _plan()
    runtime = plan["runtime"]
    diagnostic = plan["prospective_diagnostic"]
    claims = plan["claims"]
    assert type(runtime) is dict and type(diagnostic) is dict and type(claims) is dict
    assert runtime["python"] == "3.8.18"
    assert runtime["package_versions"]["torch"] == "2.1.0+cpu"
    assert runtime["package_versions"]["scipy"] == "1.10.1"
    assert "SciPy 1.11.2" in runtime["compatibility_deviations"][0]
    assert "@sha256:e796941013b" in dockerfile
    assert "--require-hashes" in dockerfile
    assert "torch-2.1.0%2Bcpu-cp38-cp38-linux_x86_64.whl" in lock
    assert "load_mnist.py" not in dockerfile
    assert "online_expr.py" not in dockerfile
    assert diagnostic["labels_permuted"] is False
    assert diagnostic["workload_executed"] is False
    assert diagnostic["dataset_in_image"] is False
    assert claims == {
        "runtime_build_only": True,
        "external_workload_executed": False,
        "execution_attested": False,
        "negative_outcome_retained": False,
        "paper_parity_claimed": False,
        "performance_metrics_computed": False,
        "scientific_promotion_allowed": False,
        "external_execution_authorized": False,
    }
    assert len(plan["blockers"]) == 9  # type: ignore[arg-type]


def test_verifier_accepts_exact_plan_and_rejects_claim_or_source_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = _module()
    for source, destination in (
        ("Dockerfile", "Dockerfile.source"),
        ("requirements.in", "requirements.in"),
        ("requirements.lock", "requirements.lock"),
        ("fetch_source.py", "fetch_source.py"),
        ("verify_runtime.py", "verify_runtime.py"),
    ):
        (tmp_path / destination).write_bytes((ROOT / source).read_bytes())
    monkeypatch.setattr(verifier, "QUALIFICATION_ROOT", tmp_path)
    plan = _plan()
    verifier._preflight(plan)
    verifier._validate_plan(plan)

    promoted = copy.deepcopy(plan)
    promoted["claims"]["external_execution_authorized"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="claims exceed"):
        verifier._validate_plan(promoted)
    forged = copy.deepcopy(plan)
    forged["authority"]["source_archive_sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(ValueError, match="source authority"):
        verifier._validate_plan(forged)

    weakened_invocation = copy.deepcopy(plan)
    weakened_invocation["runtime"]["future_invocation_requirements"][1] = (  # type: ignore[index]
        "network preferred"
    )
    with pytest.raises(ValueError, match="future invocation requirements"):
        verifier._validate_plan(weakened_invocation)

    removed_blocker = copy.deepcopy(plan)
    cast("list[object]", removed_blocker["blockers"]).pop()
    with pytest.raises(ValueError, match="all nine exact blockers"):
        verifier._validate_plan(removed_blocker)

    rewritten_blocker = copy.deepcopy(plan)
    rewritten_blocker["blockers"][0] = "authorization implied"  # type: ignore[index]
    with pytest.raises(ValueError, match="all nine exact blockers"):
        verifier._validate_plan(rewritten_blocker)


def test_verifier_rejects_hostile_or_unbounded_plan_before_hooks() -> None:
    verifier = _module()
    hostile = _HookMapping({"schema": "x"})
    _HookMapping.calls = 0
    with pytest.raises(ValueError, match="exact JSON"):
        verifier._preflight(hostile)
    assert _HookMapping.calls == 0

    deep: object = 0
    for _ in range(34):
        deep = [deep]
    with pytest.raises(ValueError, match="depth"):
        verifier._preflight(deep)
    with pytest.raises(ValueError, match="item limit"):
        verifier._preflight([0] * 2049)


def test_runtime_scripts_parse_as_python38_and_fetch_is_bounded() -> None:
    for name in ("fetch_source.py", "verify_runtime.py"):
        source = (ROOT / name).read_text(encoding="utf-8")
        ast.parse(source, filename=name, feature_version=(3, 8))
    fetch = (ROOT / "fetch_source.py").read_text(encoding="utf-8")
    assert "_MAX_ARCHIVE_BYTES = 16 * 1024 * 1024" in fetch
    assert "_MAX_EXPANDED_BYTES = 64 * 1024 * 1024" in fetch
    assert "member.isdir() or member.isreg()" in fetch
    assert "SOURCE_ARCHIVE_SHA256" in fetch
