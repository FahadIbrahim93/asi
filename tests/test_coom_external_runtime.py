from __future__ import annotations

import ast
import hashlib
from pathlib import Path

ROOT = Path("external_runtimes/coom")


def test_coom_runtime_is_source_dependency_and_base_image_pinned() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements.lock").read_text(encoding="utf-8")
    patch = ROOT / "coom-gymnasium.patch"

    assert "python:3.12.12-slim-bookworm@sha256:" in dockerfile
    assert "7929801176c6e2e036c7c1c7dd6ce9b84a9d1f3e" in dockerfile
    assert "a4736e9916468482d75831d53a12a8601c4da91cd40b9b24d313522034a15661" in dockerfile
    assert hashlib.sha256(patch.read_bytes()).hexdigest() in dockerfile
    assert "--require-hashes" in dockerfile
    assert "apt-get" not in dockerfile
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
    assert "elapsed_ns_telemetry_only" in source
