"""Package import boundaries must not hide internal import failures."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "statement",
    (
        "import alberta_framework",
        "import alberta_framework.core.learners",
        "from alberta_framework.streams import GymnasiumStream",
        "from alberta_framework.evaluation.evidence_manifest import EVIDENCE_SPECS",
    ),
)
def test_package_import_paths_are_cycle_free(statement: str) -> None:
    completed = subprocess.run(
        (sys.executable, "-c", statement),
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr


def test_base_import_does_not_require_research_extra() -> None:
    probe = textwrap.dedent(
        """
        import builtins

        blocked = {"joblib", "matplotlib", "pandas", "sklearn", "tqdm"}
        original_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.split(".", 1)[0] in blocked:
                raise ModuleNotFoundError(f"blocked optional dependency: {name}")
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded_import
        import alberta_framework
        import alberta_framework.utils

        assert alberta_framework.LinearLearner is not None
        assert alberta_framework.utils.run_multi_seed_experiment is not None
        """
    )

    completed = subprocess.run(
        (sys.executable, "-c", probe),
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr


def test_gymnasium_adapter_runtime_annotations_remain_resolvable() -> None:
    probe = textwrap.dedent(
        """
        from typing import get_type_hints

        from alberta_framework.core.learners import LinearLearner
        from alberta_framework.core.types import TimeStep
        from alberta_framework.streams.gymnasium import (
            GymnasiumStream,
            learn_from_trajectory,
        )

        assert get_type_hints(learn_from_trajectory)["learner"] is LinearLearner
        assert get_type_hints(GymnasiumStream.__next__)["return"] is TimeStep
        """
    )

    completed = subprocess.run(
        (sys.executable, "-c", probe),
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "blocked_module",
    (
        "alberta_framework.pipeline",
        "alberta_framework.streams.gymnasium",
        "alberta_framework.utils.statistics",
        "alberta_framework.utils.visualization",
    ),
)
def test_internal_import_errors_are_not_silently_suppressed(blocked_module: str) -> None:
    """A broken shipped module must fail the package import loudly."""
    probe = textwrap.dedent(
        f"""
        import builtins

        original_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == {blocked_module!r}:
                raise ImportError("synthetic internal import failure")
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded_import
        try:
            import alberta_framework
        except ImportError as exc:
            if str(exc) == "synthetic internal import failure":
                raise SystemExit(0)
            raise
        raise SystemExit("internal ImportError was silently suppressed")
        """
    )

    completed = subprocess.run(
        (sys.executable, "-c", probe),
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
