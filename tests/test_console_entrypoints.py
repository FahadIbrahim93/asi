"""Packaging-safe lazy boundaries for hard-link-sensitive console commands."""

from __future__ import annotations

import importlib
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[1]
_LAZY_SCRIPTS = {
    "alberta-forager-benchmark": (
        "alberta_framework.console_entrypoints:forager_benchmark_main"
    ),
    "alberta-historical-forager": (
        "alberta_framework.console_entrypoints:historical_forager_main"
    ),
    "alberta-foragax-oci": (
        "alberta_framework.console_entrypoints:official_foragax_oci_main"
    ),
}


def test_hard_link_sensitive_scripts_use_lazy_entrypoints() -> None:
    project = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    scripts = project["project"]["scripts"]

    assert {name: scripts[name] for name in _LAZY_SCRIPTS} == _LAZY_SCRIPTS


def test_importing_lazy_entrypoints_does_not_import_scientific_implementations() -> None:
    probe = textwrap.dedent(
        """
        import sys

        import alberta_framework.console_entrypoints

        forbidden = {
            "alberta_framework.forager_cli",
            "alberta_framework.benchmarks.official_foragax",
            "alberta_framework.benchmarks.official_foragax_oci",
        }
        imported = sorted(forbidden.intersection(sys.modules))
        if imported:
            raise SystemExit(f"eager scientific imports: {imported}")
        """
    )

    completed = subprocess.run(
        (sys.executable, "-c", probe),
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("wrapper_name", "module_name", "target_name"),
    (
        (
            "forager_benchmark_main",
            "alberta_framework.forager_cli",
            "main",
        ),
        (
            "historical_forager_main",
            "alberta_framework.forager_cli",
            "historical_main",
        ),
        (
            "official_foragax_oci_main",
            "alberta_framework.benchmarks.official_foragax_oci",
            "main",
        ),
    ),
)
def test_lazy_entrypoints_delegate_to_original_commands(
    monkeypatch: pytest.MonkeyPatch,
    wrapper_name: str,
    module_name: str,
    target_name: str,
) -> None:
    entrypoints = importlib.import_module("alberta_framework.console_entrypoints")
    target_module = ModuleType(module_name)
    calls = 0

    def target() -> int:
        nonlocal calls
        calls += 1
        return 37

    setattr(target_module, target_name, target)
    parent_name, child_name = module_name.rsplit(".", 1)
    parent = importlib.import_module(parent_name)
    monkeypatch.setitem(sys.modules, module_name, target_module)
    monkeypatch.setattr(parent, child_name, target_module, raising=False)

    assert getattr(entrypoints, wrapper_name)() == 37
    assert calls == 1
