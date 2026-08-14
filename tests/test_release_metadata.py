"""Release metadata must move as one versioned transaction."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

import alberta_framework

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[1]
_SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$")


def _citation_scalar(field: str) -> str:
    matches: list[str] = re.findall(
        rf"(?m)^{re.escape(field)}:\s*[\"']?([^\"'#\s]+)[\"']?\s*$",
        (_ROOT / "CITATION.cff").read_text(encoding="utf-8"),
    )
    if len(matches) != 1:
        raise AssertionError(f"CITATION.cff must contain exactly one scalar {field}")
    return matches[0]


def test_release_version_carriers_and_lockfile_are_synchronized() -> None:
    project = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    expected = project["project"]["version"]
    assert isinstance(expected, str)
    assert _SEMVER.fullmatch(expected)

    lock = tomllib.loads((_ROOT / "uv.lock").read_text(encoding="utf-8"))
    root_versions = [
        package["version"]
        for package in lock["package"]
        if package.get("name") == "alberta-framework"
    ]

    assert alberta_framework.__version__ == expected
    assert _citation_scalar("version") == expected
    assert root_versions == [expected]
    release_date = _citation_scalar("date-released")
    assert re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", release_date)
    assert f"## [{expected}] - {release_date}" in (_ROOT / "CHANGELOG.md").read_text(
        encoding="utf-8"
    )


def test_release_repository_and_runtime_floors_are_explicit() -> None:
    project = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert project["urls"] == {
        "Homepage": "https://github.com/elizaOS/asi",
        "Repository": "https://github.com/elizaOS/asi",
        "Issues": "https://github.com/elizaOS/asi/issues",
        "Upstream": "https://github.com/lalalune/alberta",
    }
    assert _citation_scalar("repository-code") == project["urls"]["Repository"]
    assert {"jax>=0.7.1", "jaxlib>=0.7.1", "numpy>=1.26"} <= set(project["dependencies"])
    assert project["optional-dependencies"]["gpu"] == ["jax[cuda12]>=0.7.1"]
