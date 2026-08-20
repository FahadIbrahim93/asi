from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "research" / "implementation-backlog.json"
PUBLISHER = ROOT / ".github" / "scripts" / "publish_research_backlog.py"


def _publisher() -> ModuleType:
    spec = importlib.util.spec_from_file_location("publish_research_backlog", PUBLISHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_research_backlog_is_unique_and_issue_ready() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert payload["schema"] == "asi.research_implementation_backlog.v2"
    assert payload["repository"] == "elizaOS/asi"
    assert payload["published"] == {
        "date": "2026-08-17",
        "first_issue": 1559,
        "last_issue": 1586,
    }
    issues = payload["issues"]
    assert len(issues) == 28
    assert [issue["issue"] for issue in issues] == list(range(1559, 1587))
    assert {issue["issue_state"] for issue in issues} == {"open", "closed"}
    assert {issue["issue"] for issue in issues if issue["issue_state"] == "closed"} == {
        1559,
        1572,
    }
    titles = [issue["title"] for issue in issues]
    assert len(titles) == len(set(titles))
    for issue in issues:
        assert issue["scope"].endswith(".")
        assert issue["implementation_status"]
        assert all(reference.startswith("https://") for reference in issue["references"])


def test_publisher_validates_the_complete_manifest_before_network() -> None:
    publisher = _publisher()
    payload = publisher._validate_manifest(MANIFEST, "elizaOS/asi")
    assert len(payload["issues"]) == 28


def test_publisher_rejects_symlink_and_oversized_manifest(tmp_path: Path) -> None:
    publisher = _publisher()
    link = tmp_path / "manifest-link.json"
    link.symlink_to(MANIFEST)
    with pytest.raises(OSError):
        publisher._validate_manifest(link, "elizaOS/asi")
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * (publisher.MAX_MANIFEST_BYTES + 1))
    with pytest.raises(ValueError, match="bounded regular file"):
        publisher._validate_manifest(oversized, "elizaOS/asi")


@pytest.mark.parametrize(
    "mutation",
    (
        lambda payload: payload["issues"].append(payload["issues"][0]),
        lambda payload: payload["issues"][1].update(issue=1559),
        lambda payload: payload["issues"][0].update(title=payload["issues"][1]["title"]),
        lambda payload: payload["issues"][0]["references"].append("http://example.invalid"),
        lambda payload: payload.update(repository="other/repository"),
    ),
)
def test_publisher_rejects_malformed_manifest_before_api_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[dict[str, object]], object],
) -> None:
    publisher = _publisher()
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mutation(payload)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    calls = 0

    def forbidden_request(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("network must not be reached")

    monkeypatch.setattr(publisher, "_request", forbidden_request)
    with pytest.raises(ValueError):
        publisher._validate_manifest(path, "elizaOS/asi")
    assert calls == 0
