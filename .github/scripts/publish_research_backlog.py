#!/usr/bin/env python3
"""Idempotently publish the research backlog as GitHub issues.

Dry-run is the default. Writing requires both ``--apply`` and a token with
Issues write permission in ``GITHUB_TOKEN``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "docs" / "research" / "implementation-backlog.json"
MANIFEST_SCHEMA = "asi.research_implementation_backlog.v2"
MAX_MANIFEST_BYTES = 1_048_576
MAX_ISSUES = 100
MAX_TEXT_BYTES = 8_192
_REPOSITORY_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")


def _exact_text(value: object, *, name: str, maximum_bytes: int = MAX_TEXT_BYTES) -> str:
    if type(value) is not str or not value or len(value.encode("utf-8")) > maximum_bytes:
        raise ValueError(f"{name} must be a non-empty bounded exact string")
    return value


def _read_bounded_manifest(path: Path) -> dict[str, Any]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_MANIFEST_BYTES:
            raise ValueError("manifest must be a bounded regular file")
        chunks: list[bytes] = []
        remaining = MAX_MANIFEST_BYTES + 1
        while remaining:
            chunk = os.read(fd, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(fd)
        if len(raw) > MAX_MANIFEST_BYTES:
            raise ValueError("manifest exceeds the byte limit")
        if len(raw) != after.st_size or (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError("manifest changed while it was read")
    finally:
        os.close(fd)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("manifest must contain valid UTF-8 JSON") from exc
    if type(payload) is not dict:
        raise ValueError("manifest must be an exact JSON object")
    return payload


def _validate_manifest(path: Path, repository: str) -> dict[str, Any]:
    if type(repository) is not str or _REPOSITORY_RE.fullmatch(repository) is None:
        raise ValueError("repository must be an owner/name identifier")
    payload = _read_bounded_manifest(path)
    if set(payload) != {
        "schema",
        "repository",
        "source",
        "published",
        "status_as_of",
        "issues",
    }:
        raise ValueError("manifest fields do not match the v2 schema")
    if payload["schema"] != MANIFEST_SCHEMA or payload["repository"] != repository:
        raise ValueError("manifest schema/repository does not match this invocation")
    _exact_text(payload["source"], name="source")
    _exact_text(payload["status_as_of"], name="status_as_of", maximum_bytes=32)
    published = payload["published"]
    if type(published) is not dict or set(published) != {"date", "first_issue", "last_issue"}:
        raise ValueError("published must match its exact schema")
    _exact_text(published["date"], name="published.date", maximum_bytes=32)
    first = published["first_issue"]
    last = published["last_issue"]
    if type(first) is not int or type(last) is not int or first <= 0 or last < first:
        raise ValueError("published issue range is invalid")
    issues = payload["issues"]
    if type(issues) is not list or not issues or len(issues) > MAX_ISSUES:
        raise ValueError("issues must be a non-empty bounded exact list")
    numbers: list[int] = []
    titles: list[str] = []
    allowed_states = {"open", "closed"}
    for index, issue in enumerate(issues):
        if type(issue) is not dict or set(issue) != {
            "issue",
            "issue_state",
            "implementation_status",
            "title",
            "scope",
            "references",
        }:
            raise ValueError(f"issues[{index}] fields do not match the v2 schema")
        number = issue["issue"]
        if type(number) is not int or number <= 0:
            raise ValueError(f"issues[{index}].issue must be a positive exact integer")
        if issue["issue_state"] not in allowed_states:
            raise ValueError(f"issues[{index}].issue_state is invalid")
        _exact_text(issue["implementation_status"], name=f"issues[{index}].implementation_status")
        title = _exact_text(issue["title"], name=f"issues[{index}].title")
        _exact_text(issue["scope"], name=f"issues[{index}].scope")
        references = issue["references"]
        if type(references) is not list or len(references) > 32:
            raise ValueError(f"issues[{index}].references must be a bounded exact list")
        normalized_references = [
            _exact_text(value, name=f"issues[{index}].references") for value in references
        ]
        if any(not value.startswith("https://") for value in normalized_references):
            raise ValueError(f"issues[{index}].references must use HTTPS")
        if len(normalized_references) != len(set(normalized_references)):
            raise ValueError(f"issues[{index}].references contains duplicates")
        numbers.append(number)
        titles.append(title)
    if numbers != list(range(first, last + 1)):
        raise ValueError("issue numbers must exactly cover the published range in order")
    if len(titles) != len(set(titles)):
        raise ValueError("issue titles must be unique")
    return payload


def _request(url: str, token: str, *, data: bytes | None = None) -> Any:
    method = "POST" if data is not None else "GET"
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "asi-research-backlog-publisher",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API returned {exc.code}: {detail}") from exc


def _issue_exists(repository: str, title: str, token: str) -> bool:
    query = urllib.parse.urlencode(
        {"q": f'repo:{repository} is:issue in:title "{title}"', "per_page": 100}
    )
    payload = _request(f"https://api.github.com/search/issues?{query}", token)
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise RuntimeError("GitHub issue search did not return an item list")
    return any(
        isinstance(item, dict) and item.get("title") == title for item in payload["items"]
    )


def _body(issue: dict[str, Any], source: str) -> str:
    references = issue["references"]
    lines = [
        "### Objective",
        "",
        issue["scope"],
        "",
        "### Acceptance criteria",
        "",
        "- Pin the paper/code revision and record protocol differences before a long run.",
        "- Add failing-test-first unit/parity coverage and a mechanism-off reduction.",
        "- Match seeds, updates, observations, and allowed boundary/task information.",
        "- Report persistent bytes, environment/data steps, model queries, and timing telemetry.",
        "- Keep development results nonpromoting and retain negative outcomes.",
        "",
        f"Backlog source: `{source}`",
    ]
    if references:
        lines.extend(["", "### References", ""])
        lines.extend(f"- {reference}" for reference in references)
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repo", default="elizaOS/asi")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        manifest = _validate_manifest(args.manifest, args.repo)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    issues = manifest["issues"]
    if not args.apply:
        for issue in issues:
            print(f"CREATE {issue['title']}")
        print(f"dry-run: {len(issues)} issue(s); pass --apply to write", file=sys.stderr)
        return 0
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        parser.error("--apply requires GITHUB_TOKEN with Issues write permission")
    created = 0
    skipped = 0
    for issue in issues:
        title = issue["title"]
        if _issue_exists(args.repo, title, token):
            print(f"SKIP {title}")
            skipped += 1
            continue
        data = json.dumps(
            {"title": title, "body": _body(issue, manifest["source"])},
            ensure_ascii=True,
        ).encode("utf-8")
        result = _request(f"https://api.github.com/repos/{args.repo}/issues", token, data=data)
        print(f"CREATE {result['html_url']}")
        created += 1
    print(f"created={created} skipped={skipped}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
