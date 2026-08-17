#!/usr/bin/env python3
"""Build deterministic, nonempty pytest file shards from collected node IDs."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path


def selected_file_counts(nodeids: Iterable[str]) -> Counter[str]:
    """Count selected test nodes by repository-relative test file."""
    counts: Counter[str] = Counter()
    for raw in nodeids:
        nodeid = raw.strip()
        if nodeid.startswith("tests/") and "::" in nodeid:
            counts[nodeid.split("::", 1)[0]] += 1
    if not counts:
        raise ValueError("pytest selected no test nodes")
    return counts


def balanced_file_shards(
    counts: Mapping[str, int],
    shard_limit: int,
) -> tuple[tuple[tuple[str, ...], int], ...]:
    """Assign whole files to deterministic least-loaded shards."""
    if shard_limit < 1:
        raise ValueError("shard limit must be positive")
    if not counts:
        raise ValueError("cannot shard an empty file selection")
    if any(not path or count < 1 for path, count in counts.items()):
        raise ValueError("test file counts must use nonempty paths and positive counts")

    shard_count = min(shard_limit, len(counts))
    buckets: list[list[str]] = [[] for _ in range(shard_count)]
    totals = [0 for _ in range(shard_count)]
    for path, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        index = min(
            range(shard_count),
            key=lambda candidate: (
                totals[candidate],
                len(buckets[candidate]),
                candidate,
            ),
        )
        buckets[index].append(path)
        totals[index] += count

    return tuple(
        (tuple(sorted(files)), totals[index])
        for index, files in enumerate(buckets)
        if files
    )


def shard_matrix(
    shards: Sequence[tuple[Sequence[str], int]],
    python_versions: Sequence[str],
) -> dict[str, list[dict[str, object]]]:
    """Expand one file plan over the requested Python runtimes."""
    if not python_versions or any(not version for version in python_versions):
        raise ValueError("at least one nonempty Python version is required")
    include: list[dict[str, object]] = []
    for version in python_versions:
        for index, (files, node_count) in enumerate(shards, start=1):
            if not files or node_count < 1:
                raise ValueError("shards must be nonempty and have positive node counts")
            include.append(
                {
                    "python": version,
                    "shard": index,
                    "files_json": json.dumps(list(files), separators=(",", ":")),
                    "node_count": node_count,
                }
            )
    return {"include": include}


def _append_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodeids", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--python-version", action="append", required=True)
    parser.add_argument("--label", required=True)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    counts = selected_file_counts(args.nodeids.read_text(encoding="utf-8").splitlines())
    shards = balanced_file_shards(counts, args.shards)
    matrix = json.dumps(
        shard_matrix(shards, args.python_version),
        separators=(",", ":"),
    )

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        _append_line(Path(github_output), f"matrix={matrix}")
    else:
        print(matrix)

    github_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if github_summary:
        _append_line(
            Path(github_summary),
            f"Selected {sum(counts.values())} {args.label} nodes in "
            f"{len(counts)} files across {len(shards)} shards.",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
