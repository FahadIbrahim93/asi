"""Depth ceiling for historical Forager ``result.json`` artifact loads.

``_strict_json_object`` bounds ``result.json`` by byte size
(``_MAX_JSON_BYTES``, 4MiB) but, before this fix, not by nesting depth. A
deeply nested JSON array is small in bytes (10_000 levels of ``[`` is ~20KB)
but RecursionErrors CPython's ``json`` decoder well before the byte-size gate
can reject it, so ``validate_historical_forager_artifact`` — a fail-closed
artifact validator — crashed with an uncaught ``RecursionError`` instead of
cleanly raising ``HistoricalForagerArtifactError`` on a hostile artifact
directory. The protocol ceiling matches ``security._JSON_MAX_DEPTH`` and
``reference_life_checkpoint._MAX_TREE_DEPTH`` (32).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from alberta_framework.benchmarks.historical_forager import (
    _MAX_JSON_DEPTH,
    HistoricalForagerArtifactError,
    _strict_json_object,
    validate_historical_forager_artifact,
)


def _nested_array_bytes(depth: int) -> bytes:
    return ("[" * depth + "]" * depth).encode("utf-8")


def _write_canonical_artifact_file(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)
    os.chmod(path, 0o444)


def test_protocol_ceiling_matches_json_depth_bound() -> None:
    assert _MAX_JSON_DEPTH == 32


def test_first_overflow_nest_is_artifact_error_not_recursion_error(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    _write_canonical_artifact_file(result_path, _nested_array_bytes(_MAX_JSON_DEPTH + 1))
    with pytest.raises(HistoricalForagerArtifactError, match="nesting limit"):
        _strict_json_object(result_path)


def test_origin_hang_class_10000_is_artifact_error_not_recursion_error(tmp_path: Path) -> None:
    """The 10_000-deep nest is ~20KB, far under ``_MAX_JSON_BYTES`` (4MiB)."""
    result_path = tmp_path / "result.json"
    payload = _nested_array_bytes(10_000)
    assert len(payload) < 1024 * 1024
    _write_canonical_artifact_file(result_path, payload)
    with pytest.raises(HistoricalForagerArtifactError, match="nesting limit"):
        _strict_json_object(result_path)


def test_validate_historical_forager_artifact_rejects_deep_result_json_without_crashing(
    tmp_path: Path,
) -> None:
    """The public fail-closed validator must reject, not RecursionError, a hostile artifact."""
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    _write_canonical_artifact_file(
        artifact_dir / "result.json", _nested_array_bytes(_MAX_JSON_DEPTH + 1)
    )
    _write_canonical_artifact_file(artifact_dir / "rewards.npy", b"\x00")
    with pytest.raises(HistoricalForagerArtifactError, match="nesting limit"):
        validate_historical_forager_artifact(artifact_dir)


def test_boundary_depth_is_not_rejected_by_the_depth_scan(tmp_path: Path) -> None:
    """At exactly the ceiling, the pre-check passes; JSON-object-shape errors take over."""
    result_path = tmp_path / "result.json"
    _write_canonical_artifact_file(result_path, _nested_array_bytes(_MAX_JSON_DEPTH))
    with pytest.raises(HistoricalForagerArtifactError, match="must contain an object"):
        _strict_json_object(result_path)
