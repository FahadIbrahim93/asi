"""Strict JSON contracts for the remaining evaluation evidence loaders.

FTL, scale-robust, and multiagent already refuse duplicate object keys and
non-standard numeric constants. IA and recurring used default ``json.loads``
last-wins, so a malformed file could present as the last key and reach the
validator. These checks stay in tmp and never touch pinned ``outputs/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from alberta_framework.evaluation.continual_ia_artifact import load_ia_evidence_artifact
from alberta_framework.evaluation.recurring_feature_artifact import (
    load_recurring_feature_artifact,
)

pytestmark = pytest.mark.unit

_DUPLICATE_SCHEMA = '{"schema_version":"forged.v0","schema_version":"honest.v1"}'
_DUPLICATE_DIGEST = (
    '{"content_digest":{"sha256":"forged","sha256":"honest"}}'
)
_NONFINITE = '{"value":NaN}'
_CLEAN = '{"schema_version":"honest.v1","content_digest":{"sha256":"honest"}}'


@pytest.mark.parametrize(
    "loader",
    [load_ia_evidence_artifact, load_recurring_feature_artifact],
    ids=["continual_ia", "recurring_feature"],
)
@pytest.mark.parametrize(
    "raw",
    [_DUPLICATE_SCHEMA, _DUPLICATE_DIGEST],
    ids=["top_level", "nested"],
)
def test_evidence_loaders_reject_duplicate_object_keys(
    tmp_path: Path,
    loader,
    raw: str,
) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        loader(path)


@pytest.mark.parametrize(
    "loader",
    [load_ia_evidence_artifact, load_recurring_feature_artifact],
    ids=["continual_ia", "recurring_feature"],
)
def test_evidence_loaders_still_reject_nonstandard_numeric_constants(
    tmp_path: Path,
    loader,
) -> None:
    path = tmp_path / "nan.json"
    path.write_text(_NONFINITE, encoding="utf-8")

    with pytest.raises(ValueError, match="non-standard JSON"):
        loader(path)


@pytest.mark.parametrize(
    "loader",
    [load_ia_evidence_artifact, load_recurring_feature_artifact],
    ids=["continual_ia", "recurring_feature"],
)
def test_evidence_loaders_preserve_clean_nested_objects(
    tmp_path: Path,
    loader,
) -> None:
    path = tmp_path / "clean.json"
    path.write_text(_CLEAN, encoding="utf-8")

    assert loader(path) == {
        "schema_version": "honest.v1",
        "content_digest": {"sha256": "honest"},
    }
