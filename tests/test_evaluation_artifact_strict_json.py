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
_NONFINITE = '{"value":NaN}'


@pytest.mark.parametrize(
    "loader",
    [load_ia_evidence_artifact, load_recurring_feature_artifact],
    ids=["continual_ia", "recurring_feature"],
)
def test_evidence_loaders_reject_duplicate_object_keys(
    tmp_path: Path,
    loader,
) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(_DUPLICATE_SCHEMA, encoding="utf-8")

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
