"""Strict JSON contracts for the evaluation evidence loaders.

These checks stay in ``tmp_path`` and never touch pinned ``outputs/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from alberta_framework.evaluation.continual_ia_artifact import load_ia_evidence_artifact
from alberta_framework.evaluation.continual_multiagent_artifact import (
    load_evidence_artifact as load_multiagent_evidence_artifact,
)
from alberta_framework.evaluation.ftl_decision_artifact import load_ftl_decision_artifact
from alberta_framework.evaluation.recurring_feature_artifact import (
    load_recurring_feature_artifact,
)
from alberta_framework.evaluation.scale_robust_feature_artifact import (
    load_evidence_artifact as load_scale_robust_evidence_artifact,
)

pytestmark = pytest.mark.unit

_DUPLICATE_SCHEMA = '{"schema_version":"forged.v0","schema_version":"honest.v1"}'
_DUPLICATE_DIGEST = (
    '{"content_digest":{"sha256":"forged","sha256":"honest"}}'
)
_NONFINITE = '{"value":NaN}'
_CLEAN = '{"schema_version":"honest.v1","content_digest":{"sha256":"honest"}}'
_EVIDENCE_LOADERS = (
    load_ia_evidence_artifact,
    load_recurring_feature_artifact,
    load_ftl_decision_artifact,
    load_multiagent_evidence_artifact,
    load_scale_robust_evidence_artifact,
)
_EVIDENCE_LOADER_IDS = ("continual_ia", "recurring_feature", "ftl", "multiagent", "scale")


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


@pytest.mark.parametrize("loader", _EVIDENCE_LOADERS, ids=_EVIDENCE_LOADER_IDS)
@pytest.mark.parametrize(
    "raw",
    ('{"value":1e999}', '{"nested":{"value":1e999}}'),
    ids=("top_level", "nested"),
)
def test_evidence_loaders_reject_float_exponent_overflow(
    tmp_path: Path,
    loader: Any,
    raw: str,
) -> None:
    path = tmp_path / "overflow.json"
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(ValueError, match="non-finite JSON number"):
        loader(path)


@pytest.mark.parametrize("loader", _EVIDENCE_LOADERS, ids=_EVIDENCE_LOADER_IDS)
def test_evidence_loaders_preserve_finite_json_floats(
    tmp_path: Path,
    loader: Any,
) -> None:
    path = tmp_path / "finite.json"
    path.write_text('{"large":1e200,"small":-1e-200}', encoding="utf-8")

    assert loader(path) == {"large": 1e200, "small": -1e-200}
