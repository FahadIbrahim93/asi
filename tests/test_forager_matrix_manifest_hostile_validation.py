"""Hostile input and boundary validation for ForagerMatrixManifest."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from alberta_framework.benchmarks import forager_matrix as matrix
from alberta_framework.benchmarks.forager_matrix import (
    ForagerMatrixManifest,
    ForagerMatrixStateError,
    ForagerTuningRule,
)


class _HostileString(str):
    calls = 0

    def __bool__(self) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile string truthiness must not execute")


class _ExplodingPattern:
    calls = 0

    def fullmatch(self, _value: str) -> None:
        type(self).calls += 1
        raise AssertionError("pattern matching must follow exact-type validation")


@pytest.fixture
def dummy_rule() -> ForagerTuningRule:
    return ForagerTuningRule(
        metric="mean_reward",
        direction="maximize",
        statistic="mean",
        confidence=0.95,
        bootstrap_resamples=100,
        bootstrap_seed=0,
    )


def test_tuning_rule_accepts_parser_supported_conservative_statistic() -> None:
    rule = ForagerTuningRule(
        metric="mean_reward",
        direction="maximize",
        statistic="conservative_ci_endpoint",
        confidence=0.95,
        bootstrap_resamples=100,
        bootstrap_seed=0,
    )
    assert rule.statistic == "conservative_ci_endpoint"


def test_forager_matrix_manifest_rejects_invalid_inputs(
    dummy_rule: ForagerTuningRule,
) -> None:
    with pytest.raises(ValueError, match="schema_version must be a non-empty string"):
        ForagerMatrixManifest(
            schema_version="",
            preset="field_of_view",
            stage="evaluation",
            steps=100,
            seeds=(0,),
            jax_chunk_size=1,
            seed_batch_size=1,
            mode="strict",
            source_execution_mode="live_tree_unsealed",
            metric_evidence_mode="scalar_summary_unsealed",
            selection_rule=dummy_rule,
            variants={},
        )

    with pytest.raises(ValueError, match="steps must be a positive integer"):
        ForagerMatrixManifest(
            schema_version="2.3",
            preset="field_of_view",
            stage="evaluation",
            steps=0,
            seeds=(0,),
            jax_chunk_size=1,
            seed_batch_size=1,
            mode="strict",
            source_execution_mode="live_tree_unsealed",
            metric_evidence_mode="scalar_summary_unsealed",
            selection_rule=dummy_rule,
            variants={},
        )

    with pytest.raises(TypeError, match="selection_rule must be a ForagerTuningRule"):
        ForagerMatrixManifest(
            schema_version="2.3",
            preset="field_of_view",
            stage="evaluation",
            steps=100,
            seeds=(0,),
            jax_chunk_size=1,
            seed_batch_size=1,
            mode="strict",
            source_execution_mode="live_tree_unsealed",
            metric_evidence_mode="scalar_summary_unsealed",
            selection_rule=None,  # type: ignore[arg-type]
            variants={},
        )


def test_state_payload_identity_gates_reject_string_subclasses_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hostile = _HostileString("0" * 64)
    pattern = _ExplodingPattern()
    monkeypatch.setattr(matrix, "_SHA256", pattern)

    _HostileString.calls = 0
    _ExplodingPattern.calls = 0
    with pytest.raises(ForagerMatrixStateError, match="payload_sha256"):
        matrix._verify_hashed_payload(
            {"payload_sha256": hostile},
            description="hostile payload",
        )
    assert _HostileString.calls == _ExplodingPattern.calls == 0

    with pytest.raises(ForagerMatrixStateError, match="UTC timestamp"):
        matrix._validate_utc_timestamp(hostile, "hostile timestamp")
    assert _HostileString.calls == _ExplodingPattern.calls == 0

    archive_bytes = b"x"
    inventory = {
        "schema_version": "1.0",
        "tree_hash_scheme": matrix.SOURCE_TREE_HASH_SCHEME,
        "files": [{"path": "alberta_framework/x.py", "size": 0, "sha256": hostile}],
        "tree_sha256": "tree",
    }
    tree_sha256 = matrix._json_sha256(
        {"tree_hash_scheme": matrix.SOURCE_TREE_HASH_SCHEME, "files": inventory["files"]}
    )
    inventory["tree_sha256"] = tree_sha256
    metadata: dict[str, Any] = {
        "path": matrix.SOURCE_SNAPSHOT_FILENAME,
        "archive_format": matrix.SOURCE_ARCHIVE_FORMAT,
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "archive_size": len(archive_bytes),
        "tree_sha256": tree_sha256,
        "inventory_sha256": hashlib.sha256(
            matrix._canonical_json_bytes(inventory) + b"\n"
        ).hexdigest(),
        "inventory": inventory,
        "source_execution_mode": matrix.SNAPSHOT_SOURCE_EXECUTION_MODE,
    }
    _HostileString.calls = 0
    _ExplodingPattern.calls = 0
    with pytest.raises(ForagerMatrixStateError, match="inventory digest"):
        matrix._validate_source_snapshot_bytes(
            archive_bytes,
            metadata,
            description="hostile snapshot",
        )
    assert _HostileString.calls == _ExplodingPattern.calls == 0
