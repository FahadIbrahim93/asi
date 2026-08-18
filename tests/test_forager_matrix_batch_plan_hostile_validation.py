"""Hostile input and boundary validation for matrix batch plan and source snapshot dataclasses."""

from __future__ import annotations

import hashlib

import pytest

from alberta_framework.benchmarks.forager_matrix import (
    ForagerMatrixError,
    _BatchPlan,
    _SourceSnapshot,
)


def test_batch_plan_rejects_invalid_inputs() -> None:
    with pytest.raises(ForagerMatrixError, match="variant_id must be a non-empty string"):
        _BatchPlan(
            variant_id="",
            batch_index=0,
            seeds=(0, 1),
        )

    with pytest.raises(ForagerMatrixError, match="batch_index must be a non-negative integer"):
        _BatchPlan(
            variant_id="var1",
            batch_index=-1,
            seeds=(0, 1),
        )

    with pytest.raises(ForagerMatrixError, match="seeds item must be a non-negative integer"):
        _BatchPlan(
            variant_id="var1",
            batch_index=0,
            seeds=(True,),
        )


def test_source_snapshot_rejects_invalid_inputs() -> None:
    payload = b"test payload"
    digest = hashlib.sha256(payload).hexdigest()
    valid_sha = "0" * 64

    with pytest.raises(
        ForagerMatrixError,
        match="archive_sha256 does not match archive_bytes digest",
    ):
        _SourceSnapshot(
            archive_bytes=payload,
            archive_sha256=valid_sha,
            tree_sha256=valid_sha,
            inventory_sha256=valid_sha,
            inventory={},
        )

    with pytest.raises(
        ForagerMatrixError,
        match="tree_sha256 must be a 64-character lowercase hex string",
    ):
        _SourceSnapshot(
            archive_bytes=payload,
            archive_sha256=digest,
            tree_sha256="invalid",
            inventory_sha256=valid_sha,
            inventory={},
        )
