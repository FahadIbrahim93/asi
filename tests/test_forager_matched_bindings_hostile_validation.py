"""Hostile input and boundary validation for matched protocol bindings dataclasses."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.forager_matched_protocol import (
    AllowedTransform,
    ConfigurationBinding,
    ExecutionSemantics,
    ForagerMatchedProtocolError,
    SeedContract,
    SourceBinding,
)


def test_allowed_transform_rejects_invalid_inputs() -> None:
    with pytest.raises(
        ForagerMatchedProtocolError,
        match="transform_type must be a non-empty string",
    ):
        AllowedTransform(
            transform_type="",
            target="lr",
            value_type="number",
            value=0.01,
        )

    with pytest.raises(ForagerMatchedProtocolError, match="invalid value_type"):
        AllowedTransform(
            transform_type="override",
            target="lr",
            value_type="invalid_type",  # type: ignore[arg-type]
            value=0.01,
        )


def test_source_binding_rejects_invalid_inputs() -> None:
    valid_sha = "0" * 64
    with pytest.raises(
        ForagerMatchedProtocolError,
        match="provenance_kind must be git_tree or reviewed_snapshot",
    ):
        SourceBinding(
            provenance_kind="invalid_kind",  # type: ignore[arg-type]
            repository="repo",
            base_commit="0" * 40,
            tree_git_sha1=None,
            archive_sha256=valid_sha,
            inventory_sha256=valid_sha,
            snapshot_descriptor_sha256=None,
        )

    with pytest.raises(ForagerMatchedProtocolError):
        SourceBinding(
            provenance_kind="git_tree",
            repository="repo",
            base_commit="0" * 40,
            tree_git_sha1=None,
            archive_sha256="not_a_sha",
            inventory_sha256=valid_sha,
            snapshot_descriptor_sha256=None,
        )


def test_configuration_binding_rejects_invalid_inputs() -> None:
    valid_sha = "0" * 64
    with pytest.raises(ForagerMatchedProtocolError, match="allowed_transforms must be a tuple"):
        ConfigurationBinding(
            original_path="config.json",
            original_sha256=valid_sha,
            derived_sha256=valid_sha,
            allowed_transforms=[],  # type: ignore[arg-type]
        )


def test_seed_contract_rejects_invalid_inputs() -> None:
    valid_sha = "0" * 64
    with pytest.raises(ForagerMatchedProtocolError, match="offset must be an integer"):
        SeedContract(
            transport="direct",
            offset="0",  # type: ignore[arg-type]
            effective_seed_expression="active_seed",
            effective_seed_proof_sha256=valid_sha,
        )


def test_execution_semantics_rejects_invalid_inputs() -> None:
    with pytest.raises(
        ForagerMatchedProtocolError,
        match="rollout_steps must be positive integer or None",
    ):
        ExecutionSemantics(
            rollout_steps=0,
            num_rollouts=10,
            update_semantics="sync",
        )

    with pytest.raises(
        ForagerMatchedProtocolError,
        match="update_semantics must be a non-empty string",
    ):
        ExecutionSemantics(
            rollout_steps=100,
            num_rollouts=10,
            update_semantics="",
        )
