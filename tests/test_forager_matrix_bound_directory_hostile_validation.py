"""Hostile input and boundary validation for bound directory dataclasses."""

from __future__ import annotations

from pathlib import Path

import pytest

from alberta_framework.benchmarks.forager_matrix import (
    ForagerMatrixStateError,
    _BoundDirectory,
    _BoundPathComponent,
)


def test_bound_path_component_rejects_invalid_inputs() -> None:
    with pytest.raises(ForagerMatrixStateError, match="name must be a non-empty string"):
        _BoundPathComponent(
            parent_descriptor=3,
            name="",
            child_descriptor=4,
            device=1,
            inode=2,
        )

    with pytest.raises(ForagerMatrixStateError, match="parent_descriptor must be an integer"):
        _BoundPathComponent(
            parent_descriptor=True,
            name="subdir",
            child_descriptor=4,
            device=1,
            inode=2,
        )


def test_bound_directory_rejects_invalid_inputs() -> None:
    comp = _BoundPathComponent(
        parent_descriptor=3,
        name="subdir",
        child_descriptor=4,
        device=1,
        inode=2,
    )

    with pytest.raises(ForagerMatrixStateError, match="path must be a Path"):
        _BoundDirectory(
            path="/not/a/path/object",  # type: ignore[arg-type]
            root_descriptor=5,
            bindings=(comp,),
            device=1,
            inode=2,
        )

    with pytest.raises(
        ForagerMatrixStateError,
        match=r"lock_identity must be None or a tuple of two integers",
    ):
        _BoundDirectory(
            path=Path("/tmp"),
            root_descriptor=5,
            bindings=(comp,),
            device=1,
            inode=2,
            lock_identity=(1, 2, 3),  # type: ignore[arg-type]
        )
