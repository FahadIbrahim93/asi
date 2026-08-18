"""Hostile input and boundary validation for micro continual benchmark dataclasses."""

from __future__ import annotations

import numpy as np
import pytest

from alberta_framework.benchmarks.micro_continual import (
    MicroStream,
    MicroTaskConfig,
)


def test_micro_task_config_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="MicroTaskConfig.name must be a non-empty string"):
        MicroTaskConfig(
            name="",
            kind="input_permutation",
            role="search",
            input_dim=64,
            n_classes=10,
            n_tasks=8,
            task_length=500,
            hidden1=32,
            hidden2=16,
            crop=False,
        )

    with pytest.raises(ValueError, match="MicroTaskConfig.role must be 'search' or 'holdout'"):
        MicroTaskConfig(
            name="M1",
            kind="input_permutation",
            role="invalid_role",
            input_dim=64,
            n_classes=10,
            n_tasks=8,
            task_length=500,
            hidden1=32,
            hidden2=16,
            crop=False,
        )

    with pytest.raises(ValueError, match="MicroTaskConfig.input_dim must be a positive integer"):
        MicroTaskConfig(
            name="M1",
            kind="input_permutation",
            role="search",
            input_dim=True,
            n_classes=10,
            n_tasks=8,
            task_length=500,
            hidden1=32,
            hidden2=16,
            crop=False,
        )


def test_micro_stream_rejects_invalid_inputs() -> None:
    dummy_arr = np.zeros((1, 1))
    valid_cfg = MicroTaskConfig(
        name="M1",
        kind="input_permutation",
        role="search",
        input_dim=64,
        n_classes=10,
        n_tasks=8,
        task_length=500,
        hidden1=32,
        hidden2=16,
        crop=False,
    )
    with pytest.raises(TypeError, match="MicroStream.xs must be a numpy ndarray"):
        MicroStream(
            xs=None,  # type: ignore[arg-type]
            ys=dummy_arr,
            example_indices=dummy_arr,
            config=valid_cfg,
            seed=0,
        )

    with pytest.raises(TypeError, match="MicroStream.config must be a MicroTaskConfig"):
        MicroStream(
            xs=dummy_arr,
            ys=dummy_arr,
            example_indices=dummy_arr,
            config=None,  # type: ignore[arg-type]
            seed=0,
        )
