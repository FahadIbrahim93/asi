"""Hostile input and boundary validation for micro continual benchmark dataclasses."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from alberta_framework.benchmarks.micro_continual import (
    MICRO_ARM_REGISTRY,
    MicroRunResult,
    MicroStream,
    MicroStreamConfig,
    MicroTaskConfig,
    load_micro_shard,
    micro_shard_payload,
)


def test_micro_shard_rejects_unhashable_arm_name_as_validation_error(tmp_path: Path) -> None:
    path = tmp_path / "shard.json"
    config = MicroStreamConfig(n_regimes=1, regime_length=1, dim=10)
    arm = next(iter(MICRO_ARM_REGISTRY.values()))
    result = MicroRunResult(
        family=config.family,
        arm_name=arm.name,
        mechanism=arm.mechanism,
        hyperparameters=arm.hyperparameters,
        seed=0,
        hidden1=1,
        hidden2=1,
        stream_config=config,
        per_regime_accuracy=np.asarray([0.5]),
        per_regime_loss=np.asarray([0.5]),
        per_regime_plasticity=np.asarray([0.5]),
        overall_accuracy=0.5,
        wall_clock_seconds=0.1,
    )
    payload = micro_shard_payload(result)
    payload["arm_name"] = []
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="arm_name must be an exact string"):
        load_micro_shard(path)


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
