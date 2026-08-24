"""Regression tests for #2134: rule-discovery summary rejects semantic substitution.

Both cases reproduce the exact scenarios from the issue:

1. Arm substitution — a shard whose config_name is sigma0_shiftnorm_d099
   placed under the disc_r1 filename is accepted and misattributed.
2. Stage substitution — 60-task screen shards placed under confirm filenames
   are published as 200-task confirmation measurements.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from alberta_framework.benchmarks.ipmnist_screening import IPMNISTConfig
from alberta_framework.benchmarks.rule_discovery_summary import (
    SCREEN_ARMS,
    DISCOVERY_ARMS,
    CHAMPION,
    build_legacy_rule_discovery_summary,
)

pytestmark = pytest.mark.unit


def _legacy_shard(path: Path, *, config_name: str, seed: int, n_tasks: int,
                  mean_accuracy: float, base_learner: str = "upgd_w") -> None:
    """Write a valid legacy v1 shard (schema, config, curves, environment).

    The shard passes load_shard validation but may be semantically misplaced
    (wrong config_name for the expected arm, or wrong n_tasks for the stage).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    accuracy = [mean_accuracy] * n_tasks
    payload = {
        "schema": "alberta.ipmnist_screening.shard.v1",
        "config_name": config_name,
        "base_learner": base_learner,
        "seed": seed,
        "hyperparameters": {},
        "per_task_accuracy": accuracy,
        "per_task_loss": [0.5] * n_tasks,
        "per_task_plasticity": [0.3] * n_tasks,
        "wall_clock_seconds": 1.0,
        "config": {
            "input_dim": 784,
            "hidden1": 300,
            "hidden2": 150,
            "n_classes": 10,
            "n_tasks": n_tasks,
            "task_length": 5000,
        },
        "environment": {
            "jax": "0.11.0",
            "numpy": "2.5.1",
            "python": "3.12.3",
            "platform": "Linux-7.0.0-28-generic-x86_64-with-glibc2.39",
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_arm_substitution_rejects_misattributed_shard(tmp_path: Path) -> None:
    """A sigma0_shiftnorm_d099 shard placed under disc_r1's filename is rejected.

    Exact repro from the issue: copy the sigma0_shiftnorm_d099 seed-0 shard
    to disc_r1_seed0.json, build the summary, and observe that the sigma0
    payload is accepted under disc_r1 with mean 0.8635.
    """
    screen = tmp_path / "screen"
    confirm = tmp_path / "confirm"
    screen.mkdir(parents=True)
    confirm.mkdir(parents=True)
    # Place sigma0's real payload under disc_r1's expected filename.
    _legacy_shard(
        screen / "disc_r1_seed0.json",
        config_name="sigma0_shiftnorm_d099",
        seed=0,
        n_tasks=60,
        mean_accuracy=0.8635,
    )
    # Also write the other required screen arms + confirm shards so the
    # builder gets past existence checks for the other arms.
    for name in SCREEN_ARMS:
        if name == "disc_r1":
            continue
        _legacy_shard(screen / f"{name}_seed0.json", config_name=name,
                      seed=0, n_tasks=60, mean_accuracy=0.79)
    _legacy_shard(confirm / "disc_r1_pscale_norms_seed0.json",
                  config_name="disc_r1_pscale_norms", seed=0, n_tasks=200,
                  mean_accuracy=0.8)
    _legacy_shard(confirm / f"{CHAMPION}_seed0.json", config_name=CHAMPION,
                  seed=0, n_tasks=200, mean_accuracy=0.8)
    with pytest.raises(ValueError, match="config_name"):
        build_legacy_rule_discovery_summary(screen, confirm, seeds=(0,))


@pytest.mark.parametrize("name", SCREEN_ARMS)
def test_arm_substitution_rejects_every_misattributed_arm(
    tmp_path: Path, name: str
) -> None:
    """Any shard whose config_name != expected arm is rejected for that arm."""
    screen = tmp_path / "screen"
    confirm = tmp_path / "confirm"
    screen.mkdir(parents=True)
    confirm.mkdir(parents=True)
    other = "sigma0_shiftnorm_d099" if name != "sigma0_shiftnorm_d099" else "disc_r1"
    _legacy_shard(
        screen / f"{name}_seed0.json",
        config_name=other,
        seed=0,
        n_tasks=60,
        mean_accuracy=0.8,
    )
    for n in SCREEN_ARMS:
        if n == name:
            continue
        _legacy_shard(screen / f"{n}_seed0.json", config_name=n,
                      seed=0, n_tasks=60, mean_accuracy=0.79)
    _legacy_shard(confirm / "disc_r1_pscale_norms_seed0.json",
                  config_name="disc_r1_pscale_norms", seed=0, n_tasks=200,
                  mean_accuracy=0.8)
    _legacy_shard(confirm / f"{CHAMPION}_seed0.json", config_name=CHAMPION,
                  seed=0, n_tasks=200, mean_accuracy=0.8)
    with pytest.raises(ValueError, match="config_name"):
        build_legacy_rule_discovery_summary(screen, confirm, seeds=(0,))


def test_stage_substitution_rejects_screen_as_confirmation(tmp_path: Path) -> None:
    """60-task screen shards placed under confirm filenames are rejected.

    Exact repro from the issue: copy genuine 60-task v1 screen shards for
    disc_r1_pscale_norms and sigma0_shiftnorm_d099 into the confirmation
    directory; the maintained summary publishes both as confirm_200_task.
    """
    screen = tmp_path / "screen"
    confirm = tmp_path / "confirm"
    screen.mkdir(parents=True)
    confirm.mkdir(parents=True)
    # Legitimate 60-task screen shards for all 8 screen arms.
    for name in SCREEN_ARMS:
        _legacy_shard(screen / f"{name}_seed0.json", config_name=name,
                      seed=0, n_tasks=60, mean_accuracy=0.79)
    # Misplaced: 60-task shards under confirmation filenames.
    for name in ("disc_r1_pscale_norms", CHAMPION):
        _legacy_shard(
            confirm / f"{name}_seed0.json",
            config_name=name,
            seed=0,
            n_tasks=60,
            mean_accuracy=0.8,
        )
    with pytest.raises(ValueError, match="n_tasks"):
        build_legacy_rule_discovery_summary(screen, confirm, seeds=(0,))
