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


def _write_screen_fixtures(tmp_path: Path, *, substitute: str | None = None,
                           substitute_with: str | None = None,
                           substitute_n_tasks: int | None = None,
                           substitute_accuracy: float = 0.8635) -> Path:
    """Write all 8 SCREEN_ARMS shards for seeds 0,1,2.

    If substitute is set, that arm's shard gets the wrong config_name
    (or wrong n_tasks for stage-substitution tests).
    """
    screen = tmp_path / "screen"
    screen.mkdir(parents=True)
    for name in SCREEN_ARMS:
        for seed in (0, 1, 2):
            if name == substitute:
                config_name = substitute_with or name
                n_tasks = substitute_n_tasks if substitute_n_tasks is not None else 60
            else:
                config_name = name
                n_tasks = 60
            _legacy_shard(
                screen / f"{name}_seed{seed}.json",
                config_name=config_name,
                seed=seed,
                n_tasks=n_tasks,
                mean_accuracy=substitute_accuracy if name == substitute else 0.79,
            )
    return screen


def _write_confirm_fixtures(tmp_path: Path, *, screen_substitute: bool = False) -> Path:
    """Write confirm fixtures for disc_r1_pscale_norms + CHAMPION (seeds 0,1,2).

    If screen_substitute=True, write 60-task shards instead of 200-task.
    """
    confirm = tmp_path / "confirm"
    confirm.mkdir(parents=True)
    n_tasks = 60 if screen_substitute else 200
    for name in ("disc_r1_pscale_norms", CHAMPION):
        for seed in (0, 1, 2):
            _legacy_shard(
                confirm / f"{name}_seed{seed}.json",
                config_name=name,
                seed=seed,
                n_tasks=n_tasks,
                mean_accuracy=0.8,
            )
    return confirm


def test_arm_substitution_rejects_misattributed_shard(tmp_path: Path) -> None:
    """A sigma0_shiftnorm_d099 shard placed under disc_r1's filename is rejected.

    Exact repro from the issue: copy the sigma0_shiftnorm_d099 seed-0 shard
    to disc_r1_seed0.json, build the summary, and observe that the sigma0
    payload is accepted under disc_r1 with mean 0.8635.
    """
    screen = _write_screen_fixtures(
        tmp_path,
        substitute="disc_r1",
        substitute_with="sigma0_shiftnorm_d099",
        substitute_accuracy=0.8635,
    )
    confirm = _write_confirm_fixtures(tmp_path)
    with pytest.raises(ValueError, match="config_name"):
        build_legacy_rule_discovery_summary(screen, confirm, seeds=(0, 1, 2))


@pytest.mark.parametrize("name", SCREEN_ARMS)
def test_arm_substitution_rejects_every_misattributed_arm(
    tmp_path: Path, name: str
) -> None:
    """Any shard whose config_name != expected arm is rejected for that arm."""
    screen = _write_screen_fixtures(
        tmp_path,
        substitute=name,
        substitute_with="sigma0_shiftnorm_d099" if name != "sigma0_shiftnorm_d099" else "disc_r1",
    )
    confirm = _write_confirm_fixtures(tmp_path)
    with pytest.raises(ValueError, match="config_name"):
        build_legacy_rule_discovery_summary(screen, confirm, seeds=(0, 1, 2))


def test_stage_substitution_rejects_screen_as_confirmation(tmp_path: Path) -> None:
    """60-task screen shards placed under confirm filenames are rejected.

    Exact repro from the issue: copy genuine 60-task v1 screen shards for
    disc_r1_pscale_norms and sigma0_shiftnorm_d099 into the confirmation
    directory; the maintained summary publishes both as confirm_200_task.
    """
    screen = _write_screen_fixtures(tmp_path)
    confirm = _write_confirm_fixtures(tmp_path, screen_substitute=True)
    with pytest.raises(ValueError, match="n_tasks"):
        build_legacy_rule_discovery_summary(screen, confirm, seeds=(0, 1, 2))
