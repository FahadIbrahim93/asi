"""Contracts for maintained campaign tools that supersede output-local scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pytest

from alberta_framework.benchmarks.ipmnist_campaign_tools import (
    CONFIRM_ALIGNMENT_ATOL,
    across_seed_spread,
    build_ceiling_summary,
    build_frontier,
    validate_confirm_alignment,
)
from alberta_framework.benchmarks.ipmnist_ceiling import _atomic_publish, _publish_run
from alberta_framework.benchmarks.rule_discovery_summary import (
    CHAMPION,
    SCREEN_ARMS,
    build_legacy_rule_discovery_summary,
    build_rule_discovery_summary,
)

pytestmark = pytest.mark.unit
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _shard(path: Path, *, seed: int, accuracy: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"seed": seed, "per_task_accuracy": [accuracy, accuracy]}),
        encoding="utf-8",
    )


def _ceiling_run(
    root: Path,
    *,
    prefix: str,
    seed: int,
    mean_accuracy: float,
    tasks: int = 1,
) -> None:
    tag = prefix
    (root / f"{prefix}_seed{seed}.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "tag": tag,
                "mean_accuracy": mean_accuracy,
                "per_task_accuracy": [mean_accuracy] * tasks,
            }
        ),
        encoding="utf-8",
    )
    np.save(
        root / f"{tag}_seed{seed}_per_step.npy",
        np.full((tasks, 5000), mean_accuracy, dtype=np.float64),
        allow_pickle=False,
    )


def test_across_seed_spread_uses_sample_estimator() -> None:
    values = [0.7814, 0.7842, 0.7870]
    assert across_seed_spread(values) == pytest.approx(np.std(values, ddof=1))
    assert across_seed_spread(values) > float(np.std(values))
    assert across_seed_spread([]) == 0.0
    assert across_seed_spread([0.5]) == 0.0


def test_frontier_requires_and_uses_exact_paired_seed_sets(tmp_path: Path) -> None:
    screen = tmp_path / "screen"
    confirm = tmp_path / "confirm"
    for seed in (0, 1):
        _shard(screen / f"base_seed{seed}.json", seed=seed, accuracy=0.80)
        _shard(screen / f"candidate_seed{seed}.json", seed=seed, accuracy=0.81)
        _shard(confirm / f"base_seed{seed}.json", seed=seed, accuracy=0.80)
        _shard(confirm / f"candidate_seed{seed}.json", seed=seed, accuracy=0.82)
    frontier = build_frontier(
        screen,
        confirm,
        base="base",
        arms=["candidate"],
        created_unix=0.0,
    )
    row = frontier["results"][0]

    assert row["n_confirm_seeds"] == 2
    assert row["confirm_mean"] == pytest.approx(0.82)
    assert row["confirm_paired_delta_vs_base"] == pytest.approx(0.02)

    assert frontier["schema"] == "asi.ipmnist.frontier.v2"
    assert len(frontier["provenance"]["inputs"]) == 8
    assert len(frontier["provenance"]["sources"]["campaign_tools"]["sha256"]) == 64
    assert "ipmnist_provenance" in frontier["provenance"]["sources"]
    assert {
        item["path"] for item in frontier["provenance"]["environment_specifications"]
    } == {"pyproject.toml", "uv.lock"}


def test_frontier_rejects_mismatched_confirm_seed_sets(tmp_path: Path) -> None:
    screen = tmp_path / "screen"
    confirm = tmp_path / "confirm"
    _shard(screen / "base_seed0.json", seed=0, accuracy=0.80)
    _shard(screen / "candidate_seed0.json", seed=0, accuracy=0.81)
    _shard(confirm / "base_seed0.json", seed=0, accuracy=0.80)
    _shard(confirm / "candidate_seed7.json", seed=7, accuracy=0.99)

    with pytest.raises(ValueError, match="confirm seed sets differ"):
        build_frontier(
            screen,
            confirm,
            base="base",
            arms=["candidate"],
            created_unix=0.0,
        )


def test_frontier_rejects_empty_or_no_overlap_screen_seed_sets(tmp_path: Path) -> None:
    screen = tmp_path / "screen"
    confirm = tmp_path / "confirm"
    _shard(screen / "base_seed0.json", seed=0, accuracy=0.80)
    _shard(screen / "candidate_seed7.json", seed=7, accuracy=0.81)

    with pytest.raises(ValueError, match="screen seed sets differ"):
        build_frontier(screen, confirm, base="base", arms=["candidate"])

    with pytest.raises(ValueError, match="has no seeds"):
        build_frontier(tmp_path / "empty", confirm, base="base", arms=["candidate"])


def test_ceiling_summary_records_sample_spread_from_explicit_paths(tmp_path: Path) -> None:
    ceiling = tmp_path / "ceiling"
    confirm = tmp_path / "confirm"
    ceiling.mkdir()
    confirm.mkdir()
    for seed, accuracy in enumerate((0.78, 0.82)):
        _ceiling_run(
            ceiling,
            prefix="stationary_sigma0_ndecay099",
            seed=seed,
            mean_accuracy=accuracy,
        )

    summary = build_ceiling_summary(ceiling, confirm)
    result = summary["stationary_sigma0_ndecay099"]

    assert result["sample_standard_deviation"] == pytest.approx(
        np.std([0.78, 0.82], ddof=1)
    )
    assert summary["schema"] == "asi.ipmnist.ceiling_summary.v2"
    assert len(summary["provenance"]["inputs"]) == 4
    assert summary["provenance"]["runtime"]["dependencies"]["numpy"] is not None


def test_current_stored_ceiling_reconstructs_with_rounded_confirmation() -> None:
    campaign = _REPO_ROOT / "outputs" / "ipmnist_screening"

    summary = build_ceiling_summary(campaign / "ceiling", campaign / "confirm_full")
    alignment = summary["error_budget"]["confirm_alignment"]

    assert alignment["relative_tolerance"] == 0.0
    assert alignment["absolute_tolerance"] == CONFIRM_ALIGNMENT_ATOL
    assert alignment["max_abs_delta"] == pytest.approx(6.000000007944095e-08)


def test_ceiling_confirm_alignment_rejects_delta_above_frozen_tolerance(
    tmp_path: Path,
) -> None:
    confirm = tmp_path / "confirm"
    _shard(confirm / "sigma0_ndecay099_seed0.json", seed=0, accuracy=0.8)
    run = {"seed": 0, "per_task_accuracy": [0.8, 0.8 + 2 * CONFIRM_ALIGNMENT_ATOL]}

    with pytest.raises(ValueError, match="exceeds atol"):
        validate_confirm_alignment([run], confirm)


def test_ceiling_publication_refuses_to_replace_existing_bytes(tmp_path: Path) -> None:
    per_step = np.ones((1, 5000), dtype=np.uint8)
    per_task = np.asarray([1.0], dtype=np.float64)
    artifact_path = _publish_run(
        tmp_path,
        tag="stationary_sigma0_ndecay099",
        spec_name="test",
        seed=0,
        permutation_mode="identity",
        n_tasks=1,
        per_step=per_step,
        per_task=per_task,
        wall_seconds=1.0,
        provenance={"schema": "test.provenance.v1"},
    )
    before = artifact_path.read_bytes()
    with np.load(artifact_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        assert np.array_equal(archive["per_step"], per_step)
    assert metadata["schema"] == "asi.ipmnist_ceiling.run.v2"
    assert metadata["provenance"]["schema"] == "test.provenance.v1"
    reconstructed = build_ceiling_summary(tmp_path, tmp_path / "confirm")
    assert reconstructed["stationary_sigma0_ndecay099"]["avg_online_mean"] == 1.0

    with pytest.raises(FileExistsError, match="refusing to replace"):
        _publish_run(
            tmp_path,
            tag="stationary_sigma0_ndecay099",
            spec_name="test",
            seed=0,
            permutation_mode="identity",
            n_tasks=1,
            per_step=per_step,
            per_task=per_task,
            wall_seconds=1.0,
            provenance={"schema": "test.provenance.v1"},
        )

    assert artifact_path.read_bytes() == before


def test_atomic_publication_cleans_up_writer_failure(tmp_path: Path) -> None:
    destination = tmp_path / "result.bin"

    def fail_after_write(stream: BinaryIO) -> None:
        stream.write(b"partial")
        raise RuntimeError("injected writer fault")

    with pytest.raises(RuntimeError, match="injected writer fault"):
        _atomic_publish(destination, fail_after_write)

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_atomic_publication_cleans_up_link_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "result.bin"

    def fail_link(source: object, target: object) -> None:
        del source, target
        raise OSError("injected link fault")

    monkeypatch.setattr("alberta_framework.benchmarks.ipmnist_ceiling.os.link", fail_link)
    with pytest.raises(OSError, match="injected link fault"):
        _atomic_publish(destination, lambda stream: stream.write(b"complete"))

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_atomic_publication_rolls_back_directory_sync_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "result.bin"
    real_fsync = os.fsync
    calls = 0

    def fail_second_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected directory sync fault")
        real_fsync(descriptor)

    monkeypatch.setattr("alberta_framework.benchmarks.ipmnist_ceiling.os.fsync", fail_second_fsync)
    with pytest.raises(OSError, match="injected directory sync fault"):
        _atomic_publish(destination, lambda stream: stream.write(b"complete"))

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_rule_discovery_summary_uses_explicit_directories(tmp_path: Path) -> None:
    screen = tmp_path / "screen"
    confirm = tmp_path / "confirm"
    for name in SCREEN_ARMS:
        for seed in (0, 1, 2):
            accuracy = 0.8 if name == CHAMPION else 0.79
            _shard(screen / f"{name}_seed{seed}.json", seed=seed, accuracy=accuracy)

    summary = build_rule_discovery_summary(screen, confirm)

    assert summary["schema"] == "asi.rule_discovery.real_screen.v2"
    assert summary["legacy_compatibility"]["schema"] == (
        "alberta.rule_discovery.real_screen.v1"
    )
    assert summary["screen_60_task"][CHAMPION]["mean"] == pytest.approx(0.8)
    assert summary["paired_vs_champion_60_task"]["disc_r1"]["mean"] == pytest.approx(
        -0.01
    )
    assert summary["provenance"]["schema"] == "asi.ipmnist.analysis_provenance.v1"
    assert len(summary["provenance"]["inputs"]) == 24
    assert "rule_discovery" in summary["provenance"]["sources"]
    assert "ipmnist_provenance" in summary["provenance"]["sources"]


def test_current_rule_discovery_legacy_payload_reconstructs_exactly() -> None:
    campaign = _REPO_ROOT / "outputs" / "ipmnist_screening"
    expected = json.loads(
        (_REPO_ROOT / "outputs" / "rule_discovery" / "real_screen_v1.json").read_text()
    )

    actual = build_legacy_rule_discovery_summary(
        campaign / "shards", campaign / "confirm_full"
    )

    assert actual == expected
