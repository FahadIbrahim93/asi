from __future__ import annotations

from pathlib import Path

WORKFLOW = Path(".github/workflows/reference-life-scorecard-dev.yml")


def test_reference_life_scorecard_is_manual_only() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    trigger_block = text.split("permissions:", maxsplit=1)[0]
    assert "workflow_dispatch:" in trigger_block
    assert "pull_request:" not in trigger_block
    assert "push:" not in trigger_block


def test_reference_life_scorecard_keeps_the_frozen_cross_product() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert (
        "seed: [70000, 70001, 70002, 70003, 70004, 70005, 70006, "
        "70007, 70008, 70009, 70010, 70011]"
    ) in text
    assert "environment: [switching_two_state, riverswim]" in text
    assert (
        "arm: [prototype, prototype_frozen, random, privileged_oracle, "
        "differential_sarsa, sarsa]"
    ) in text
    assert "Run and validate one fresh-process shard" in text
    assert "Run twelve fresh-process shards" not in text
    assert "timeout-minutes: 90" in text
    assert 'test "$count" = 144' in text


def test_reference_life_scorecard_fails_closed_on_source_and_runtime() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "launch_sha:" in text
    assert text.count("if: github.repository == 'SlopDotCash/asi'") == 2
    assert text.count('test "$GITHUB_REF" = "refs/heads/main"') == 2
    assert text.count('test "$GITHUB_SHA" = "$LAUNCH_SHA"') == 2
    assert text.count('test "$WORKFLOW_SHA" = "$LAUNCH_SHA"') == 2
    assert text.count('test "$(git rev-parse HEAD)" = "$LAUNCH_SHA"') == 2
    assert text.count('python-version: "3.12.12"') == 2
    assert text.count('test "$uv_version" = "0.9.24"') == 2
    assert text.count('python_path="$GITHUB_WORKSPACE/.venv/bin/python"') == 2
    assert 'python_path="$(realpath' not in text
    assert '"$PYTHON_PATH" -m alberta_framework.benchmarks.reference_life_scorecard' in text
    assert "import alberta_framework, jax" in text
    assert "retention-days: 90" in text
    assert "retention-days: 30" not in text


def test_reference_life_scorecard_artifacts_and_receipt_bind_the_run() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "reference-life-shard-${{ inputs.launch_sha }}-${{ github.run_id }}-" in text
    assert (
        "reference-life-scorecard-${{ inputs.launch_sha }}-"
        "${{ github.run_id }}-validated"
    ) in text
    assert '"schema": "asi.reference_life_scorecard.github_run.v1"' in text
    for field in (
        "workflow_blob_sha1",
        "workflow_commit",
        "dispatch_ref",
        "repository",
        "uv_lock_sha256",
        "python_executable_sha256",
        "uv_executable_sha256",
        "campaign_inventory_sha256",
        "artifact_sha256",
        "shard_count",
    ):
        assert field in text
    assert "path: scorecard\n" not in text
    assert "scorecard/run-receipt.v1.json" in text
