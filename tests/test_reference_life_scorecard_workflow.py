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

    assert all(f"- {seed}" in text for seed in range(70_000, 70_012))
    assert 'test "$count" = 144' in text
