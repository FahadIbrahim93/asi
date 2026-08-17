"""Timer must not report a failed block as completed."""

from __future__ import annotations

import pytest

from alberta_framework.utils.timing import Timer

pytestmark = pytest.mark.unit


def test_timer_reports_completed_only_on_success() -> None:
    messages: list[str] = []
    with Timer("Training", print_fn=messages.append):
        pass

    assert len(messages) == 1
    assert messages[0].startswith("Training completed in ")
    assert "failed after" not in messages[0]


def test_timer_reports_failed_after_on_exception() -> None:
    messages: list[str] = []
    with pytest.raises(RuntimeError, match="boom"):
        with Timer("Training", print_fn=messages.append):
            raise RuntimeError("boom")

    assert len(messages) == 1
    assert messages[0].startswith("Training failed after ")
    assert "completed in" not in messages[0]


def test_timer_still_records_duration_when_the_block_raises() -> None:
    with pytest.raises(ValueError):
        with Timer("Experiment", verbose=False) as timer:
            raise ValueError("no result")

    assert timer.duration > 0.0
    assert timer.end_time >= timer.start_time
