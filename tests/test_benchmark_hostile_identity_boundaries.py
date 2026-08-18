"""Hostile identity regressions for development benchmark host boundaries."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks import foragax_open_screen as screen
from alberta_framework.benchmarks import reference_life_scorecard as scorecard


class _HostileString(str):
    calls = 0

    def __bool__(self) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile string truth hook executed")

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile string comparison hook executed")

    __hash__ = str.__hash__


class _HostileDict(dict[str, object]):
    def items(self):  # type: ignore[no-untyped-def]
        raise AssertionError("hostile mapping iteration hook executed")


class _HostileList(list[object]):
    def __iter__(self):  # type: ignore[no-untyped-def]
        raise AssertionError("hostile sequence iteration hook executed")


def test_open_screen_helpers_reject_hostile_identities_before_hooks() -> None:
    hostile = _HostileString("results")
    _HostileString.calls = 0
    with pytest.raises(screen.ScreenError, match="relative path"):
        screen._normalized_relative_path(hostile, "result_root")
    with pytest.raises(screen.ScreenError, match="object"):
        screen._require_dict(_HostileDict(), "payload")
    with pytest.raises(screen.ScreenError, match="array"):
        screen._require_list(_HostileList(), "records")
    assert _HostileString.calls == 0


def test_scorecard_canonical_json_rejects_hostile_identities_before_hooks() -> None:
    hostile = _HostileString("value")
    _HostileString.calls = 0
    with pytest.raises(ValueError, match="canonical JSON"):
        scorecard.canonical_json_bytes({"field": hostile})
    with pytest.raises(ValueError, match="canonical JSON"):
        scorecard.canonical_json_bytes(_HostileDict({"field": "value"}))
    assert scorecard._is_sha256(_HostileString("a" * 64)) is False
    assert _HostileString.calls == 0
