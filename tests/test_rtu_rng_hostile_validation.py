"""Hostile validation for RTU PPO RNG isolation."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.forager_rtu_ppo_rng_isolation import (
    RTUPPORngIsolationError,
    _require_exact_str,
)


class _EvilStr(str):
    calls = 0

    def __str__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("EvilStr.__str__ must not be called")

    def __repr__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("EvilStr.__repr__ must not be called")


class _StringSubclass(str):
    pass


def test_require_exact_str_rejects_evil_without_hooks() -> None:
    evil = _EvilStr("id")
    _EvilStr.calls = 0
    with pytest.raises(RTUPPORngIsolationError, match="exact string") as exc:
        _require_exact_str("replacement_id", evil)  # type: ignore[arg-type]
    assert _EvilStr.calls == 0
    assert "!r" not in str(exc.value)
    assert "EvilStr" not in str(exc.value)


def test_require_exact_str_rejects_subclass() -> None:
    with pytest.raises(RTUPPORngIsolationError, match="exact string"):
        _require_exact_str("replacement_id", _StringSubclass("id"))  # type: ignore[arg-type]


def test_require_exact_str_rejects_hostile_name() -> None:
    evil = _EvilStr("replacement_id")
    _EvilStr.calls = 0
    with pytest.raises(RTUPPORngIsolationError, match="exact string"):
        _require_exact_str(evil, "value")  # type: ignore[arg-type]
    assert _EvilStr.calls == 0


def test_valid_require_exact_str_passes() -> None:
    assert _require_exact_str("replacement_id", "valid_id") == "valid_id"
