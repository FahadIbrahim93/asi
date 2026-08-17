"""Hostile validation for Step 5 prediction facade."""

from fractions import Fraction
from typing import Any, cast

import numpy as np
import pytest

from alberta_framework.steps.step5 import Step5AverageRewardTDConfig


class _EvilStr(str):
    def __str__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__str__ must not be called")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__repr__ must not be called")


class _StringSubclass(str):
    pass


class _HostileInt(int):
    calls = 0

    def __index__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileInt.__index__ must not be called")

    def __repr__(self) -> str:
        type(self).calls += 1
        raise AssertionError("HostileInt.__repr__ must not be called")


class _HostileFloat(float):
    calls = 0

    def as_integer_ratio(self):  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileFloat.as_integer_ratio must not be called")

    def __float__(self) -> float:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileFloat.__float__ must not be called")

    def __repr__(self) -> str:
        type(self).calls += 1
        raise AssertionError("HostileFloat.__repr__ must not be called")


def test_rejects_string_subclass_for_step_size() -> None:
    with pytest.raises(ValueError, match="must be a real number"):
        Step5AverageRewardTDConfig(step_size=_StringSubclass("0.05"))  # type: ignore[arg-type]


def test_hostile_str_for_step_size_without_repr_leak() -> None:
    evil = _EvilStr("0.05")
    with pytest.raises(ValueError, match="must be a real number") as exc:
        Step5AverageRewardTDConfig(step_size=evil)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_rejects_hostile_float_without_hook_and_repr_leak() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="must be finite") as exc:
        Step5AverageRewardTDConfig(step_size=_HostileFloat(0.05))  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0
    assert "HostileFloat" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_does_not_invoke_hostile_value_when_name_is_evil_via_sink() -> None:
    from alberta_framework.steps._float32_validation import finite_real_and_float32

    evil = _EvilStr("x")
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="must be an exact string"):
        finite_real_and_float32(evil, _HostileFloat(1.0))  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0


def test_rejects_plain_string_for_trace_decay() -> None:
    with pytest.raises(ValueError, match="must be a real number"):
        Step5AverageRewardTDConfig(trace_decay="0.5")  # type: ignore[arg-type]


def test_rejects_string_subclass_for_trace_decay() -> None:
    with pytest.raises(ValueError, match="must be a real number"):
        Step5AverageRewardTDConfig(trace_decay=_StringSubclass("0.5"))  # type: ignore[arg-type]


def test_rejects_out_of_range_trace_decay_without_repr() -> None:
    with pytest.raises(ValueError, match="must be in \\[0, 1\\]") as exc:
        Step5AverageRewardTDConfig(trace_decay=2.0)
    assert "2.0" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_rejects_negative_step_size_without_repr() -> None:
    with pytest.raises(ValueError, match="must be non-negative") as exc:
        Step5AverageRewardTDConfig(step_size=-1.0)
    assert "!r" not in str(exc.value)


def test_rejects_bool_and_hostile_int_via_require_int() -> None:
    from alberta_framework.steps.step5 import _require_int

    with pytest.raises(ValueError, match="must be an integer"):
        _require_int("steps", True)  # type: ignore[arg-type]
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="must be an integer") as exc:
        _require_int("steps", _HostileInt(32))  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    assert "HostileInt" not in str(exc.value)


def test_rejects_bool_gate_without_repr() -> None:
    from alberta_framework.steps.step5 import _require_bool

    with pytest.raises(ValueError, match="must be an exact string") as exc:
        _require_bool(_EvilStr("finite"), True)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)
    with pytest.raises(ValueError, match="must be a built-in bool") as exc2:
        _require_bool("finite", _StringSubclass("true"))  # type: ignore[arg-type]
    assert "StringSubclass" not in str(exc2.value)
    assert "!r" not in str(exc2.value)


def test_valid_configs_still_pass() -> None:
    cfg = Step5AverageRewardTDConfig(step_size=0.05, trace_decay=0.5)
    assert cfg.step_size == pytest.approx(0.05)
    assert cfg.trace_decay == pytest.approx(0.5)


def test_numpy_scalars_pass() -> None:
    cfg = Step5AverageRewardTDConfig(
        step_size=cast(Any, np.float32(0.05)),
        trace_decay=cast(Any, np.float64(0.5)),
    )
    assert cfg.step_size == pytest.approx(0.05)
    cfg2 = Step5AverageRewardTDConfig(step_size=cast(Any, Fraction(1, 20)))
    assert cfg2.step_size == pytest.approx(0.05)


def test_float_subclass_with_lying_ratio_is_rejected() -> None:
    class RatioFloat(float):
        calls = 0

        def as_integer_ratio(self) -> tuple[int, int]:
            type(self).calls += 1
            return (3, 4)

    with pytest.raises(ValueError, match="must be finite"):
        Step5AverageRewardTDConfig(step_size=RatioFloat(0.05))
    assert RatioFloat.calls == 0
