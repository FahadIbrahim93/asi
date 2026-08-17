"""Hostile validation for Step 9 guarded-dreaming facade."""

from fractions import Fraction
from typing import Any, cast

import numpy as np
import pytest

from alberta_framework.steps.step9 import Step9DreamingConfig


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


def test_rejects_string_subclass_for_observation_dim() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        Step9DreamingConfig(observation_dim=_StringSubclass("4"))  # type: ignore[arg-type]


def test_hostile_str_for_observation_dim_without_repr_leak() -> None:
    evil = _EvilStr("4")
    with pytest.raises(ValueError, match="must be an integer") as exc:
        Step9DreamingConfig(observation_dim=evil)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_rejects_bool_and_hostile_int() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        Step9DreamingConfig(observation_dim=True)  # type: ignore[arg-type]
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="must be an integer") as exc:
        Step9DreamingConfig(observation_dim=_HostileInt(4))  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    assert "HostileInt" not in str(exc.value)


def test_rejects_hostile_float_without_hook_and_repr_leak() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="must be finite") as exc:
        Step9DreamingConfig(model_step_size=_HostileFloat(0.05))  # type: ignore[arg-type]
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


def test_rejects_plain_string_for_model_gamma() -> None:
    with pytest.raises(ValueError, match="must be a real number"):
        Step9DreamingConfig(model_gamma="0.99")  # type: ignore[arg-type]


def test_rejects_string_subclass_for_model_gamma() -> None:
    with pytest.raises(ValueError, match="must be a real number"):
        Step9DreamingConfig(model_gamma=_StringSubclass("0.99"))  # type: ignore[arg-type]


def test_rejects_out_of_range_model_gamma_without_repr() -> None:
    with pytest.raises(ValueError, match="must be in \\[0, 1\\]") as exc:
        Step9DreamingConfig(model_gamma=2.0)
    assert "2.0" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_rejects_hidden_sizes_non_tuple_without_repr() -> None:
    with pytest.raises(ValueError, match="must be a tuple of integers") as exc:
        Step9DreamingConfig(model_hidden_sizes=[64])  # type: ignore[arg-type]
    assert "!r" not in str(exc.value)


def test_rejects_model_error_decay_out_of_range_without_repr() -> None:
    with pytest.raises(ValueError, match="must be in \\[0, 1\\)") as exc:
        Step9DreamingConfig(model_error_decay=1.0)
    assert "!r" not in str(exc.value)


def test_valid_configs_still_pass() -> None:
    cfg = Step9DreamingConfig(observation_dim=4, model_gamma=0.99)
    assert cfg.observation_dim == 4
    assert cfg.model_gamma == pytest.approx(0.99)
    cfg2 = Step9DreamingConfig(model_hidden_sizes=(32, 16), observation_dim=4)
    assert cfg2.model_hidden_sizes == (32, 16)


def test_numpy_scalars_pass() -> None:
    cfg = Step9DreamingConfig(
        observation_dim=cast(Any, np.int32(4)),
        model_step_size=cast(Any, np.float32(0.05)),
        model_gamma=cast(Any, np.float64(0.9)),
    )
    assert cfg.observation_dim == 4
    cfg2 = Step9DreamingConfig(model_step_size=cast(Any, Fraction(1, 20)))
    assert cfg2.model_step_size == pytest.approx(0.05)


def test_float_subclass_with_lying_ratio_is_rejected() -> None:
    class RatioFloat(float):
        calls = 0

        def as_integer_ratio(self) -> tuple[int, int]:
            type(self).calls += 1
            return (3, 4)

    with pytest.raises(ValueError, match="must be finite"):
        Step9DreamingConfig(model_step_size=RatioFloat(0.05))
    assert RatioFloat.calls == 0
