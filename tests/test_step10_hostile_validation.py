"""Hostile validation for Step 10 STOMP facade."""

from fractions import Fraction
from typing import Any, cast

import numpy as np
import pytest

from alberta_framework.core.options import SubtaskSpec
from alberta_framework.steps.step10 import (
    Step10STOMPConfig,
    make_step10_stomp_agent,
    run_step10_smoke,
)


class _EvilStr(str):
    def __str__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__str__ must not be called")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("EvilStr.__repr__ must not be called")


class _StringSubclass(str):
    pass


class _HostileInt(int):
    calls = 0

    def __int__(self) -> int:
        type(self).calls += 1
        raise AssertionError("HostileInt.__int__ must not be called")

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


class _HostileIntMeta(type):
    calls = 0

    def __hash__(cls) -> int:
        type(cls).calls += 1
        raise AssertionError("HostileIntMeta.__hash__ must not be called")


class _MetaclassHostileInt(int, metaclass=_HostileIntMeta):
    pass


class _HostileFloatMeta(type):
    calls = 0

    def __hash__(cls) -> int:
        type(cls).calls += 1
        raise AssertionError("HostileFloatMeta.__hash__ must not be called")


class _MetaclassHostileFloat(float, metaclass=_HostileFloatMeta):
    pass


def test_rejects_string_subclass_for_observation_dim() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        Step10STOMPConfig(observation_dim=_StringSubclass("4"))  # type: ignore[arg-type]


def test_hostile_str_for_observation_dim_without_repr_leak() -> None:
    evil = _EvilStr("4")
    with pytest.raises(ValueError, match="must be an integer") as exc:
        Step10STOMPConfig(observation_dim=evil)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_rejects_bool_and_hostile_int() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        Step10STOMPConfig(observation_dim=True)  # type: ignore[arg-type]
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="must be an integer") as exc:
        Step10STOMPConfig(observation_dim=_HostileInt(4))  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    assert "HostileInt" not in str(exc.value)


def test_rejects_hostile_int_metaclass_without_hooks() -> None:
    _HostileIntMeta.calls = 0
    with pytest.raises(ValueError, match="must be an integer"):
        Step10STOMPConfig(observation_dim=_MetaclassHostileInt(4))
    assert _HostileIntMeta.calls == 0


def test_rejects_hostile_float_without_hook_and_repr_leak() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="must be finite") as exc:
        Step10STOMPConfig(base_step_size=_HostileFloat(0.05))  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0
    assert "HostileFloat" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_rejects_hostile_float_metaclass_without_hooks() -> None:
    _HostileFloatMeta.calls = 0
    with pytest.raises(ValueError, match="must be finite"):
        Step10STOMPConfig(base_step_size=_MetaclassHostileFloat(0.05))
    assert _HostileFloatMeta.calls == 0


def test_rejects_plain_string_for_option_gamma() -> None:
    with pytest.raises(ValueError, match="must be a real number"):
        Step10STOMPConfig(option_gamma="0.99")  # type: ignore[arg-type]


def test_rejects_string_subclass_for_option_gamma() -> None:
    with pytest.raises(ValueError, match="must be a real number"):
        Step10STOMPConfig(option_gamma=_StringSubclass("0.99"))  # type: ignore[arg-type]


def test_rejects_out_of_range_option_gamma_without_repr() -> None:
    with pytest.raises(ValueError, match="must be in \\[0, 1\\]") as exc:
        Step10STOMPConfig(option_gamma=2.0)
    assert "2.0" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_rejects_subtask_specs_non_tuple_without_repr() -> None:
    with pytest.raises(ValueError, match="must be a tuple of SubtaskSpec") as exc:
        Step10STOMPConfig(subtask_specs=[SubtaskSpec(feature_index=0)])  # type: ignore[arg-type]
    assert "!r" not in str(exc.value)


def test_rejects_tuple_and_subtask_spec_subclasses_without_hooks() -> None:
    calls: list[str] = []

    class HostileTuple(tuple[SubtaskSpec, ...]):
        def __iter__(self):  # type: ignore[override]
            calls.append("iter")
            raise AssertionError("HostileTuple.__iter__ must not be called")

        def __repr__(self) -> str:
            calls.append("repr")
            raise AssertionError("HostileTuple.__repr__ must not be called")

    class HostileSpec(SubtaskSpec):
        def __repr__(self) -> str:
            calls.append("spec-repr")
            raise AssertionError("HostileSpec.__repr__ must not be called")

    with pytest.raises(ValueError, match="must be a tuple of SubtaskSpec"):
        Step10STOMPConfig(subtask_specs=HostileTuple((SubtaskSpec(feature_index=0),)))
    with pytest.raises(ValueError, match="must contain SubtaskSpec values"):
        Step10STOMPConfig(subtask_specs=(HostileSpec(feature_index=0),))
    assert calls == []


def test_smoke_preflights_steps_and_seed_without_hostile_hooks() -> None:
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="steps must be an integer"):
        run_step10_smoke(steps=_HostileInt(1))
    with pytest.raises(ValueError, match="seed must be a built-in integer"):
        run_step10_smoke(steps=1, seed=_HostileInt(0))
    assert _HostileInt.calls == 0


def test_smoke_accepts_full_uint32_seed_and_bounds_steps() -> None:
    assert run_step10_smoke(steps=1, seed=2**32 - 1).seed == 2**32 - 1
    with pytest.raises(ValueError, match="steps must be <="):
        run_step10_smoke(steps=2**31)


def test_rejects_feature_index_out_of_range_without_repr() -> None:
    with pytest.raises(ValueError, match="must be < observation_dim") as exc:
        Step10STOMPConfig(
            observation_dim=2,
            subtask_specs=(SubtaskSpec(feature_index=5, threshold=1.0),),
        )
    assert "!r" not in str(exc.value)


def test_valid_configs_still_pass() -> None:
    cfg = Step10STOMPConfig(observation_dim=4, option_gamma=0.99)
    assert cfg.observation_dim == 4
    assert cfg.option_gamma == pytest.approx(0.99)
    cfg2 = Step10STOMPConfig(
        subtask_specs=(SubtaskSpec(feature_index=0, threshold=0.5),),
        observation_dim=4,
    )
    assert cfg2.subtask_specs[0].feature_index == 0


def test_numpy_scalars_pass() -> None:
    cfg = Step10STOMPConfig(
        observation_dim=cast(Any, np.int32(4)),
        base_step_size=cast(Any, np.float32(0.05)),
        option_gamma=cast(Any, np.float64(0.9)),
    )
    assert cfg.observation_dim == 4
    cfg2 = Step10STOMPConfig(base_step_size=cast(Any, Fraction(1, 20)))
    assert cfg2.base_step_size == pytest.approx(0.05)


def test_float_subclass_with_lying_ratio_is_rejected() -> None:
    class RatioFloat(float):
        calls = 0

        def as_integer_ratio(self) -> tuple[int, int]:
            type(self).calls += 1
            return (3, 4)

    with pytest.raises(ValueError, match="must be finite"):
        Step10STOMPConfig(base_step_size=RatioFloat(0.05))
    assert RatioFloat.calls == 0


def test_from_config_requires_exact_complete_nested_schema_without_hooks() -> None:
    class HostileDict(dict[object, object]):
        calls = 0

        def __iter__(self):  # pragma: no cover - must not run
            type(self).calls += 1
            raise AssertionError("mapping hook must not run")

    with pytest.raises(ValueError, match="exact dictionary"):
        Step10STOMPConfig.from_config(cast(Any, HostileDict()))
    assert HostileDict.calls == 0
    payload = Step10STOMPConfig().to_config()
    payload["type"] = "wrong"
    with pytest.raises(ValueError, match="payload type"):
        Step10STOMPConfig.from_config(payload)
    payload = Step10STOMPConfig(
        subtask_specs=(SubtaskSpec(feature_index=0),)
    ).to_config()
    payload["subtask_specs"] = [HostileDict()]
    with pytest.raises(ValueError, match="exact dictionary"):
        Step10STOMPConfig.from_config(payload)
    assert HostileDict.calls == 0


def test_subtask_iteration_and_planning_work_are_bounded() -> None:
    spec = SubtaskSpec(feature_index=0)
    with pytest.raises(ValueError, match="at most 4096"):
        Step10STOMPConfig(subtask_specs=(spec,) * 4_097)
    with pytest.raises(ValueError, match="option_planning_backups_per_step"):
        Step10STOMPConfig(option_planning_backups_per_step=4_097)


def test_runtime_entry_points_require_exact_config_without_truthiness_hooks() -> None:
    calls = 0

    class HostileConfig:
        def __bool__(self) -> bool:  # pragma: no cover - must not run
            nonlocal calls
            calls += 1
            raise AssertionError("truthiness hook must not run")

    value = HostileConfig()
    with pytest.raises(TypeError, match="exact Step10STOMPConfig"):
        make_step10_stomp_agent(cast(Any, value))
    with pytest.raises(TypeError, match="exact Step10STOMPConfig"):
        run_step10_smoke(cast(Any, value), steps=1)
    assert calls == 0


def test_smoke_preflights_output_resources_before_allocation() -> None:
    with pytest.raises(ValueError, match="observation row count"):
        run_step10_smoke(steps=2**31 - 1)
