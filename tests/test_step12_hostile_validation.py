"""Hostile validation for Step 12 IA facade."""

from fractions import Fraction
from typing import Any, cast

import numpy as np
import pytest

from alberta_framework.core.options import SubtaskSpec
from alberta_framework.steps.step12 import (
    Step12IAConfig,
    Step12SmokeResult,
    make_step12_ia_agent,
    run_step12_smoke,
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

    def __index__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("HostileInt.__index__ must not be called")

    def __int__(self) -> int:
        type(self).calls += 1
        raise AssertionError("HostileInt.__int__ must not be called")

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


class _HostileTypeName(type):
    calls = 0

    def __getattribute__(cls, name: str) -> Any:
        if name == "__name__":
            _HostileTypeName.calls += 1
            raise AssertionError("metaclass __name__ hook must not be called")
        return super().__getattribute__(name)


class _HostileSpecsContainer(metaclass=_HostileTypeName):
    calls = 0

    def __iter__(self):
        type(self).calls += 1
        raise AssertionError("container iteration hook must not be called")

    def __repr__(self) -> str:
        type(self).calls += 1
        raise AssertionError("container repr hook must not be called")


def test_rejects_string_subclass_for_n_demons() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        Step12IAConfig(n_demons=_StringSubclass("4"))  # type: ignore[arg-type]


def test_hostile_str_for_n_demons_without_repr_leak() -> None:
    evil = _EvilStr("4")
    with pytest.raises(ValueError, match="must be an integer") as exc:
        Step12IAConfig(n_demons=evil)  # type: ignore[arg-type]
    assert "EvilStr" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_rejects_bool_and_hostile_int() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        Step12IAConfig(n_demons=True)  # type: ignore[arg-type]
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="must be an integer") as exc:
        Step12IAConfig(n_demons=_HostileInt(4))  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    assert "HostileInt" not in str(exc.value)


def test_rejects_hostile_float_without_hook_and_repr_leak() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="must be finite") as exc:
        Step12IAConfig(cerebellum_step_size=_HostileFloat(0.05))  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0
    assert "HostileFloat" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_rejects_plain_string_for_option_gamma() -> None:
    with pytest.raises(ValueError, match="must be a real number"):
        Step12IAConfig(option_gamma="0.99")  # type: ignore[arg-type]


def test_rejects_string_subclass_for_option_gamma() -> None:
    with pytest.raises(ValueError, match="must be a real number"):
        Step12IAConfig(option_gamma=_StringSubclass("0.99"))  # type: ignore[arg-type]


def test_rejects_out_of_range_option_gamma_without_repr() -> None:
    with pytest.raises(ValueError, match="must be in \\[0, 1\\]") as exc:
        Step12IAConfig(option_gamma=2.0)
    assert "2.0" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_rejects_subtask_specs_non_tuple_without_repr() -> None:
    with pytest.raises(ValueError, match="must be a tuple of SubtaskSpec") as exc:
        Step12IAConfig(subtask_specs=[SubtaskSpec(feature_index=0)])  # type: ignore[arg-type]
    assert "!r" not in str(exc.value)


def test_rejects_hostile_subtask_container_without_metadata_hooks() -> None:
    _HostileTypeName.calls = 0
    _HostileSpecsContainer.calls = 0
    with pytest.raises(ValueError, match="must be a tuple of SubtaskSpec"):
        Step12IAConfig(subtask_specs=_HostileSpecsContainer())  # type: ignore[arg-type]
    assert _HostileTypeName.calls == 0
    assert _HostileSpecsContainer.calls == 0


def test_rejects_feature_index_out_of_range_without_repr() -> None:
    with pytest.raises(ValueError, match="must be < observation_dim") as exc:
        Step12IAConfig(
            observation_dim=2,
            subtask_specs=(SubtaskSpec(feature_index=5, threshold=1.0),),
        )
    assert "!r" not in str(exc.value)


def test_valid_configs_still_pass() -> None:
    cfg = Step12IAConfig(n_demons=4, observation_dim=4, option_gamma=0.99)
    assert cfg.n_demons == 4
    assert cfg.option_gamma == pytest.approx(0.99)
    cfg2 = Step12IAConfig(
        subtask_specs=(SubtaskSpec(feature_index=0, threshold=0.5),),
        observation_dim=4,
    )
    assert cfg2.subtask_specs[0].feature_index == 0


def test_numpy_scalars_pass() -> None:
    cfg = Step12IAConfig(
        n_demons=cast(Any, np.int32(4)),
        cerebellum_step_size=cast(Any, np.float32(0.05)),
        option_gamma=cast(Any, np.float64(0.9)),
    )
    assert cfg.n_demons == 4
    cfg2 = Step12IAConfig(cerebellum_step_size=cast(Any, Fraction(1, 20)))
    assert cfg2.cerebellum_step_size == pytest.approx(0.05)


def test_float_subclass_with_lying_ratio_is_rejected() -> None:
    class RatioFloat(float):
        calls = 0

        def as_integer_ratio(self) -> tuple[int, int]:
            type(self).calls += 1
            return (3, 4)

    with pytest.raises(ValueError, match="must be finite"):
        Step12IAConfig(cerebellum_step_size=RatioFloat(0.05))
    assert RatioFloat.calls == 0


def test_from_config_requires_complete_exact_schema_without_mapping_hooks() -> None:
    class HostileDict(dict[object, object]):
        calls = 0

        def __iter__(self):  # pragma: no cover - must not run
            type(self).calls += 1
            raise AssertionError("mapping hook must not run")

    with pytest.raises(ValueError, match="exact dictionary"):
        Step12IAConfig.from_config(cast(Any, HostileDict()))
    assert HostileDict.calls == 0

    payload = Step12IAConfig().to_config()
    for mutation in (
        lambda value: value.pop("type"),
        lambda value: value.__setitem__("extra", 1),
        lambda value: value.__setitem__("type", "wrong"),
    ):
        malformed = dict(payload)
        mutation(malformed)
        with pytest.raises(ValueError, match="(schema|payload type)"):
            Step12IAConfig.from_config(malformed)


def test_from_config_requires_exact_nested_records_before_hooks() -> None:
    class HostileRecord(dict[object, object]):
        calls = 0

        def __iter__(self):  # pragma: no cover - must not run
            type(self).calls += 1
            raise AssertionError("record hook must not run")

    payload = Step12IAConfig().to_config()
    payload["subtask_specs"] = [HostileRecord()]
    with pytest.raises(ValueError, match="exact dictionary"):
        Step12IAConfig.from_config(payload)
    assert HostileRecord.calls == 0


def test_from_config_and_direct_paths_have_matching_canonical_values() -> None:
    direct = Step12IAConfig(
        n_demons=np.int32(3),  # type: ignore[arg-type]
        cerebellum_step_size=np.float64(0.05),  # type: ignore[arg-type]
        subtask_specs=(SubtaskSpec(feature_index=0, threshold=Fraction(1, 2)),),
    )
    parsed = Step12IAConfig.from_config(direct.to_config())
    assert parsed == direct
    assert parsed.to_config() == direct.to_config()


def test_subtask_validation_work_is_bounded_before_iteration() -> None:
    spec = SubtaskSpec(feature_index=0)
    with pytest.raises(ValueError, match="at most 4096"):
        Step12IAConfig(subtask_specs=(spec,) * 4_097)

    payload = Step12IAConfig().to_config()
    raw_spec = {
        "feature_index": 0,
        "threshold": 0.5,
        "pseudo_reward_scale": 1.0,
        "max_option_steps": 8,
    }
    payload["subtask_specs"] = [raw_spec] * 4_097
    with pytest.raises(ValueError, match="at most 4096"):
        Step12IAConfig.from_config(payload)


def test_runtime_entry_points_require_exact_config_without_hooks() -> None:
    calls = 0

    class HostileConfig:
        def __bool__(self) -> bool:  # pragma: no cover - must not run
            nonlocal calls
            calls += 1
            raise AssertionError("truthiness hook must not run")

        def to_ia_config(self) -> object:  # pragma: no cover - must not run
            nonlocal calls
            calls += 1
            raise AssertionError("config hook must not run")

    value = HostileConfig()
    with pytest.raises(TypeError, match="exact Step12IAConfig"):
        make_step12_ia_agent(cast(Any, value))
    with pytest.raises(TypeError, match="exact Step12IAConfig"):
        run_step12_smoke(cast(Any, value), steps=1)
    assert calls == 0


def test_runtime_preflights_derived_resources_before_jax_allocation() -> None:
    with pytest.raises(ValueError, match="augmented observation dimension"):
        Step12IAConfig(n_demons=2**31 - 1, observation_dim=1)
    with pytest.raises(ValueError, match="cerebellum weight bytes"):
        Step12IAConfig(n_demons=536_870_912, observation_dim=1)
    with pytest.raises(ValueError, match="cerebellum weight count"):
        Step12IAConfig(n_demons=50_000, observation_dim=50_000)
    with pytest.raises(ValueError, match="observation row count"):
        run_step12_smoke(steps=2**31 - 1)


def test_smoke_record_requires_exact_config_and_agent_config_identities() -> None:
    legal: dict[str, object] = {
        "config": Step12IAConfig(),
        "steps": 1,
        "seed": 0,
        "predictions_shape": (1, 4),
        "cerebellum_errors_shape": (1, 4),
        "recommendations_shape": (1,),
        "augmented_obs_shape": (1, 8),
        "cortex_td_errors_shape": (1,),
        "finite": True,
        "agent_config": {},
    }
    with pytest.raises(TypeError, match="config must be an exact"):
        Step12SmokeResult(**{**legal, "config": object()})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="agent_config must be an exact"):
        Step12SmokeResult(**{**legal, "agent_config": {"x": 1}.keys()})  # type: ignore[arg-type]
