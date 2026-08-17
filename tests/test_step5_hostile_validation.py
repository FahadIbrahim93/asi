"""Hostile validation for Step 5 prediction facade."""

from fractions import Fraction
from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.steps.step5 import (
    Step5AverageRewardTDConfig,
    Step5SmokeResult,
    make_step5_td_learner,
    run_step5_scan,
    run_step5_smoke,
)


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


class _HostileMeta(type):
    calls = 0

    def __hash__(cls) -> int:
        _HostileMeta.calls += 1
        raise AssertionError("HostileMeta.__hash__ must not be called")


class _HostileMetaclassInt(int, metaclass=_HostileMeta):
    pass


class _HostileTuple(tuple):
    calls = 0

    def __iter__(self):
        type(self).calls += 1
        raise AssertionError("HostileTuple.__iter__ must not be called")

    def __repr__(self) -> str:
        type(self).calls += 1
        raise AssertionError("HostileTuple.__repr__ must not be called")


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


def test_rejects_hostile_float_without_hook_and_repr_leak() -> None:
    _HostileFloat.calls = 0
    with pytest.raises(ValueError, match="must be finite") as exc:
        Step5AverageRewardTDConfig(step_size=_HostileFloat(0.05))  # type: ignore[arg-type]
    assert _HostileFloat.calls == 0
    assert "HostileFloat" not in str(exc.value)
    assert "!r" not in str(exc.value)


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

    _HostileMeta.calls = 0
    with pytest.raises(ValueError, match="must be an integer"):
        _require_int("steps", _HostileMetaclassInt(32))
    assert _HostileMeta.calls == 0


def test_rejects_bool_gate_without_repr() -> None:
    from alberta_framework.steps.step5 import _require_bool

    with pytest.raises(ValueError, match="must be a built-in bool") as exc2:
        _require_bool("finite", _StringSubclass("true"))  # type: ignore[arg-type]
    assert "StringSubclass" not in str(exc2.value)
    assert "!r" not in str(exc2.value)


def test_smoke_result_rejects_hostile_shape_without_hooks() -> None:
    _HostileTuple.calls = 0
    with pytest.raises(ValueError, match="predictions_shape"):
        Step5SmokeResult(
            config=Step5AverageRewardTDConfig(),
            steps=2,
            seed=0,
            predictions_shape=_HostileTuple((2,)),
            td_errors_shape=(2,),
            average_rewards_shape=(2,),
            finite=True,
            learner_config={},
        )
    assert _HostileTuple.calls == 0


@pytest.mark.parametrize("seed", [2**31, 2**32 - 1])
def test_smoke_accepts_full_uint32_seed(seed: int) -> None:
    assert run_step5_smoke(steps=1, feature_dim=1, seed=seed).seed == seed


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


def test_config_subclass_and_dict_subclass_are_rejected() -> None:
    class ConfigSubclass(Step5AverageRewardTDConfig):
        pass

    with pytest.raises(ValueError, match="actual Step5AverageRewardTDConfig"):
        ConfigSubclass()

    class DictSubclass(dict[str, object]):
        pass

    with pytest.raises(ValueError, match="actual dict"):
        Step5AverageRewardTDConfig.from_dict(
            DictSubclass(Step5AverageRewardTDConfig().to_dict())
        )


class _HostileArray:
    calls = 0

    @property
    def shape(self) -> tuple[int, ...]:
        type(self).calls += 1
        raise AssertionError("shape hook must not run")


def test_scan_rejects_hostile_arrays_without_hooks() -> None:
    learner = make_step5_td_learner()
    state = learner.init(2)
    _HostileArray.calls = 0
    with pytest.raises(TypeError, match="trusted array"):
        run_step5_scan(
            learner,
            state,
            cast(Any, _HostileArray()),
            jnp.zeros(1, dtype=jnp.float32),
            jnp.zeros((1, 2), dtype=jnp.float32),
        )
    assert _HostileArray.calls == 0


def test_scan_requires_exact_float32_shapes() -> None:
    learner = make_step5_td_learner()
    state = learner.init(2)
    with pytest.raises(TypeError, match="rewards.*float32"):
        run_step5_scan(
            learner,
            state,
            jnp.zeros((1, 2), dtype=jnp.float32),
            jnp.zeros(1, dtype=jnp.int32),
            jnp.zeros((1, 2), dtype=jnp.float32),
        )
    with pytest.raises(ValueError, match="next_observations must have shape"):
        run_step5_scan(
            learner,
            state,
            jnp.zeros((1, 2), dtype=jnp.float32),
            jnp.zeros(1, dtype=jnp.float32),
            jnp.zeros((2, 2), dtype=jnp.float32),
        )


def test_smoke_rejects_derived_allocation_before_jax() -> None:
    with pytest.raises(ValueError, match="derived Step 5 smoke"):
        run_step5_smoke(steps=2**20, feature_dim=2**20)
