"""Exact binary32 canonicalization shared by the merged Step facades."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from fractions import Fraction
from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.options import SubtaskSpec
from alberta_framework.steps.step3 import Step3HordeConfig
from alberta_framework.steps.step4 import Step4SARSAConfig
from alberta_framework.steps.step5 import Step5AverageRewardTDConfig
from alberta_framework.steps.step7 import Step7DynaConfig
from alberta_framework.steps.step8 import Step8WorldModelConfig
from alberta_framework.steps.step9 import Step9DreamingConfig
from alberta_framework.steps.step10 import Step10STOMPConfig

_FacadeValues = Callable[[object], tuple[float, ...]]


def _step3_values(value: object) -> tuple[float, ...]:
    scalar = cast(Any, value)
    config = Step3HordeConfig(
        gammas=(scalar,),
        lamdas=(scalar,),
        step_size=scalar,
        obgd_kappa=scalar,
        sparsity=scalar,
    )
    return (
        config.gammas[0],
        config.lamdas[0],
        config.step_size,
        config.obgd_kappa,
        config.sparsity,
    )


def _step4_values(value: object) -> tuple[float, ...]:
    scalar = cast(Any, value)
    config = Step4SARSAConfig(
        gamma=scalar,
        epsilon_start=scalar,
        epsilon_end=scalar,
        lamda=scalar,
        step_size=scalar,
        meta_step_size=scalar,
        bounder_kappa=scalar,
        sparsity=scalar,
    )
    return (
        config.gamma,
        config.epsilon_start,
        config.epsilon_end,
        config.lamda,
        config.step_size,
        config.meta_step_size,
        config.bounder_kappa,
        config.sparsity,
    )


def _step5_values(value: object) -> tuple[float, ...]:
    scalar = cast(Any, value)
    config = Step5AverageRewardTDConfig(
        step_size=scalar,
        average_reward_step_size=scalar,
        trace_decay=scalar,
    )
    return (
        config.step_size,
        config.average_reward_step_size,
        config.trace_decay,
    )


def _step7_values(value: object) -> tuple[float, ...]:
    scalar = cast(Any, value)
    config = Step7DynaConfig(
        planning_importance_ratio_clip=scalar,
        planning_priority_propagation=scalar,
        planning_utility_step_size=scalar,
    )
    return (
        config.planning_importance_ratio_clip,
        config.planning_priority_propagation,
        config.planning_utility_step_size,
    )


def _step8_values(value: object) -> tuple[float, ...]:
    scalar = cast(Any, value)
    config = Step8WorldModelConfig(
        step_size=scalar,
        sparsity=scalar,
        leaky_relu_slope=scalar,
        utility_decay=scalar,
    )
    return (
        config.step_size,
        config.sparsity,
        config.leaky_relu_slope,
        config.utility_decay,
    )


def _step9_values(value: object) -> tuple[float, ...]:
    scalar = cast(Any, value)
    config = Step9DreamingConfig(
        model_step_size=scalar,
        model_sparsity=scalar,
        model_gamma=scalar,
        dreaming_max_model_error=scalar,
        model_error_decay=scalar,
        behavior_model_step_size=scalar,
        dream_surprise_weight=scalar,
        dream_utility_weight=scalar,
    )
    return (
        config.model_step_size,
        config.model_sparsity,
        config.model_gamma,
        config.dreaming_max_model_error,
        config.model_error_decay,
        config.behavior_model_step_size,
        config.dream_surprise_weight,
        config.dream_utility_weight,
    )


def _step10_values(value: object) -> tuple[float, ...]:
    scalar = cast(Any, value)
    config = Step10STOMPConfig(
        subtask_specs=(
            SubtaskSpec(
                feature_index=0,
                threshold=scalar,
                pseudo_reward_scale=scalar,
            ),
        ),
        base_step_size=scalar,
        base_avg_reward_step_size=scalar,
        base_trace_decay=scalar,
        option_step_size=scalar,
        option_avg_reward_step_size=scalar,
        option_trace_decay=scalar,
        option_gamma=scalar,
        option_model_decay=scalar,
        option_model_step_size=scalar,
        epsilon_base=scalar,
        epsilon_option=scalar,
        option_target_epsilon=scalar,
        option_importance_clip=scalar,
    )
    spec = config.subtask_specs[0]
    return (
        spec.threshold,
        spec.pseudo_reward_scale,
        config.base_step_size,
        config.base_avg_reward_step_size,
        config.base_trace_decay,
        config.option_step_size,
        config.option_avg_reward_step_size,
        config.option_trace_decay,
        config.option_gamma,
        config.option_model_decay,
        config.option_model_step_size,
        config.epsilon_base,
        config.epsilon_option,
        cast(float, config.option_target_epsilon),
        config.option_importance_clip,
    )


@pytest.mark.parametrize(
    "facade_values",
    [
        pytest.param(_step3_values, id="step3"),
        pytest.param(_step4_values, id="step4"),
        pytest.param(_step5_values, id="step5"),
        pytest.param(_step7_values, id="step7"),
        pytest.param(_step8_values, id="step8"),
        pytest.param(_step9_values, id="step9"),
        pytest.param(_step10_values, id="step10"),
    ],
)
def test_all_float32_bound_fields_round_exact_fraction_once(
    facade_values: _FacadeValues,
) -> None:
    above_half_midpoint = (
        Fraction(1, 2) + Fraction(1, 2**25) + Fraction(1, 2**70)
    )
    expected = float(np.nextafter(np.float32(0.5), np.float32(1.0)))

    values = facade_values(above_half_midpoint)

    assert values
    assert all(value == expected for value in values)
    json.dumps(values, allow_nan=False)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            Fraction(1, 1) + Fraction(1, 2**24) - Fraction(1, 2**60),
            1.0,
        ),
        (Fraction(1, 1) + Fraction(1, 2**24), 1.0),
        (
            Fraction(1, 1) + Fraction(1, 2**24) + Fraction(1, 2**60),
            float(np.nextafter(np.float32(1.0), np.float32(2.0))),
        ),
    ],
    ids=("below", "tie-even", "above"),
)
def test_fraction_midpoint_neighborhood_rounds_once(
    value: Fraction,
    expected: float,
) -> None:
    assert Step5AverageRewardTDConfig(step_size=value).step_size == expected


def test_fraction_odd_lower_significand_tie_rounds_up() -> None:
    odd_lower_tie = Fraction(1, 1) + Fraction(3, 2**24)
    expected = float(np.nextafter(np.float32(1.0), np.float32(2.0), dtype=np.float32))
    expected = float(np.nextafter(np.float32(expected), np.float32(2.0)))

    assert Step8WorldModelConfig(step_size=odd_lower_tie).step_size == expected


def test_subnormal_tie_and_above_round_with_even_zero() -> None:
    half_min_subnormal = Fraction(1, 2**150)
    just_above = half_min_subnormal + Fraction(1, 2**200)
    min_subnormal = float(np.nextafter(np.float32(0.0), np.float32(1.0)))

    assert Step8WorldModelConfig(step_size=half_min_subnormal).step_size == 0.0
    assert Step8WorldModelConfig(step_size=just_above).step_size == min_subnormal


def test_positive_field_rejects_value_that_rounds_to_zero() -> None:
    with pytest.raises(ValueError, match="option_importance_clip must be positive"):
        Step10STOMPConfig(option_importance_clip=Fraction(1, 2**150))


def test_nonnegative_field_rejects_negative_value_that_underflows() -> None:
    with pytest.raises(ValueError, match="model_step_size must be non-negative"):
        Step9DreamingConfig(model_step_size=Fraction(-1, 2**1200))


def test_unit_interval_rejects_exact_value_above_endpoint() -> None:
    with pytest.raises(ValueError, match=r"gamma must be in \[0, 1\]"):
        Step4SARSAConfig(gamma=Fraction(1, 1) + Fraction(1, 2**200))


def test_finite_overflow_boundary_accepts_below_and_rejects_tie() -> None:
    overflow_midpoint = (2**25 - 1) * 2**103
    maximum = float(np.finfo(np.float32).max)

    config = Step7DynaConfig(
        planning_importance_ratio_clip=overflow_midpoint - 1,
    )
    assert config.planning_importance_ratio_clip == maximum
    with pytest.raises(ValueError, match="must be finite"):
        Step7DynaConfig(planning_importance_ratio_clip=overflow_midpoint)


def test_large_integral_midpoint_uses_binary32_ties_to_even() -> None:
    lower = 2**64
    tie = lower + 2**40
    above = tie + 1
    expected_upper = float(np.nextafter(np.float32(lower), np.float32(np.inf)))

    assert Step9DreamingConfig(dreaming_max_model_error=tie).dreaming_max_model_error == float(
        lower
    )
    assert (
        Step9DreamingConfig(dreaming_max_model_error=above).dreaming_max_model_error
        == expected_upper
    )


def test_numpy_integral_midpoint_uses_exact_integer_ratio() -> None:
    lower = 2**62
    tie = np.int64(lower + 2**38)
    above = np.int64(int(tie) + 1)
    expected_upper = float(np.nextafter(np.float32(lower), np.float32(np.inf)))

    assert Step9DreamingConfig(dreaming_max_model_error=tie).dreaming_max_model_error == float(
        lower
    )
    assert (
        Step9DreamingConfig(dreaming_max_model_error=above).dreaming_max_model_error
        == expected_upper
    )


def test_numpy_float64_midpoint_neighborhood_rounds_once() -> None:
    tie = np.float64(float.fromhex("0x1.000001p0"))
    above = np.nextafter(tie, np.float64(np.inf))
    expected_upper = float(np.nextafter(np.float32(1.0), np.float32(2.0)))

    tie_config = Step8WorldModelConfig(step_size=tie)
    above_config = Step8WorldModelConfig(step_size=above)

    assert type(tie_config.step_size) is float
    assert tie_config.step_size == 1.0
    assert above_config.step_size == expected_upper


@pytest.mark.parametrize(
    "value",
    [np.float16(0.1), np.float32(0.1), np.float64(0.1), np.int32(1), np.int64(1)],
)
def test_numpy_real_and_integral_scalars_canonicalize_for_json(value: object) -> None:
    config = Step3HordeConfig(step_size=cast(Any, value))

    assert type(config.step_size) is float
    json.dumps(config.to_dict(), allow_nan=False)


@pytest.mark.parametrize("value", [jnp.float32(0.5), jnp.int32(1)])
def test_jax_array_scalars_remain_outside_the_real_facade(value: object) -> None:
    with pytest.raises(ValueError, match="step_size must be a real number"):
        Step3HordeConfig(step_size=cast(Any, value))


@pytest.mark.parametrize(
    "config_and_payload",
    [
        pytest.param(
            lambda: (
                Step3HordeConfig(step_size=0.1),
                Step3HordeConfig(step_size=0.1).to_dict(),
            ),
            id="step3",
        ),
        pytest.param(
            lambda: (
                Step4SARSAConfig(step_size=0.1),
                Step4SARSAConfig(step_size=0.1).to_dict(),
            ),
            id="step4",
        ),
        pytest.param(
            lambda: (
                Step5AverageRewardTDConfig(step_size=0.1),
                Step5AverageRewardTDConfig(step_size=0.1).to_dict(),
            ),
            id="step5",
        ),
        pytest.param(
            lambda: (
                Step7DynaConfig(planning_priority_propagation=0.1),
                Step7DynaConfig(planning_priority_propagation=0.1).to_dict(),
            ),
            id="step7",
        ),
        pytest.param(
            lambda: (
                Step8WorldModelConfig(step_size=0.1),
                Step8WorldModelConfig(step_size=0.1).to_dict(),
            ),
            id="step8",
        ),
        pytest.param(
            lambda: (
                Step9DreamingConfig(model_step_size=0.1),
                Step9DreamingConfig(model_step_size=0.1).to_dict(),
            ),
            id="step9",
        ),
        pytest.param(
            lambda: (
                Step10STOMPConfig(base_step_size=0.1),
                Step10STOMPConfig(base_step_size=0.1).to_config(),
            ),
            id="step10",
        ),
    ],
)
def test_builtin_float_serialization_value_is_preserved(
    config_and_payload: Callable[[], tuple[object, dict[str, Any]]],
) -> None:
    _, payload = config_and_payload()

    selected = next(value for value in payload.values() if value == 0.1)
    assert selected == 0.1
    json.dumps(payload, allow_nan=False)


def test_signed_zero_survives_validation_and_serialization() -> None:
    config = Step3HordeConfig(gammas=(-0.0,), lamdas=(0.0,))
    payload = config.to_dict()

    assert math.copysign(1.0, config.gammas[0]) == -1.0
    assert math.copysign(1.0, cast(list[float], payload["gammas"])[0]) == -1.0
