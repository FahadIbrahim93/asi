"""Hostile-safe validation for pipeline trusts."""

from __future__ import annotations

from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.pipeline import (
    AlbertaPipeline,
    AlbertaPipelineConfig,
    Step2FeatureConfig,
    Step2UPGDConfig,
    _require_bool,
    _require_int,
    _require_str_choice,
)
from alberta_framework.steps import Step3HordeConfig, Step4SARSAConfig


class _StringSubclass(str):
    pass


class _EvilStr(str):
    def __str__(self) -> str:  # pragma: no cover
        raise RuntimeError("str hook")

    def __repr__(self) -> str:  # pragma: no cover
        raise RuntimeError("repr hook")


class _HostileInt(int):
    def __index__(self) -> int:  # pragma: no cover
        raise AssertionError("index hook")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("repr hook")


class _HostileFloat(float):
    calls = 0

    def as_integer_ratio(self) -> tuple[int, int]:
        type(self).calls += 1
        raise RuntimeError("ratio hook")


class _HostileArray:
    calls = 0

    @property
    def __class__(self) -> type[np.ndarray]:
        type(self).calls += 1
        return np.ndarray

    @property
    def shape(self) -> tuple[int, ...]:  # pragma: no cover
        raise AssertionError("shape hook")

    @property
    def dtype(self) -> np.dtype:  # pragma: no cover
        raise AssertionError("dtype hook")


def _pipeline() -> AlbertaPipeline:
    return AlbertaPipeline(
        AlbertaPipelineConfig(
            features=Step2FeatureConfig.identity(3),
            horde=Step3HordeConfig(
                gammas=(0.0, 0.5),
                lamdas=(0.0, 0.0),
                hidden_sizes=(),
            ),
            control=Step4SARSAConfig(
                n_actions=2,
                hidden_sizes=(),
                epsilon_start=0.0,
                epsilon_end=0.0,
            ),
        )
    )


def _without_timing(state: Any) -> Any:
    horde_state = state.horde_state.replace(birth_timestamp=0.0, uptime_s=0.0)
    learner_state = state.control_state.learner_state.replace(
        birth_timestamp=0.0, uptime_s=0.0
    )
    return state.replace(
        horde_state=horde_state,
        control_state=state.control_state.replace(learner_state=learner_state),
    )


def test_require_bool_rejects_string_subclass_name() -> None:
    with pytest.raises(ValueError, match="exact string"):
        _require_bool(_StringSubclass("use_layer_norm"), True)
    with pytest.raises(ValueError, match="exact string"):
        _require_int(_StringSubclass("observation_dim"), 4)


def test_require_bool_does_not_invoke_hostile_name_repr() -> None:
    with pytest.raises(ValueError, match="exact string"):
        _require_bool(_EvilStr("include_raw"), True)
    with pytest.raises(ValueError, match="exact string"):
        _require_str_choice(_EvilStr("step2"), "upgd", ("upgd", "identity"))


def test_require_bool_rejects_non_bool_without_repr() -> None:
    with pytest.raises(ValueError, match="must be a bool"):
        _require_bool("include_raw", 1)
    with pytest.raises(ValueError, match="must be a bool"):
        _require_bool("include_raw", _StringSubclass("true"))


def test_require_int_rejects_bool_and_hostile() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        _require_int("observation_dim", True)
    with pytest.raises(ValueError, match="must be an integer"):
        _require_int("observation_dim", _HostileInt(4))


def test_require_int_does_not_invoke_hostile_repr() -> None:
    evil = _EvilStr("evil")
    # value is EvilStr but _require_int will reject before formatting it with !r
    with pytest.raises(ValueError, match="must be an integer"):
        _require_int("observation_dim", evil)


def test_require_str_choice_rejects_string_subclass() -> None:
    with pytest.raises(ValueError, match="unknown"):
        _require_str_choice("step2", _StringSubclass("upgd"), ("upgd", "identity"))
    with pytest.raises(ValueError, match="exact string"):
        _require_str_choice(_StringSubclass("step2"), "upgd", ("upgd",))


def test_step2feature_rejects_string_subclass_periods() -> None:
    with pytest.raises(ValueError, match="periods must be a tuple"):
        Step2FeatureConfig(periods=cast(Any, _StringSubclass("bad")))


def test_step2feature_does_not_invoke_hostile_periods_repr() -> None:
    # periods with hostile repr should not invoke it (no !r)
    with pytest.raises(ValueError, match="periods must be a tuple"):
        Step2FeatureConfig(periods=cast(Any, _EvilStr("bad")))


def test_step2upgd_rejects_string_subclass_hidden_sizes() -> None:
    with pytest.raises(ValueError, match="hidden_sizes must contain"):
        Step2UPGDConfig(hidden_sizes=cast(Any, _StringSubclass("bad")))
    with pytest.raises(ValueError, match="hidden_sizes must contain"):
        Step2UPGDConfig(hidden_sizes=cast(Any, _EvilStr("bad")))


def test_valid_configs_still_pass() -> None:
    cfg = Step2FeatureConfig(observation_dim=4, periods=(32.0, 64.0))
    assert cfg.observation_dim == 4
    cfg2 = Step2UPGDConfig(observation_dim=4, hidden_sizes=(8,), step_size=0.03)
    assert cfg2.observation_dim == 4
    assert _require_bool("flag", True) is True
    assert _require_int("n", cast(Any, np.int32(7))) == 7
    assert _require_str_choice("m", "a", ("a", "b")) == "a"


def test_numpy_int_and_domain_still_pass() -> None:
    # valid numpy int types still canonicalize
    assert _require_int("x", cast(Any, np.int64(5))) == 5
    cfg = Step2FeatureConfig(observation_dim=cast(Any, np.int32(3)), periods=())
    assert cfg.observation_dim == 3


def test_pipeline_rejects_hostile_array_before_class_or_metadata_hooks() -> None:
    _HostileArray.calls = 0
    with pytest.raises(TypeError, match="trusted array"):
        _pipeline().init(jr.key(0), cast(Any, _HostileArray()))
    assert _HostileArray.calls == 0


def test_pipeline_rejects_array_dtype_and_shape_before_jax_narrowing() -> None:
    pipeline = _pipeline()
    with jax.enable_x64():
        with pytest.raises(TypeError, match="dtype float32"):
            pipeline.init(jr.key(0), jnp.zeros(3, dtype=jnp.float64))
    with pytest.raises(ValueError, match="shape"):
        pipeline.init(jr.key(0), jnp.zeros((3, 1), dtype=jnp.float32))


@pytest.mark.parametrize(
    ("reward", "terminated"),
    [
        (jnp.asarray(jnp.inf, dtype=jnp.float32), jnp.asarray(0.0, dtype=jnp.float32)),
        (jnp.asarray(1.0, dtype=jnp.float32), jnp.asarray(0.5, dtype=jnp.float32)),
    ],
)
def test_pipeline_invalid_transition_is_atomic_eager_and_jit(
    reward: jax.Array, terminated: jax.Array
) -> None:
    pipeline = _pipeline()
    state = pipeline.init(jr.key(0), jnp.zeros(3, dtype=jnp.float32))
    observation = jnp.ones(3, dtype=jnp.float32)
    cumulants = jnp.ones(2, dtype=jnp.float32)

    eager = pipeline.update(state, observation, reward, terminated, cumulants)
    compiled_input = jax.jit(lambda current: current)(state)
    compiled = jax.jit(
        lambda current, r, done: pipeline.update(
            current, observation, r, done, cumulants
        )
    )(state, reward, terminated)

    for rejected, expected in ((eager, state), (compiled, compiled_input)):
        chex.assert_trees_all_equal(
            _without_timing(rejected.state), _without_timing(expected)
        )
        assert int(rejected.action) == -1
        assert int(rejected.state.step_count) == 0
        chex.assert_trees_all_equal(
            rejected.horde_predictions,
            jnp.zeros_like(rejected.horde_predictions),
        )


def test_pipeline_counter_saturates_and_scan_shapes_are_preflighted() -> None:
    pipeline = _pipeline()
    state = pipeline.init(jr.key(0), jnp.zeros(3, dtype=jnp.float32)).replace(
        step_count=jnp.asarray(2**31 - 1, dtype=jnp.int32)
    )
    accepted = pipeline.update(
        state,
        jnp.ones(3, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.ones(2, dtype=jnp.float32),
    )
    assert int(accepted.state.step_count) == 2**31 - 1

    with pytest.raises(ValueError, match="rewards must have shape"):
        pipeline.run_arrays(
            state,
            jnp.ones((2, 3), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
            jnp.zeros((2,), dtype=jnp.float32),
            jnp.ones((2, 2), dtype=jnp.float32),
        )


def test_pipeline_config_schema_and_record_types_are_exact() -> None:
    class DictSubclass(dict[str, object]):
        pass

    class FeatureSubclass(Step2FeatureConfig):
        pass

    with pytest.raises(ValueError, match="exact Step2FeatureConfig"):
        AlbertaPipelineConfig(features=cast(Any, FeatureSubclass()))
    with pytest.raises(ValueError, match="exact dictionary"):
        Step2FeatureConfig.from_dict(cast(Any, DictSubclass()))
    payload = Step2FeatureConfig().to_dict()
    payload["extra"] = 1
    with pytest.raises(ValueError, match="schema"):
        Step2FeatureConfig.from_dict(payload)
