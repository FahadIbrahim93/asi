"""Total public-boundary tests for the production Step 1 facade."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import pytest

from alberta_framework.steps.step1 import (
    Step1KernelConfig,
    Step1SmokeResult,
    make_step1_learner,
    make_step1_optimizer,
    make_step1_stream,
    run_step1_smoke,
)


class _HostileDict(dict[object, object]):
    calls = 0

    def __iter__(self) -> Iterator[object]:
        type(self).calls += 1
        raise AssertionError("mapping hook must not run")


class _HostileConfig(Step1KernelConfig):
    calls = 0

    def __getattribute__(self, name: str) -> Any:
        type(self).calls += 1
        raise AssertionError(f"config hook must not run: {name}")

    def __bool__(self) -> bool:
        type(self).calls += 1
        raise AssertionError("truthiness hook must not run")


def _hostile_config() -> object:
    return object.__new__(_HostileConfig)


def _result(**overrides: object) -> Step1SmokeResult:
    payload: dict[str, object] = {
        "config": Step1KernelConfig(),
        "steps": 4,
        "seed": 0,
        "final_window_mse": 0.25,
        "metrics_shape": (4, 4),
        "finite": True,
    }
    payload.update(overrides)
    return Step1SmokeResult(**cast(Any, payload))


def test_config_parser_requires_exact_complete_schema_without_mapping_hooks() -> None:
    payload = Step1KernelConfig().to_dict()
    missing = {key: value for key, value in payload.items() if key != "stream"}
    for bad in ({**payload, "extra": 1}, missing):
        with pytest.raises(ValueError, match="schema"):
            Step1KernelConfig.from_dict(cast(Any, bad))

    hostile = _HostileDict(payload)
    _HostileDict.calls = 0
    with pytest.raises(ValueError, match="exact dictionary"):
        Step1KernelConfig.from_dict(cast(Any, hostile))
    assert _HostileDict.calls == 0


@pytest.mark.parametrize(
    "factory",
    [make_step1_optimizer, make_step1_stream, make_step1_learner],
)
def test_factories_reject_config_subclasses_before_attribute_hooks(factory: Any) -> None:
    _HostileConfig.calls = 0
    with pytest.raises(TypeError, match="exact Step1KernelConfig"):
        factory(cast(Any, _hostile_config()))
    assert _HostileConfig.calls == 0


def test_smoke_rejects_config_subclass_without_truthiness_or_attribute_hooks() -> None:
    _HostileConfig.calls = 0
    with pytest.raises(TypeError, match="exact Step1KernelConfig"):
        run_step1_smoke(cast(Any, _hostile_config()), steps=1, final_window=1)
    assert _HostileConfig.calls == 0


def test_selected_state_and_smoke_outputs_are_preflighted_before_jax_allocation() -> None:
    huge = Step1KernelConfig(
        feature_dim=2**31 - 1,
        num_relevant=1,
        optimizer="lms",
        normalizer="none",
    )
    with pytest.raises(ValueError, match="resource"):
        make_step1_learner(huge)
    with pytest.raises(ValueError, match="resource"):
        make_step1_stream(huge)
    with pytest.raises(ValueError, match="resource"):
        run_step1_smoke(steps=2**31 - 1)


def test_smoke_record_requires_exact_config_metric_schema_and_nonnegative_mse() -> None:
    with pytest.raises(TypeError, match="exact Step1KernelConfig"):
        _result(config=cast(Any, _hostile_config()))
    with pytest.raises(ValueError, match="metrics_shape"):
        _result(metrics_shape=(4, 3))
    with pytest.raises(ValueError, match="final_window_mse"):
        _result(final_window_mse=-0.25)

    without_normalizer = _result(
        config=Step1KernelConfig(normalizer="none"), metrics_shape=(4, 3)
    )
    assert without_normalizer.metrics_shape == (4, 3)
