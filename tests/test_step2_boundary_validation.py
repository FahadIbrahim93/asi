"""Exact schema, resource, and runtime tests for the Step 2 facade."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.steps.step2 import (
    Step2HybridConfig,
    Step2KernelConfig,
    Step2MemoryConfig,
    Step2StrictDigitReadoutConfig,
    Step2TemporalContextConfig,
    collect_step2_arrays,
    make_step2_learner,
    make_step2_stream,
)


@pytest.mark.parametrize(
    "config",
    [
        Step2KernelConfig(),
        Step2StrictDigitReadoutConfig(),
        Step2MemoryConfig(),
        Step2HybridConfig(),
        Step2TemporalContextConfig(),
    ],
)
def test_serialized_configs_require_exact_complete_json_schema(config: Any) -> None:
    payload = config.to_dict()

    class DictSubclass(dict[str, object]):
        pass

    with pytest.raises(ValueError, match="exact dict"):
        type(config).from_dict(DictSubclass(payload))
    first = next(iter(payload))
    with pytest.raises(ValueError, match="keys"):
        type(config).from_dict({key: value for key, value in payload.items() if key != first})
    with pytest.raises(ValueError, match="keys"):
        type(config).from_dict({**payload, "extra": 1})


@pytest.mark.parametrize(
    "mutation",
    [
        {"feature_dim": np.int32(4)},
        {"hidden_sizes": (8,)},
        {"upgd_step_size": np.float32(0.03)},
        {"readout_mode": np.str_("softmax_ce")},
    ],
)
def test_hybrid_serialized_fields_are_json_exact(mutation: dict[str, object]) -> None:
    payload = Step2HybridConfig(feature_dim=4, hidden_sizes=(8,)).to_dict()
    payload.update(mutation)
    with pytest.raises(ValueError, match="serialized"):
        Step2HybridConfig.from_dict(payload)


def test_hybrid_direct_config_is_total_and_canonical() -> None:
    config = Step2HybridConfig(
        feature_dim=np.int32(4),
        n_heads=np.int64(2),
        hidden_sizes=(np.int32(8),),
        upgd_step_size=np.float32(0.03),
    )
    assert type(config.feature_dim) is int
    assert type(config.upgd_step_size) is float
    assert config.hidden_sizes == (8,)

    with pytest.raises(ValueError, match="feature_dim"):
        Step2HybridConfig(feature_dim=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="upgd_step_size"):
        Step2HybridConfig(upgd_step_size=float("nan"))


def test_temporal_config_validates_derived_output_bytes() -> None:
    last_feature_dim = ((2**31 - 1) // 4 - 2) // 3
    Step2TemporalContextConfig(
        feature_dim=last_feature_dim,
        hidden_sizes=(),
        periods=(1.0,),
    )
    with pytest.raises(ValueError, match="temporal feature bytes"):
        Step2TemporalContextConfig(
            feature_dim=last_feature_dim + 1,
            hidden_sizes=(),
            periods=(1.0,),
        )


def test_factory_rejects_hostile_config_without_truthiness_hook() -> None:
    class Hostile:
        calls = 0

        def __bool__(self) -> bool:
            type(self).calls += 1
            raise AssertionError("truthiness hook executed")

    hostile = Hostile()
    with pytest.raises(ValueError, match="config must be"):
        make_step2_learner(hostile)  # type: ignore[arg-type]
    assert Hostile.calls == 0


def test_collector_rejects_hostile_stream_and_legacy_key_without_hooks() -> None:
    class Hostile:
        calls = 0

        def __getattribute__(self, name: str) -> Any:
            if name == "calls":
                return object.__getattribute__(self, name)
            type(self).calls += 1
            raise AssertionError("attribute hook executed")

    hostile = Hostile()
    with pytest.raises(ValueError, match="stream must be"):
        collect_step2_arrays(hostile, steps=1, key=jax.random.key(0))
    assert Hostile.calls == 0

    stream = make_step2_stream(Step2KernelConfig(feature_dim=3))
    with pytest.raises(ValueError, match="typed JAX PRNG key"):
        collect_step2_arrays(
            stream,
            steps=1,
            key=jnp.asarray([0, 0], dtype=jnp.uint32),
        )


def test_collector_preflights_output_bytes_before_stream_init() -> None:
    stream = make_step2_stream(Step2KernelConfig(feature_dim=3, n_heads=1))
    stream._feature_dim = 2**31 - 1  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="collection bytes"):
        collect_step2_arrays(stream, steps=1, key=jax.random.key(0))
