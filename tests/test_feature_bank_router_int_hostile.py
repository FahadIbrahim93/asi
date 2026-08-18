"""Hostile integer validation for feature bank router."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.unit


class _HostileInt(int):
    calls = 0

    def __lt__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile lt")

    def __ge__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile ge")

    def __hash__(self) -> int:
        type(self).calls += 1
        raise AssertionError("hostile hash")

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile eq")

    def __index__(self) -> int:
        type(self).calls += 1
        raise AssertionError("hostile index")

    def __repr__(self) -> str:
        type(self).calls += 1
        raise AssertionError("hostile repr")


def test_feature_axis_rejects_hostile_before_lt() -> None:
    from alberta_framework.core.feature_bank_router import (
        FeatureBankRouter,
        FeatureBankRouterConfig,
    )

    config = FeatureBankRouterConfig(base_dim=2, active_slots=1)
    router = FeatureBankRouter(config)
    hostile = _HostileInt(0)
    _HostileInt.calls = 0
    consumers = [jnp.ones((2, 3))]
    with pytest.raises(TypeError, match="must be an integer"):
        router._consumer_layout(consumers, [hostile])  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    with pytest.raises(TypeError, match="must be an integer"):
        router._consumer_layout(consumers, [True])  # type: ignore[arg-type]
    # valid
    arrays, _, axes = router._consumer_layout(consumers, [1])
    assert axes == (1,)
    # also negative axis allowed via ndim calc after type check, but still int
    arrays, _, axes = router._consumer_layout(consumers, [-1])
    assert axes == (1,)


def test_config_and_resource_counts_reject_hostile_before_runtime_hooks() -> None:
    from alberta_framework.core.feature_bank_router import (
        FeatureBankRouterConfig,
        _require_host_count,
    )

    hostile = _HostileInt(2)
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="base_dim must be an integer"):
        FeatureBankRouterConfig(base_dim=hostile, active_slots=1)  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    with pytest.raises(ValueError, match="consumer_total_scalars must be an integer"):
        _require_host_count("consumer_total_scalars", hostile)
    assert _HostileInt.calls == 0
