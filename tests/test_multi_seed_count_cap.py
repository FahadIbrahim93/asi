"""Protocol ceilings for public multi-seed experiment counts.

Documented last-fit in tests is seeds=3. Origin handed unbounded counts to
list(range(seeds)) with no last-fit reject — hang, not leftover INT32 math.
"""

from __future__ import annotations

import numpy as np
import pytest

from alberta_framework.utils.experiments import (
    _MULTI_SEED_MAX_COUNT,
    ExperimentConfig,
    run_multi_seed_experiment,
)


class _HostileInt(int):
    def __index__(self) -> int:  # pragma: no cover
        raise AssertionError("index hook executed")

    def __int__(self) -> int:  # pragma: no cover
        raise AssertionError("int hook executed")


def test_documented_protocol_ceiling() -> None:
    assert _MULTI_SEED_MAX_COUNT == 4096


@pytest.mark.parametrize("value", [0, -1, 10**12, 2**31 - 1, 10_001])
def test_rejects_oversized_or_non_positive_seed_counts(value: object) -> None:
    with pytest.raises(ValueError, match="seeds count must be"):
        run_multi_seed_experiment([], seeds=value)  # type: ignore[arg-type]


def test_rejects_numpy_and_subclass_seed_counts_without_index_hooks() -> None:
    with pytest.raises(ValueError, match="seeds"):
        run_multi_seed_experiment([], seeds=np.int64(10))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seeds"):
        run_multi_seed_experiment([], seeds=_HostileInt(10))  # type: ignore[arg-type]


def test_rejects_oversized_explicit_sequence_before_seed_iteration() -> None:
    with pytest.raises(ValueError, match="seeds length must be an integer"):
        run_multi_seed_experiment([], seeds=list(range(4097)))


def test_rejects_oversized_config_seed_product_before_factories() -> None:
    def fail() -> None:
        raise AssertionError("factory executed")

    configs = [
        ExperimentConfig(f"config-{index}", fail, fail, 1)  # type: ignore[arg-type]
        for index in range(2)
    ]
    with pytest.raises(ValueError, match="step-units exceed"):
        run_multi_seed_experiment(configs, seeds=3000, parallel=False)
