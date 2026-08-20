# mypy: disable-error-code="arg-type"
"""Periodic mask schedules reject oversized tuples before asarray/stack hang."""

from __future__ import annotations

import time

import numpy as np
import pytest

from alberta_framework.streams import partial_observation
from alberta_framework.streams.partial_observation import (
    _MAX_PERIODIC_SCHEDULE_LENGTH,
    _MAX_PERIODIC_SCHEDULE_VALUES,
    MaskMode,
    PartialObservationWrapper,
)
from alberta_framework.streams.synthetic import RandomWalkStream

pytestmark = pytest.mark.unit


def _inner() -> RandomWalkStream:
    return RandomWalkStream(feature_dim=2, drift_rate=0.0, noise_std=0.0)


def test_periodic_schedule_rejects_origin_hang_class_before_stack() -> None:
    row = np.array([True, False], dtype=bool)
    started = time.perf_counter()
    with pytest.raises(ValueError, match="periodic schedule length"):
        PartialObservationWrapper(
            _inner(),
            mode=MaskMode.PERIODIC,
            schedule=(row,) * (_MAX_PERIODIC_SCHEDULE_LENGTH + 1),
        )
    assert time.perf_counter() - started < 0.25


def test_periodic_schedule_rejects_pointer_repeat_origin_hang_n() -> None:
    row = np.array([True, False], dtype=bool)
    started = time.perf_counter()
    with pytest.raises(ValueError, match="periodic schedule length"):
        PartialObservationWrapper(
            _inner(),
            mode=MaskMode.PERIODIC,
            schedule=(row,) * 5_000,
        )
    assert time.perf_counter() - started < 0.25


def test_periodic_schedule_accepts_public_last_fit() -> None:
    row_a = np.array([True, False], dtype=bool)
    row_b = np.array([False, True], dtype=bool)
    wrapper = PartialObservationWrapper(
        _inner(),
        mode=MaskMode.PERIODIC,
        schedule=(row_a, row_b, row_a),
    )
    assert wrapper.mode is MaskMode.PERIODIC


def test_periodic_schedule_rejects_large_rows_before_array_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pointer-repeated row cannot amplify into a multi-gigabyte stack."""
    feature_dim = 1_000_000
    row = np.zeros((feature_dim,), dtype=bool)
    schedule_length = _MAX_PERIODIC_SCHEDULE_VALUES // feature_dim + 1

    def fail_asarray(*args: object, **kwargs: object) -> object:
        raise AssertionError("array conversion ran before the working-set gate")

    monkeypatch.setattr(partial_observation.jnp, "asarray", fail_asarray)
    with pytest.raises(ValueError, match="periodic schedule working set"):
        PartialObservationWrapper(
            RandomWalkStream(feature_dim=feature_dim, drift_rate=0.0, noise_std=0.0),
            mode=MaskMode.PERIODIC,
            schedule=(row,) * schedule_length,
        )


def test_periodic_schedule_rejects_hostile_container_before_len() -> None:
    class HostileSchedule(tuple[object, ...]):
        calls = 0

        def __len__(self) -> int:
            type(self).calls += 1
            raise AssertionError("hostile length")

    with pytest.raises(ValueError, match="exact tuple"):
        PartialObservationWrapper(
            _inner(), mode=MaskMode.PERIODIC, schedule=HostileSchedule()
        )
    assert HostileSchedule.calls == 0


def test_periodic_schedule_rejects_hostile_row_before_array_hooks() -> None:
    class HostileRow:
        calls = 0

        def __jax_array__(self) -> object:
            type(self).calls += 1
            raise AssertionError("hostile JAX conversion")

        def __array__(self, dtype: object = None) -> object:
            type(self).calls += 1
            raise AssertionError("hostile NumPy conversion")

    with pytest.raises(ValueError, match="exact NumPy or JAX array"):
        PartialObservationWrapper(
            _inner(), mode=MaskMode.PERIODIC, schedule=(HostileRow(),)
        )
    assert HostileRow.calls == 0


def test_fixed_mask_rejects_numeric_dtype_instead_of_boolean_laundering() -> None:
    with pytest.raises(ValueError, match="dtype bool"):
        PartialObservationWrapper(
            _inner(),
            mode=MaskMode.FIXED,
            fixed_mask=np.asarray([1, 0], dtype=np.int32),
        )
