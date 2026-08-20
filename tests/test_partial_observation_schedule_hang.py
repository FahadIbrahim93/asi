# mypy: disable-error-code="arg-type"
"""Resource and type boundaries for partial-observation masks."""

from __future__ import annotations

import numpy as np
import pytest

from alberta_framework.streams import partial_observation
from alberta_framework.streams.partial_observation import (
    _MAX_PERIODIC_SCHEDULE_LENGTH,
    MaskMode,
    PartialObservationWrapper,
)
from alberta_framework.streams.synthetic import RandomWalkStream

pytestmark = pytest.mark.unit


def _inner() -> RandomWalkStream:
    return RandomWalkStream(feature_dim=2, drift_rate=0.0, noise_std=0.0)


def test_periodic_schedule_rejects_oversized_tuple_before_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(partial_observation, "_MAX_PERIODIC_SCHEDULE_LENGTH", 8)

    def fail_asarray(*args: object, **kwargs: object) -> object:
        raise AssertionError("conversion ran before the schedule-length gate")

    monkeypatch.setattr(partial_observation.jnp, "asarray", fail_asarray)
    row = np.array([True, False], dtype=bool)
    with pytest.raises(ValueError, match="periodic schedule length"):
        PartialObservationWrapper(
            _inner(),
            mode=MaskMode.PERIODIC,
            schedule=(row,) * 9,
        )


def test_periodic_schedule_maximum_uses_one_array_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = np.array([True, False], dtype=bool)
    original_asarray = partial_observation.jnp.asarray
    calls = 0

    def counted_asarray(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original_asarray(*args, **kwargs)

    monkeypatch.setattr(partial_observation.jnp, "asarray", counted_asarray)
    wrapper = PartialObservationWrapper(
        _inner(),
        mode=MaskMode.PERIODIC,
        schedule=(row,) * _MAX_PERIODIC_SCHEDULE_LENGTH,
    )
    assert wrapper.mode is MaskMode.PERIODIC
    assert calls == 1


def test_periodic_schedule_rejects_total_values_before_array_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(partial_observation, "_MAX_PERIODIC_SCHEDULE_VALUES", 16)
    row = np.zeros((3,), dtype=bool)

    def fail_asarray(*args: object, **kwargs: object) -> object:
        raise AssertionError("array conversion ran before the working-set gate")

    monkeypatch.setattr(partial_observation.jnp, "asarray", fail_asarray)
    with pytest.raises(ValueError, match="periodic schedule working set"):
        PartialObservationWrapper(
            RandomWalkStream(feature_dim=3, drift_rate=0.0, noise_std=0.0),
            mode=MaskMode.PERIODIC,
            schedule=(row,) * 6,
        )


def test_periodic_schedule_rejects_wrong_large_row_before_array_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A huge caller-owned row is checked by metadata before conversion."""
    row = np.zeros((1_000_000,), dtype=bool)

    def fail_asarray(*args: object, **kwargs: object) -> object:
        raise AssertionError("array conversion ran before the row-shape gate")

    monkeypatch.setattr(partial_observation.jnp, "asarray", fail_asarray)
    with pytest.raises(ValueError, match="schedule masks"):
        PartialObservationWrapper(
            _inner(),
            mode=MaskMode.PERIODIC,
            schedule=(row,),
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


def test_periodic_schedule_rejects_hostile_feature_dim_before_arithmetic() -> None:
    class HostileInt(int):
        calls = 0

        def __mul__(self, other: object) -> int:
            type(self).calls += 1
            raise AssertionError("untrusted multiplication hook executed")

        def __le__(self, other: object) -> bool:
            type(self).calls += 1
            raise AssertionError("untrusted comparison hook executed")

    inner = _inner()
    object.__setattr__(inner, "_feature_dim", HostileInt(2))
    with pytest.raises(ValueError, match="inner.feature_dim"):
        PartialObservationWrapper(
            inner,
            mode=MaskMode.PERIODIC,
            schedule=(np.array([True, False], dtype=bool),),
        )
    assert HostileInt.calls == 0


def test_periodic_schedule_rejects_numeric_dtype_without_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_asarray(*args: object, **kwargs: object) -> object:
        raise AssertionError("conversion ran before dtype validation")

    monkeypatch.setattr(partial_observation.jnp, "asarray", fail_asarray)
    with pytest.raises(ValueError, match="dtype bool"):
        PartialObservationWrapper(
            _inner(),
            mode=MaskMode.PERIODIC,
            schedule=(np.asarray([1, 0], dtype=np.int32),),
        )


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
