"""List ceilings for history-feature decay rates and channel indices.

Origin enumerated every decay/channel before the INT32 feature_dim product.
A cheap pointer-repeat still walks the validator; public last-fit is 4096.
"""

from __future__ import annotations

import time

import pytest

from alberta_framework.core.history_features import (
    _MAX_HISTORY_CHANNELS,
    _MAX_HISTORY_DECAY_RATES,
    HistoryFeatureExtractor,
)


def test_documented_protocol_ceilings() -> None:
    assert _MAX_HISTORY_DECAY_RATES == 4096
    assert _MAX_HISTORY_CHANNELS == 4096


def test_last_fit_decay_count_is_accepted() -> None:
    extractor = HistoryFeatureExtractor(
        raw_dim=1,
        decay_rates=(0.5,) * _MAX_HISTORY_DECAY_RATES,
        channels=(0,),
        include_raw=False,
    )
    assert len(extractor.decay_rates) == _MAX_HISTORY_DECAY_RATES


def test_last_fit_channel_count_is_accepted() -> None:
    extractor = HistoryFeatureExtractor(
        raw_dim=_MAX_HISTORY_CHANNELS,
        decay_rates=(0.5,),
        channels=tuple(range(_MAX_HISTORY_CHANNELS)),
        include_raw=False,
    )
    assert extractor.channels is not None
    assert len(extractor.channels) == _MAX_HISTORY_CHANNELS


@pytest.mark.parametrize("count", [4097, 400_000])
def test_rejects_oversized_decay_rates_before_per_rate_walk(count: int) -> None:
    started = time.perf_counter()
    with pytest.raises(ValueError, match="at most 4096"):
        HistoryFeatureExtractor(raw_dim=1, decay_rates=(0.5,) * count)
    assert time.perf_counter() - started < 0.5


@pytest.mark.parametrize("count", [4097, 400_000])
def test_rejects_oversized_channels_before_per_index_walk(count: int) -> None:
    started = time.perf_counter()
    with pytest.raises(ValueError, match="at most 4096"):
        HistoryFeatureExtractor(
            raw_dim=1,
            decay_rates=(0.5,),
            channels=(0,) * count,
        )
    assert time.perf_counter() - started < 0.5


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("decay_rates", [0.5] * 4097),
        ("channels", [0] * 4097),
    ],
)
def test_from_config_rejects_oversized_lists_before_tuple_copy(
    field: str, value: list[object]
) -> None:
    config = HistoryFeatureExtractor(raw_dim=1).to_config()
    config[field] = value
    with pytest.raises(ValueError, match="at most 4096"):
        HistoryFeatureExtractor.from_config(config)
