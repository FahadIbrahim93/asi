"""Canonical JSON configs reject oversized width before the 1 MiB encoding walk."""

from __future__ import annotations

import time

import pytest

from alberta_framework.reference_agent import (
    _MAX_JSON_VALUES,
    _validate_json_value,
    canonical_config_sha256,
)


def test_frozen_json_value_bound_matches_strict_json_last_fit() -> None:
    assert _MAX_JSON_VALUES == 1_000_000


def test_last_fit_json_list_validates_without_width_hang() -> None:
    payload: dict[str, object] = {"k": [0] * (_MAX_JSON_VALUES - 2)}
    started = time.perf_counter()
    nodes = _validate_json_value(payload, path="config")
    assert time.perf_counter() - started < 0.5
    assert nodes == _MAX_JSON_VALUES


@pytest.mark.parametrize("count", [_MAX_JSON_VALUES, 2_000_000, 20_000_000])
def test_canonical_config_rejects_oversized_list_before_encoding(
    count: int,
) -> None:
    started = time.perf_counter()
    with pytest.raises(ValueError, match="JSON value resource limit"):
        canonical_config_sha256({"k": [0] * count})
    assert time.perf_counter() - started < 0.5


def test_canonical_config_rejects_pointer_repeated_nested_lists() -> None:
    row = [0] * 5_000
    payload: dict[str, object] = {"k": [row] * 5_000}
    started = time.perf_counter()
    with pytest.raises(ValueError, match="JSON value resource limit"):
        canonical_config_sha256(payload)
    assert time.perf_counter() - started < 0.5
