"""Canonical JSON configs reject oversized width before the 1 MiB encoding walk."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import pytest

from alberta_framework.reference_agent import (
    _MAX_JSON_VALUES,
    _validate_json_value,
    canonical_config_sha256,
)


def test_frozen_json_value_bound_matches_strict_json_last_fit() -> None:
    assert _MAX_JSON_VALUES == 1_000_000


class _HookList(list[object]):
    calls = 0

    def __len__(self) -> int:
        type(self).calls += 1
        raise AssertionError("hostile list hook ran")


class _HookMapping(Mapping[str, object]):
    calls = 0

    def __getitem__(self, key: str) -> object:
        type(self).calls += 1
        raise AssertionError("hostile mapping hook ran")

    def __iter__(self) -> Iterator[str]:
        type(self).calls += 1
        raise AssertionError("hostile mapping hook ran")

    def __len__(self) -> int:
        type(self).calls += 1
        raise AssertionError("hostile mapping hook ran")


def test_small_json_tree_reports_exact_node_count() -> None:
    payload: dict[str, object] = {"k": [0, True, None]}
    nodes = _validate_json_value(payload, path="config")
    assert nodes == 5


def test_oversized_list_rejects_from_length_before_child_traversal() -> None:
    payload = {"k": [object()] * 600_000}
    with pytest.raises(ValueError, match="encoding work limit"):
        canonical_config_sha256(payload)


def test_canonical_config_rejects_pointer_repeated_nested_lists() -> None:
    row = [0] * 2_048
    payload: dict[str, object] = {"k": [row] * 512}
    with pytest.raises(ValueError, match="encoding work limit"):
        canonical_config_sha256(payload)


def test_hostile_container_aliases_reject_without_hooks() -> None:
    hostile_list = _HookList()
    _HookList.calls = 0
    with pytest.raises(ValueError, match="not a canonical JSON value"):
        _validate_json_value(hostile_list, path="config")
    assert _HookList.calls == 0

    hostile_mapping = _HookMapping()
    _HookMapping.calls = 0
    with pytest.raises(ValueError, match="not a canonical JSON value"):
        _validate_json_value(hostile_mapping, path="config")
    assert _HookMapping.calls == 0
    with pytest.raises(ValueError, match="JSON mapping"):
        canonical_config_sha256(hostile_mapping)
    assert _HookMapping.calls == 0


def test_aggregate_text_escaping_and_integer_bounds_fail_closed() -> None:
    shared = "a" * 600_000
    with pytest.raises(ValueError, match="aggregate UTF-8"):
        canonical_config_sha256({"k": [shared, shared]})
    with pytest.raises(ValueError, match="encoding work limit"):
        canonical_config_sha256({"k": "\x00" * 200_000})
    with pytest.raises(ValueError, match="signed 64-bit"):
        canonical_config_sha256({"k": 1 << 63})
