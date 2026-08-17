"""Hostile-safe validation for JAX seed boundaries."""

from __future__ import annotations

import json
from collections.abc import Iterator

import numpy as np
import pytest

from alberta_framework._seed_validation import (
    JAX_KEY_SEED_MAX,
    JAX_SEED_SEQUENCE_MAX_LENGTH,
    require_jax_seed,
    require_unique_jax_seeds,
)


class _HostileInt(int):
    def __index__(self) -> int:  # pragma: no cover
        raise AssertionError("index hook executed")

    def __repr__(self) -> str:  # pragma: no cover
        raise AssertionError("repr hook")


class _StringSubclass(str):
    pass


class _HostileList(list[int]):
    calls = 0

    def __iter__(self) -> Iterator[int]:
        type(self).calls += 1
        raise AssertionError("iter hook executed")


class _HostileTuple(tuple[int, ...]):
    calls = 0

    def __iter__(self) -> Iterator[int]:
        type(self).calls += 1
        raise AssertionError("iter hook executed")


def test_require_jax_seed_rejects_bool() -> None:
    with pytest.raises(ValueError, match="built-in integer"):
        require_jax_seed(True)
    with pytest.raises(ValueError, match="built-in integer"):
        require_jax_seed(False)


def test_require_jax_seed_rejects_hostile_int() -> None:
    with pytest.raises(ValueError, match="built-in integer"):
        require_jax_seed(_HostileInt(0))


@pytest.mark.parametrize(
    "value",
    [np.int64(0), np.uint32(0), np.bool_(False), 0.0, "0", None, -1, 2**32],
)
def test_require_jax_seed_rejects_every_non_builtin_or_out_of_domain_family(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="built-in integer.*uint32"):
        require_jax_seed(value)


def test_require_jax_seed_accepts_exact_uint32_endpoints() -> None:
    assert require_jax_seed(0) == 0
    assert require_jax_seed(JAX_KEY_SEED_MAX) == JAX_KEY_SEED_MAX


def test_require_jax_seed_rejects_string_subclass_name() -> None:
    with pytest.raises(ValueError, match="exact string"):
        require_jax_seed(0, name=_StringSubclass("seed"))


def test_require_jax_seed_does_not_invoke_hostile_name_repr() -> None:
    class EvilStr(str):
        def __repr__(self) -> str:  # pragma: no cover
            raise RuntimeError("repr hook")

        def __str__(self) -> str:  # pragma: no cover
            raise RuntimeError("str hook")

    with pytest.raises(ValueError, match="exact string"):
        require_jax_seed(0, name=EvilStr("seed"))


def test_require_unique_rejects_string_values() -> None:
    with pytest.raises(ValueError, match="exact list or tuple"):
        require_unique_jax_seeds("abc")


def test_require_unique_rejects_string_subclass_values() -> None:
    with pytest.raises(ValueError, match="exact list or tuple"):
        require_unique_jax_seeds(_StringSubclass("abc"))


def test_require_unique_rejects_hostile_list_subclass() -> None:
    _HostileList.calls = 0
    hostile = _HostileList([0, 1])
    with pytest.raises(ValueError, match="list or tuple"):
        require_unique_jax_seeds(hostile)
    assert _HostileList.calls == 0


def test_require_unique_rejects_hostile_tuple_subclass() -> None:
    _HostileTuple.calls = 0
    hostile = _HostileTuple((0, 1))
    with pytest.raises(ValueError, match="list or tuple"):
        require_unique_jax_seeds(hostile)
    assert _HostileTuple.calls == 0


def test_require_unique_rejects_range_sequence() -> None:
    with pytest.raises(ValueError, match="list or tuple"):
        require_unique_jax_seeds(range(3))


def test_require_unique_rejects_bytes_subclass() -> None:
    class BytesSubclass(bytes):
        pass

    with pytest.raises(ValueError, match="exact list or tuple"):
        require_unique_jax_seeds(BytesSubclass(b"ab"))


def test_require_unique_does_not_invoke_hostile_name_repr() -> None:
    class EvilStr(str):
        def __repr__(self) -> str:  # pragma: no cover
            raise RuntimeError("repr hook")

    evil = EvilStr("seeds")
    with pytest.raises(ValueError, match="exact string"):
        require_unique_jax_seeds([0, 1], name=evil)


def test_require_unique_rejects_string_subclass_name() -> None:
    with pytest.raises(ValueError, match="exact string"):
        require_unique_jax_seeds([0, 1], name=_StringSubclass("seeds"))


def test_require_unique_valid_list_passes() -> None:
    result = require_unique_jax_seeds([0, 1, JAX_KEY_SEED_MAX])
    assert result == (0, 1, JAX_KEY_SEED_MAX)


def test_require_unique_rejects_bool_element() -> None:
    with pytest.raises(ValueError, match="built-in integer"):
        require_unique_jax_seeds([0, True])


def test_require_unique_rejects_hostile_int_element() -> None:
    _HostileInt(1)  # ensure class works
    with pytest.raises(ValueError, match="built-in integer"):
        require_unique_jax_seeds([0, _HostileInt(1)])


def test_require_unique_preserves_order_and_canonical_tuple_identity() -> None:
    raw = (JAX_KEY_SEED_MAX, 0, 17)
    result = require_unique_jax_seeds(raw)
    assert result is raw
    assert result == (JAX_KEY_SEED_MAX, 0, 17)


def test_require_unique_canonicalizes_exact_list_without_mutating_it() -> None:
    raw = [7, 3, 11]
    result = require_unique_jax_seeds(raw)
    assert raw == [7, 3, 11]
    assert result == (7, 3, 11)
    assert all(type(seed) is int for seed in result)


def test_require_unique_json_roundtrip_preserves_seed_identities() -> None:
    seeds = require_unique_jax_seeds([0, 17, JAX_KEY_SEED_MAX])
    encoded = json.dumps(seeds, allow_nan=False)
    assert json.loads(encoded) == [0, 17, JAX_KEY_SEED_MAX]


def test_require_unique_rejects_oversized_exact_list_before_reading_elements() -> None:
    values: list[object] = [_HostileInt(0)] * (JAX_SEED_SEQUENCE_MAX_LENGTH + 1)
    with pytest.raises(ValueError, match="at most 4096"):
        require_unique_jax_seeds(values)


def test_require_unique_accepts_exact_ceiling_without_sorting() -> None:
    values = list(range(JAX_SEED_SEQUENCE_MAX_LENGTH - 1, -1, -1))
    result = require_unique_jax_seeds(values)
    assert len(result) == JAX_SEED_SEQUENCE_MAX_LENGTH
    assert result[0] == JAX_SEED_SEQUENCE_MAX_LENGTH - 1
    assert result[-1] == 0


def test_require_unique_rejects_first_duplicate_without_touching_later_hostile() -> None:
    with pytest.raises(ValueError, match="unique"):
        require_unique_jax_seeds([5, 5, _HostileInt(1)])
