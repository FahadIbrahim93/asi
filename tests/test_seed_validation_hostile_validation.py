"""Hostile-safe validation for JAX seed boundaries."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from alberta_framework._seed_validation import (
    JAX_KEY_SEED_MAX,
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
    with pytest.raises(ValueError, match="non-string sequence"):
        require_unique_jax_seeds("abc")


def test_require_unique_rejects_string_subclass_values() -> None:
    with pytest.raises(ValueError, match="non-string sequence"):
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

    with pytest.raises(ValueError, match="non-string sequence"):
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
