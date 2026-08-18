"""Hostile validation for streams __getattr__ facade."""

import pytest

import alberta_framework.streams as streams


class _EvilStr(str):
    calls = 0

    def __str__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("EvilStr.__str__ must not be called")

    def __repr__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("EvilStr.__repr__ must not be called")


class _StringSubclass(str):
    pass


def test_hostile_name_without_repr_leak() -> None:
    evil = _EvilStr("bad_attr")
    _EvilStr.calls = 0
    with pytest.raises(AttributeError, match="must be a string") as exc:
        getattr(streams, evil)  # type: ignore[arg-type]
    assert _EvilStr.calls == 0
    assert "EvilStr" not in str(exc.value)
    assert "!r" not in str(exc.value)


def test_string_subclass_rejected() -> None:
    with pytest.raises(AttributeError, match="must be a string"):
        getattr(streams, _StringSubclass("bad_attr"))  # type: ignore[arg-type]


def test_unknown_attribute_sanitized_without_repr() -> None:
    with pytest.raises(AttributeError, match="has no attribute") as exc:
        getattr(streams, "unknown_bad_attr")
    assert "!r" not in str(exc.value)
    assert "unknown_bad_attr" in str(exc.value)
    assert "'" in str(exc.value)


def test_known_lazy_attribute_still_loads() -> None:
    # Should not raise for known export, should lazy-load gymnasium
    # Check one known export exists in _GYMNASIUM_EXPORTS

    assert hasattr(streams, "ScanStream")
    # unknown should still raise
    with pytest.raises(AttributeError):
        getattr(streams, "TotallyUnknown123")
