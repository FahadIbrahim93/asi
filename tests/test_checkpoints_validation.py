"""Validation hardening for checkpoints (path + metadata)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from alberta_framework import LinearLearner, save_checkpoint
from alberta_framework.core.checkpoints import (
    checkpoint_exists,
    load_checkpoint,
    load_checkpoint_metadata,
)


def _learner_state():
    learner = LinearLearner()
    return learner.init(feature_dim=3)


class _StringSubclass(str):
    pass


class _HostilePath(Path):
    def __str__(self) -> str:  # pragma: no cover
        raise AssertionError("str hook executed")

    def __fspath__(self) -> str:  # pragma: no cover
        raise AssertionError("fspath hook executed")


class _RaisingRepr:
    def __repr__(self) -> str:  # pragma: no cover
        raise RuntimeError("repr hook")


class _HostileMapping(Mapping[str, Any]):  # type: ignore[type-arg]
    def __getitem__(self, key: str) -> Any:
        raise RuntimeError("hook")

    def __iter__(self):  # type: ignore[no-untyped-def]
        raise RuntimeError("iter hook")

    def __len__(self) -> int:
        return 0


class _HostileInner(Mapping[str, Any]):  # type: ignore[type-arg]
    def __getitem__(self, key: str) -> Any:
        raise RuntimeError("hook")

    def __iter__(self):  # type: ignore[no-untyped-def]
        raise RuntimeError("hook")

    def __len__(self) -> int:
        return 1


def test_save_rejects_string_subclass_path(tmp_path: Path) -> None:
    state = _learner_state()
    with pytest.raises(ValueError, match="exact str or Path"):
        save_checkpoint(state, _StringSubclass(str(tmp_path / "ckpt")))


def test_save_rejects_hostile_path_without_hook(tmp_path: Path) -> None:
    state = _learner_state()
    hostile = object.__new__(_HostilePath)
    # Make it look like a Path but keep module __main__ so it is rejected
    # without invoking __str__/__fspath__
    with pytest.raises(ValueError, match="exact str or Path"):
        save_checkpoint(state, hostile)  # type: ignore[arg-type]


def test_save_rejects_hostile_repr_path(tmp_path: Path) -> None:
    state = _learner_state()
    with pytest.raises(ValueError, match="exact str or Path"):
        save_checkpoint(state, _RaisingRepr())  # type: ignore[arg-type]


def test_save_rejects_hostile_mapping_metadata(tmp_path: Path) -> None:
    state = _learner_state()
    with pytest.raises(ValueError, match="mapping"):
        save_checkpoint(state, tmp_path / "ckpt", metadata=_HostileMapping())  # type: ignore[arg-type]


def test_save_rejects_string_subclass_metadata_key(tmp_path: Path) -> None:
    state = _learner_state()
    hostile: dict[Any, Any] = {_StringSubclass("epoch"): 1}
    with pytest.raises(ValueError, match="exact strings"):
        save_checkpoint(state, tmp_path / "ckpt", metadata=hostile)


def test_save_rejects_hostile_inner_mapping_key(tmp_path: Path) -> None:
    state = _learner_state()
    # MappingProxy is ok, but hostile inner should be caught via exact keys
    # Use a normal dict with one hostile key subclass
    good = {"epoch": 1}
    # Also test that MappingProxy preserves
    proxy = MappingProxyType(good)
    save_checkpoint(state, tmp_path / "proxy_ok", metadata=proxy)
    _, meta = load_checkpoint(state, tmp_path / "proxy_ok")
    assert meta == good


def test_save_preserves_mapping_proxy(tmp_path: Path) -> None:
    state = _learner_state()
    payload = {"epoch": 7, "ok": True}
    proxy = MappingProxyType(payload)
    save_checkpoint(state, tmp_path / "proxy", metadata=proxy)
    _, loaded = load_checkpoint(state, tmp_path / "proxy")
    assert loaded == payload


def test_load_rejects_hostile_path(tmp_path: Path) -> None:
    state = _learner_state()
    save_checkpoint(state, tmp_path / "good", metadata={"epoch": 1})
    hostile = object.__new__(_HostilePath)
    with pytest.raises(ValueError, match="exact str or Path"):
        load_checkpoint(state, hostile)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exact str or Path"):
        load_checkpoint_metadata(hostile)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exact str or Path"):
        checkpoint_exists(hostile)  # type: ignore[arg-type]


def test_checkpoint_exists_rejects_string_subclass(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exact str or Path"):
        checkpoint_exists(_StringSubclass(str(tmp_path / "x")))  # type: ignore[arg-type]


def test_save_rejects_nonfinite_metadata(tmp_path: Path) -> None:
    state = _learner_state()
    with pytest.raises(ValueError, match="JSON-safe"):
        save_checkpoint(state, tmp_path / "nan", metadata={"loss": float("nan")})


def test_roundtrip_with_exact_types(tmp_path: Path) -> None:
    state = _learner_state()
    # str path
    save_checkpoint(state, str(tmp_path / "str_ckpt"), metadata={"epoch": 2})
    loaded, meta = load_checkpoint(state, str(tmp_path / "str_ckpt"))
    assert meta == {"epoch": 2}
    # Path path
    save_checkpoint(state, tmp_path / "path_ckpt", metadata={"epoch": 3})
    loaded2, meta2 = load_checkpoint(state, tmp_path / "path_ckpt")
    assert meta2 == {"epoch": 3}
    assert checkpoint_exists(tmp_path / "path_ckpt")
    assert checkpoint_exists(str(tmp_path / "path_ckpt"))


def test_checkpoint_exists_true_after_save(tmp_path: Path) -> None:
    state = _learner_state()
    p = tmp_path / "exists"
    assert not checkpoint_exists(p)
    save_checkpoint(state, p)
    assert checkpoint_exists(p)
