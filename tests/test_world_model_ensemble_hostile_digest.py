"""Hostile string validation for world-model ensemble digest."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class _HostileStr(str):
    calls = 0
    __hash__ = str.__hash__

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile eq executed")

    def __ne__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile ne executed")


class _HostileDict(dict[str, object]):
    calls = 0

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile dict eq executed")

    def __ne__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile dict ne executed")


def test_ensemble_digest_rejects_hostile_before_eq() -> None:
    from alberta_framework.core.world_model_ensemble import _ensemble_config_digest

    config = {"ensemble": "test"}
    digest = _ensemble_config_digest(config)
    hostile = _HostileStr(digest)
    _HostileStr.calls = 0
    assert (type(hostile) is not str or hostile != digest) is True
    assert _HostileStr.calls == 0


def test_ensemble_checkpoint_rejects_hostile_digest() -> None:
    import tempfile
    from pathlib import Path
    from unittest.mock import patch

    from alberta_framework.core.world_model_ensemble import (
        WORLD_MODEL_ENSEMBLE_CHECKPOINT_SCHEMA,
        _ensemble_config_digest,
        load_world_model_ensemble_checkpoint,
    )

    config = {"ensemble": "test"}
    expected = _ensemble_config_digest(config)
    hostile = _HostileStr(expected)
    _HostileStr.calls = 0
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "chkpt"
        path.write_bytes(b"dummy")
        fake_meta = {
            "schema": WORLD_MODEL_ENSEMBLE_CHECKPOINT_SCHEMA,
            "ensemble_config": config,
            "config_sha256": hostile,  # type: ignore[dict-item]
        }
        with patch(
            "alberta_framework.core.world_model_ensemble.load_checkpoint_metadata",
            return_value=fake_meta,
        ):
            with pytest.raises(ValueError, match="config digest does not match"):
                load_world_model_ensemble_checkpoint(str(path))
        assert _HostileStr.calls == 0


def test_ensemble_checkpoint_rejects_hostile_schema(tmp_path: object) -> None:
    from pathlib import Path
    from unittest.mock import patch

    from alberta_framework.core import world_model_ensemble as world

    path = Path(str(tmp_path)) / "checkpoint"
    path.write_bytes(b"dummy")
    metadata = {"schema": _HostileStr(world.WORLD_MODEL_ENSEMBLE_CHECKPOINT_SCHEMA)}
    _HostileStr.calls = 0
    with patch.object(world, "load_checkpoint_metadata", return_value=metadata):
        with pytest.raises(ValueError, match="not a WorldModelEnsemble"):
            world.load_world_model_ensemble_checkpoint(path)
    assert _HostileStr.calls == 0


def test_ensemble_checkpoint_rejects_hostile_resource_metadata(tmp_path: object) -> None:
    from pathlib import Path
    from unittest.mock import Mock, patch

    from alberta_framework.core import world_model_ensemble as world

    config: dict[str, object] = {"ensemble": "test"}
    ensemble = Mock()
    ensemble.to_config.return_value = config
    ensemble.init.return_value = object()
    ensemble.resource_budget.return_value.to_config.return_value = {"bytes": 1}
    metadata = {
        "schema": world.WORLD_MODEL_ENSEMBLE_CHECKPOINT_SCHEMA,
        "ensemble_config": config,
        "config_sha256": world._ensemble_config_digest(config),
        "resource_budget": _HostileDict({"bytes": 1}),
    }
    path = Path(str(tmp_path)) / "checkpoint"
    path.write_bytes(b"dummy")
    _HostileDict.calls = 0
    with (
        patch.object(world, "load_checkpoint_metadata", return_value=metadata),
        patch.object(world.WorldModelEnsemble, "from_config", return_value=ensemble),
    ):
        with pytest.raises(ValueError, match="resource budget"):
            world.load_world_model_ensemble_checkpoint(path)
    assert _HostileDict.calls == 0


def test_ensemble_checkpoint_rejects_hostile_config_mapping(tmp_path: object) -> None:
    from pathlib import Path
    from unittest.mock import patch

    from alberta_framework.core import world_model_ensemble as world

    metadata = {
        "schema": world.WORLD_MODEL_ENSEMBLE_CHECKPOINT_SCHEMA,
        "ensemble_config": _HostileDict({"ensemble": "test"}),
    }
    path = Path(str(tmp_path)) / "checkpoint"
    path.write_bytes(b"dummy")
    _HostileDict.calls = 0
    with patch.object(world, "load_checkpoint_metadata", return_value=metadata):
        with pytest.raises(ValueError, match="missing ensemble_config"):
            world.load_world_model_ensemble_checkpoint(path)
    assert _HostileDict.calls == 0
