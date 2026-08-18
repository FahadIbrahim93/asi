"""Hostile string validation for prototype checkpoint digest."""

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

    def __str__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("hostile str executed")

    def __repr__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("hostile repr executed")


class _HostileDict(dict[str, object]):
    calls = 0

    def items(self):  # type: ignore[no-untyped-def, override]
        type(self).calls += 1
        raise AssertionError("hostile items executed")


def test_prototype_checkpoint_rejects_hostile_digest_without_dispatch() -> None:
    import tempfile
    from pathlib import Path
    from unittest.mock import patch

    from alberta_framework.core.prototype_agent import (
        PROTOTYPE_CHECKPOINT_SCHEMA,
        _prototype_config_digest,
        load_prototype_checkpoint,
    )

    real_config = {"prototype": "minimal"}
    expected = _prototype_config_digest(real_config)
    hostile = _HostileStr(expected)
    _HostileStr.calls = 0

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "chkpt"
        path.write_bytes(b"dummy")

        fake_metadata = {
            "schema": PROTOTYPE_CHECKPOINT_SCHEMA,
            "agent_config": real_config,
            "config_sha256": hostile,  # type: ignore[dict-item]
        }

        with patch(
            "alberta_framework.core.prototype_agent.load_checkpoint_metadata",
            return_value=fake_metadata,
        ):
            with pytest.raises(ValueError, match="config digest does not match"):
                load_prototype_checkpoint(str(path))
        assert _HostileStr.calls == 0


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("schema", "checkpoint is not an Alberta PrototypeAgent"),
        ("empty_array_codec", "unknown empty-array codec"),
        ("prng_impl", "unsupported PRNG implementation"),
    ],
)
def test_prototype_checkpoint_rejects_other_hostile_metadata_strings(
    tmp_path: object,
    field: str,
    message: str,
) -> None:
    from pathlib import Path
    from unittest.mock import patch

    from alberta_framework.core import prototype_agent as prototype

    path = Path(str(tmp_path)) / "checkpoint"
    path.write_bytes(b"dummy")
    metadata = {
        "schema": prototype.PROTOTYPE_CHECKPOINT_SCHEMA,
        "agent_config": {"prototype": "minimal"},
        "config_sha256": "0" * 64,
        "empty_array_codec": None,
        "prng_impl": None,
    }
    valid_value = {
        "schema": prototype.PROTOTYPE_CHECKPOINT_SCHEMA,
        "empty_array_codec": prototype._PROTOTYPE_EMPTY_ARRAY_CODEC,
        "prng_impl": next(iter(prototype._PROTOTYPE_SUPPORTED_PRNG_IMPLS)),
    }[field]
    metadata[field] = _HostileStr(valid_value)
    _HostileStr.calls = 0
    with patch.object(prototype, "load_checkpoint_metadata", return_value=metadata):
        with pytest.raises(ValueError, match=message):
            prototype.load_prototype_checkpoint(path)
    assert _HostileStr.calls == 0


def test_prototype_checkpoint_rejects_hostile_config_mapping(tmp_path: object) -> None:
    from pathlib import Path
    from unittest.mock import patch

    from alberta_framework.core import prototype_agent as prototype

    path = Path(str(tmp_path)) / "checkpoint"
    path.write_bytes(b"dummy")
    metadata = {
        "schema": prototype.PROTOTYPE_CHECKPOINT_SCHEMA,
        "agent_config": _HostileDict({"prototype": "minimal"}),
    }
    _HostileDict.calls = 0
    with patch.object(prototype, "load_checkpoint_metadata", return_value=metadata):
        with pytest.raises(ValueError, match="missing agent_config"):
            prototype.load_prototype_checkpoint(path)
    assert _HostileDict.calls == 0


def test_prototype_digest_text_has_no_repr_leak() -> None:
    import pathlib

    text = pathlib.Path(
        "alberta_framework/core/prototype_agent.py"
    ).read_text()
    # Ensure the digest error does not interpolate hostile via !r
    assert 'config_sha256' in text
