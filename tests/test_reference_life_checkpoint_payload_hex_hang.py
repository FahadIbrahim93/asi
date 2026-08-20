"""Checkpoint array hex is sized from shape/dtype before the host charset walk.

Origin ``_decode_array_value`` / ``_decode_jax_array`` scanned every character of
``payload_hex`` with a Python ``any(c not in ...)`` test before comparing the
decoded length to the declared shape. A cheap ``'a' * (2 * 64 MiB)`` scalar
float32 payload took 3.057s on origin/main.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import alberta_framework.reference_life_checkpoint as checkpoint_module
from alberta_framework.reference_life_checkpoint import (
    _decode_array_value,
    _decode_jax_array,
)

pytestmark = pytest.mark.unit

_SCALAR_FLOAT32 = {
    "semantic_id": "obs",
    "dtype": "float32",
    "shape": [1],
    "payload_hex": "0000803f",
}


class _ForbiddenPattern:
    def fullmatch(self, _value: str) -> None:
        raise AssertionError("payload characters were inspected before its length")


def test_array_value_rejects_mismatched_hex_before_charset_walk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "semantic_id": "obs",
        "dtype": "float32",
        "shape": [1],
        "payload_hex": "a" * 10_000,
    }
    monkeypatch.setattr(checkpoint_module, "_LOWER_HEX_PATTERN", _ForbiddenPattern())
    with pytest.raises(ValueError, match="payload_hex length does not match"):
        _decode_array_value(payload, path="observation")


def test_array_value_rejects_semantic_id_before_charset_walk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = dict(_SCALAR_FLOAT32)
    payload["semantic_id"] = 1
    monkeypatch.setattr(checkpoint_module, "_LOWER_HEX_PATTERN", _ForbiddenPattern())
    with pytest.raises(ValueError, match="semantic_id must be a string"):
        _decode_array_value(payload, path="observation")


def test_array_value_accepts_matching_lowercase_hex() -> None:
    value = _decode_array_value(_SCALAR_FLOAT32, path="observation")
    assert value.shape == (1,)
    assert value.dtype == "float32"
    assert value.payload == bytes.fromhex("0000803f")


def test_array_value_rejects_uppercase_hex_of_matching_length() -> None:
    payload = dict(_SCALAR_FLOAT32)
    payload["payload_hex"] = "0000803F"
    with pytest.raises(ValueError, match="lowercase hexadecimal"):
        _decode_array_value(payload, path="observation")


def test_jax_array_accepts_matching_lowercase_hex() -> None:
    payload = {
        "dtype": "float32",
        "shape": [1],
        "payload_hex": "0000803f",
    }
    array = _decode_jax_array(payload, path="state_index")
    assert array.shape == (1,)
    assert array.dtype == jnp.float32
    assert float(array[0]) == pytest.approx(1.0)


def test_jax_array_rejects_mismatched_hex_before_charset_walk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "dtype": "float32",
        "shape": [1],
        "payload_hex": "a" * 10_000,
    }
    monkeypatch.setattr(checkpoint_module, "_LOWER_HEX_PATTERN", _ForbiddenPattern())
    with pytest.raises(ValueError, match="payload_hex length does not match"):
        _decode_jax_array(payload, path="state_index")
