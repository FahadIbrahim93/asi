"""Checkpoint array hex is sized from shape/dtype before the host charset walk.

Origin ``_decode_array_value`` / ``_decode_jax_array`` scanned every character of
``payload_hex`` with a Python ``any(c not in ...)`` test before comparing the
decoded length to the declared shape. A cheap ``'a' * (2 * 64 MiB)`` scalar
float32 payload took 3.057s on origin/main.
"""

from __future__ import annotations

import time

import jax.numpy as jnp
import pytest

from alberta_framework.reference_life_checkpoint import (
    _MAX_ARRAY_BYTES,
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


def _at_cap_hex() -> str:
    return "a" * (2 * _MAX_ARRAY_BYTES)


def test_array_value_rejects_at_cap_hex_before_charset_walk() -> None:
    payload = {
        "semantic_id": "obs",
        "dtype": "float32",
        "shape": [1],
        "payload_hex": _at_cap_hex(),
    }
    started = time.perf_counter()
    with pytest.raises(ValueError, match="payload_hex length does not match"):
        _decode_array_value(payload, path="observation")
    assert time.perf_counter() - started < 0.5


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


def test_jax_array_rejects_at_cap_hex_before_charset_walk() -> None:
    payload = {
        "dtype": "float32",
        "shape": [1],
        "payload_hex": _at_cap_hex(),
    }
    started = time.perf_counter()
    with pytest.raises(ValueError, match="payload_hex length does not match"):
        _decode_jax_array(payload, path="state_index")
    assert time.perf_counter() - started < 0.5
