"""Exact string and hostile runtime-type gates for Forager matrix artifacts."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks import forager_matrix as matrix

pytestmark = pytest.mark.unit


class _HostileString(str):
    calls = 0

    def __bool__(self) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile truth hook executed")

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile comparison hook executed")

    def startswith(self, prefix: object, *args: object) -> bool:
        del prefix, args
        type(self).calls += 1
        raise AssertionError("hostile startswith hook executed")

    def strip(self, chars: object = None) -> str:
        del chars
        type(self).calls += 1
        raise AssertionError("hostile strip hook executed")

    __hash__ = str.__hash__


class _ExplodingHashMeta(type):
    def __hash__(cls) -> int:
        raise AssertionError("hostile runtime-class hash executed")


class _HostileObject(metaclass=_ExplodingHashMeta):
    pass


def test_central_json_string_gates_reject_subclasses_before_hooks() -> None:
    hostile = _HostileString("a" * 64)
    payload = {hostile: "value"}
    _HostileString.calls = 0

    with pytest.raises(matrix.ForagerMatrixManifestError, match="exact strings"):
        matrix._require_object(payload, "payload")
    with pytest.raises(matrix.ForagerMatrixManifestError, match="exact string"):
        matrix._require_string(hostile, "value")
    with pytest.raises(matrix.ForagerMatrixStateError, match="payload_sha256"):
        matrix._verify_hashed_payload(
            {"payload_sha256": hostile}, description="payload"
        )
    with pytest.raises(matrix.ForagerMatrixStateError, match="UTC timestamp"):
        matrix._validate_utc_timestamp(hostile, "timestamp")
    assert _HostileString.calls == 0

    hostile_path = _HostileString("/tmp/hostile")
    _HostileString.calls = 0
    with pytest.raises(matrix.ForagerMatrixError, match="host path"):
        matrix._assert_path_sanitized(hostile_path)
    assert _HostileString.calls == 0


def test_object_gate_does_not_hash_hostile_runtime_classes() -> None:
    with pytest.raises(matrix.ForagerMatrixManifestError, match="JSON object"):
        matrix._require_object(_HostileObject(), "payload")


def test_trace_descriptor_rejects_hostile_location_before_path_hooks() -> None:
    hostile = _HostileString("seed-0.npz")
    descriptor = {
        "schema_version": matrix._RAW_TRACE_SCHEMA,
        "seed": 0,
        "path": hostile,
        "format": matrix._RAW_TRACE_FORMAT,
        "steps": 1,
        "biome_regret_present": True,
        "arrays": {
            "rewards": {
                "member": matrix._RAW_TRACE_MEMBERS[0],
                "dtype": matrix._RAW_TRACE_DTYPE.str,
                "shape": [1],
            },
            "biome_regrets": {
                "member": matrix._RAW_TRACE_MEMBERS[1],
                "dtype": matrix._RAW_TRACE_DTYPE.str,
                "shape": [1],
            },
        },
        "sha256": "0" * 64,
        "size": 1,
    }
    _HostileString.calls = 0

    with pytest.raises(matrix.ForagerMatrixStateError, match="identity"):
        matrix._validate_trace_descriptor(
            descriptor,
            path="trace",
            expected_seed=0,
            expected_steps=1,
            expected_output_path=None,
            exchange=False,
        )

    assert _HostileString.calls == 0
