"""Hostile validation for historical provenance facade."""

import pytest

from alberta_framework.benchmarks.historical_forager_provenance import (
    HISTORICAL_FORAGER_FAMILY_ID,
    HistoricalForagerFamilyMismatchError,
    HistoricalForagerProvenanceError,
    assert_historical_family_pairing,
    historical_forager_provenance,
    validate_historical_forager_provenance,
)


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


def test_hostile_left_without_repr_leak() -> None:
    evil = _EvilStr(HISTORICAL_FORAGER_FAMILY_ID)
    _EvilStr.calls = 0
    with pytest.raises(HistoricalForagerProvenanceError, match="must be a string") as exc:
        assert_historical_family_pairing(evil, HISTORICAL_FORAGER_FAMILY_ID)  # type: ignore[arg-type]
    assert _EvilStr.calls == 0
    assert "EvilStr" not in str(exc.value)


def test_hostile_right_without_repr_leak() -> None:
    evil = _EvilStr(HISTORICAL_FORAGER_FAMILY_ID)
    _EvilStr.calls = 0
    with pytest.raises(HistoricalForagerProvenanceError, match="must be a string"):
        assert_historical_family_pairing(HISTORICAL_FORAGER_FAMILY_ID, evil)  # type: ignore[arg-type]
    assert _EvilStr.calls == 0


def test_string_subclass_rejected() -> None:
    with pytest.raises(HistoricalForagerProvenanceError, match="must be a string"):
        assert_historical_family_pairing(
            _StringSubclass(HISTORICAL_FORAGER_FAMILY_ID),
            HISTORICAL_FORAGER_FAMILY_ID,
        )  # type: ignore[arg-type]


def test_mismatch_sanitized_without_repr() -> None:
    with pytest.raises(
        HistoricalForagerFamilyMismatchError,
        match="historical reconstructed results pair only with",
    ) as exc:
        assert_historical_family_pairing("bad_left", "bad_right")
    assert "!r" not in str(exc.value)
    assert "bad_left" not in str(exc.value)
    assert "bad_right" not in str(exc.value)
    msg = str(exc.value)
    assert "'" in msg


def test_valid_pairing_passes() -> None:
    assert_historical_family_pairing(
        HISTORICAL_FORAGER_FAMILY_ID, HISTORICAL_FORAGER_FAMILY_ID
    )


def test_mismatch_even_one_bad_rejected_sanitized() -> None:
    with pytest.raises(HistoricalForagerFamilyMismatchError) as exc:
        assert_historical_family_pairing(HISTORICAL_FORAGER_FAMILY_ID, "other")
    assert "!r" not in str(exc.value)


class _HostileDict(dict[str, object]):
    calls = 0

    def items(self):
        type(self).calls += 1
        raise AssertionError("items hook must not run")


def test_provenance_validator_rejects_dict_subclass_without_hooks() -> None:
    _HostileDict.calls = 0
    with pytest.raises(HistoricalForagerProvenanceError, match="actual dictionary"):
        validate_historical_forager_provenance(_HostileDict())
    assert _HostileDict.calls == 0


def test_provenance_validator_rejects_nested_hostile_json_identity() -> None:
    payload = historical_forager_provenance()
    payload["agents"] = _HostileDict()
    _HostileDict.calls = 0
    with pytest.raises(HistoricalForagerProvenanceError, match="exact JSON values"):
        validate_historical_forager_provenance(payload)
    assert _HostileDict.calls == 0


def test_provenance_validator_bounds_work_before_serialization() -> None:
    payload = historical_forager_provenance()
    payload["agents"] = {str(index): index for index in range(4097)}
    with pytest.raises(HistoricalForagerProvenanceError, match="too large"):
        validate_historical_forager_provenance(payload)
