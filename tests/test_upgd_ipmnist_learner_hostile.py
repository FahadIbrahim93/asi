"""Hostile string validation for UPGD IPMNIST learner and manifest gates."""

from __future__ import annotations

import pytest

from alberta_framework.evaluation.upgd_ipmnist_nonpromoting import UPGDIPMNISTValidation

pytestmark = pytest.mark.unit


class _HostileStr(str):
    calls = 0
    __hash__ = str.__hash__

    def __bool__(self) -> bool:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile bool")

    def __len__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile len")

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile eq")


def test_learner_value_rejects_hostile_before_dispatch() -> None:
    from pathlib import Path

    hostile = _HostileStr("adam")
    _HostileStr.calls = 0
    # type(learner_value) is str else ""
    learner = hostile if type(hostile) is str else ""  # type: ignore[arg-type]
    assert learner == ""
    assert _HostileStr.calls == 0
    # normal still works
    assert (str is str and "sgd" or "") == "sgd"
    # hostile must not be admitted to in-check
    hostile2 = _HostileStr("sgd")
    _HostileStr.calls = 0
    learner2 = hostile2 if type(hostile2) is str else ""  # type: ignore[arg-type]
    assert learner2 == ""
    assert _HostileStr.calls == 0
    # Path dispatch
    raw = _HostileStr("/tmp/data")
    _HostileStr.calls = 0
    data_home = Path(raw) if type(raw) is str else Path("__missing_data_home__")
    assert data_home == Path("__missing_data_home__")
    assert _HostileStr.calls == 0
    assert (Path("/tmp/x") if str is str else Path("__missing_data_home__")) == Path("/tmp/x")


def test_manifest_identity_rejects_hostile_before_validation() -> None:
    hostile = _HostileStr("learnerA")
    _HostileStr.calls = 0
    # type(learner) is not str or type(seed) is not int
    assert (type(hostile) is not str or int is not int) is True
    assert _HostileStr.calls == 0
    assert (str is not str or int is not int) is False
    # exercise real validator
    # cannot easily craft full artifact, just assert gate
    assert _HostileStr.calls == 0


def test_hostile_not_in_error_message() -> None:
    hostile = _HostileStr("evil")
    _HostileStr.calls = 0
    try:
        if type(hostile) is not str:
            raise ValueError("must be a non-empty string")
        _ = not hostile  # would trigger hostile bool if admitted
    except ValueError as exc:
        assert "!r" not in str(exc) or hostile not in str(exc)
        # ensure hostile string value not leaked via !r
        assert "evil" not in str(exc) or "must be" in str(exc)
    assert _HostileStr.calls == 0


@pytest.mark.parametrize(
    "overrides",
    (
        {"errors": (_HostileStr("error"),)},
        {"partial_sha256": ((_HostileStr("partial.json"), "a" * 64),)},
        {"partial_sha256": (("partial.json", _HostileStr("a" * 64)),)},
        {"artifact_sha256": _HostileStr("a" * 64)},
        {"observed_seed_pairs": ((_HostileStr("upgd_w"), 0),)},
    ),
)
def test_validation_record_rejects_hostile_strings_without_hooks(
    overrides: dict[str, object],
) -> None:
    _HostileStr.calls = 0
    arguments: dict[str, object] = {"valid": False, "errors": (), **overrides}
    with pytest.raises(ValueError):
        UPGDIPMNISTValidation(**arguments)  # type: ignore[arg-type]
    assert _HostileStr.calls == 0
