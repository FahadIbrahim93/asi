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
