"""Hostile string-subclass checks for Forager result import boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from alberta_framework.benchmarks.forager_results import (
    FORAGAX_DISTRIBUTION,
    FORAGAX_INSTALL_TREE_HASH_SCHEME,
    LegacyFOVSQLiteRunSpec,
    OfficialForagaxRunSpec,
    _validated_environment_provenance,
)


class _HookedStr(str):
    calls: int

    def __new__(cls, value: str) -> _HookedStr:
        instance = super().__new__(cls, value)
        instance.calls = 0
        return instance

    def _called(self) -> Any:
        self.calls += 1
        raise AssertionError("hostile string hook must not run")

    def __bool__(self) -> bool:
        return self._called()

    def __eq__(self, other: object) -> bool:
        return self._called()

    def __hash__(self) -> int:
        return self._called()

    def strip(self, chars: str | None = None) -> str:
        return self._called()


def _legacy_spec(**overrides: object) -> LegacyFOVSQLiteRunSpec:
    values: dict[str, object] = {
        "agent": "DQN",
        "path": Path("results.sqlite"),
        "config_path": Path("config.json"),
        "run_index": 0,
        "stored_seed": 0,
        "expected_config_agent": "DQN-7",
        "expected_aperture_size": 7,
        "expected_stored_seeds": (0,),
    }
    values.update(overrides)
    return LegacyFOVSQLiteRunSpec(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("expected_config_agent", "DQN-7"),
        ("agent", "DQN"),
    ],
)
def test_legacy_spec_rejects_string_subclasses_without_hooks(
    field: str, value: str
) -> None:
    hostile = _HookedStr(value)
    with pytest.raises(ValueError):
        _legacy_spec(**{field: hostile})
    assert hostile.calls == 0


def test_official_spec_rejects_agent_string_subclass_without_hooks() -> None:
    hostile = _HookedStr("DQN")
    with pytest.raises(ValueError, match="agent must be a non-empty string"):
        OfficialForagaxRunSpec(agent=hostile, seed=0, path=Path("result.npz"))
    assert hostile.calls == 0


def test_provenance_json_copy_normalizes_version_without_hooks() -> None:
    """JSON provenance stays compatible by normalizing strings before validation."""

    semantic = {"preset": "paper_relearning"}
    hostile = _HookedStr("1.2.3")
    provenance = _validated_environment_provenance(
        {
            "semantic": semantic,
            "implementation": {
                "distribution": FORAGAX_DISTRIBUTION,
                "package": "foragax",
                "version": hostile,
                "direct_url": None,
                "install_tree_hash_scheme": FORAGAX_INSTALL_TREE_HASH_SCHEME,
                "install_tree_sha256": "a" * 64,
            },
        },
        expected_semantic=semantic,
        required=True,
    )

    assert provenance is not None
    assert type(provenance["implementation"]["version"]) is str
    assert provenance["implementation"]["version"] == "1.2.3"
    assert hostile.calls == 0
