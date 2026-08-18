"""Complete ScorecardRunSpec identity contract: leftover, roster, and type gates."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.reference_life_scorecard import (
    ARM_ROSTER,
    ENVIRONMENT_ROSTER,
    SEED_ROSTER,
    ScorecardRunSpec,
    build_development_plan,
    iter_run_specs,
)


class _StringSubclass(str):
    """Leftover string identity that must not cross the spec boundary."""

    calls = 0

    def __repr__(self) -> str:
        type(self).calls += 1
        raise AssertionError("hostile repr")


def _legal_spec(**overrides: object) -> ScorecardRunSpec:
    canonical = iter_run_specs(build_development_plan())[0]
    payload = {
        "schedule_index": canonical.schedule_index,
        "environment_kind": canonical.environment_kind,
        "arm": canonical.arm,
        "seed": canonical.seed,
        "lifecycle_id": canonical.lifecycle_id,
    }
    payload.update(overrides)
    return ScorecardRunSpec(**payload)  # type: ignore[arg-type]


def test_scorecard_run_spec_accepts_canonical_schedule_identity() -> None:
    spec = iter_run_specs(build_development_plan())[0]
    assert spec.schedule_index == 0
    assert spec.environment_kind == ENVIRONMENT_ROSTER[0]
    assert spec.arm in ARM_ROSTER
    assert spec.seed == SEED_ROSTER[0]
    assert spec.lifecycle_id.startswith("prototype.")
    assert len(spec.lifecycle_id) == len("prototype.") + 16


def test_scorecard_run_spec_rejects_leftover_integer_identities() -> None:
    with pytest.raises(ValueError, match="schedule_index must be a nonnegative integer"):
        _legal_spec(schedule_index=True)
    with pytest.raises(ValueError, match="seed is not in the fixed scorecard roster"):
        _legal_spec(seed=True)
    with pytest.raises(ValueError, match="schedule_index must be a nonnegative integer"):
        _legal_spec(schedule_index=-1)
    with pytest.raises(ValueError, match="seed is not in the fixed scorecard roster"):
        _legal_spec(seed=0)


def test_scorecard_run_spec_rejects_leftover_and_unknown_string_identities() -> None:
    with pytest.raises(ValueError, match="environment_kind is not"):
        _legal_spec(environment_kind=True)
    with pytest.raises(ValueError, match="arm is not"):
        _legal_spec(arm=True)
    _StringSubclass.calls = 0
    with pytest.raises(ValueError, match="environment_kind is not"):
        _legal_spec(environment_kind=_StringSubclass(ENVIRONMENT_ROSTER[0]))
    with pytest.raises(ValueError, match="arm is not"):
        _legal_spec(arm=_StringSubclass(ARM_ROSTER[0]))
    with pytest.raises(ValueError, match="lifecycle_id must be a prototype"):
        _legal_spec(lifecycle_id=True)
    with pytest.raises(ValueError, match="lifecycle_id must be a prototype"):
        _legal_spec(lifecycle_id=_StringSubclass("prototype." + ("a" * 16)))
    with pytest.raises(ValueError, match="lifecycle_id must be a prototype"):
        _legal_spec(lifecycle_id="prototype.not-hex")
    assert _StringSubclass.calls == 0


def test_scorecard_run_spec_cross_binds_schedule_and_lifecycle() -> None:
    specs = iter_run_specs(build_development_plan())
    assert len(specs) == 144
    first = specs[0]
    second = specs[1]
    with pytest.raises(ValueError, match="does not match its fixed schedule_index"):
        _legal_spec(arm=second.arm)
    with pytest.raises(ValueError, match="does not match the fixed run identity"):
        _legal_spec(lifecycle_id=second.lifecycle_id)
    assert first.lifecycle_id != second.lifecycle_id
