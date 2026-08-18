"""Complete EvidenceSpec identity contract: leftover, types, and roster gates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest

from alberta_framework.evaluation.evidence_manifest import (
    EVIDENCE_SPECS,
    EvidenceSpec,
    _validated_evidence_specs,
)


class StringSubclass(str):
    """Leftover string identity that must not cross the spec boundary."""


class PathSubclass(type(Path())):  # type: ignore[misc]
    """Leftover concrete-path identity that must not cross the spec boundary."""


class CallableObject:
    def __call__(self, _value: object) -> object:
        return _value


class HostileList(list[object]):
    calls = 0

    def __iter__(self):  # type: ignore[no-untyped-def]
        type(self).calls += 1
        raise AssertionError("hostile list iteration ran")


class HostileCallable:
    calls = 0

    def __call__(self, *_args: object, **_kwargs: object) -> object:
        type(self).calls += 1
        raise AssertionError("hostile callable ran")


class HostilePath(type(Path())):  # type: ignore[misc]
    calls = 0

    def is_absolute(self) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile path method ran")


class HostileTuple(tuple[object, ...]):
    calls = 0

    def __iter__(self):  # type: ignore[no-untyped-def]
        type(self).calls += 1
        raise AssertionError("hostile tuple iteration ran")


def _load(_path: Path) -> dict[str, object]:
    return {"ok": True}


def _validate(_artifact: Mapping[str, object]) -> object:
    class _Result:
        valid = True
        accepted = True
        errors: tuple[str, ...] = ()

    return _Result()


def _legal(**overrides: object) -> EvidenceSpec:
    payload: dict[str, object] = {
        "name": "fixture_claim",
        "claim_scope": "test-only scope",
        "evidence_class": "scientific",
        "evidence_level": "L2",
        "promotes_scientific_claim": True,
        "relative_path": Path("fixture.json"),
        "expected_schema": "test.schema.v1",
        "command_argv": ("python", "-m", "fixture"),
        "protocol": {"protocol_version": "test.protocol.v1"},
        "configuration": {"steps": 1},
        "seeds": {"development": [0]},
        "thresholds": {"minimum_effect": 0.25},
        "limitations": ("test fixture only",),
        "source_paths": (Path("fixture.py"),),
        "required_environment_fields": ("python",),
        "loader": _load,
        "validator": _validate,
    }
    payload.update(overrides)
    return EvidenceSpec(**payload)  # type: ignore[arg-type]


def test_evidence_spec_accepts_canonical_and_registered_identities() -> None:
    spec = _legal()
    assert spec.name == "fixture_claim"
    assert spec.evidence_class == "scientific"
    assert spec.promotes_scientific_claim is True
    assert spec.relative_path == Path("fixture.json")
    names = {registered.name for registered in EVIDENCE_SPECS}
    assert names == {
        "recurring_pair_features",
        "scale_robust_pair_features",
        "ftl_world_model_decision_fidelity",
        "recurring_multiagent_coadaptation",
        "continual_intelligence_amplification",
    }


def test_evidence_spec_rejects_leftover_string_identities() -> None:
    with pytest.raises(ValueError, match="name must be a non-empty string"):
        _legal(name=True)
    with pytest.raises(ValueError, match="name must be a non-empty string"):
        _legal(name=StringSubclass("fixture_claim"))
    with pytest.raises(ValueError, match="claim_scope must be a non-empty string"):
        _legal(claim_scope=True)
    with pytest.raises(ValueError, match="expected_schema must be a non-empty string"):
        _legal(expected_schema="")


def test_evidence_spec_rejects_leftover_class_level_and_bool_identities() -> None:
    with pytest.raises(ValueError, match="evidence_class must be a known evidence class"):
        _legal(evidence_class=True)
    with pytest.raises(ValueError, match="evidence_class must be a known evidence class"):
        _legal(evidence_class=StringSubclass("scientific"))
    with pytest.raises(ValueError, match="evidence_class must be a known evidence class"):
        _legal(evidence_class="anecdote")
    with pytest.raises(ValueError, match="evidence_level must be a known evidence level"):
        _legal(evidence_level="L9")
    with pytest.raises(ValueError, match="promotes_scientific_claim must be a boolean"):
        _legal(promotes_scientific_claim=1)


def test_evidence_spec_rejects_leftover_path_tuple_and_mapping_identities() -> None:
    with pytest.raises(ValueError, match="relative_path must be an exact platform Path"):
        _legal(relative_path="fixture.json")
    with pytest.raises(ValueError, match="relative_path must be an exact platform Path"):
        _legal(relative_path=PathSubclass("fixture.json"))
    with pytest.raises(ValueError, match="repository-relative Path"):
        _legal(relative_path=Path("../fixture.json"))
    with pytest.raises(ValueError, match="command_argv must be a non-empty tuple"):
        _legal(command_argv=["python"])
    with pytest.raises(ValueError, match="limitations must be a non-empty tuple"):
        _legal(limitations=["test fixture only"])
    with pytest.raises(ValueError, match="source_paths must be a non-empty tuple"):
        _legal(source_paths=["fixture.py"])
    with pytest.raises(ValueError, match="protocol must be a non-empty dict"):
        _legal(protocol=True)
    with pytest.raises(ValueError, match="seeds key must be a non-empty string"):
        _legal(seeds={True: 0})
    with pytest.raises(ValueError, match="loader must be an exact function"):
        _legal(loader=None)
    with pytest.raises(ValueError, match="loader must be an exact function"):
        _legal(loader=CallableObject())
    with pytest.raises(ValueError, match="validator must be an exact function"):
        _legal(validator="validate")


def test_evidence_spec_preflights_sequence_and_mapping_counts() -> None:
    with pytest.raises(ValueError, match="command_argv must be a non-empty tuple"):
        _legal(command_argv=("x",) * 257)
    with pytest.raises(ValueError, match="protocol exceeds the mapping item limit"):
        _legal(protocol={f"key-{index}": index for index in range(257)})
    with pytest.raises(ValueError, match="exceeds 65536 UTF-8 bytes"):
        _legal(name="x" * 65_537)


def test_evidence_spec_rejects_unsafe_and_hostile_paths_without_hooks() -> None:
    hostile = HostilePath("fixture.json")
    HostilePath.calls = 0
    with pytest.raises(ValueError, match="relative_path must be an exact platform Path"):
        _legal(relative_path=hostile)
    with pytest.raises(ValueError, match=r"source_paths\[0\] must be an exact platform Path"):
        _legal(source_paths=(hostile,))
    assert HostilePath.calls == 0

    for path in (Path("/tmp/fixture.json"), Path("../fixture.json"), Path(".")):
        with pytest.raises(ValueError, match="repository-relative Path"):
            _legal(relative_path=path)


def test_evidence_spec_rejects_nested_non_json_identities_without_hooks() -> None:
    hostile = HostileList([1])
    HostileList.calls = 0
    with pytest.raises(ValueError, match="finite exact JSON values"):
        _legal(protocol={"nested": hostile})
    assert HostileList.calls == 0

    with pytest.raises(ValueError, match="finite exact JSON values"):
        _legal(configuration={"nested": StringSubclass("value")})
    with pytest.raises(ValueError, match="finite exact JSON values"):
        _legal(thresholds={"value": float("nan")})

    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    with pytest.raises(ValueError, match="must not contain a cycle"):
        _legal(protocol=cyclic)


def test_evidence_spec_rejects_callable_objects_without_calling_them() -> None:
    hostile = HostileCallable()
    HostileCallable.calls = 0
    with pytest.raises(ValueError, match="loader must be an exact function"):
        _legal(loader=hostile)
    assert HostileCallable.calls == 0


def test_registry_rejects_hostile_rosters_and_duplicate_identities_without_hooks() -> None:
    HostileTuple.calls = 0
    with pytest.raises(ValueError, match="EVIDENCE_SPECS must be a non-empty exact tuple"):
        _validated_evidence_specs(HostileTuple(EVIDENCE_SPECS))
    assert HostileTuple.calls == 0

    first = EVIDENCE_SPECS[0]
    duplicate_name = replace(EVIDENCE_SPECS[1], name=first.name)
    with pytest.raises(ValueError, match="claim names must be unique"):
        _validated_evidence_specs((first, duplicate_name))

    duplicate_path = replace(EVIDENCE_SPECS[1], relative_path=first.relative_path)
    with pytest.raises(ValueError, match="artifact paths must be unique"):
        _validated_evidence_specs((first, duplicate_path))


def test_evidence_spec_recursively_bounds_nested_json_values() -> None:
    with pytest.raises(ValueError, match="sequence item limit"):
        _legal(protocol={"nested": [0] * 257})
    with pytest.raises(ValueError, match="65536 UTF-8 bytes"):
        _legal(protocol={"nested": "x" * 65_537})

    nested: object = None
    for _ in range(65):
        nested = [nested]
    with pytest.raises(ValueError, match="maximum nesting depth"):
        _legal(protocol={"nested": nested})

    aggregate = [[0] * 16 for _ in range(256)]
    with pytest.raises(ValueError, match="aggregate value limit"):
        _legal(protocol={"nested": aggregate})
