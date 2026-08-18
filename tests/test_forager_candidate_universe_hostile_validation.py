"""Hostile input and boundary validation for candidate universe bindings."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks import forager_matched_candidate_universe as universe
from alberta_framework.benchmarks.forager_matched_candidate_universe import (
    BoundJsonArtifact,
    CandidateUniverseVerification,
    ForagerMatchedCandidateUniverseError,
    LocalCandidateGenerationBinding,
)


class _HostileString(str):
    calls = 0

    def __bool__(self) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile string truthiness executed")

    def __hash__(self) -> int:
        type(self).calls += 1
        raise AssertionError("hostile string hashing executed")


class _HostileRealMeta(type):
    calls = 0

    def __eq__(cls, other: object) -> bool:
        cls.calls += 1
        raise AssertionError("hostile metaclass hook dispatched")


class _HostileReal(metaclass=_HostileRealMeta):
    pass


def test_bound_json_artifact_validation() -> None:
    art = BoundJsonArtifact(role="plan", path="path/to/plan.json", sha256="a" * 64)
    assert art.role == "plan"
    assert art.path == "path/to/plan.json"
    assert art.sha256 == "a" * 64

    with pytest.raises(
        ForagerMatchedCandidateUniverseError, match="role must be a non-empty string"
    ):
        BoundJsonArtifact(role="", path="path/to/plan.json", sha256="a" * 64)

    with pytest.raises(
        ForagerMatchedCandidateUniverseError, match="path must be a non-empty string"
    ):
        BoundJsonArtifact(role="plan", path="", sha256="a" * 64)

    with pytest.raises(
        ForagerMatchedCandidateUniverseError,
        match="sha256 must be a 64-character lowercase hexadecimal",
    ):
        BoundJsonArtifact(role="plan", path="path/to/plan.json", sha256="invalid")


def test_candidate_universe_verification_validation() -> None:
    ver = CandidateUniverseVerification(
        candidate_universe_sha256="b" * 64,
        verified_json_paths=("path1.json", "path2.json"),
    )
    assert ver.candidate_universe_sha256 == "b" * 64
    assert len(ver.verified_json_paths) == 2

    with pytest.raises(
        ForagerMatchedCandidateUniverseError,
        match="candidate_universe_sha256 must be a 64-character lowercase hexadecimal",
    ):
        CandidateUniverseVerification(
            candidate_universe_sha256="invalid",
            verified_json_paths=("path1.json",),
        )

    with pytest.raises(
        ForagerMatchedCandidateUniverseError,
        match="verified_json_paths must be a tuple of non-empty strings",
    ):
        CandidateUniverseVerification(
            candidate_universe_sha256="b" * 64,
            verified_json_paths=["path1.json"],  # type: ignore[arg-type]
        )


def test_candidate_universe_string_boundaries_reject_subclasses_without_dispatch() -> None:
    hostile = _HostileString("path.json")
    _HostileString.calls = 0
    operations = (
        lambda: BoundJsonArtifact(role=hostile, path="path.json", sha256="a" * 64),
        lambda: CandidateUniverseVerification(
            candidate_universe_sha256="b" * 64,
            verified_json_paths=(hostile,),
        ),
    )
    for operation in operations:
        with pytest.raises(ForagerMatchedCandidateUniverseError):
            operation()
    assert _HostileString.calls == 0


def test_finite_real_rejects_hostile_runtime_type_without_metaclass_hooks() -> None:
    _HostileRealMeta.calls = 0
    with pytest.raises(ForagerMatchedCandidateUniverseError, match="must be a finite number"):
        universe._require_finite_real(_HostileReal(), "value")
    assert _HostileRealMeta.calls == 0


def test_local_candidate_generation_binding_artifact_tuple_validation() -> None:
    with pytest.raises(
        ForagerMatchedCandidateUniverseError,
        match="artifacts must be a tuple of BoundJsonArtifact instances",
    ):
        LocalCandidateGenerationBinding(
            screen_id="local_screen_v1",  # type: ignore[arg-type]
            family="linear",
            seeds=(1, 2, 3),
            horizon_per_seed=1000,
            candidate_count=5,
            normalized_matrix_sha256="c" * 64,
            source_tree_sha256="d" * 64,
            source_archive_sha256="e" * 64,
            source_inventory_sha256="f" * 64,
            artifacts=[BoundJsonArtifact(role="plan", path="plan.json", sha256="a" * 64)],  # type: ignore[arg-type]
        )
