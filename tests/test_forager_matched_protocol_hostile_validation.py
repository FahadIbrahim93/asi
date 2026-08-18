"""Hostile input, leftover identity, and type validation for protocol records."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.forager_matched_protocol import (
    AgentRNGContract,
    DescriptiveContext,
    EnvironmentRNGContract,
    ForagerMatchedProtocolError,
    ForagerMatchedSelectionResult,
    RankedSelectionGroup,
    ResolvedSelectionSlot,
    SelectionSlot,
    decode_strict_json,
    parse_forager_matched_protocol,
    parse_forager_matched_selection_result,
)


class _HostileString(str):
    calls = 0

    def __bool__(self) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile string truthiness executed")

    def __len__(self) -> int:
        type(self).calls += 1
        raise AssertionError("hostile string length executed")

    def __hash__(self) -> int:
        type(self).calls += 1
        raise AssertionError("hostile string hashing executed")

    def encode(self, *args: object, **kwargs: object) -> bytes:
        type(self).calls += 1
        raise AssertionError("hostile string encoding executed")


class _HostileInputMeta(type):
    calls = 0

    def __eq__(cls, other: object) -> bool:
        cls.calls += 1
        raise AssertionError("hostile metaclass equality executed")


class _HostileInput(metaclass=_HostileInputMeta):
    pass


class _HostileSelectionResult(ForagerMatchedSelectionResult):
    calls = 0

    def to_dict(self) -> dict[str, object]:
        type(self).calls += 1
        raise AssertionError("hostile selection conversion executed")


def test_environment_rng_contract_validation() -> None:
    contract = EnvironmentRNGContract(
        identity="env_rng.v1",
        schedule_sha256="a" * 64,
    )
    assert contract.identity == "env_rng.v1"
    assert contract.schedule_sha256 == "a" * 64

    with pytest.raises(
        ForagerMatchedProtocolError, match="must be a non-empty string of at most 128 characters"
    ):
        EnvironmentRNGContract(
            identity="",
            schedule_sha256="a" * 64,
        )

    with pytest.raises(
        ForagerMatchedProtocolError,
        match="must be a lowercase 64-character SHA-256 digest",
    ):
        EnvironmentRNGContract(
            identity="env_rng.v1",
            schedule_sha256="invalid",
        )


def test_agent_rng_contract_validation() -> None:
    contract = AgentRNGContract(identity="agent_rng.v1", environment_key_shared=False)
    assert contract.identity == "agent_rng.v1"
    assert contract.environment_key_shared is False

    with pytest.raises(
        ForagerMatchedProtocolError, match="must be a non-empty string of at most 128 characters"
    ):
        AgentRNGContract(identity="", environment_key_shared=False)

    with pytest.raises(ForagerMatchedProtocolError, match="must be a boolean"):
        AgentRNGContract(identity="agent_rng.v1", environment_key_shared=1)  # type: ignore[arg-type]


def test_selection_slot_validation() -> None:
    slot = SelectionSlot(selection_group="group_a", rank=1)
    assert slot.selection_group == "group_a"
    assert slot.rank == 1

    with pytest.raises(
        ForagerMatchedProtocolError, match="must be a non-empty string of at most 128 characters"
    ):
        SelectionSlot(selection_group="", rank=1)

    with pytest.raises(ForagerMatchedProtocolError, match="must lie in"):
        SelectionSlot(selection_group="group_a", rank=0)


def test_resolved_selection_slot_validation() -> None:
    slot = ResolvedSelectionSlot(selection_group="group_a", rank=1, candidate_id="cand_1")
    assert slot.selection_group == "group_a"
    assert slot.rank == 1
    assert slot.candidate_id == "cand_1"

    with pytest.raises(
        ForagerMatchedProtocolError, match="must be a non-empty string of at most 128 characters"
    ):
        ResolvedSelectionSlot(selection_group="group_a", rank=1, candidate_id="")


def test_ranked_selection_group_validation() -> None:
    group = RankedSelectionGroup(
        selection_group="group_a",
        ranked_candidate_ids=("cand_1", "cand_2"),
        ranking_evidence_sha256="b" * 64,
    )
    assert group.selection_group == "group_a"
    assert len(group.ranked_candidate_ids) == 2

    with pytest.raises(ForagerMatchedProtocolError, match="must be a tuple of candidate IDs"):
        RankedSelectionGroup(
            selection_group="group_a",
            ranked_candidate_ids=["cand_1"],  # type: ignore[arg-type]
            ranking_evidence_sha256="b" * 64,
        )


def test_descriptive_context_validation() -> None:
    ctx = DescriptiveContext(
        candidate_ids=("cand_1",),
        analysis_role="descriptive_only",
        selection_eligible=False,
        pairing_eligible=False,
    )
    assert ctx.analysis_role == "descriptive_only"

    with pytest.raises(ForagerMatchedProtocolError, match="selection_eligible must be False"):
        DescriptiveContext(
            candidate_ids=("cand_1",),
            analysis_role="descriptive_only",
            selection_eligible=True,  # type: ignore[arg-type]
            pairing_eligible=False,
        )

    with pytest.raises(ForagerMatchedProtocolError, match="pairing_eligible must be False"):
        DescriptiveContext(
            candidate_ids=("cand_1",),
            analysis_role="descriptive_only",
            selection_eligible=False,
            pairing_eligible=True,  # type: ignore[arg-type]
        )


def test_protocol_string_boundaries_reject_subclasses_without_dispatch() -> None:
    hostile = _HostileString("candidate")
    _HostileString.calls = 0
    operations = (
        lambda: EnvironmentRNGContract(
            identity=hostile,
            schedule_sha256="a" * 64,
        ),
        lambda: RankedSelectionGroup(
            selection_group="group",
            ranked_candidate_ids=(hostile,),
            ranking_evidence_sha256="b" * 64,
        ),
        lambda: DescriptiveContext(
            candidate_ids=(hostile,),
            analysis_role="descriptive_only",
            selection_eligible=False,
            pairing_eligible=False,
        ),
        lambda: decode_strict_json(hostile),
        lambda: parse_forager_matched_protocol({"hostile": hostile}),
        lambda: parse_forager_matched_protocol(hostile),
        lambda: parse_forager_matched_selection_result(hostile),
    )
    for operation in operations:
        with pytest.raises(ForagerMatchedProtocolError):
            operation()
    assert _HostileString.calls == 0


def test_protocol_decoder_rejects_hostile_runtime_type_without_metaclass_hooks() -> None:
    _HostileInputMeta.calls = 0
    with pytest.raises(ForagerMatchedProtocolError, match="exact bytes or string JSON"):
        decode_strict_json(_HostileInput())  # type: ignore[arg-type]
    assert _HostileInputMeta.calls == 0


def test_selection_result_parser_rejects_subclass_without_conversion() -> None:
    hostile = object.__new__(_HostileSelectionResult)
    _HostileSelectionResult.calls = 0
    with pytest.raises(ForagerMatchedProtocolError):
        parse_forager_matched_selection_result(hostile)
    assert _HostileSelectionResult.calls == 0
