"""Fail-closed contracts for external benchmark qualification plans."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.external_qualification import (
    COMMON_GATES,
    EXTERNAL_QUALIFICATION_PLANS,
    ExternalCodeRevision,
    ExternalQualificationPlan,
    qualification_plan,
)

pytestmark = pytest.mark.unit


def test_registry_exactly_covers_research_wave_and_is_not_run_ready() -> None:
    assert tuple(plan.issue for plan in EXTERNAL_QUALIFICATION_PLANS) == tuple(range(1574, 1584))
    assert len({plan.lane_id for plan in EXTERNAL_QUALIFICATION_PLANS}) == 10
    for plan in EXTERNAL_QUALIFICATION_PLANS:
        assert plan.blockers == plan.required_gates
        with pytest.raises(RuntimeError, match=f"external lane {plan.lane_id} is not qualified"):
            plan.require_ready()


def test_ready_plan_requires_every_gate_without_reordering() -> None:
    plan = ExternalQualificationPlan(
        issue=1,
        lane_id="fixture",
        paper_revisions=("paper-v1",),
        code_revisions=(),
        required_gates=("first", "second"),
        completed_gates=("second", "first"),
    )
    assert plan.blockers == ()
    plan.require_ready()


@pytest.mark.parametrize("issue", [True, 1574.0, "1574"])
def test_lookup_rejects_scalar_aliases(issue: object) -> None:
    with pytest.raises(ValueError, match="exact integer"):
        qualification_plan(issue)


def test_plan_rejects_hostile_container_and_string_subclasses() -> None:
    class StringSubclass(str):
        pass

    with pytest.raises(ValueError, match="lane_id"):
        ExternalQualificationPlan(1, StringSubclass("lane"), ("paper",), (), COMMON_GATES)
    with pytest.raises(ValueError, match="paper_revisions"):
        ExternalQualificationPlan(1, "lane", ["paper"], (), COMMON_GATES)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="entries"):
        ExternalQualificationPlan(1, "lane", (StringSubclass("paper"),), (), COMMON_GATES)


def test_code_revision_requires_credential_free_url_and_full_commit() -> None:
    with pytest.raises(ValueError, match="credential-free"):
        ExternalCodeRevision("https://token@github.com/org/repo.git", "a" * 40)
    with pytest.raises(ValueError, match="full lowercase"):
        ExternalCodeRevision("https://github.com/org/repo.git", "A" * 40)


def test_completed_gates_must_be_known_and_unique() -> None:
    with pytest.raises(ValueError, match="subset"):
        ExternalQualificationPlan(1, "lane", ("paper",), (), ("known",), ("unknown",))
    with pytest.raises(ValueError, match="duplicates"):
        ExternalQualificationPlan(1, "lane", ("paper",), (), ("known", "known"))
