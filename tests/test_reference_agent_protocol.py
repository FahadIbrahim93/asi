"""Contract tests for the host-facing ASI reference-agent API.

These tests were introduced before the production module and define a small
transaction boundary that can be implemented by distinct PrototypeAgent,
Forager, and robot adapters without claiming that their action spaces,
algorithms, or evidence are equivalent.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace
from typing import Any

import numpy as np
import pytest

from alberta_framework.reference_agent import (
    REFERENCE_AGENT_MANIFEST_SCHEMA,
    AgentCapabilities,
    AgentManifest,
    Decision,
    DecisionOwnershipError,
    DispatchAck,
    DispatchStatus,
    ReferenceAgentSession,
    SpaceSpec,
    StepResult,
    Transition,
    canonical_config_sha256,
)


def _observation_spec() -> SpaceSpec:
    return SpaceSpec.box(
        shape=(3,),
        dtype="float32",
        low=(-100.0, -100.0, -100.0),
        high=(100.0, 100.0, 100.0),
        semantic_id="tests.reference_observation.v1",
    )


def _action_spec() -> SpaceSpec:
    return SpaceSpec.box(
        shape=(2,),
        dtype="float32",
        low=(-1.0, -1.0),
        high=(1.0, 1.0),
        semantic_id="tests.normalized_action.v1",
    )


def _manifest(*, dispatch_rebinding: bool = False) -> AgentManifest:
    return AgentManifest(
        schema=REFERENCE_AGENT_MANIFEST_SCHEMA,
        implementation_id="tests.fake_reference_session.v1",
        config_sha256=canonical_config_sha256(
            {"agent": "fake", "action_scale": 1.0, "seed": 7}
        ),
        observation_spec=_observation_spec(),
        action_spec=_action_spec(),
        capabilities=AgentCapabilities(
            explicit_discount=True,
            exact_checkpoint_resume=False,
            dispatch_rebinding=dispatch_rebinding,
            compiled_rollout=False,
            context_inputs=(),
        ),
    )


def _decision(
    *,
    lifecycle_id: str = "life-a",
    decision_id: str = "life-a:0",
    observation_id: str | None = None,
    action_codec_id: str = "tests.normalized_action.v1",
    armed: bool = True,
    observation: tuple[float, ...] = (0.0, 0.0, 0.0),
    action: tuple[float, ...] = (0.25, -0.25),
) -> Decision:
    return Decision(
        lifecycle_id=lifecycle_id,
        decision_id=decision_id,
        observation_id=(f"{decision_id}:observation" if observation_id is None else observation_id),
        action_codec_id=action_codec_id,
        armed=armed,
        observation=observation,
        proposed_action=action,
    )


def _exact_dispatch(decision: Decision) -> DispatchAck:
    return DispatchAck.exact(
        decision,
        authority_id="tests.dispatch_authority.v1",
        policy_version="tests.dispatch_policy.v1",
        dispatch_receipt_id=f"{decision.decision_id}:dispatch",
    )


def _transition(
    dispatch: DispatchAck,
    *,
    reward: float = 1.0,
    discount: float = 0.9,
    terminated: bool = False,
    truncated: bool = False,
    bootstrap_observation: tuple[float, ...] = (0.1, 0.2, 0.3),
    bootstrap_observation_id: str = "tests.observation:next",
    autoreset: bool = False,
    next_decision_observation: tuple[float, ...] | None = (0.1, 0.2, 0.3),
    next_decision_observation_id: str | None = "tests.observation:next",
) -> Transition:
    return Transition(
        dispatch=dispatch,
        reward=reward,
        discount=discount,
        terminated=terminated,
        truncated=truncated,
        bootstrap_observation=bootstrap_observation,
        bootstrap_observation_id=bootstrap_observation_id,
        autoreset=autoreset,
        next_decision_observation=next_decision_observation,
        next_decision_observation_id=next_decision_observation_id,
    )


def test_manifest_schema_config_digest_and_records_are_immutable() -> None:
    assert REFERENCE_AGENT_MANIFEST_SCHEMA == "asi.reference_agent_manifest.v1"

    first = {"seed": 7, "nested": {"beta": 2, "alpha": 1}}
    reordered = {"nested": {"alpha": 1, "beta": 2}, "seed": 7}
    assert canonical_config_sha256(first) == canonical_config_sha256(reordered)
    assert canonical_config_sha256(first) != canonical_config_sha256(
        {"seed": 8, "nested": {"alpha": 1, "beta": 2}}
    )
    with pytest.raises(ValueError, match="finite|JSON"):
        canonical_config_sha256({"invalid": math.nan})

    manifest = _manifest()
    assert len(manifest.config_sha256) == 64
    assert manifest.capabilities.explicit_discount
    with pytest.raises(FrozenInstanceError):
        setattr(manifest, "implementation_id", "changed")

    values = {
        "schema": REFERENCE_AGENT_MANIFEST_SCHEMA,
        "implementation_id": "tests.fake_reference_session.v1",
        "config_sha256": "a" * 64,
        "observation_spec": _observation_spec(),
        "action_spec": _action_spec(),
        "capabilities": AgentCapabilities(
            explicit_discount=True,
            exact_checkpoint_resume=False,
            dispatch_rebinding=False,
            compiled_rollout=False,
            context_inputs=(),
        ),
    }
    with pytest.raises(ValueError, match="schema"):
        AgentManifest(**{**values, "schema": "alberta.reference_agent.v0"})
    with pytest.raises(ValueError, match="config_sha256"):
        AgentManifest(**{**values, "config_sha256": "not-a-digest"})
    with pytest.raises(ValueError, match="implementation_id"):
        AgentManifest(**{**values, "implementation_id": ""})


def test_space_specs_keep_discrete_and_box_actions_distinct_and_validated() -> None:
    discrete = SpaceSpec.discrete(
        cardinality=4,
        dtype="int32",
        semantic_id="forager.direction.v1",
    )
    box = _action_spec()

    discrete.validate_value(3)
    box.validate_value((0.25, -0.5))
    with pytest.raises(ValueError, match="cardinality|range"):
        discrete.validate_value(4)
    with pytest.raises((TypeError, ValueError), match="integer|bool"):
        discrete.validate_value(True)
    with pytest.raises(ValueError, match="shape"):
        box.validate_value((0.25,))
    with pytest.raises(ValueError, match="finite"):
        box.validate_value((0.25, math.nan))
    with pytest.raises(ValueError, match="bounds|range"):
        box.validate_value((0.25, 1.5))

    with pytest.raises(ValueError, match="cardinality"):
        SpaceSpec.discrete(cardinality=0, dtype="int32", semantic_id="bad")
    with pytest.raises(ValueError, match="shape|bounds"):
        SpaceSpec.box(
            shape=(2,),
            dtype="float32",
            low=(-1.0,),
            high=(1.0, 1.0),
            semantic_id="bad",
        )
    with pytest.raises(ValueError, match="bounds|low|high"):
        SpaceSpec.box(
            shape=(2,),
            dtype="float32",
            low=(2.0, -1.0),
            high=(1.0, 1.0),
            semantic_id="bad",
        )


def test_space_specs_reject_values_not_representable_by_declared_dtype() -> None:
    with pytest.raises(ValueError, match="cardinality|int8|represent"):
        SpaceSpec.discrete(
            cardinality=300,
            dtype="int8",
            semantic_id="tests.too_wide_discrete.v1",
        )
    with pytest.raises(ValueError, match="bounds|float16|represent|finite"):
        SpaceSpec.box(
            shape=(1,),
            dtype="float16",
            low=(-1.0e20,),
            high=(1.0e20,),
            semantic_id="tests.too_wide_box.v1",
        )

    narrow = SpaceSpec.box(
        shape=(1,),
        dtype="float16",
        low=(-10.0,),
        high=(10.0,),
        semantic_id="tests.float16_box.v1",
    )
    with pytest.raises(ValueError, match="float16|represent|finite"):
        narrow.validate_value((1.0e10,))


def test_decision_identity_is_explicit_nonempty_and_immutable() -> None:
    decision = _decision()
    assert decision.lifecycle_id == "life-a"
    assert decision.decision_id == "life-a:0"
    assert decision.observation_id == "life-a:0:observation"
    assert decision.action_codec_id == "tests.normalized_action.v1"
    assert decision.armed
    with pytest.raises(FrozenInstanceError):
        setattr(decision, "proposed_action", (0.0, 0.0))
    with pytest.raises(ValueError, match="lifecycle_id"):
        _decision(lifecycle_id="")
    with pytest.raises(ValueError, match="decision_id"):
        _decision(decision_id="")
    with pytest.raises(ValueError, match="observation_id"):
        Decision(
            lifecycle_id="life-a",
            decision_id="life-a:0",
            observation_id="",
            action_codec_id="tests.normalized_action.v1",
            armed=True,
            observation=(0.0, 0.0, 0.0),
            proposed_action=(0.0, 0.0),
        )
    with pytest.raises(ValueError, match="armed"):
        DispatchAck.exact(
            Decision(
                lifecycle_id="life-a",
                decision_id="life-a:0",
                observation_id="life-a:observation:0",
                action_codec_id="tests.normalized_action.v1",
                armed=False,
                observation=(0.0, 0.0, 0.0),
                proposed_action=(0.0, 0.0),
            ),
            authority_id="tests.dispatch_authority.v1",
            policy_version="tests.dispatch_policy.v1",
            dispatch_receipt_id="life-a:0:dispatch",
        )
    with pytest.raises(ValueError, match="action codec"):
        _manifest().validate_decision(
            replace(decision, action_codec_id="tests.incompatible_action.v1")
        )


def test_numeric_record_payloads_are_defensively_copied_and_immutable() -> None:
    observation = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    action = np.array([0.25, -0.25], dtype=np.float32)
    decision = Decision(
        lifecycle_id="life-copy",
        decision_id="life-copy:0",
        observation_id="life-copy:observation:0",
        action_codec_id="tests.normalized_action.v1",
        armed=True,
        observation=observation,
        proposed_action=action,
    )
    dispatch = _exact_dispatch(decision)

    observation[:] = 9.0
    action[:] = 9.0
    assert decision.observation == (0.0, 0.5, 1.0)
    assert decision.proposed_action == (0.25, -0.25)
    assert dispatch.effective_action == (0.25, -0.25)

    bootstrap = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    transition = Transition(
        dispatch=dispatch,
        reward=1.0,
        discount=0.9,
        terminated=False,
        truncated=False,
        bootstrap_observation=bootstrap,
        bootstrap_observation_id="life-copy:observation:1",
        autoreset=False,
        next_decision_observation=bootstrap,
        next_decision_observation_id="life-copy:observation:1",
    )
    bootstrap[:] = 8.0
    assert transition.bootstrap_observation != (8.0, 8.0, 8.0)
    assert transition.next_decision_observation != (8.0, 8.0, 8.0)


def test_dispatch_ack_statuses_have_fail_closed_credit_semantics() -> None:
    decision = _decision()

    exact = _exact_dispatch(decision)
    assert exact.status is DispatchStatus.EXACT
    assert exact.effective_action == decision.proposed_action
    assert exact.dispatched and exact.transition_expected
    assert exact.learning_credit_allowed
    assert exact.authority_id == "tests.dispatch_authority.v1"
    assert exact.policy_version == "tests.dispatch_policy.v1"
    assert exact.dispatch_receipt_id == "life-a:0:dispatch"

    rebound = DispatchAck.rebound(
        decision,
        effective_action=(-0.5, 0.5),
        authority_id="tests.dispatch_authority.v1",
        policy_version="tests.dispatch_policy.v1",
        dispatch_receipt_id="life-a:0:dispatch-rebound",
    )
    assert rebound.status is DispatchStatus.REBOUND
    assert rebound.effective_action == (-0.5, 0.5)
    assert rebound.dispatched and rebound.transition_expected
    assert rebound.learning_credit_allowed
    with pytest.raises(ValueError, match="differ"):
        DispatchAck.rebound(
            decision,
            effective_action=decision.proposed_action,
            authority_id="tests.dispatch_authority.v1",
            policy_version="tests.dispatch_policy.v1",
            dispatch_receipt_id="life-a:0:dispatch-invalid",
        )

    vetoed = DispatchAck.vetoed(
        decision,
        reason="safety envelope",
        authority_id="tests.dispatch_authority.v1",
        policy_version="tests.dispatch_policy.v1",
    )
    assert vetoed.status is DispatchStatus.VETOED
    assert vetoed.effective_action is None
    assert not vetoed.dispatched and not vetoed.transition_expected
    assert not vetoed.learning_credit_allowed

    unsupported = DispatchAck.unsupported(
        decision,
        reason="adapter cannot rebind the credited action",
        authority_id="tests.dispatch_authority.v1",
        policy_version="tests.dispatch_policy.v1",
    )
    assert unsupported.status is DispatchStatus.UNSUPPORTED
    assert unsupported.effective_action is None
    assert not unsupported.dispatched and not unsupported.transition_expected
    assert not unsupported.learning_credit_allowed

    with pytest.raises(ValueError, match="dispatch|transition"):
        _transition(vetoed)
    with pytest.raises(ValueError, match="dispatch|transition"):
        _transition(unsupported)


def test_transition_validates_explicit_discount_boundaries_and_autoreset() -> None:
    exact = _exact_dispatch(_decision())
    continuing = _transition(exact)
    assert not continuing.is_boundary
    assert not continuing.is_autoreset
    assert continuing.lifecycle_id == exact.lifecycle_id
    assert continuing.decision_id == exact.decision_id
    assert continuing.effective_action == exact.effective_action

    terminated = _transition(
        exact,
        discount=0.0,
        terminated=True,
        bootstrap_observation=(9.0, 9.0, 9.0),
        bootstrap_observation_id="tests.observation:terminal",
        autoreset=True,
        next_decision_observation=(0.0, 0.0, 0.0),
        next_decision_observation_id="tests.observation:reset",
    )
    assert terminated.is_boundary and terminated.is_autoreset

    truncated = _transition(
        exact,
        discount=0.9,
        truncated=True,
        bootstrap_observation=(9.0, 9.0, 9.0),
        bootstrap_observation_id="tests.observation:truncated",
        autoreset=True,
        next_decision_observation=(0.0, 0.0, 0.0),
        next_decision_observation_id="tests.observation:reset",
    )
    assert truncated.is_boundary and truncated.is_autoreset

    for invalid_discount in (-0.1, 1.1, math.nan, math.inf):
        with pytest.raises(ValueError, match="discount"):
            _transition(exact, discount=invalid_discount)
    with pytest.raises(ValueError, match="terminated|discount"):
        _transition(exact, discount=0.9, terminated=True)
    with pytest.raises(ValueError, match="truncated|discount"):
        _transition(exact, discount=0.0, truncated=True)
    with pytest.raises(ValueError, match="next_decision_observation|boundary"):
        _transition(
            exact,
            bootstrap_observation=(1.0, 1.0, 1.0),
            bootstrap_observation_id="tests.observation:a",
            next_decision_observation=(2.0, 2.0, 2.0),
            next_decision_observation_id="tests.observation:a",
        )
    with pytest.raises(ValueError, match="reward"):
        _transition(exact, reward=math.nan)


def test_autoreset_is_explicit_even_when_terminal_and_reset_values_match() -> None:
    dispatch = _exact_dispatch(_decision())
    same_value = (0.0, 0.0, 0.0)
    reset = _transition(
        dispatch,
        discount=0.0,
        terminated=True,
        bootstrap_observation=same_value,
        bootstrap_observation_id="tests.observation:terminal",
        autoreset=True,
        next_decision_observation=same_value,
        next_decision_observation_id="tests.observation:reset",
    )
    assert reset.is_boundary and reset.is_autoreset

    no_reset = _transition(
        dispatch,
        discount=0.0,
        terminated=True,
        bootstrap_observation=same_value,
        bootstrap_observation_id="tests.observation:terminal",
        autoreset=False,
        next_decision_observation=None,
        next_decision_observation_id=None,
    )
    assert no_reset.is_boundary and not no_reset.is_autoreset
    with pytest.raises(ValueError, match="autoreset|next_decision_observation"):
        _transition(
            dispatch,
            discount=0.0,
            terminated=True,
            bootstrap_observation=same_value,
            bootstrap_observation_id="tests.observation:terminal",
            autoreset=True,
            next_decision_observation=None,
            next_decision_observation_id=None,
        )
    with pytest.raises(ValueError, match="autoreset|boundary"):
        _transition(dispatch, autoreset=True)
    with pytest.raises(ValueError, match="next_decision_observation"):
        _transition(
            dispatch,
            discount=0.0,
            terminated=True,
            bootstrap_observation=same_value,
            bootstrap_observation_id="tests.observation:terminal",
            autoreset=False,
            next_decision_observation=same_value,
            next_decision_observation_id="tests.observation:reset",
        )


class _FakeSession:
    """Tiny state machine specifying ownership behavior expected of adapters."""

    def __init__(self, *, dispatch_rebinding: bool = False) -> None:
        self._manifest = _manifest(dispatch_rebinding=dispatch_rebinding)
        self._current: Decision | None = None
        self._dispatch: DispatchAck | None = None
        self._consumed: set[tuple[str, str]] = set()
        self._decision_index = 0

    @property
    def manifest(self) -> AgentManifest:
        return self._manifest

    def start(self, lifecycle_id: str, initial_observation: Any) -> Decision:
        self.manifest.observation_spec.validate_value(initial_observation)
        self._decision_index = 0
        self._dispatch = None
        self._current = Decision(
            lifecycle_id=lifecycle_id,
            decision_id=f"{lifecycle_id}:0",
            observation_id=f"{lifecycle_id}:observation:0",
            action_codec_id=self.manifest.action_spec.semantic_id,
            armed=True,
            observation=initial_observation,
            proposed_action=(0.25, -0.25),
        )
        return self._current

    def bind_dispatch(
        self,
        decision: Decision,
        effective_action: Any | None = None,
        *,
        veto_reason: str | None = None,
    ) -> DispatchAck:
        if self._current is None or decision != self._current:
            raise DecisionOwnershipError("stale decision")
        self.manifest.validate_decision(decision)
        if self._dispatch is not None:
            raise DecisionOwnershipError("decision already has a pending settlement")
        if veto_reason is not None:
            self._dispatch = DispatchAck.vetoed(
                decision,
                reason=veto_reason,
                authority_id="tests.dispatch_authority.v1",
                policy_version="tests.dispatch_policy.v1",
            )
            return self._dispatch
        effective = decision.proposed_action if effective_action is None else effective_action
        self.manifest.action_spec.validate_value(effective)
        if effective == decision.proposed_action:
            self._dispatch = _exact_dispatch(decision)
        elif self.manifest.capabilities.dispatch_rebinding:
            self._dispatch = DispatchAck.rebound(
                decision,
                effective_action=effective,
                authority_id="tests.dispatch_authority.v1",
                policy_version="tests.dispatch_policy.v1",
                dispatch_receipt_id=f"{decision.decision_id}:dispatch",
            )
        else:
            self._dispatch = DispatchAck.unsupported(
                decision,
                reason="adapter cannot rebind the credited action",
                authority_id="tests.dispatch_authority.v1",
                policy_version="tests.dispatch_policy.v1",
            )
        return self._dispatch

    def advance(self, transition: Transition) -> StepResult:
        owner = (transition.lifecycle_id, transition.decision_id)
        if owner in self._consumed:
            raise DecisionOwnershipError("duplicate decision")
        if self._current is None or transition.lifecycle_id != self._current.lifecycle_id:
            raise DecisionOwnershipError("stale lifecycle")
        if transition.decision_id != self._current.decision_id:
            raise DecisionOwnershipError("stale decision")
        if self._dispatch is None or transition.dispatch != self._dispatch:
            raise DecisionOwnershipError("transition does not own the bound dispatch")
        self._consumed.add(owner)
        self._dispatch = None
        self._decision_index += 1
        next_decision = (
            None
            if transition.is_boundary and not transition.is_autoreset
            else Decision(
                lifecycle_id=self._current.lifecycle_id,
                decision_id=f"{self._current.lifecycle_id}:{self._decision_index}",
                observation_id=transition.next_decision_observation_id,
                action_codec_id=self.manifest.action_spec.semantic_id,
                armed=True,
                observation=transition.next_decision_observation,
                proposed_action=(0.25, -0.25),
            )
        )
        self._current = next_decision
        return StepResult(
            transition=transition,
            next_decision=next_decision,
            learning_applied=True,
            retry_required=False,
            rejection_reason=None,
        )


def test_rejected_step_requires_retry_and_cannot_arm_a_next_decision() -> None:
    decision = _decision(lifecycle_id="life-retry", decision_id="life-retry:0")
    transition = _transition(_exact_dispatch(decision))
    retry = StepResult(
        transition=transition,
        next_decision=None,
        learning_applied=False,
        retry_required=True,
        rejection_reason="atomic learner update rejected",
    )
    assert retry.retry_required
    assert retry.next_decision is None

    next_decision = _decision(
        lifecycle_id="life-retry",
        decision_id="life-retry:1",
        observation=transition.next_decision_observation,
    )
    with pytest.raises(ValueError, match="retry|next decision|next_decision"):
        StepResult(
            transition=transition,
            next_decision=next_decision,
            learning_applied=False,
            retry_required=True,
            rejection_reason="atomic learner update rejected",
        )


def test_applied_step_rejects_disarmed_next_decision_and_codec_drift() -> None:
    transition = _transition(_exact_dispatch(_decision()))
    next_decision = _decision(
        decision_id="life-a:1",
        observation_id=transition.next_decision_observation_id,
        observation=transition.next_decision_observation,
    )
    with pytest.raises(DecisionOwnershipError, match="armed"):
        StepResult(
            transition=transition,
            next_decision=replace(next_decision, armed=False),
            learning_applied=True,
            retry_required=False,
            rejection_reason=None,
        )
    with pytest.raises(DecisionOwnershipError, match="codec"):
        StepResult(
            transition=transition,
            next_decision=replace(
                next_decision,
                action_codec_id="tests.changed_action_codec.v1",
            ),
            learning_applied=True,
            retry_required=False,
            rejection_reason=None,
        )
    with pytest.raises(ValueError, match="retry_required"):
        StepResult(
            transition=transition,
            next_decision=None,
            learning_applied=False,
            retry_required=False,
            rejection_reason="atomic learner update rejected",
        )


def test_fake_session_is_structurally_host_facing() -> None:
    assert isinstance(_FakeSession(), ReferenceAgentSession)


def test_fake_session_rejects_duplicate_and_stale_decisions() -> None:
    session = _FakeSession()
    decision = session.start("life-a", (0.0, 0.0, 0.0))
    dispatch = session.bind_dispatch(decision)
    transition = _transition(dispatch)

    with pytest.raises(DecisionOwnershipError, match="pending|settled|duplicate"):
        session.bind_dispatch(decision)

    result = session.advance(transition)
    assert result.learning_applied
    assert result.next_decision is not None
    assert result.next_decision.decision_id == "life-a:1"
    with pytest.raises(DecisionOwnershipError, match="duplicate"):
        session.advance(transition)

    stale = session.start("life-stale", (0.0, 0.0, 0.0))
    stale_dispatch = session.bind_dispatch(stale)
    stale_transition = _transition(stale_dispatch)
    session.start("life-current", (0.0, 0.0, 0.0))
    with pytest.raises(DecisionOwnershipError, match="stale"):
        session.advance(stale_transition)


def test_fake_session_distinguishes_rebound_from_unsupported_dispatch() -> None:
    exact_only = _FakeSession(dispatch_rebinding=False)
    decision = exact_only.start("life-exact", (0.0, 0.0, 0.0))
    unsupported = exact_only.bind_dispatch(decision, effective_action=(-0.5, 0.5))
    assert unsupported.status is DispatchStatus.UNSUPPORTED

    rebinding = _FakeSession(dispatch_rebinding=True)
    decision = rebinding.start("life-rebind", (0.0, 0.0, 0.0))
    rebound = rebinding.bind_dispatch(decision, effective_action=(-0.5, 0.5))
    assert rebound.status is DispatchStatus.REBOUND
    result = rebinding.advance(_transition(rebound))
    assert result.learning_applied
