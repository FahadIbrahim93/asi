"""Host-facing interoperability contract for ASI reference-agent adapters.

This module defines transaction records and structural validation for agents
with different internal learners and action spaces.  Conformance is an L0
integration property only: it does not establish learning benefit, retention,
safety, robotics readiness, scientific evidence, or ``reference-dev`` status.

The first version deliberately keeps workload schedules, resource budgets,
checkpoint storage, and evidence policy outside the agent boundary.  Those
belong to the whole-life runner and its separately versioned protocols.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

REFERENCE_AGENT_MANIFEST_SCHEMA = "asi.reference_agent_manifest.v1"

_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._:-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _validate_json_value(value: Any, *, path: str) -> None:
    """Reject values that do not have one stable, finite JSON encoding."""

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite JSON numbers")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} JSON object keys must be strings")
            _validate_json_value(item, path=f"{path}.{key}")
        return
    raise ValueError(f"{path} is not a canonical JSON value")


def canonical_config_sha256(config: Mapping[str, Any]) -> str:
    """Return the SHA-256 of a canonical, finite JSON configuration.

    The helper intentionally rejects Python-specific containers and numeric
    objects instead of guessing how to serialize them.  Materialize every
    behavior-affecting default into ordinary JSON values before hashing.
    """

    if not isinstance(config, Mapping):
        raise ValueError("config must be a JSON mapping")
    _validate_json_value(config, path="config")
    try:
        payload = json.dumps(
            config,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("config must have a finite canonical JSON encoding") from exc
    return hashlib.sha256(payload).hexdigest()


def _require_safe_id(value: str, *, name: str) -> None:
    if not isinstance(value, str) or not value or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(
            f"{name} must be a nonempty lowercase identifier containing only "
            "letters, digits, '.', '_', ':', or '-'"
        )


def _shape_size(shape: tuple[int, ...]) -> int:
    size = 1
    for dimension in shape:
        size *= dimension
    return size


def _numeric_array(value: Any, *, name: str) -> np.ndarray[Any, Any]:
    """Materialize one finite real numeric value without retaining caller aliases."""

    try:
        array = np.array(value, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real numeric scalar or array") from exc
    if array.dtype.kind not in {"f", "i", "u"}:
        raise ValueError(f"{name} must contain only real numeric values, not bool")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _cast_representable(
    array: np.ndarray[Any, Any],
    *,
    dtype: np.dtype[Any],
    name: str,
) -> np.ndarray[Any, Any]:
    """Cast only values that the declared dtype can represent without overflow."""

    if dtype.kind in {"i", "u"}:
        if array.dtype.kind == "f" and np.any(array != np.trunc(array)):
            raise ValueError(f"{name} must contain integral values representable by {dtype.name}")
        limits = np.iinfo(dtype)
        if np.any(array < limits.min) or np.any(array > limits.max):
            raise ValueError(f"{name} contains a value not representable by {dtype.name}")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            cast = array.astype(dtype, copy=True)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains a value not representable by {dtype.name}") from exc
    if dtype.kind == "f" and not np.all(np.isfinite(cast)):
        raise ValueError(f"{name} contains a value not finitely representable by {dtype.name}")
    return cast


def _immutable_numeric(value: Any, *, name: str) -> int | float | tuple[Any, ...]:
    """Return a deeply immutable, finite snapshot of a host numeric payload."""

    array = _numeric_array(value, name=name)
    if array.shape == ():
        scalar = array.item()
        if array.dtype.kind == "f":
            frozen = float(scalar)
            if not math.isfinite(frozen):
                raise ValueError(f"{name} must be representable as a finite Python float")
            return frozen
        return int(scalar)

    def freeze_nested(item: Any) -> Any:
        if isinstance(item, list):
            return tuple(freeze_nested(child) for child in item)
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError(f"{name} must contain only finite values")
            return item
        if isinstance(item, int) and not isinstance(item, bool):
            return item
        raise ValueError(f"{name} must contain only real numeric values")

    frozen = freeze_nested(array.tolist())
    assert isinstance(frozen, tuple)
    return frozen


@dataclasses.dataclass(frozen=True, slots=True)
class SpaceSpec:
    """Flat scalar/tensor space used at the common agent boundary.

    ``discrete`` spaces are scalar integer indices in ``[0, cardinality)``.
    ``box`` spaces have one finite lower and upper bound per flattened value.
    Structured and dynamically shaped spaces require a future protocol version
    and must not be hidden in an unvalidated extras mapping.
    """

    kind: str
    shape: tuple[int, ...]
    dtype: str
    semantic_id: str
    cardinality: int | None = None
    low: tuple[float | int, ...] | None = None
    high: tuple[float | int, ...] | None = None

    def __post_init__(self) -> None:
        _require_safe_id(self.semantic_id, name="semantic_id")
        if not isinstance(self.shape, tuple) or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            for dimension in self.shape
        ):
            raise ValueError("shape must be a tuple of positive integer dimensions")
        try:
            dtype = np.dtype(self.dtype)
        except TypeError as exc:
            raise ValueError(f"unsupported dtype {self.dtype!r}") from exc
        if dtype.kind not in {"f", "i", "u"}:
            raise ValueError("dtype must be a real floating-point or integer dtype")
        if dtype.name != self.dtype:
            raise ValueError(f"dtype must use canonical spelling {dtype.name!r}")

        if self.kind == "discrete":
            if self.shape != ():
                raise ValueError("a discrete space must have scalar shape ()")
            if dtype.kind not in {"i", "u"}:
                raise ValueError("a discrete space dtype must be integer")
            if (
                isinstance(self.cardinality, bool)
                or not isinstance(self.cardinality, int)
                or self.cardinality <= 0
            ):
                raise ValueError("discrete cardinality must be a positive integer")
            if self.cardinality - 1 > int(np.iinfo(dtype).max):
                raise ValueError(
                    f"discrete cardinality is not representable by declared dtype {dtype.name}"
                )
            if self.low is not None or self.high is not None:
                raise ValueError("a discrete space cannot declare box bounds")
            return

        if self.kind != "box":
            raise ValueError("space kind must be 'discrete' or 'box'")
        if self.cardinality is not None:
            raise ValueError("a box space cannot declare a cardinality")
        if self.low is None or self.high is None:
            raise ValueError("a box space requires low and high bounds")
        lows = _numeric_array(self.low, name="box low bounds")
        highs = _numeric_array(self.high, name="box high bounds")
        size = _shape_size(self.shape)
        if lows.shape != (size,) or highs.shape != (size,):
            raise ValueError("box bounds must contain one value per flattened shape entry")
        if np.any(lows > highs):
            raise ValueError("every box low bound must be <= its high bound")
        cast_lows = _cast_representable(lows, dtype=dtype, name="box low bounds")
        cast_highs = _cast_representable(highs, dtype=dtype, name="box high bounds")
        if np.any(cast_lows > cast_highs):
            raise ValueError("box bounds collapse or reverse in the declared dtype")
        object.__setattr__(
            self,
            "low",
            tuple(
                float(value) if dtype.kind == "f" else int(value)
                for value in cast_lows.tolist()
            ),
        )
        object.__setattr__(
            self,
            "high",
            tuple(
                float(value) if dtype.kind == "f" else int(value)
                for value in cast_highs.tolist()
            ),
        )

    @classmethod
    def discrete(
        cls,
        *,
        cardinality: int,
        dtype: str,
        semantic_id: str,
    ) -> SpaceSpec:
        """Construct a scalar discrete-action specification."""

        return cls(
            kind="discrete",
            shape=(),
            dtype=dtype,
            semantic_id=semantic_id,
            cardinality=cardinality,
        )

    @classmethod
    def box(
        cls,
        *,
        shape: tuple[int, ...],
        dtype: str,
        low: Sequence[float | int],
        high: Sequence[float | int],
        semantic_id: str,
    ) -> SpaceSpec:
        """Construct a fixed-shape bounded tensor specification."""

        return cls(
            kind="box",
            shape=tuple(shape),
            dtype=dtype,
            semantic_id=semantic_id,
            low=tuple(low),
            high=tuple(high),
        )

    def validate_value(self, value: Any) -> None:
        """Raise when ``value`` cannot inhabit this space without ambiguity."""

        array = _numeric_array(value, name="space value")
        dtype = np.dtype(self.dtype)

        if self.kind == "discrete":
            if array.shape != ():
                raise ValueError(f"discrete value shape must be (), got {array.shape}")
            if array.dtype.kind not in {"i", "u"}:
                raise TypeError("discrete value must be an integer")
            integer = int(array)
            assert self.cardinality is not None
            if integer < 0 or integer >= self.cardinality:
                raise ValueError(
                    f"discrete value {integer} is outside cardinality range "
                    f"[0, {self.cardinality})"
                )
            _cast_representable(array, dtype=dtype, name="discrete value")
            return

        if array.shape != self.shape:
            raise ValueError(f"box value shape must be {self.shape}, got {array.shape}")
        cast = _cast_representable(array, dtype=dtype, name="box value")
        assert self.low is not None and self.high is not None
        flattened = cast.reshape(-1)
        lows = np.asarray(self.low, dtype=dtype)
        highs = np.asarray(self.high, dtype=dtype)
        if np.any(flattened < lows) or np.any(flattened > highs):
            raise ValueError("box value is outside the declared bounds/range")


@dataclasses.dataclass(frozen=True, slots=True)
class AgentCapabilities:
    """Mechanism disclosures; none of these fields is a performance claim."""

    explicit_discount: bool
    exact_checkpoint_resume: bool
    dispatch_rebinding: bool
    compiled_rollout: bool
    context_inputs: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "explicit_discount",
            "exact_checkpoint_resume",
            "dispatch_rebinding",
            "compiled_rollout",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if not isinstance(self.context_inputs, tuple):
            raise ValueError("context_inputs must be an immutable tuple")
        if len(set(self.context_inputs)) != len(self.context_inputs):
            raise ValueError("context_inputs must not contain duplicates")
        for context_input in self.context_inputs:
            _require_safe_id(context_input, name="context input")


@dataclasses.dataclass(frozen=True, slots=True)
class AgentManifest:
    """Versioned structural declaration for one reference-agent adapter."""

    schema: str
    implementation_id: str
    config_sha256: str
    observation_spec: SpaceSpec
    action_spec: SpaceSpec
    capabilities: AgentCapabilities

    def __post_init__(self) -> None:
        if self.schema != REFERENCE_AGENT_MANIFEST_SCHEMA:
            raise ValueError(
                f"schema must be {REFERENCE_AGENT_MANIFEST_SCHEMA!r}, got {self.schema!r}"
            )
        _require_safe_id(self.implementation_id, name="implementation_id")
        if not isinstance(self.config_sha256, str) or _SHA256.fullmatch(
            self.config_sha256
        ) is None:
            raise ValueError("config_sha256 must be a lowercase 64-character SHA-256 digest")
        if not isinstance(self.observation_spec, SpaceSpec):
            raise ValueError("observation_spec must be a SpaceSpec")
        if not isinstance(self.action_spec, SpaceSpec):
            raise ValueError("action_spec must be a SpaceSpec")
        if not isinstance(self.capabilities, AgentCapabilities):
            raise ValueError("capabilities must be AgentCapabilities")
        if not self.capabilities.explicit_discount:
            raise ValueError("reference-agent adapters must consume an explicit discount")

    def validate_decision(self, decision: Decision) -> None:
        """Validate a decision against this adapter's declared codecs and spaces."""

        if not isinstance(decision, Decision):
            raise ValueError("decision must be a Decision")
        if not decision.armed:
            raise ValueError("decision must be armed")
        if decision.action_codec_id != self.action_spec.semantic_id:
            raise ValueError("decision action codec does not match the manifest action spec")
        self.observation_spec.validate_value(decision.observation)
        self.action_spec.validate_value(decision.proposed_action)


@dataclasses.dataclass(frozen=True, slots=True)
class Decision:
    """One lifecycle-scoped proposed action awaiting dispatch settlement."""

    lifecycle_id: str
    decision_id: str
    observation_id: str
    action_codec_id: str
    armed: bool
    observation: Any
    proposed_action: Any

    def __post_init__(self) -> None:
        _require_safe_id(self.lifecycle_id, name="lifecycle_id")
        _require_safe_id(self.decision_id, name="decision_id")
        _require_safe_id(self.observation_id, name="observation_id")
        _require_safe_id(self.action_codec_id, name="action_codec_id")
        if not isinstance(self.armed, bool):
            raise ValueError("armed must be boolean")
        object.__setattr__(
            self,
            "observation",
            _immutable_numeric(self.observation, name="decision observation"),
        )
        object.__setattr__(
            self,
            "proposed_action",
            _immutable_numeric(self.proposed_action, name="decision proposed_action"),
        )


class DecisionOwnershipError(ValueError):
    """A decision, dispatch, or transition does not own the current event."""


class DispatchStatus(enum.Enum):
    """How independent dispatch authority settled a proposed action."""

    EXACT = "exact"
    REBOUND = "rebound"
    VETOED = "vetoed"
    UNSUPPORTED = "unsupported"


def _values_equal(left: Any, right: Any) -> bool:
    try:
        return bool(np.array_equal(np.asarray(left), np.asarray(right), equal_nan=False))
    except (TypeError, ValueError):
        return bool(left == right)


@dataclasses.dataclass(frozen=True, slots=True)
class DispatchAck:
    """Decision-bound settlement from dispatch/safety authority.

    ``REBOUND`` means the adapter has explicitly rebound learning ownership to
    the effective action.  Merely clipping or replacing an action is not a
    rebound.  Adapters without that mechanism must return ``UNSUPPORTED`` and
    the host must not dispatch or construct a learning transition.
    """

    decision: Decision
    status: DispatchStatus
    effective_action: Any | None
    authority_id: str
    policy_version: str
    dispatch_receipt_id: str | None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.decision, Decision):
            raise ValueError("dispatch must reference a Decision")
        if not isinstance(self.status, DispatchStatus):
            raise ValueError("dispatch status must be a DispatchStatus")
        if not self.decision.armed:
            raise ValueError("dispatch requires an armed decision")
        _require_safe_id(self.authority_id, name="authority_id")
        _require_safe_id(self.policy_version, name="policy_version")
        if self.effective_action is not None:
            object.__setattr__(
                self,
                "effective_action",
                _immutable_numeric(self.effective_action, name="dispatch effective_action"),
            )
        if self.status is DispatchStatus.EXACT:
            if self.effective_action is None or not _values_equal(
                self.effective_action, self.decision.proposed_action
            ):
                raise ValueError("exact dispatch action must equal the proposal")
        elif self.status is DispatchStatus.REBOUND:
            if self.effective_action is None:
                raise ValueError("rebound dispatch requires an effective action")
            if _values_equal(self.effective_action, self.decision.proposed_action):
                raise ValueError("rebound effective action must differ from the proposal")
        else:
            if self.effective_action is not None:
                raise ValueError("vetoed/unsupported dispatch cannot carry an effective action")
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise ValueError("vetoed/unsupported dispatch requires a nonempty reason")
        if self.dispatched:
            if (
                not isinstance(self.dispatch_receipt_id, str)
                or not self.dispatch_receipt_id
            ):
                raise ValueError("a dispatched action requires a dispatch_receipt_id")
            _require_safe_id(self.dispatch_receipt_id, name="dispatch_receipt_id")
            if self.reason is not None:
                raise ValueError("an exact/rebound dispatch cannot carry a rejection reason")
        elif self.dispatch_receipt_id is not None:
            raise ValueError("vetoed/unsupported dispatch cannot carry a dispatch_receipt_id")

    @classmethod
    def exact(
        cls,
        decision: Decision,
        *,
        authority_id: str,
        policy_version: str,
        dispatch_receipt_id: str,
    ) -> DispatchAck:
        """Acknowledge that the proposed action will be dispatched unchanged."""

        return cls(
            decision=decision,
            status=DispatchStatus.EXACT,
            effective_action=decision.proposed_action,
            authority_id=authority_id,
            policy_version=policy_version,
            dispatch_receipt_id=dispatch_receipt_id,
        )

    @classmethod
    def rebound(
        cls,
        decision: Decision,
        *,
        effective_action: Any,
        authority_id: str,
        policy_version: str,
        dispatch_receipt_id: str,
    ) -> DispatchAck:
        """Acknowledge explicit credit rebinding to a changed action."""

        return cls(
            decision=decision,
            status=DispatchStatus.REBOUND,
            effective_action=effective_action,
            authority_id=authority_id,
            policy_version=policy_version,
            dispatch_receipt_id=dispatch_receipt_id,
        )

    @classmethod
    def vetoed(
        cls,
        decision: Decision,
        *,
        reason: str,
        authority_id: str,
        policy_version: str,
    ) -> DispatchAck:
        """Record an independent safety/dispatch veto without execution."""

        return cls(
            decision=decision,
            status=DispatchStatus.VETOED,
            effective_action=None,
            authority_id=authority_id,
            policy_version=policy_version,
            dispatch_receipt_id=None,
            reason=reason,
        )

    @classmethod
    def unsupported(
        cls,
        decision: Decision,
        *,
        reason: str,
        authority_id: str,
        policy_version: str,
    ) -> DispatchAck:
        """Refuse a changed action when the adapter cannot rebind credit."""

        return cls(
            decision=decision,
            status=DispatchStatus.UNSUPPORTED,
            effective_action=None,
            authority_id=authority_id,
            policy_version=policy_version,
            dispatch_receipt_id=None,
            reason=reason,
        )

    @property
    def lifecycle_id(self) -> str:
        return self.decision.lifecycle_id

    @property
    def decision_id(self) -> str:
        return self.decision.decision_id

    @property
    def dispatched(self) -> bool:
        return self.status in {DispatchStatus.EXACT, DispatchStatus.REBOUND}

    @property
    def transition_expected(self) -> bool:
        return self.dispatched

    @property
    def learning_credit_allowed(self) -> bool:
        return self.dispatched


def _finite_scalar(value: Any, *, name: str) -> float:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if array.shape != () or array.dtype.kind not in {"f", "i", "u"}:
        raise ValueError(f"{name} must be a finite real scalar")
    scalar = float(array)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


@dataclasses.dataclass(frozen=True, slots=True)
class Transition:
    """One dispatched outcome with explicit continuing-boundary semantics."""

    dispatch: DispatchAck
    reward: float
    discount: float
    terminated: bool
    truncated: bool
    bootstrap_observation: Any
    bootstrap_observation_id: str
    autoreset: bool
    next_decision_observation: Any | None
    next_decision_observation_id: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.dispatch, DispatchAck):
            raise ValueError("transition dispatch must be a DispatchAck")
        if not self.dispatch.transition_expected or not self.dispatch.learning_credit_allowed:
            raise ValueError("transition requires a dispatched, credit-bearing dispatch ack")
        reward = _finite_scalar(self.reward, name="reward")
        discount = _finite_scalar(self.discount, name="discount")
        if discount < 0.0 or discount > 1.0:
            raise ValueError("discount must be in [0, 1]")
        if not isinstance(self.terminated, (bool, np.bool_)):
            raise ValueError("terminated must be boolean")
        if not isinstance(self.truncated, (bool, np.bool_)):
            raise ValueError("truncated must be boolean")
        if not isinstance(self.autoreset, (bool, np.bool_)):
            raise ValueError("autoreset must be boolean")
        terminated = bool(self.terminated)
        truncated = bool(self.truncated)
        autoreset = bool(self.autoreset)
        _require_safe_id(self.bootstrap_observation_id, name="bootstrap_observation_id")
        bootstrap_observation = _immutable_numeric(
            self.bootstrap_observation,
            name="bootstrap_observation",
        )
        if terminated and discount != 0.0:
            raise ValueError("terminated transitions require discount == 0")
        if truncated and not terminated and discount <= 0.0:
            raise ValueError("a truncated nonterminal transition requires discount > 0")
        boundary = terminated or truncated
        if autoreset and not boundary:
            raise ValueError("autoreset is valid only at a terminated/truncated boundary")
        if boundary:
            if autoreset:
                if self.next_decision_observation is None:
                    raise ValueError("autoreset requires a next_decision_observation")
                if self.next_decision_observation_id is None:
                    raise ValueError("autoreset requires a next_decision_observation_id")
                _require_safe_id(
                    self.next_decision_observation_id,
                    name="next_decision_observation_id",
                )
                if self.next_decision_observation_id == self.bootstrap_observation_id:
                    raise ValueError(
                        "autoreset final and reset observations require distinct identities"
                    )
                next_observation = _immutable_numeric(
                    self.next_decision_observation,
                    name="next_decision_observation",
                )
            else:
                if self.next_decision_observation is not None:
                    raise ValueError(
                        "a boundary without autoreset cannot carry next_decision_observation"
                    )
                if self.next_decision_observation_id is not None:
                    raise ValueError(
                        "a boundary without autoreset cannot carry "
                        "next_decision_observation_id"
                    )
                next_observation = None
        else:
            if self.next_decision_observation is None:
                raise ValueError("a continuing transition requires next_decision_observation")
            if self.next_decision_observation_id is None:
                raise ValueError("a continuing transition requires next_decision_observation_id")
            _require_safe_id(
                self.next_decision_observation_id,
                name="next_decision_observation_id",
            )
            if self.next_decision_observation_id != self.bootstrap_observation_id:
                raise ValueError(
                    "continuing bootstrap and next-decision observations must share identity"
                )
            next_observation = _immutable_numeric(
                self.next_decision_observation,
                name="next_decision_observation",
            )
            if not _values_equal(bootstrap_observation, next_observation):
                raise ValueError(
                    "next_decision_observation must equal the bootstrap observation "
                    "away from a boundary"
                )
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "discount", discount)
        object.__setattr__(self, "terminated", terminated)
        object.__setattr__(self, "truncated", truncated)
        object.__setattr__(self, "autoreset", autoreset)
        object.__setattr__(self, "bootstrap_observation", bootstrap_observation)
        object.__setattr__(self, "next_decision_observation", next_observation)

    @property
    def lifecycle_id(self) -> str:
        return self.dispatch.lifecycle_id

    @property
    def decision_id(self) -> str:
        return self.dispatch.decision_id

    @property
    def effective_action(self) -> Any:
        return self.dispatch.effective_action

    @property
    def dispatch_receipt_id(self) -> str:
        assert self.dispatch.dispatch_receipt_id is not None
        return self.dispatch.dispatch_receipt_id

    @property
    def is_boundary(self) -> bool:
        return self.terminated or self.truncated

    @property
    def is_autoreset(self) -> bool:
        return self.autoreset


@dataclasses.dataclass(frozen=True, slots=True)
class StepResult:
    """Atomic adapter result for one owned transition."""

    transition: Transition
    next_decision: Decision | None
    learning_applied: bool
    retry_required: bool
    rejection_reason: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.transition, Transition):
            raise ValueError("step result transition must be a Transition")
        if not isinstance(self.learning_applied, bool):
            raise ValueError("learning_applied must be boolean")
        if not isinstance(self.retry_required, bool):
            raise ValueError("retry_required must be boolean")
        if not self.learning_applied:
            if not self.retry_required:
                raise ValueError("a rejected update requires retry_required=True")
            if not isinstance(self.rejection_reason, str) or not self.rejection_reason.strip():
                raise ValueError("a rejected update requires a nonempty rejection_reason")
            if self.next_decision is not None:
                raise ValueError("a retry-required update cannot arm a next decision")
            return
        if self.retry_required:
            raise ValueError("an applied update requires retry_required=False")
        if self.rejection_reason is not None:
            raise ValueError("an applied update cannot carry a rejection_reason")
        if self.next_decision is None:
            if not self.transition.is_boundary or self.transition.is_autoreset:
                raise ValueError("a continuing/autoreset transition must arm a next decision")
            return
        if not isinstance(self.next_decision, Decision):
            raise ValueError("next_decision must be a Decision or None")
        if not self.next_decision.armed:
            raise DecisionOwnershipError("next decision must be armed")
        if self.next_decision.lifecycle_id != self.transition.lifecycle_id:
            raise DecisionOwnershipError("next decision belongs to a different lifecycle")
        if (
            self.next_decision.action_codec_id
            != self.transition.dispatch.decision.action_codec_id
        ):
            raise DecisionOwnershipError("next decision action codec changed within the life")
        if self.next_decision.decision_id == self.transition.decision_id:
            raise DecisionOwnershipError("next decision must have a fresh decision_id")
        if not _values_equal(
            self.next_decision.observation,
            self.transition.next_decision_observation,
        ):
            raise DecisionOwnershipError(
                "next decision does not own next_decision_observation"
            )
        if (
            self.next_decision.observation_id
            != self.transition.next_decision_observation_id
        ):
            raise DecisionOwnershipError(
                "next decision does not own next_decision_observation_id"
            )


@runtime_checkable
class ReferenceAgentSession(Protocol):
    """Stateful host adapter; structural conformance makes no evidence claim."""

    @property
    def manifest(self) -> AgentManifest: ...

    def start(self, lifecycle_id: str, initial_observation: Any) -> Decision: ...

    def bind_dispatch(
        self,
        decision: Decision,
        effective_action: Any | None = None,
        *,
        veto_reason: str | None = None,
    ) -> DispatchAck: ...

    def advance(self, transition: Transition) -> StepResult: ...


__all__ = [
    "REFERENCE_AGENT_MANIFEST_SCHEMA",
    "AgentCapabilities",
    "AgentManifest",
    "Decision",
    "DecisionOwnershipError",
    "DispatchAck",
    "DispatchStatus",
    "ReferenceAgentSession",
    "SpaceSpec",
    "StepResult",
    "Transition",
    "canonical_config_sha256",
]
