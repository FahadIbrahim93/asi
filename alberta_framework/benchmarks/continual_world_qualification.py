"""Strict setup receipts for an isolated Continual World CW20 smoke lane.

This module deliberately does not import the legacy TensorFlow/Gym/MuJoCo
stack.  A separately built immutable runtime executes the official benchmark;
the host validates its fixed-action trace before any learning comparison is
allowed.  The smoke lane is permanently nonpromoting.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import operator
import sys
from typing import Any, SupportsIndex, cast

import numpy as np

import alberta_framework.benchmarks.external_qualification as external_qualification_module
from alberta_framework.benchmarks.qualification_provenance import (
    QualificationIdentity,
    collect_qualification_identity,
    identity_from_payload,
    require_current_identity,
)

SCHEMA = "asi.continual-world.fixed-action-smoke.v1"
QUALIFICATION_ISSUE = 1580
QUALIFICATION_LANE_ID = "continual-world-cw20"
PAPER_REVISION = "arXiv:2105.10919v3"
OFFICIAL_REPOSITORY = "https://github.com/awarelab/continual_world.git"
OFFICIAL_COMMIT = "73f63bb4fa0b5d00bda973e20dfb783bfcf1b8aa"
METAWORLD_COMMIT = "0875192baaa91c43523708f55866d98eaf3facaf"
OFFICIAL_SETUP_BLOB = "125705947e810f8a41e7f9560d429d444c06694f"
OFFICIAL_DOCKERFILE_BLOB = "6282da996f68e58549f3fcf15ba5d326b1f5ef50"
FROZEN_DEVELOPMENT_SEED = 1_580_000
FROZEN_STEPS_PER_TASK = 2
CW10_TASKS = (
    "hammer-v1",
    "push-wall-v1",
    "faucet-close-v1",
    "push-back-v1",
    "stick-pull-v1",
    "handle-press-side-v1",
    "push-v1",
    "shelf-place-v1",
    "window-close-v1",
    "peg-unplug-side-v1",
)
CW20_TASKS = CW10_TASKS + CW10_TASKS
FIXED_ACTION = (0.0, 0.0, 0.0, 0.0)
OUTCOME_SCOPE = "official_runtime_fixed_action_trace_only"
OUTCOMES = ("inconclusive", "rejected", "supported")
PERSISTENT_ENVIRONMENT_BYTE_SCOPE = "external_runtime_numeric_arrays_nbytes_sum"
_MAX_BYTES = 1 << 30
_MAX_JSON_BYTES = 1 << 20
_MAX_JSON_CONTAINER_ITEMS = 10_000
_MAX_JSON_DEPTH = 64
_MAX_JSON_NODES = 50_000
_MAX_JSON_STRING_BYTES = 100_000
_INT32_MAX = 2**31 - 1
_WORKLOAD_REGISTRY = (
    ("action_dtype", "float32"),
    ("action_shape", (4,)),
    ("fixed_action", FIXED_ACTION),
    ("observation_dtype", "float32"),
    ("observation_shape", (32,)),
    ("outcome_scope", OUTCOME_SCOPE),
    ("outcomes", OUTCOMES),
    ("persistent_environment_byte_scope", PERSISTENT_ENVIRONMENT_BYTE_SCOPE),
    ("seed", FROZEN_DEVELOPMENT_SEED),
    ("steps_per_task", FROZEN_STEPS_PER_TASK),
    ("success_dtype", "bool"),
    ("task_index_dtype", "int32"),
    ("tasks", CW20_TASKS),
    ("timing_is_telemetry_only", True),
    ("learner_boundary_information", ()),
    ("learner_task_information", ()),
    ("negative_outcome_retained", True),
    ("scientific_promotion_allowed", False),
)
_PAPER_REGISTRY = (
    ("metaworld_commit", METAWORLD_COMMIT),
    ("official_commit", OFFICIAL_COMMIT),
    ("official_dockerfile_blob", OFFICIAL_DOCKERFILE_BLOB),
    ("official_repository", OFFICIAL_REPOSITORY),
    ("official_setup_blob", OFFICIAL_SETUP_BLOB),
    ("paper_revision", PAPER_REVISION),
    ("qualification_issue", QUALIFICATION_ISSUE),
    ("qualification_lane_id", QUALIFICATION_LANE_ID),
)


def _require_authoritative_external_plan() -> None:
    plan = external_qualification_module.qualification_plan(QUALIFICATION_ISSUE)
    revisions = plan.code_revisions
    if (
        plan.lane_id != QUALIFICATION_LANE_ID
        or len(revisions) != 1
        or revisions[0].repository != OFFICIAL_REPOSITORY
        or revisions[0].commit != OFFICIAL_COMMIT
    ):
        raise ValueError("Continual World pin differs from the external qualification authority")


def _current_identity() -> QualificationIdentity:
    _require_authoritative_external_plan()
    return collect_qualification_identity(
        lane_module=sys.modules[__name__],
        dependency_modules=(external_qualification_module,),
        workload_registry=_WORKLOAD_REGISTRY,
        paper_registry=_PAPER_REGISTRY,
    )


def _int(value: object, *, name: str, minimum: int = 0, maximum: int = _INT32_MAX) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an exact integer")
    result = operator.index(cast(SupportsIndex, value))
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} lies outside the bounded protocol")
    return result


def _digest(value: object, *, name: str, prefixed: bool = True) -> str:
    prefix = "sha256:" if prefixed else ""
    if (
        type(value) is not str
        or not value.startswith(prefix)
        or len(value) != len(prefix) + 64
        or any(character not in "0123456789abcdef" for character in value[len(prefix) :])
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256 identity")
    return value


def _preflight_json_tree(value: object) -> None:
    """Bound an exact primitive JSON tree before recursive serialization."""
    stack: list[tuple[object, int]] = [(value, 0)]
    seen_containers: set[int] = set()
    nodes = 0
    conservative_bytes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            raise ValueError("payload exceeds its JSON node or depth limit")
        if item is None or type(item) is bool:
            conservative_bytes += 5
        elif type(item) is int:
            if not -(1 << 63) <= item <= (1 << 63) - 1:
                raise ValueError("payload integers must be signed 64-bit values")
            conservative_bytes += 21
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError("payload must contain only finite floats")
            conservative_bytes += 32
        elif type(item) is str:
            encoded_length = len(item.encode("utf-8"))
            if len(item) > _MAX_JSON_STRING_BYTES or encoded_length > _MAX_JSON_STRING_BYTES:
                raise ValueError("payload string exceeds its byte or character limit")
            conservative_bytes += 2 + 12 * len(item)
        elif type(item) is list:
            identity = id(item)
            if identity in seen_containers:
                raise ValueError("payload cannot contain cycles or container aliases")
            seen_containers.add(identity)
            if len(item) > _MAX_JSON_CONTAINER_ITEMS:
                raise ValueError("payload list exceeds its item limit")
            conservative_bytes += len(item) + 2
            stack.extend((child, depth + 1) for child in item)
        elif type(item) is dict:
            identity = id(item)
            if identity in seen_containers:
                raise ValueError("payload cannot contain cycles or container aliases")
            seen_containers.add(identity)
            if len(item) > _MAX_JSON_CONTAINER_ITEMS:
                raise ValueError("payload object exceeds its item limit")
            conservative_bytes += len(item) + 2
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError("payload object keys must be exact strings")
                encoded_length = len(key.encode("utf-8"))
                if len(key) > _MAX_JSON_STRING_BYTES or encoded_length > _MAX_JSON_STRING_BYTES:
                    raise ValueError("payload object key exceeds its byte or character limit")
                nodes += 1
                if nodes > _MAX_JSON_NODES:
                    raise ValueError("payload exceeds its JSON node limit")
                conservative_bytes += 3 + 12 * len(key)
                stack.append((child, depth + 1))
        else:
            raise ValueError("payload must use exact primitive JSON values")
        if conservative_bytes > _MAX_JSON_BYTES:
            raise ValueError("payload exceeds the one-MiB receipt ceiling")


def _canonical(value: object) -> bytes:
    _preflight_json_tree(value)
    try:
        encoded = json.dumps(
            value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError, UnicodeEncodeError) as error:
        raise ValueError("payload must be finite canonical JSON") from error
    if len(encoded) > _MAX_JSON_BYTES:
        raise ValueError("payload exceeds the one-MiB receipt ceiling")
    return encoded


@dataclasses.dataclass(frozen=True, slots=True)
class IsolatedRuntimeIdentity:
    """Content identity supplied only after the legacy runtime is built."""

    image_digest: str
    mujoco_archive_sha256: str
    python_version: str
    tensorflow_version: str
    mujoco_py_version: str
    gym_version: str
    numpy_version: str
    platform: str

    def __post_init__(self) -> None:
        _digest(self.image_digest, name="image_digest")
        _digest(self.mujoco_archive_sha256, name="mujoco_archive_sha256", prefixed=False)
        for name in (
            "python_version",
            "tensorflow_version",
            "mujoco_py_version",
            "gym_version",
            "numpy_version",
            "platform",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value or len(value.encode("utf-8")) > 128:
                raise ValueError(f"{name} must be bounded exact text")


@dataclasses.dataclass(frozen=True, slots=True)
class ContinualWorldSmokePlan:
    """Frozen workload plus one immutable, externally built runtime identity."""

    runtime: IsolatedRuntimeIdentity
    seed: int = FROZEN_DEVELOPMENT_SEED
    steps_per_task: int = FROZEN_STEPS_PER_TASK

    def __post_init__(self) -> None:
        _require_authoritative_external_plan()
        if type(self.runtime) is not IsolatedRuntimeIdentity:
            raise ValueError("runtime must be an exact IsolatedRuntimeIdentity")
        IsolatedRuntimeIdentity.__post_init__(self.runtime)
        if _int(self.seed, name="seed") != FROZEN_DEVELOPMENT_SEED:
            raise ValueError("development seed is frozen")
        if _int(self.steps_per_task, name="steps_per_task", minimum=1, maximum=200) != 2:
            raise ValueError("smoke steps_per_task is frozen")

    def payload(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "paper_revision": PAPER_REVISION,
            "qualification_issue": QUALIFICATION_ISSUE,
            "qualification_lane_id": QUALIFICATION_LANE_ID,
            "official_repository": OFFICIAL_REPOSITORY,
            "official_commit": OFFICIAL_COMMIT,
            "metaworld_commit": METAWORLD_COMMIT,
            "official_setup_blob": OFFICIAL_SETUP_BLOB,
            "official_dockerfile_blob": OFFICIAL_DOCKERFILE_BLOB,
            "runtime": dataclasses.asdict(self.runtime),
            "seed": self.seed,
            "tasks": list(CW20_TASKS),
            "steps_per_task": self.steps_per_task,
            "fixed_action": list(FIXED_ACTION),
            "allowed_boundary_information": ["evaluator_task_index", "evaluator_task_boundary"],
            "allowed_task_information": ["evaluator_task_name"],
            "learner_boundary_information": [],
            "learner_task_information": [],
            "development_only": True,
            "scientific_promotion_allowed": False,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical(self.payload())).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class ContinualWorldSmokeReceipt:
    """Bounded result of one official-runtime, fixed-action CW20 smoke trace."""

    schema: str
    plan: ContinualWorldSmokePlan
    identity: QualificationIdentity
    plan_sha256: str
    action_sha256: str
    observation_sha256: str
    reward_sha256: str
    success_sha256: str
    task_index_sha256: str
    environment_steps: int
    data_steps: int
    learner_updates: int
    model_queries: int
    persistent_mechanism_bytes: int
    persistent_environment_numeric_bytes: int
    persistent_environment_byte_scope: str
    action_bytes: int
    observation_bytes: int
    reward_bytes: int
    success_bytes: int
    task_index_bytes: int
    timing_ns: int
    timing_is_telemetry_only: bool
    mechanism_off: bool
    outcome: str
    outcome_scope: str
    negative_outcome_retained: bool
    development_only: bool
    scientific_promotion_allowed: bool

    def __post_init__(self) -> None:
        if type(self.schema) is not str or self.schema != SCHEMA:
            raise ValueError("unsupported Continual World receipt schema")
        if type(self.plan) is not ContinualWorldSmokePlan:
            raise ValueError("plan must be an exact ContinualWorldSmokePlan")
        ContinualWorldSmokePlan.__post_init__(self.plan)
        require_current_identity(self.identity, _current_identity())
        if _digest(self.plan_sha256, name="plan_sha256", prefixed=False) != self.plan.sha256:
            raise ValueError("plan_sha256 does not bind the exact plan")
        for name in (
            "action_sha256",
            "observation_sha256",
            "reward_sha256",
            "success_sha256",
            "task_index_sha256",
        ):
            _digest(getattr(self, name), name=name, prefixed=False)
        horizon = len(CW20_TASKS) * self.plan.steps_per_task
        if _int(self.environment_steps, name="environment_steps", minimum=1) != horizon:
            raise ValueError("environment_steps do not match the frozen CW20 smoke horizon")
        exact_zero = ("data_steps", "learner_updates", "model_queries")
        if any(_int(getattr(self, name), name=name) != 0 for name in exact_zero):
            raise ValueError("fixed-action mechanism-off cannot train or query a model")
        expected_mechanism_bytes = len(FIXED_ACTION) * np.dtype(np.float32).itemsize
        if (
            _int(self.persistent_mechanism_bytes, name="persistent_mechanism_bytes")
            != expected_mechanism_bytes
        ):
            raise ValueError("persistent_mechanism_bytes must equal the fixed float32 action")
        _int(
            self.persistent_environment_numeric_bytes,
            name="persistent_environment_numeric_bytes",
            minimum=1,
            maximum=_MAX_BYTES,
        )
        if (
            type(self.persistent_environment_byte_scope) is not str
            or self.persistent_environment_byte_scope != PERSISTENT_ENVIRONMENT_BYTE_SCOPE
        ):
            raise ValueError("environment byte scope differs from the frozen accounting contract")
        trace_bytes = {
            "action_bytes": horizon * len(FIXED_ACTION) * np.dtype(np.float32).itemsize,
            "observation_bytes": horizon * 32 * np.dtype(np.float32).itemsize,
            "reward_bytes": horizon * np.dtype(np.float32).itemsize,
            "success_bytes": horizon * np.dtype(np.bool_).itemsize,
            "task_index_bytes": horizon * np.dtype(np.int32).itemsize,
        }
        if any(
            _int(getattr(self, name), name=name, maximum=_MAX_BYTES) != expected
            for name, expected in trace_bytes.items()
        ):
            raise ValueError("trace byte receipt differs from the exact frozen arrays")
        _int(self.timing_ns, name="timing_ns", maximum=2**63 - 1)
        if self.timing_is_telemetry_only is not True or self.mechanism_off is not True:
            raise ValueError("smoke timing/mechanism policy drifted")
        if type(self.outcome) is not str or self.outcome not in OUTCOMES:
            raise ValueError("outcome must use the frozen retention vocabulary")
        if type(self.outcome_scope) is not str or self.outcome_scope != OUTCOME_SCOPE:
            raise ValueError("outcome scope must remain limited to the fixed-action runtime smoke")
        if (
            self.negative_outcome_retained is not True
            or self.development_only is not True
            or self.scientific_promotion_allowed is not False
        ):
            raise ValueError("Continual World smoke receipts are permanently nonpromoting")

    def payload(self) -> dict[str, object]:
        self.__post_init__()
        return cast(dict[str, object], json.loads(json.dumps(dataclasses.asdict(self))))


def _array_hash(domain: bytes, value: np.ndarray) -> str:
    digest = hashlib.sha256(domain)
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def build_smoke_receipt(
    plan: ContinualWorldSmokePlan,
    *,
    actions: np.ndarray,
    observations: np.ndarray,
    rewards: np.ndarray,
    successes: np.ndarray,
    task_indices: np.ndarray,
    persistent_environment_numeric_bytes: int,
    timing_ns: int,
    outcome: str,
) -> ContinualWorldSmokeReceipt:
    """Snapshot one external official-runtime trace into a strict host receipt."""
    if type(plan) is not ContinualWorldSmokePlan:
        raise ValueError("plan must be exact")
    ContinualWorldSmokePlan.__post_init__(plan)
    horizon = len(CW20_TASKS) * plan.steps_per_task
    arrays = (
        ("actions", actions, np.dtype(np.float32), (horizon, 4)),
        ("observations", observations, np.dtype(np.float32), (horizon, 32)),
        ("rewards", rewards, np.dtype(np.float32), (horizon,)),
        ("successes", successes, np.dtype(np.bool_), (horizon,)),
        ("task_indices", task_indices, np.dtype(np.int32), (horizon,)),
    )
    snapshots: dict[str, np.ndarray] = {}
    for name, value, dtype, shape in arrays:
        if type(value) is not np.ndarray or value.dtype != dtype or value.shape != shape:
            raise ValueError(f"{name} must be an exact {dtype} array of shape {shape}")
        snapshot = value.copy()
        if dtype.kind == "f" and not np.isfinite(snapshot).all():
            raise ValueError(f"{name} must be finite")
        snapshots[name] = snapshot
    expected_actions = np.zeros((horizon, 4), dtype=np.float32)
    if not np.array_equal(snapshots["actions"], expected_actions):
        raise ValueError("actions do not match the exact fixed-action mechanism-off schedule")
    expected_indices = np.repeat(np.arange(20, dtype=np.int32), plan.steps_per_task)
    if not np.array_equal(snapshots["task_indices"], expected_indices):
        raise ValueError("task_indices do not follow the exact CW20 boundary schedule")
    return ContinualWorldSmokeReceipt(
        schema=SCHEMA,
        plan=plan,
        identity=_current_identity(),
        plan_sha256=plan.sha256,
        action_sha256=_array_hash(b"asi.cw.actions.v1\0", snapshots["actions"]),
        observation_sha256=_array_hash(b"asi.cw.observations.v1\0", snapshots["observations"]),
        reward_sha256=_array_hash(b"asi.cw.rewards.v1\0", snapshots["rewards"]),
        success_sha256=_array_hash(b"asi.cw.successes.v1\0", snapshots["successes"]),
        task_index_sha256=_array_hash(b"asi.cw.tasks.v1\0", snapshots["task_indices"]),
        environment_steps=horizon,
        data_steps=0,
        learner_updates=0,
        model_queries=0,
        persistent_mechanism_bytes=16,
        persistent_environment_numeric_bytes=persistent_environment_numeric_bytes,
        persistent_environment_byte_scope=PERSISTENT_ENVIRONMENT_BYTE_SCOPE,
        action_bytes=snapshots["actions"].nbytes,
        observation_bytes=snapshots["observations"].nbytes,
        reward_bytes=snapshots["rewards"].nbytes,
        success_bytes=snapshots["successes"].nbytes,
        task_index_bytes=snapshots["task_indices"].nbytes,
        timing_ns=timing_ns,
        timing_is_telemetry_only=True,
        mechanism_off=True,
        outcome=outcome,
        outcome_scope=OUTCOME_SCOPE,
        negative_outcome_retained=True,
        development_only=True,
        scientific_promotion_allowed=False,
    )


def validate_smoke_payload(payload: object) -> ContinualWorldSmokeReceipt:
    """Reject expanded or hostile serialized smoke receipts."""
    if type(payload) is not dict:
        raise ValueError("payload must be an exact object")
    root = cast(dict[object, object], payload)
    fields = {field.name for field in dataclasses.fields(ContinualWorldSmokeReceipt)}
    if any(type(key) is not str for key in root) or set(root) != fields:
        raise ValueError("payload fields differ from the exact schema")
    _canonical(root)
    plan_raw = root["plan"]
    if type(plan_raw) is not dict:
        raise ValueError("serialized plan must be an exact object")
    plan_dict = cast(dict[object, object], plan_raw)
    expected_plan_fields = {field.name for field in dataclasses.fields(ContinualWorldSmokePlan)}
    if (
        any(type(key) is not str for key in plan_dict)
        or set(plan_dict) != expected_plan_fields
    ):
        raise ValueError("serialized plan fields differ from the exact schema")
    runtime_raw = plan_dict["runtime"]
    if type(runtime_raw) is not dict:
        raise ValueError("serialized runtime must be an exact object")
    runtime_dict = cast(dict[object, object], runtime_raw)
    runtime_fields = {field.name for field in dataclasses.fields(IsolatedRuntimeIdentity)}
    if any(type(key) is not str for key in runtime_dict) or set(runtime_dict) != runtime_fields:
        raise ValueError("serialized runtime fields differ from the exact schema")
    runtime = IsolatedRuntimeIdentity(**cast(dict[str, str], runtime_dict))
    plan = ContinualWorldSmokePlan(
        runtime=runtime,
        seed=plan_dict["seed"],  # type: ignore[arg-type]
        steps_per_task=plan_dict["steps_per_task"],  # type: ignore[arg-type]
    )
    if plan_dict != dataclasses.asdict(plan):
        raise ValueError("serialized plan differs from the frozen protocol")
    identity = identity_from_payload(root["identity"])
    kwargs: dict[str, Any] = {
        cast(str, key): value
        for key, value in root.items()
        if key not in {"plan", "identity"}
    }
    return ContinualWorldSmokeReceipt(
        plan=plan,
        identity=identity,
        **kwargs,
    )


def protocol_gap_record() -> tuple[str, ...]:
    """Material gaps that keep the smoke receipt from becoming a paper comparison."""
    return (
        "the official runtime recipe leaves most transitive package versions and "
        "archive bytes unpinned",
        "the official stack uses legacy mujoco-py, Gym, TensorFlow, and Meta-World v1",
        "the smoke uses two fixed-action steps per task, not one million training steps per task",
        "no SAC learner, replay buffer, multihead network, or continual-learning method "
        "is executed",
        "paper metrics require success evaluations, single-task references, forgetting, "
        "and forward transfer",
        "paper results use twenty seeds and 90% bootstrap confidence intervals",
        "environment numeric bytes exclude interpreter, simulator native heap, and renderer "
        "allocations",
        "timing is telemetry only and the consistency hashes are not execution attestation",
    )
