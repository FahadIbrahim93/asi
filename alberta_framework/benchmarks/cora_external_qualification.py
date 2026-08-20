"""Fail-closed external CORA Procgen qualification contract.

This module never imports or launches CORA, PyTorch, Gym, or Procgen.  An
isolated provider may use the frozen plan to produce one bounded fixed-action
trace.  ASI validates that trace and its declared source/runtime/asset identity
without confusing it with the native recurring-bandit analogue or a paper run.
"""

from __future__ import annotations

import argparse
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
    exact_qualification_object,
    identity_from_payload,
    preflight_qualification_tree,
    require_current_identity,
)

SCHEMA = "asi.cora.procgen.fixed_action_smoke.v1"
QUALIFICATION_ISSUE = 1581
QUALIFICATION_LANE_ID = "cora"
PAPER_REVISION = "arXiv:2110.10067v2"
CORA_REPOSITORY = "https://github.com/AGI-Labs/continual_rl.git"
CORA_COMMIT = "f2754bb282757829765beb4703f24b87efa13ff9"
CORA_LICENSE = "MIT"
PROCGEN_TASKS = ("climber", "dodgeball", "ninja", "starpilot", "bigfish", "fruitbot")
PAPER_TRAINING_CYCLES = 5
PAPER_TRAIN_STEPS_PER_TASK = 5_000_000
PAPER_TRAIN_LEVELS = 200
PAPER_EVALUATION_LEVEL_DISTRIBUTION = "full"
PAPER_METRIC_WINDOW_STEPS = 250_000
PAPER_METRIC_WINDOWS = 20
PAPER_SEEDS = 20
OBSERVATION_SHAPE = (64, 64, 3)
ACTION_SPACE_N = 15
FIXED_ACTION = 0
TRAINING_LEVEL_SEEDS = tuple(range(len(PROCGEN_TASKS)))
EVALUATION_LEVEL_SEEDS = tuple(10_000 + index for index in range(len(PROCGEN_TASKS)))
SMOKE_LEVEL_SEEDS = tuple(
    seed
    for pair in zip(TRAINING_LEVEL_SEEDS, EVALUATION_LEVEL_SEEDS, strict=True)
    for seed in pair
)
OUTCOMES = ("inconclusive", "rejected", "supported")
OUTCOME_SCOPE = "caller_supplied_procgen_provider_contract_trace_only"
PERSISTENT_ENVIRONMENT_BYTE_SCOPE = (
    "caller_reported_unattested_provider_numeric_arrays_nbytes_sum"
)
REQUIRED_SOURCE_FILES = (
    "LICENSE",
    "README.md",
    "continual_rl/experiment_specs.py",
    "continual_rl/utils/metrics.py",
    "environment.yml",
    "main.py",
    "setup.py",
)
PROVIDER_SCHEMA = "asi.cora.procgen.provider.v1"

_MAX_JSON_BYTES = 1 << 20
_MAX_JSON_CONTAINER_ITEMS = 10_000
_MAX_JSON_DEPTH = 64
_MAX_JSON_NODES = 50_000
_MAX_JSON_STRING_BYTES = 100_000
_MAX_RESOURCE_BYTES = 1 << 40
_INT32_MAX = 2**31 - 1

_WORKLOAD_REGISTRY = (
    ("action_dtype", "int32"),
    ("action_space_n", ACTION_SPACE_N),
    ("evaluation_split_order", (False, True)),
    ("fixed_action", FIXED_ACTION),
    ("smoke_level_seeds", SMOKE_LEVEL_SEEDS),
    ("learner_boundary_information", ()),
    ("learner_task_information", ()),
    ("observation_dtype", "uint8"),
    ("observation_shape", OBSERVATION_SHAPE),
    ("outcome_scope", OUTCOME_SCOPE),
    ("provider_schema", PROVIDER_SCHEMA),
    ("tasks", PROCGEN_TASKS),
)
_PAPER_REGISTRY = (
    ("commit", CORA_COMMIT),
    ("license", CORA_LICENSE),
    ("metric_window_steps", PAPER_METRIC_WINDOW_STEPS),
    ("metric_windows", PAPER_METRIC_WINDOWS),
    ("paper_revision", PAPER_REVISION),
    ("paper_seeds", PAPER_SEEDS),
    ("repository", CORA_REPOSITORY),
    ("train_levels", PAPER_TRAIN_LEVELS),
    ("train_steps_per_task", PAPER_TRAIN_STEPS_PER_TASK),
    ("training_cycles", PAPER_TRAINING_CYCLES),
)


def _require_authoritative_plan() -> None:
    literal_constants = (
        "asi.cora.procgen.fixed_action_smoke.v1",
        1581,
        "cora",
        "arXiv:2110.10067v2",
        "https://github.com/AGI-Labs/continual_rl.git",
        "f2754bb282757829765beb4703f24b87efa13ff9",
        "MIT",
        ("climber", "dodgeball", "ninja", "starpilot", "bigfish", "fruitbot"),
        5,
        5_000_000,
        200,
        "full",
        250_000,
        20,
        20,
        (64, 64, 3),
        15,
        0,
        (0, 1, 2, 3, 4, 5),
        (10_000, 10_001, 10_002, 10_003, 10_004, 10_005),
        (0, 10_000, 1, 10_001, 2, 10_002, 3, 10_003, 4, 10_004, 5, 10_005),
        ("inconclusive", "rejected", "supported"),
        "caller_supplied_procgen_provider_contract_trace_only",
        "caller_reported_unattested_provider_numeric_arrays_nbytes_sum",
        (
            "LICENSE",
            "README.md",
            "continual_rl/experiment_specs.py",
            "continual_rl/utils/metrics.py",
            "environment.yml",
            "main.py",
            "setup.py",
        ),
        "asi.cora.procgen.provider.v1",
        1 << 20,
        10_000,
        64,
        50_000,
        100_000,
        1 << 40,
        2**31 - 1,
    )
    current_constants = (
        SCHEMA,
        QUALIFICATION_ISSUE,
        QUALIFICATION_LANE_ID,
        PAPER_REVISION,
        CORA_REPOSITORY,
        CORA_COMMIT,
        CORA_LICENSE,
        PROCGEN_TASKS,
        PAPER_TRAINING_CYCLES,
        PAPER_TRAIN_STEPS_PER_TASK,
        PAPER_TRAIN_LEVELS,
        PAPER_EVALUATION_LEVEL_DISTRIBUTION,
        PAPER_METRIC_WINDOW_STEPS,
        PAPER_METRIC_WINDOWS,
        PAPER_SEEDS,
        OBSERVATION_SHAPE,
        ACTION_SPACE_N,
        FIXED_ACTION,
        TRAINING_LEVEL_SEEDS,
        EVALUATION_LEVEL_SEEDS,
        SMOKE_LEVEL_SEEDS,
        OUTCOMES,
        OUTCOME_SCOPE,
        PERSISTENT_ENVIRONMENT_BYTE_SCOPE,
        REQUIRED_SOURCE_FILES,
        PROVIDER_SCHEMA,
        _MAX_JSON_BYTES,
        _MAX_JSON_CONTAINER_ITEMS,
        _MAX_JSON_DEPTH,
        _MAX_JSON_NODES,
        _MAX_JSON_STRING_BYTES,
        _MAX_RESOURCE_BYTES,
        _INT32_MAX,
    )
    literal_workload_registry = (
        ("action_dtype", "int32"),
        ("action_space_n", 15),
        ("evaluation_split_order", (False, True)),
        ("fixed_action", 0),
        (
            "smoke_level_seeds",
            (0, 10_000, 1, 10_001, 2, 10_002, 3, 10_003, 4, 10_004, 5, 10_005),
        ),
        ("learner_boundary_information", ()),
        ("learner_task_information", ()),
        ("observation_dtype", "uint8"),
        ("observation_shape", (64, 64, 3)),
        ("outcome_scope", "caller_supplied_procgen_provider_contract_trace_only"),
        ("provider_schema", "asi.cora.procgen.provider.v1"),
        ("tasks", ("climber", "dodgeball", "ninja", "starpilot", "bigfish", "fruitbot")),
    )
    literal_paper_registry = (
        ("commit", "f2754bb282757829765beb4703f24b87efa13ff9"),
        ("license", "MIT"),
        ("metric_window_steps", 250_000),
        ("metric_windows", 20),
        ("paper_revision", "arXiv:2110.10067v2"),
        ("paper_seeds", 20),
        ("repository", "https://github.com/AGI-Labs/continual_rl.git"),
        ("train_levels", 200),
        ("train_steps_per_task", 5_000_000),
        ("training_cycles", 5),
    )
    if (
        current_constants != literal_constants
        or _WORKLOAD_REGISTRY != literal_workload_registry
        or _PAPER_REGISTRY != literal_paper_registry
    ):
        raise ValueError("runtime state differs from the literal frozen CORA contract")
    plan = external_qualification_module.qualification_plan(QUALIFICATION_ISSUE)
    if (
        plan.lane_id != QUALIFICATION_LANE_ID
        or plan.paper_revisions != (PAPER_REVISION,)
        or len(plan.code_revisions) != 1
        or plan.code_revisions[0].repository != CORA_REPOSITORY
        or plan.code_revisions[0].commit != CORA_COMMIT
    ):
        raise ValueError("CORA source authority differs from the external qualification plan")


def _current_identity() -> QualificationIdentity:
    _require_authoritative_plan()
    return collect_qualification_identity(
        lane_module=sys.modules[__name__],
        dependency_modules=(external_qualification_module,),
        workload_registry=_WORKLOAD_REGISTRY,
        paper_registry=_PAPER_REGISTRY,
    )


def _exact_int(
    value: object, *, name: str, minimum: int = 0, maximum: int = _INT32_MAX
) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an exact integer")
    result = operator.index(cast(SupportsIndex, value))
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} lies outside the bounded qualification contract")
    return result


def _bounded_text(value: object, *, name: str, maximum: int = 256) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be non-empty exact text")
    try:
        length = len(value.encode("utf-8", errors="strict"))
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must be valid UTF-8") from error
    if length > maximum:
        raise ValueError(f"{name} exceeds its UTF-8 limit")
    return value


def _sha256(value: object, *, name: str, prefixed: bool = False) -> str:
    prefix = "sha256:" if prefixed else ""
    if (
        type(value) is not str
        or not value.startswith(prefix)
        or len(value) != len(prefix) + 64
        or any(character not in "0123456789abcdef" for character in value[len(prefix) :])
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256 identity")
    return value


def _git_object(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-1 Git object identity")
    return value


def _preflight_json_tree(value: object) -> None:
    preflight_qualification_tree(value)


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
class SourceIdentity:
    """Exact manifest declared by a read-only official CORA checkout provider."""

    repository: str
    commit: str
    git_tree: str
    source_archive_sha256: str
    install_tree_sha256: str
    required_file_sha256: tuple[tuple[str, str], ...]
    clean_checkout: bool
    commit_verified: bool
    license: str
    authenticated_attestation: bool

    def __post_init__(self) -> None:
        _require_authoritative_plan()
        if self.repository != CORA_REPOSITORY or self.commit != CORA_COMMIT:
            raise ValueError("CORA source authority does not match the pinned official revision")
        _git_object(self.git_tree, name="git_tree")
        _sha256(self.source_archive_sha256, name="source_archive_sha256")
        _sha256(self.install_tree_sha256, name="install_tree_sha256")
        if type(self.required_file_sha256) is not tuple:
            raise ValueError("required file manifest must be an exact tuple")
        for item in self.required_file_sha256:
            if type(item) is not tuple or len(item) != 2 or type(item[0]) is not str:
                raise ValueError("required file manifest entries must be exact string pairs")
            _sha256(item[1], name=f"required_file_sha256.{item[0]}")
        if tuple(item[0] for item in self.required_file_sha256) != REQUIRED_SOURCE_FILES:
            raise ValueError("required file manifest differs from the frozen official interface")
        if self.clean_checkout is not True or self.commit_verified is not True:
            raise ValueError("source provider must declare a clean exact checkout")
        if self.license != CORA_LICENSE:
            raise ValueError("source license differs from the pinned MIT license")
        if self.authenticated_attestation is not False:
            raise ValueError("content manifests are not authenticated execution attestations")


@dataclasses.dataclass(frozen=True, slots=True)
class IsolatedRuntimeIdentity:
    """Immutable external image and exact discovered dependency versions."""

    image_digest: str
    lock_sha256: str
    python_version: str
    torch_version: str
    torchvision_version: str
    gym_version: str
    procgen_version: str
    numpy_version: str
    platform: str
    accelerator: str
    network_disabled: bool
    root_filesystem_read_only: bool

    def __post_init__(self) -> None:
        _sha256(self.image_digest, name="image_digest", prefixed=True)
        _sha256(self.lock_sha256, name="lock_sha256")
        for name in (
            "python_version",
            "torch_version",
            "torchvision_version",
            "gym_version",
            "procgen_version",
            "numpy_version",
            "platform",
            "accelerator",
        ):
            _bounded_text(getattr(self, name), name=name, maximum=128)
        if self.network_disabled is not True or self.root_filesystem_read_only is not True:
            raise ValueError("CORA must run in a network-disabled, read-only isolated runtime")


@dataclasses.dataclass(frozen=True, slots=True)
class AssetIdentity:
    """Exact Procgen distribution and compiled-data identities."""

    procgen_distribution_archive_sha256: str
    procgen_install_tree_sha256: str
    procgen_compiled_data_sha256: str
    procgen_license_sha256: str
    package_version: str
    asset_rights_reviewed: bool

    def __post_init__(self) -> None:
        for name in (
            "procgen_distribution_archive_sha256",
            "procgen_install_tree_sha256",
            "procgen_compiled_data_sha256",
            "procgen_license_sha256",
        ):
            _sha256(getattr(self, name), name=name)
        _bounded_text(self.package_version, name="package_version", maximum=64)
        if self.asset_rights_reviewed is not True:
            raise ValueError("Procgen asset rights must be reviewed before qualification")


@dataclasses.dataclass(frozen=True, slots=True)
class CORAProcgenSmokePlan:
    """Provider-neutral plan for one bounded official-runtime smoke trace."""

    source: SourceIdentity
    runtime: IsolatedRuntimeIdentity
    assets: AssetIdentity
    provider_schema: str = PROVIDER_SCHEMA

    def __post_init__(self) -> None:
        if type(self.source) is not SourceIdentity:
            raise ValueError("source must be an exact SourceIdentity")
        if type(self.runtime) is not IsolatedRuntimeIdentity:
            raise ValueError("runtime must be an exact IsolatedRuntimeIdentity")
        if type(self.assets) is not AssetIdentity:
            raise ValueError("assets must be an exact AssetIdentity")
        SourceIdentity.__post_init__(self.source)
        IsolatedRuntimeIdentity.__post_init__(self.runtime)
        AssetIdentity.__post_init__(self.assets)
        if self.runtime.procgen_version != self.assets.package_version:
            raise ValueError("runtime and asset Procgen versions differ")
        if self.provider_schema != PROVIDER_SCHEMA:
            raise ValueError("provider schema differs from the frozen interface")

    def payload(self) -> dict[str, object]:
        self.__post_init__()
        source_payload = dataclasses.asdict(self.source)
        source_payload["required_file_sha256"] = [
            list(item) for item in self.source.required_file_sha256
        ]
        return {
            "qualification_issue": QUALIFICATION_ISSUE,
            "qualification_lane_id": QUALIFICATION_LANE_ID,
            "paper_revision": PAPER_REVISION,
            "source": source_payload,
            "runtime": dataclasses.asdict(self.runtime),
            "assets": dataclasses.asdict(self.assets),
            "provider_schema": self.provider_schema,
            "tasks": list(PROCGEN_TASKS),
            "paper_training_cycles": PAPER_TRAINING_CYCLES,
            "paper_train_steps_per_task": PAPER_TRAIN_STEPS_PER_TASK,
            "paper_train_levels": PAPER_TRAIN_LEVELS,
            "paper_evaluation_level_distribution": PAPER_EVALUATION_LEVEL_DISTRIBUTION,
            "paper_metric_window_steps": PAPER_METRIC_WINDOW_STEPS,
            "paper_metric_windows": PAPER_METRIC_WINDOWS,
            "paper_seeds": PAPER_SEEDS,
            "smoke_events_per_task": 2,
            "smoke_event_order": ["training_distribution", "evaluation_distribution"],
            "smoke_level_seeds": list(SMOKE_LEVEL_SEEDS),
            "observation_shape": list(OBSERVATION_SHAPE),
            "observation_dtype": "uint8",
            "action_space_n": ACTION_SPACE_N,
            "action_dtype": "int32",
            "fixed_action": FIXED_ACTION,
            "evaluator_task_information": ["task_index", "task_name", "distribution_split"],
            "learner_task_information": [],
            "learner_boundary_information": [],
            "mechanism_off": True,
            "development_only": True,
            "scientific_promotion_allowed": False,
            "cora_parity_claimed": False,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical(self.payload())).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class CORAProcgenSmokeReceipt:
    """Strict host receipt for arrays emitted by the isolated provider."""

    schema: str
    plan: CORAProcgenSmokePlan
    identity: QualificationIdentity
    plan_sha256: str
    task_index_sha256: str
    evaluation_split_sha256: str
    level_seed_sha256: str
    action_sha256: str
    observation_sha256: str
    reward_sha256: str
    terminated_sha256: str
    truncated_sha256: str
    environment_steps: int
    learner_updates: int
    model_queries: int
    persistent_mechanism_bytes: int
    persistent_environment_numeric_bytes: int
    persistent_environment_byte_scope: str
    persistent_environment_numeric_bytes_is_provider_reported_unattested: bool
    task_index_bytes: int
    evaluation_split_bytes: int
    level_seed_bytes: int
    action_bytes: int
    observation_bytes: int
    reward_bytes: int
    terminated_bytes: int
    truncated_bytes: int
    timing_ns: int
    timing_is_telemetry_only: bool
    mechanism_off: bool
    provider_trace_supplied_by_caller: bool
    external_runtime_execution_verified: bool
    outcome: str
    outcome_scope: str
    negative_outcome_retained: bool
    development_only: bool
    scientific_promotion_allowed: bool
    cora_parity_claimed: bool

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise ValueError("unsupported CORA Procgen receipt schema")
        if type(self.plan) is not CORAProcgenSmokePlan:
            raise ValueError("plan must be an exact CORAProcgenSmokePlan")
        CORAProcgenSmokePlan.__post_init__(self.plan)
        require_current_identity(self.identity, _current_identity())
        if _sha256(self.plan_sha256, name="plan_sha256") != self.plan.sha256:
            raise ValueError("plan_sha256 does not bind the exact smoke plan")
        for name in (
            "task_index_sha256",
            "evaluation_split_sha256",
            "level_seed_sha256",
            "action_sha256",
            "observation_sha256",
            "reward_sha256",
            "terminated_sha256",
            "truncated_sha256",
        ):
            _sha256(getattr(self, name), name=name)
        horizon = len(PROCGEN_TASKS) * 2
        expected_schedule_hashes = {
            "task_index_sha256": _array_hash(
                "task_indices",
                np.repeat(np.arange(len(PROCGEN_TASKS), dtype=np.int32), 2),
            ),
            "evaluation_split_sha256": _array_hash(
                "evaluation_split",
                np.tile(np.asarray([False, True], dtype=np.bool_), len(PROCGEN_TASKS)),
            ),
            "action_sha256": _array_hash(
                "actions", np.zeros(horizon, dtype=np.int32)
            ),
            "level_seed_sha256": _array_hash(
                "level_seeds", np.asarray(SMOKE_LEVEL_SEEDS, dtype=np.int32)
            ),
        }
        if any(
            getattr(self, name) != expected
            for name, expected in expected_schedule_hashes.items()
        ):
            raise ValueError("trace hashes do not bind the frozen task/split/fixed-action schedule")
        if _exact_int(self.environment_steps, name="environment_steps", minimum=1) != horizon:
            raise ValueError("environment_steps differ from the frozen smoke horizon")
        if any(
            _exact_int(getattr(self, name), name=name) != 0
            for name in ("learner_updates", "model_queries")
        ):
            raise ValueError("fixed-action mechanism-off cannot learn or query a model")
        if _exact_int(self.persistent_mechanism_bytes, name="persistent_mechanism_bytes") != 4:
            raise ValueError("persistent mechanism bytes must equal one int32 fixed action")
        _exact_int(
            self.persistent_environment_numeric_bytes,
            name="persistent_environment_numeric_bytes",
            minimum=1,
            maximum=_MAX_RESOURCE_BYTES,
        )
        if self.persistent_environment_byte_scope != PERSISTENT_ENVIRONMENT_BYTE_SCOPE:
            raise ValueError("persistent environment byte scope differs from the contract")
        if self.persistent_environment_numeric_bytes_is_provider_reported_unattested is not True:
            raise ValueError(
                "environment numeric bytes must remain provider-reported and unattested"
            )
        expected_bytes = {
            "task_index_bytes": horizon * np.dtype(np.int32).itemsize,
            "evaluation_split_bytes": horizon * np.dtype(np.bool_).itemsize,
            "level_seed_bytes": horizon * np.dtype(np.int32).itemsize,
            "action_bytes": horizon * np.dtype(np.int32).itemsize,
            "observation_bytes": horizon * math.prod(OBSERVATION_SHAPE),
            "reward_bytes": horizon * np.dtype(np.float32).itemsize,
            "terminated_bytes": horizon * np.dtype(np.bool_).itemsize,
            "truncated_bytes": horizon * np.dtype(np.bool_).itemsize,
        }
        if any(
            _exact_int(getattr(self, name), name=name, maximum=_MAX_RESOURCE_BYTES) != expected
            for name, expected in expected_bytes.items()
        ):
            raise ValueError("trace byte receipt differs from the exact frozen arrays")
        _exact_int(self.timing_ns, name="timing_ns", maximum=2**63 - 1)
        if (
            self.timing_is_telemetry_only is not True
            or self.mechanism_off is not True
            or self.provider_trace_supplied_by_caller is not True
            or self.external_runtime_execution_verified is not False
        ):
            raise ValueError("caller-supplied trace/mechanism/timing policy drifted")
        if type(self.outcome) is not str or self.outcome not in OUTCOMES:
            raise ValueError("outcome differs from the frozen retention vocabulary")
        if self.outcome_scope != OUTCOME_SCOPE:
            raise ValueError("outcome scope exceeds the caller-supplied provider contract")
        if (
            self.negative_outcome_retained is not True
            or self.development_only is not True
            or self.scientific_promotion_allowed is not False
            or self.cora_parity_claimed is not False
        ):
            raise ValueError("CORA Procgen smoke receipts must remain nonpromoting and nonparity")

    def payload(self) -> dict[str, object]:
        self.__post_init__()
        raw = dataclasses.asdict(self)
        raw["plan"] = self.plan.payload()
        raw["identity"] = self.identity.to_payload()
        return cast(dict[str, object], json.loads(_canonical(raw)))


def _array_hash(domain: str, value: np.ndarray) -> str:
    digest = hashlib.sha256(domain.encode("ascii"))
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def build_smoke_receipt(
    plan: CORAProcgenSmokePlan,
    *,
    task_indices: np.ndarray,
    evaluation_split: np.ndarray,
    level_seeds: np.ndarray,
    actions: np.ndarray,
    observations: np.ndarray,
    rewards: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    persistent_environment_numeric_bytes: object,
    timing_ns: object,
    outcome: str,
) -> CORAProcgenSmokeReceipt:
    """Copy and bind one caller-supplied provider-contract trace.

    This boundary does not invoke or authenticate an external runtime. A caller
    cannot turn supplied arrays into an execution claim.
    """
    if type(plan) is not CORAProcgenSmokePlan:
        raise ValueError("plan must be exact")
    CORAProcgenSmokePlan.__post_init__(plan)
    horizon = len(PROCGEN_TASKS) * 2
    arrays = (
        ("task_indices", task_indices, np.dtype(np.int32), (horizon,)),
        ("evaluation_split", evaluation_split, np.dtype(np.bool_), (horizon,)),
        ("level_seeds", level_seeds, np.dtype(np.int32), (horizon,)),
        ("actions", actions, np.dtype(np.int32), (horizon,)),
        ("observations", observations, np.dtype(np.uint8), (horizon, *OBSERVATION_SHAPE)),
        ("rewards", rewards, np.dtype(np.float32), (horizon,)),
        ("terminated", terminated, np.dtype(np.bool_), (horizon,)),
        ("truncated", truncated, np.dtype(np.bool_), (horizon,)),
    )
    snapshots: dict[str, np.ndarray] = {}
    for name, value, dtype, shape in arrays:
        if type(value) is not np.ndarray or value.dtype != dtype or value.shape != shape:
            raise ValueError(f"{name} must be an exact {dtype} array of shape {shape}")
        snapshot = value.copy(order="C")
        if dtype.kind == "f" and not np.isfinite(snapshot).all():
            raise ValueError(f"{name} must contain only finite values")
        snapshots[name] = snapshot
    expected_tasks = np.repeat(np.arange(len(PROCGEN_TASKS), dtype=np.int32), 2)
    if not np.array_equal(snapshots["task_indices"], expected_tasks):
        raise ValueError("task schedule differs from the frozen Procgen order")
    expected_splits = np.tile(np.asarray([False, True], dtype=np.bool_), len(PROCGEN_TASKS))
    if not np.array_equal(snapshots["evaluation_split"], expected_splits):
        raise ValueError("distribution split schedule differs from the provider contract")
    expected_level_seeds = np.asarray(SMOKE_LEVEL_SEEDS, dtype=np.int32)
    if not np.array_equal(snapshots["level_seeds"], expected_level_seeds):
        raise ValueError("level seeds differ from the frozen train/evaluation smoke schedule")
    if not np.array_equal(snapshots["actions"], np.zeros(horizon, dtype=np.int32)):
        raise ValueError("actions differ from the fixed-action mechanism-off schedule")
    persistent = _exact_int(
        persistent_environment_numeric_bytes,
        name="persistent_environment_numeric_bytes",
        minimum=1,
        maximum=_MAX_RESOURCE_BYTES,
    )
    timing = _exact_int(timing_ns, name="timing_ns", maximum=2**63 - 1)
    if type(outcome) is not str or outcome not in OUTCOMES:
        raise ValueError("outcome differs from the frozen vocabulary")
    return CORAProcgenSmokeReceipt(
        schema=SCHEMA,
        plan=plan,
        identity=_current_identity(),
        plan_sha256=plan.sha256,
        task_index_sha256=_array_hash("task_indices", snapshots["task_indices"]),
        evaluation_split_sha256=_array_hash("evaluation_split", snapshots["evaluation_split"]),
        level_seed_sha256=_array_hash("level_seeds", snapshots["level_seeds"]),
        action_sha256=_array_hash("actions", snapshots["actions"]),
        observation_sha256=_array_hash("observations", snapshots["observations"]),
        reward_sha256=_array_hash("rewards", snapshots["rewards"]),
        terminated_sha256=_array_hash("terminated", snapshots["terminated"]),
        truncated_sha256=_array_hash("truncated", snapshots["truncated"]),
        environment_steps=horizon,
        learner_updates=0,
        model_queries=0,
        persistent_mechanism_bytes=np.dtype(np.int32).itemsize,
        persistent_environment_numeric_bytes=persistent,
        persistent_environment_byte_scope=PERSISTENT_ENVIRONMENT_BYTE_SCOPE,
        persistent_environment_numeric_bytes_is_provider_reported_unattested=True,
        task_index_bytes=snapshots["task_indices"].nbytes,
        evaluation_split_bytes=snapshots["evaluation_split"].nbytes,
        level_seed_bytes=snapshots["level_seeds"].nbytes,
        action_bytes=snapshots["actions"].nbytes,
        observation_bytes=snapshots["observations"].nbytes,
        reward_bytes=snapshots["rewards"].nbytes,
        terminated_bytes=snapshots["terminated"].nbytes,
        truncated_bytes=snapshots["truncated"].nbytes,
        timing_ns=timing,
        timing_is_telemetry_only=True,
        mechanism_off=True,
        provider_trace_supplied_by_caller=True,
        external_runtime_execution_verified=False,
        outcome=outcome,
        outcome_scope=OUTCOME_SCOPE,
        negative_outcome_retained=True,
        development_only=True,
        scientific_promotion_allowed=False,
        cora_parity_claimed=False,
    )


def _source_from_payload(value: object) -> SourceIdentity:
    expected = tuple(field.name for field in dataclasses.fields(SourceIdentity))
    raw = dict(exact_qualification_object(value, expected, name="source identity"))
    entries = raw["required_file_sha256"]
    if type(entries) is not list or any(
        type(item) is not list
        or len(item) != 2
        or type(item[0]) is not str
        or type(item[1]) is not str
        for item in entries
    ):
        raise ValueError("required file manifest payload must contain exact string pairs")
    raw["required_file_sha256"] = tuple((item[0], item[1]) for item in entries)
    return SourceIdentity(**cast(dict[str, Any], raw))


def _plan_from_payload(value: object) -> CORAProcgenSmokePlan:
    expected = (
        "qualification_issue",
        "qualification_lane_id",
        "paper_revision",
        "source",
        "runtime",
        "assets",
        "provider_schema",
        "tasks",
        "paper_training_cycles",
        "paper_train_steps_per_task",
        "paper_train_levels",
        "paper_evaluation_level_distribution",
        "paper_metric_window_steps",
        "paper_metric_windows",
        "paper_seeds",
        "smoke_events_per_task",
        "smoke_event_order",
        "smoke_level_seeds",
        "observation_shape",
        "observation_dtype",
        "action_space_n",
        "action_dtype",
        "fixed_action",
        "evaluator_task_information",
        "learner_task_information",
        "learner_boundary_information",
        "mechanism_off",
        "development_only",
        "scientific_promotion_allowed",
        "cora_parity_claimed",
    )
    raw = dict(exact_qualification_object(value, expected, name="plan payload"))
    plan = CORAProcgenSmokePlan(
        source=_source_from_payload(raw["source"]),
        runtime=_runtime_from_payload(raw["runtime"]),
        assets=_assets_from_payload(raw["assets"]),
        provider_schema=cast(str, raw["provider_schema"]),
    )
    if raw != plan.payload():
        raise ValueError("plan payload differs from the frozen official Procgen contract")
    return plan


def _runtime_from_payload(value: object) -> IsolatedRuntimeIdentity:
    expected = tuple(field.name for field in dataclasses.fields(IsolatedRuntimeIdentity))
    admitted = exact_qualification_object(value, expected, name="runtime identity")
    return IsolatedRuntimeIdentity(**cast(dict[str, Any], admitted))


def _assets_from_payload(value: object) -> AssetIdentity:
    expected = tuple(field.name for field in dataclasses.fields(AssetIdentity))
    admitted = exact_qualification_object(value, expected, name="assets identity")
    return AssetIdentity(**cast(dict[str, Any], admitted))


def validate_smoke_payload(value: object) -> CORAProcgenSmokeReceipt:
    """Strictly reconstruct a primitive receipt and bind it to the current tree."""
    _preflight_json_tree(value)
    expected = tuple(field.name for field in dataclasses.fields(CORAProcgenSmokeReceipt))
    raw = dict(exact_qualification_object(value, expected, name="CORA Procgen receipt"))
    raw["plan"] = _plan_from_payload(raw["plan"])
    raw["identity"] = identity_from_payload(raw["identity"])
    receipt = CORAProcgenSmokeReceipt(**cast(dict[str, Any], raw))
    receipt.__post_init__()
    return receipt


def qualification_blocker_manifest() -> dict[str, object]:
    """Return current blockers without probing, importing, or downloading anything."""
    _require_authoritative_plan()
    blockers = (
        "official_source_checkout_verified",
        "isolated_runtime_lock_built_and_verified",
        "procgen_distribution_assets_and_rights_verified",
        "official_provider_fixed_action_trace_captured_and_validated",
        "paper_metric_implementation_parity_verified",
        "matched_learning_baselines_and_resource_contract_frozen",
        "paper_scale_execution_completed",
        "external_execution_separately_authorized",
    )
    return {
        "schema": "asi.cora.external_qualification_blockers.v1",
        "qualification_issue": QUALIFICATION_ISSUE,
        "qualification_lane_id": QUALIFICATION_LANE_ID,
        "paper_revision": PAPER_REVISION,
        "repository": CORA_REPOSITORY,
        "commit": CORA_COMMIT,
        "first_family": "procgen",
        "provider_schema": PROVIDER_SCHEMA,
        "completed_interface_work": [
            "paper_and_official_code_revision_pinned",
            "procgen_catalog_and_information_contract_frozen",
            "source_runtime_asset_identity_schema_implemented",
            "bounded_fixed_action_trace_validator_implemented",
        ],
        "blockers": list(blockers),
        "ready": False,
        "external_execution_authorized": False,
        "native_analogue_is_cora": False,
        "development_only": True,
        "scientific_promotion_allowed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect blocked official CORA Procgen qualification metadata"
    )
    parser.add_argument("--blockers", action="store_true")
    args = parser.parse_args(argv)
    if not args.blockers:
        parser.error("only --blockers is available; external execution is not authorized")
    print(json.dumps(qualification_blocker_manifest(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTION_SPACE_N",
    "CORA_COMMIT",
    "CORAProcgenSmokePlan",
    "CORAProcgenSmokeReceipt",
    "CORA_REPOSITORY",
    "AssetIdentity",
    "IsolatedRuntimeIdentity",
    "PAPER_REVISION",
    "PROCGEN_TASKS",
    "PROVIDER_SCHEMA",
    "SCHEMA",
    "SMOKE_LEVEL_SEEDS",
    "SourceIdentity",
    "build_smoke_receipt",
    "main",
    "qualification_blocker_manifest",
    "validate_smoke_payload",
]
