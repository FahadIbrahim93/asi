# mypy: disable-error-code="call-arg"
"""Fail-closed COOM/ViZDoom qualification and native contract smoke.

No COOM code, WAD, checkpoint, or ViZDoom runtime is imported or executed.
The executable smoke exercises a synthetic pixel-shaped adapter contract and a
current SARSA consumer so dependency isolation and accounting can be reviewed
before external execution is authorized.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import operator
import sys
from collections.abc import Sequence
from typing import Any, SupportsIndex, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import alberta_framework.benchmarks.external_qualification as external_qualification_module
import alberta_framework.core.sarsa as sarsa_module
from alberta_framework.benchmarks.external_qualification import qualification_plan
from alberta_framework.benchmarks.qualification_provenance import (
    QualificationIdentity,
    collect_qualification_identity,
    identity_from_payload,
    require_current_identity,
)
from alberta_framework.core.sarsa import SARSAAgent, SARSAConfig

COOM_SMOKE_SCHEMA = "asi.coom_qualification_smoke.development.v1"
COOM_PAPER = "NeurIPS2023-DatasetsAndBenchmarks:d61d9f4fe4357296cb658795fd7999f0"
COOM_REPOSITORY = "https://github.com/TTomilin/COOM.git"
COOM_COMMIT = "7929801176c6e2e036c7c1c7dd6ce9b84a9d1f3e"
COOM_METRICS = ("average_performance", "forgetting", "forward_transfer")
CD8_TASKS = (
    "run_and_gun-obstacles-v0",
    "run_and_gun-green-v0",
    "run_and_gun-resized-v0",
    "run_and_gun-monsters-v0",
    "run_and_gun-default-v0",
    "run_and_gun-red-v0",
    "run_and_gun-blue-v0",
    "run_and_gun-shadows-v0",
)
CO8_TASKS = (
    "pitfall-default-v0",
    "arms_dealer-default-v0",
    "hide_and_seek-default-v0",
    "floor_is_lava-default-v0",
    "chainsaw-default-v0",
    "raise_the_roof-default-v0",
    "run_and_gun-default-v0",
    "health_gathering-default-v0",
)
FROZEN_DEVELOPMENT_SEEDS = (1_582_000, 1_582_001, 1_582_002)
FROZEN_ARMS = (
    "cyclic_adapter_control",
    "native_sarsa_contract_control",
    "mechanism_off",
    "fixed_action_parity",
)
_OBSERVATION_SHAPE = (8, 8, 3)
_OBSERVATION_BYTES = 8 * 8 * 3
_MAX_SMOKE_STEPS_PER_TASK = 16
_INT32_MAX = 2**31 - 1
_SEQUENCES = ("CD4", "CD8", "CD16", "CO4", "CO8", "CO16")
_ENVIRONMENT_DEPENDENCIES = (
    "vizdoom",
    "opencv-python",
    "scipy==1.11.4",
    "gymnasium==0.28.1",
)
_LEARNING_EXTRAS = (
    "tensorflow==2.11",
    "tensorflow-probability==0.19",
    "wandb",
)
_WORKLOAD_REGISTRY = (
    ("arms", FROZEN_ARMS),
    ("cd8_tasks", CD8_TASKS),
    ("co8_tasks", CO8_TASKS),
    ("development_seeds", FROZEN_DEVELOPMENT_SEEDS),
    ("max_steps_per_task", _MAX_SMOKE_STEPS_PER_TASK),
    ("observation_shape", _OBSERVATION_SHAPE),
)
_PAPER_REGISTRY = (
    ("commit", COOM_COMMIT),
    ("paper", COOM_PAPER),
    ("repository", COOM_REPOSITORY),
)


def _current_identity() -> QualificationIdentity:
    return collect_qualification_identity(
        lane_module=sys.modules[__name__],
        dependency_modules=(external_qualification_module, sarsa_module),
        workload_registry=_WORKLOAD_REGISTRY,
        paper_registry=_PAPER_REGISTRY,
    )


def _exact_int(value: object, *, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an exact integer")
    result = operator.index(cast(SupportsIndex, value))
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must lie in [{minimum}, {maximum}]")
    return result


def _bounded_string(value: object, *, name: str, maximum: int = 128) -> str:
    if type(value) is not str or not value or len(value.encode("utf-8")) > maximum:
        raise ValueError(f"{name} must be a bounded exact string")
    return value


def _tree_nbytes(tree: object) -> int:
    leaves = jax.tree_util.tree_leaves(tree)
    if len(leaves) > 4096:
        raise ValueError("state has too many leaves")
    total = 0
    for leaf in leaves:
        try:
            value = np.asarray(leaf)
        except TypeError:
            value = np.asarray(jr.key_data(leaf))
        total += int(value.nbytes)
    if not 0 <= total <= _INT32_MAX:
        raise ValueError("state bytes exceed signed-int32 capacity")
    return total


def _sha(values: Sequence[int | float], *, floats: bool = False) -> str:
    dtype = np.dtype("<f4") if floats else np.dtype("<i4")
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes()).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class COOMCatalogEntry:
    """Machine-readable setup authority for this external lane."""

    benchmark_id: str = "coom"
    paper: str = COOM_PAPER
    repository: str = COOM_REPOSITORY
    commit: str = COOM_COMMIT
    license: str = "MIT"
    sequences: tuple[str, ...] = _SEQUENCES
    core_sequence_tasks: dict[str, tuple[str, ...]] = dataclasses.field(
        default_factory=lambda: {"CD8": CD8_TASKS, "CO8": CO8_TASKS}
    )
    metrics: tuple[str, ...] = COOM_METRICS
    environment_dependencies: tuple[str, ...] = _ENVIRONMENT_DEPENDENCIES
    learning_extras: tuple[str, ...] = _LEARNING_EXTRAS
    paper_seeds: tuple[int, ...] = tuple(range(10))
    default_steps_per_task: int = 200_000
    default_replay_capacity: int = 50_000
    default_frame_skip: int = 4
    default_frame_stack: int = 4
    default_frame_height: int = 84
    default_frame_width: int = 84
    default_test_episodes: int = 3
    task_boundaries_available: bool = True
    task_id_visible_by_default: bool = True
    integration: str = "isolated"
    status: str = "scaffolded"

    def __post_init__(self) -> None:
        for name in ("benchmark_id", "paper", "repository", "commit", "license"):
            _bounded_string(getattr(self, name), name=name, maximum=256)
        if (
            self.benchmark_id != "coom"
            or self.paper != COOM_PAPER
            or self.repository != COOM_REPOSITORY
            or self.commit != COOM_COMMIT
            or self.license != "MIT"
        ):
            raise ValueError("COOM source identity differs from the audited pin")
        if self.core_sequence_tasks != {"CD8": CD8_TASKS, "CO8": CO8_TASKS}:
            raise ValueError("core task sequences differ from the audited order")
        if self.metrics != COOM_METRICS:
            raise ValueError("COOM metrics differ from the audited roster")
        if (
            self.sequences != _SEQUENCES
            or self.environment_dependencies != _ENVIRONMENT_DEPENDENCIES
            or self.learning_extras != _LEARNING_EXTRAS
        ):
            raise ValueError("COOM catalog protocol/runtime fields differ from the audit")
        if (
            self.paper_seeds != tuple(range(10))
            or self.default_steps_per_task != 200_000
            or self.default_replay_capacity != 50_000
            or self.default_frame_skip != 4
            or self.default_frame_stack != 4
            or self.default_frame_height != 84
            or self.default_frame_width != 84
            or self.default_test_episodes != 3
            or self.task_boundaries_available is not True
            or self.task_id_visible_by_default is not True
        ):
            raise ValueError("COOM catalog defaults differ from the pinned source")
        if self.integration != "isolated" or self.status != "scaffolded":
            raise ValueError("COOM must remain an isolated scaffold")


@dataclasses.dataclass(frozen=True, slots=True)
class COOMSmokeProtocol:
    """Frozen synthetic adapter axes; not a paper protocol."""

    sequence: str = "CO8"
    steps_per_task: int = 4
    action_space_n: int = 3
    observation_shape: tuple[int, int, int] = _OBSERVATION_SHAPE
    task_boundaries_available: bool = True
    task_id_available: bool = True
    future_task_evaluation_allowed: bool = True
    previous_environment_access_during_training: bool = False
    seeds: tuple[int, ...] = FROZEN_DEVELOPMENT_SEEDS

    def __post_init__(self) -> None:
        if type(self.sequence) is not str or self.sequence not in ("CD8", "CO8"):
            raise ValueError("sequence must be an exact core-sequence identifier")
        _exact_int(
            self.steps_per_task,
            name="steps_per_task",
            minimum=1,
            maximum=_MAX_SMOKE_STEPS_PER_TASK,
        )
        _exact_int(self.action_space_n, name="action_space_n", minimum=2, maximum=16)
        if self.observation_shape != _OBSERVATION_SHAPE:
            raise ValueError("synthetic observation shape is frozen")
        for name, required in (
            ("task_boundaries_available", True),
            ("task_id_available", True),
            ("future_task_evaluation_allowed", True),
            ("previous_environment_access_during_training", False),
        ):
            value = getattr(self, name)
            if type(value) is not bool or value is not required:
                raise ValueError(f"{name} differs from the declared information contract")
        if type(self.seeds) is not tuple or self.seeds != FROZEN_DEVELOPMENT_SEEDS:
            raise ValueError("development seeds are frozen")

    @property
    def tasks(self) -> tuple[str, ...]:
        return CD8_TASKS if self.sequence == "CD8" else CO8_TASKS


@dataclasses.dataclass(frozen=True, slots=True)
class DependencyReceipt:
    coom_discoverable: bool
    vizdoom_discoverable: bool
    tensorflow_discoverable: bool
    imports_attempted: int = 0
    external_runtime_executed: bool = False
    assets_downloaded: bool = False

    def __post_init__(self) -> None:
        for name in ("coom_discoverable", "vizdoom_discoverable", "tensorflow_discoverable"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be an exact bool")
        if (
            type(self.imports_attempted) is not int
            or self.imports_attempted != 0
            or self.external_runtime_executed is not False
            or self.assets_downloaded is not False
        ):
            raise ValueError("qualification smoke may not import, execute, or download externals")


@dataclasses.dataclass(frozen=True, slots=True)
class COOMResourceReceipt:
    task_resets: int
    environment_steps: int
    environment_reset_queries: int
    environment_step_queries: int
    policy_queries: int
    observation_bytes: int
    action_bytes: int
    reward_bytes: int
    terminal_bytes: int
    task_id_bytes: int
    persistent_agent_bytes: int
    persistent_environment_bytes: int

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            _exact_int(getattr(self, field.name), name=field.name, minimum=0, maximum=_INT32_MAX)
        if self.task_resets < 1 or self.environment_steps < 1:
            raise ValueError("the contract trace must consume tasks and steps")
        if self.environment_reset_queries != self.task_resets:
            raise ValueError("reset accounting must be exact")
        if self.environment_step_queries != self.environment_steps:
            raise ValueError("step accounting must be exact")


@dataclasses.dataclass(frozen=True, slots=True)
class COOMArmReceipt:
    arm_id: str
    seed: int
    action_sha256: str
    reward_sha256: str
    observation_sha256: str
    resources: COOMResourceReceipt
    performance_metrics_computed: bool = False
    negative_outcome_retained: bool = True

    def __post_init__(self) -> None:
        if type(self.arm_id) is not str or self.arm_id not in FROZEN_ARMS:
            raise ValueError("unsupported smoke arm")
        _exact_int(self.seed, name="seed", minimum=0, maximum=2**32 - 1)
        for name in ("action_sha256", "reward_sha256", "observation_sha256"):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if type(self.resources) is not COOMResourceReceipt:
            raise ValueError("resources must use the exact receipt type")
        if self.performance_metrics_computed is not False:
            raise ValueError("synthetic traces may not compute COOM performance metrics")
        if self.negative_outcome_retained is not True:
            raise ValueError("negative outcomes must remain retained")


@dataclasses.dataclass(frozen=True, slots=True)
class COOMSmokeResult:
    schema: str
    catalog: COOMCatalogEntry
    protocol: COOMSmokeProtocol
    dependencies: DependencyReceipt
    qualification_blockers: tuple[str, ...]
    arms: tuple[COOMArmReceipt, ...]
    identity: QualificationIdentity
    synthetic_contract_trace: bool = True
    development_only: bool = True
    scientific_promotion_allowed: bool = False
    benchmark_result_claimed: bool = False

    def __post_init__(self) -> None:
        if type(self.schema) is not str or self.schema != COOM_SMOKE_SCHEMA:
            raise ValueError("unsupported COOM smoke schema")
        if (
            type(self.catalog) is not COOMCatalogEntry
            or type(self.protocol) is not COOMSmokeProtocol
        ):
            raise ValueError("catalog and protocol must use exact types")
        if type(self.dependencies) is not DependencyReceipt:
            raise ValueError("dependencies must use the exact receipt type")
        COOMCatalogEntry.__post_init__(self.catalog)
        COOMSmokeProtocol.__post_init__(self.protocol)
        DependencyReceipt.__post_init__(self.dependencies)
        require_current_identity(self.identity, _current_identity())
        expected_blockers = qualification_plan(1582).blockers
        if (
            type(self.qualification_blockers) is not tuple
            or self.qualification_blockers != expected_blockers
        ):
            raise ValueError("qualification blockers differ from the fail-closed catalog")
        expected = tuple(
            (arm_id, seed) for seed in self.protocol.seeds for arm_id in FROZEN_ARMS
        )
        if type(self.arms) is not tuple or any(
            type(arm) is not COOMArmReceipt for arm in self.arms
        ):
            raise ValueError("arms must be exact receipts")
        if tuple((arm.arm_id, arm.seed) for arm in self.arms) != expected:
            raise ValueError("arm/seed roster differs from the frozen order")
        self._validate_resources_and_parity()
        if (
            self.synthetic_contract_trace is not True
            or self.development_only is not True
            or self.scientific_promotion_allowed is not False
            or self.benchmark_result_claimed is not False
        ):
            raise ValueError("COOM smoke is permanently synthetic and nonpromoting")

    def _validate_resources_and_parity(self) -> None:
        tasks = len(self.protocol.tasks)
        steps = tasks * self.protocol.steps_per_task
        expected_observation_bytes = (tasks + steps) * _OBSERVATION_BYTES
        for seed_index in range(len(self.protocol.seeds)):
            group = self.arms[
                seed_index * len(FROZEN_ARMS) : (seed_index + 1) * len(FROZEN_ARMS)
            ]
            for arm_index, arm in enumerate(group):
                resources = arm.resources
                if (
                    resources.task_resets != tasks
                    or resources.environment_steps != steps
                    or resources.observation_bytes != expected_observation_bytes
                    or resources.action_bytes != 4 * steps
                    or resources.reward_bytes != 4 * steps
                    or resources.terminal_bytes != steps
                    or resources.task_id_bytes != 4 * (tasks + steps)
                ):
                    raise ValueError("resource receipt differs from the frozen contract")
                expected_policy_queries = (
                    tasks * (self.protocol.steps_per_task + 1)
                    if arm_index == 1
                    else (steps if arm_index == 0 else 0)
                )
                if resources.policy_queries != expected_policy_queries:
                    raise ValueError("policy queries differ from the frozen contract")
                if resources.persistent_environment_bytes != 8:
                    raise ValueError("synthetic environment bytes differ from the contract")
                if (arm_index == 1) != (resources.persistent_agent_bytes > 0):
                    raise ValueError("agent bytes disagree with the enabled mechanism")
                expected_arm = _run_arm(self.protocol, arm.seed, arm.arm_id)
                if arm != expected_arm:
                    raise ValueError("deterministic arm replay or exact resource receipt mismatch")
            mechanism_off, fixed_parity = group[2], group[3]
            if (
                mechanism_off.action_sha256 != fixed_parity.action_sha256
                or mechanism_off.reward_sha256 != fixed_parity.reward_sha256
                or mechanism_off.observation_sha256 != fixed_parity.observation_sha256
            ):
                raise ValueError("mechanism-off must reduce exactly to fixed-action parity")

    def to_payload(self) -> dict[str, object]:
        return cast(dict[str, object], json.loads(json.dumps(dataclasses.asdict(self))))


def _synthetic_observation(seed: int, task: int, step: int, action: int) -> np.ndarray:
    """Return a deterministic pixel-shaped adapter fixture, never a Doom frame."""
    base = (seed + 31 * task + 17 * step + 13 * action) % 256
    offsets = np.arange(_OBSERVATION_BYTES, dtype=np.uint16).reshape(_OBSERVATION_SHAPE)
    return ((offsets + base) % 256).astype(np.uint8)


def _feature(observation: np.ndarray, task_index: int) -> jax.Array:
    pixels = jnp.asarray(observation.reshape(-1), dtype=jnp.float32) / 255.0
    task = jax.nn.one_hot(task_index, 8, dtype=jnp.float32)
    return jnp.concatenate((pixels, task))


def _run_arm(protocol: COOMSmokeProtocol, seed: int, arm_id: str) -> COOMArmReceipt:
    steps_total = len(protocol.tasks) * protocol.steps_per_task
    actions: list[int] = []
    rewards: list[float] = []
    observation_bytes = bytearray()
    persistent_agent_bytes = 0
    agent: SARSAAgent | None = None
    state: Any = None
    if arm_id == "native_sarsa_contract_control":
        agent = SARSAAgent(
            SARSAConfig(
                n_actions=protocol.action_space_n,
                gamma=0.99,
                epsilon_start=0.1,
                epsilon_end=0.1,
                epsilon_decay_steps=1,
            ),
            hidden_sizes=(),
            sparsity=0.0,
            use_layer_norm=False,
        )
        state = agent.init(_OBSERVATION_BYTES + len(protocol.tasks), jr.key(seed))
        persistent_agent_bytes = _tree_nbytes(state)
    policy_queries = 0
    for task_index, _task_name in enumerate(protocol.tasks):
        observation = _synthetic_observation(seed, task_index, 0, 0)
        observation_bytes.extend(observation.tobytes())
        feature = _feature(observation, task_index)
        action: Any = None
        if agent is not None:
            action, key = agent.select_action(state, feature)
            policy_queries += 1
            state = state.replace(last_action=action, last_observation=feature, rng_key=key)
        for step in range(protocol.steps_per_task):
            if arm_id == "cyclic_adapter_control":
                selected = (task_index + step) % protocol.action_space_n
                policy_queries += 1
            elif agent is not None:
                selected = int(action)
            else:
                selected = 0
            target = (task_index + step) % protocol.action_space_n
            reward = np.float32(1.0 if selected == target else 0.0)
            next_observation = _synthetic_observation(
                seed, task_index, step + 1, selected
            )
            observation_bytes.extend(next_observation.tobytes())
            next_feature = _feature(next_observation, task_index)
            if agent is not None:
                next_action, key = agent.select_action(state, next_feature)
                policy_queries += 1
                result = agent.update(
                    state,
                    reward=jnp.asarray(reward),
                    observation=next_feature,
                    terminated=jnp.asarray(step + 1 == protocol.steps_per_task),
                    next_action=next_action,
                )
                float(result.td_error)
                state = result.state.replace(
                    last_action=next_action,
                    last_observation=next_feature,
                    rng_key=key,
                )
                action = next_action
            actions.append(selected)
            rewards.append(float(reward))
        if agent is not None:
            persistent_agent_bytes = max(persistent_agent_bytes, _tree_nbytes(state))
    resources = COOMResourceReceipt(
        task_resets=len(protocol.tasks),
        environment_steps=steps_total,
        environment_reset_queries=len(protocol.tasks),
        environment_step_queries=steps_total,
        policy_queries=policy_queries,
        observation_bytes=len(observation_bytes),
        action_bytes=4 * steps_total,
        reward_bytes=4 * steps_total,
        terminal_bytes=steps_total,
        task_id_bytes=4 * (len(protocol.tasks) + steps_total),
        persistent_agent_bytes=persistent_agent_bytes,
        persistent_environment_bytes=8,
    )
    return COOMArmReceipt(
        arm_id=arm_id,
        seed=seed,
        action_sha256=_sha(actions),
        reward_sha256=_sha(rewards, floats=True),
        observation_sha256=hashlib.sha256(observation_bytes).hexdigest(),
        resources=resources,
    )


def _dependency_receipt() -> DependencyReceipt:
    """Inspect discoverability without importing incompatible packages."""
    return DependencyReceipt(
        coom_discoverable=importlib.util.find_spec("COOM") is not None,
        vizdoom_discoverable=importlib.util.find_spec("vizdoom") is not None,
        tensorflow_discoverable=importlib.util.find_spec("tensorflow") is not None,
    )


def run_coom_qualification_smoke(
    protocol: COOMSmokeProtocol | None = None,
) -> COOMSmokeResult:
    """Execute only the native synthetic contract; external execution stays blocked."""
    protocol = COOMSmokeProtocol() if protocol is None else protocol
    if type(protocol) is not COOMSmokeProtocol:
        raise ValueError("protocol must be an exact COOMSmokeProtocol")
    plan = qualification_plan(1582)
    if (
        plan.code_revisions[0].repository != COOM_REPOSITORY
        or plan.code_revisions[0].commit != COOM_COMMIT
    ):
        raise RuntimeError("external qualification catalog disagrees with COOM pin")
    arms = tuple(
        _run_arm(protocol, seed, arm_id)
        for seed in protocol.seeds
        for arm_id in FROZEN_ARMS
    )
    return COOMSmokeResult(
        schema=COOM_SMOKE_SCHEMA,
        catalog=COOMCatalogEntry(),
        protocol=protocol,
        dependencies=_dependency_receipt(),
        qualification_blockers=plan.blockers,
        arms=arms,
        identity=_current_identity(),
    )


def validate_coom_smoke_payload(payload: object) -> COOMSmokeResult:
    """Fail closed on hostile, expanded, or type-aliased JSON receipts."""
    if type(payload) is not dict:
        raise ValueError("payload must be an exact dict")
    root = cast(dict[str, Any], payload)
    if set(root) != {field.name for field in dataclasses.fields(COOMSmokeResult)}:
        raise ValueError("payload fields differ from the schema")

    def exact_nested(name: str, cls: type[Any]) -> dict[str, Any]:
        raw = root[name]
        if type(raw) is not dict or set(raw) != {field.name for field in dataclasses.fields(cls)}:
            raise ValueError(f"serialized {name} differs from the schema")
        return cast(dict[str, Any], raw)

    catalog_raw = exact_nested("catalog", COOMCatalogEntry)
    catalog_raw = dict(catalog_raw)
    for name in (
        "sequences",
        "metrics",
        "environment_dependencies",
        "learning_extras",
        "paper_seeds",
    ):
        if type(catalog_raw[name]) is not list:
            raise ValueError(f"serialized catalog {name} must be an exact list")
        catalog_raw[name] = tuple(catalog_raw[name])
    core = catalog_raw["core_sequence_tasks"]
    if type(core) is not dict or set(core) != {"CD8", "CO8"}:
        raise ValueError("serialized core task sequences differ from the schema")
    catalog_raw["core_sequence_tasks"] = {
        name: tuple(values) for name, values in cast(dict[str, list[str]], core).items()
    }
    catalog = COOMCatalogEntry(**catalog_raw)
    protocol_raw = exact_nested("protocol", COOMSmokeProtocol)
    protocol_raw = dict(protocol_raw)
    for name in ("observation_shape", "seeds"):
        if type(protocol_raw[name]) is not list:
            raise ValueError(f"serialized protocol {name} must be an exact list")
        protocol_raw[name] = tuple(protocol_raw[name])
    protocol = COOMSmokeProtocol(**protocol_raw)
    dependencies = DependencyReceipt(**exact_nested("dependencies", DependencyReceipt))
    identity = identity_from_payload(root["identity"])
    blockers = root["qualification_blockers"]
    if type(blockers) is not list:
        raise ValueError("serialized blockers must be an exact list")
    arms_raw = root["arms"]
    if type(arms_raw) is not list or len(arms_raw) != len(FROZEN_ARMS) * len(protocol.seeds):
        raise ValueError("serialized arms have the wrong bounded length")
    arm_fields = {field.name for field in dataclasses.fields(COOMArmReceipt)}
    resource_fields = {field.name for field in dataclasses.fields(COOMResourceReceipt)}
    arms: list[COOMArmReceipt] = []
    for raw in arms_raw:
        if type(raw) is not dict or set(raw) != arm_fields:
            raise ValueError("serialized arm differs from the schema")
        values = dict(cast(dict[str, Any], raw))
        resources = values["resources"]
        if type(resources) is not dict or set(resources) != resource_fields:
            raise ValueError("serialized resources differ from the schema")
        values["resources"] = COOMResourceReceipt(**resources)
        arms.append(COOMArmReceipt(**values))
    return COOMSmokeResult(
        schema=root["schema"],
        catalog=catalog,
        protocol=protocol,
        dependencies=dependencies,
        qualification_blockers=tuple(blockers),
        arms=tuple(arms),
        identity=identity,
        synthetic_contract_trace=root["synthetic_contract_trace"],
        development_only=root["development_only"],
        scientific_promotion_allowed=root["scientific_promotion_allowed"],
        benchmark_result_claimed=root["benchmark_result_claimed"],
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", choices=("CD8", "CO8"), default="CO8")
    parser.add_argument("--steps-per-task", type=int, default=4)
    parser.add_argument("--catalog-only", action="store_true")
    args = parser.parse_args(argv)
    if args.catalog_only:
        print(json.dumps(dataclasses.asdict(COOMCatalogEntry()), sort_keys=True))
        return 0
    result = run_coom_qualification_smoke(
        COOMSmokeProtocol(sequence=args.sequence, steps_per_task=args.steps_per_task)
    )
    payload = result.to_payload()
    validate_coom_smoke_payload(payload)
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CD8_TASKS",
    "CO8_TASKS",
    "COOM_COMMIT",
    "COOM_METRICS",
    "COOM_PAPER",
    "COOM_REPOSITORY",
    "COOM_SMOKE_SCHEMA",
    "FROZEN_ARMS",
    "FROZEN_DEVELOPMENT_SEEDS",
    "COOMArmReceipt",
    "COOMCatalogEntry",
    "COOMResourceReceipt",
    "COOMSmokeProtocol",
    "COOMSmokeResult",
    "DependencyReceipt",
    "run_coom_qualification_smoke",
    "validate_coom_smoke_payload",
]
