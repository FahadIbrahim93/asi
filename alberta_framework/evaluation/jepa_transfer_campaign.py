"""Create and strictly replay the nonpromoting #1577 JEPA-transfer campaign."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.benchmarks import jepa_transfer_feasibility as lane
from alberta_framework.benchmarks.jepa_transfer_feasibility import (
    FROZEN_ARM_IDS,
    FROZEN_DEVELOPMENT_SEEDS,
    JEPATransferProtocol,
    run_jepa_transfer_feasibility,
    validate_jepa_transfer_payload,
)
from alberta_framework.core.sarsa import SARSAAgent, SARSAConfig
from alberta_framework.streams.closed_loop import (
    SwitchingTwoStateConfig,
    SwitchingTwoStateMDP,
)

CAMPAIGN_SCHEMA: Final = "asi.jepa_transfer_feasibility.matched-development.v1"
CAMPAIGN_SEEDS: Final = FROZEN_DEVELOPMENT_SEEDS
_PRETRAINED_ARMS: Final = frozenset(
    {
        "asi_encoder_transfer",
        "encoder_permuted",
        "full_warm_start_ceiling",
        "transfer_decision_off",
    }
)
_POLICY: Final = {
    "development_only": True,
    "scientific_promotion_allowed": False,
    "sota_claim_allowed": False,
    "visual_robotics_parity_claimed": False,
    "negative_results_retained": True,
}
_MAX_RESULT_BYTES: Final = 32 * 1024 * 1024
_MAX_JSON_NODES: Final = 150_000


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _same_json(actual: object, expected: object) -> bool:
    try:
        return _canonical(actual) == _canonical(expected)
    except (TypeError, ValueError):
        return False


def _preflight_json(value: object) -> None:
    pending: list[tuple[object, int]] = [(value, 0)]
    seen: set[int] = set()
    nodes = 0
    while pending:
        current, depth = pending.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > 24:
            raise ValueError("campaign JSON exceeds its structural bound")
        if current is None or type(current) in {bool, int}:
            continue
        if type(current) is float:
            if not math.isfinite(current):
                raise ValueError("campaign JSON contains a non-finite float")
            continue
        if type(current) is str:
            if len(current.encode("utf-8")) > 16_384:
                raise ValueError("campaign JSON contains an oversized string")
            continue
        if type(current) not in {dict, list}:
            raise ValueError("campaign must contain only exact JSON values")
        identity = id(current)
        if identity in seen:
            raise ValueError("campaign contains an aliased or cyclic container")
        seen.add(identity)
        if type(current) is dict:
            mapping = cast(dict[object, object], current)
            if len(mapping) > 4096 or any(type(key) is not str for key in mapping):
                raise ValueError("campaign object exceeds its field bound")
            pending.extend((item, depth + 1) for item in mapping.values())
        else:
            sequence = cast(list[object], current)
            if len(sequence) > 4096:
                raise ValueError("campaign list exceeds its item bound")
            pending.extend((item, depth + 1) for item in sequence)


def _exact_object(value: object, fields: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict or set(cast(dict[object, object], value)) != fields:
        raise ValueError(f"{label} fields drifted")
    return cast(dict[str, object], value)


def _source_identity() -> dict[str, object]:
    package_root = Path(lane.__file__).resolve().parents[1]
    paths = tuple(
        path
        for path in sorted(package_root.rglob("*.py"))
        if "__pycache__" not in path.parts
    )
    digest = hashlib.sha256(b"asi-jepa-transfer-python-tree-v1\0")
    for path in paths:
        relative = path.relative_to(package_root.parent).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    return {"package_python_tree_sha256": digest.hexdigest(), "file_count": len(paths)}


def _runtime_identity() -> dict[str, object]:
    environment_names = (
        "JAX_COMPILATION_CACHE_DIR",
        "JAX_DEFAULT_MATMUL_PRECISION",
        "JAX_DEFAULT_PRNG_IMPL",
        "JAX_ENABLE_X64",
        "JAX_NUM_CPU_DEVICES",
        "JAX_PLATFORMS",
        "JAX_PLATFORM_NAME",
        "JAX_RANDOM_SEED_OFFSET",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "TF_NUM_INTRAOP_THREADS",
        "TF_NUM_INTEROP_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "XLA_FLAGS",
    )
    return {
        "schema": "asi.jepa_transfer_feasibility.runtime.v1",
        "python": list(sys.version_info[:3]),
        "python_build": sys.version,
        "python_implementation": platform.python_implementation(),
        "python_cache_tag": sys.implementation.cache_tag,
        "byteorder": sys.byteorder,
        "system": platform.system(),
        "system_release": platform.release(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("chex", "jax", "jaxlib", "jaxtyping", "numpy")
        },
        "backend": jax.default_backend(),
        "devices": [
            {
                "platform": device.platform,
                "device_kind": device.device_kind,
                "id": device.id,
                "process_index": device.process_index,
            }
            for device in jax.devices()
        ],
        "jax_config": {
            "jax_default_matmul_precision": str(jax.config.jax_default_matmul_precision),
            "jax_default_prng_impl": str(jax.config.jax_default_prng_impl),
            "jax_disable_jit": bool(jax.config.jax_disable_jit),
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "jax_numpy_dtype_promotion": str(jax.config.jax_numpy_dtype_promotion.value),
            "jax_numpy_rank_promotion": str(jax.config.jax_numpy_rank_promotion),
            "jax_random_seed_offset": int(jax.config.jax_random_seed_offset),
            "jax_threefry_partitionable": bool(jax.config.jax_threefry_partitionable),
        },
        "environment": {name: os.environ.get(name) for name in environment_names},
    }


def _array_bundle_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256(b"asi-jepa-transfer-array-bundle-v1\0")
    for index, raw in enumerate(arrays):
        array = np.asarray(raw)
        digest.update(index.to_bytes(4, "little"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        dtype = array.dtype.str.encode("ascii")
        digest.update(len(dtype).to_bytes(4, "little"))
        digest.update(dtype)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _tree_hash(tree: object) -> str:
    digest = hashlib.sha256(b"asi-jepa-transfer-state-tree-v1\0")
    flattened, _structure = jax.tree_util.tree_flatten_with_path(tree)
    for path, leaf in flattened:
        try:
            array = np.asarray(leaf)
        except TypeError:
            array = np.asarray(jr.key_data(leaf))
        path_text = "/".join(str(item) for item in path)
        if "birth_timestamp" in path_text or "uptime_s" in path_text:
            array = np.zeros_like(array)
        encoded_path = path_text.encode("utf-8")
        digest.update(len(encoded_path).to_bytes(4, "little"))
        digest.update(encoded_path)
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        dtype = array.dtype.str.encode("ascii")
        digest.update(len(dtype).to_bytes(4, "little"))
        digest.update(dtype)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _protocol_payload(protocol: JEPATransferProtocol) -> dict[str, object]:
    return cast(dict[str, object], json.loads(json.dumps(dataclasses.asdict(protocol))))


def _workload_identity(protocol: JEPATransferProtocol) -> dict[str, object]:
    payload = {
        "protocol": _protocol_payload(protocol),
        "arm_ids": list(FROZEN_ARM_IDS),
        "research_pins": dict(lane.PINNED_RESEARCH),
        "external_assets": {
            "imported_checkpoint_sha256": None,
            "imported_pretraining_bytes": 0,
            "visual_dataset_sha256": None,
            "robot_dataset_sha256": None,
        },
    }
    return {
        "schema": "asi.jepa_transfer_feasibility.workload.v1",
        "sha256": hashlib.sha256(_canonical(payload)).hexdigest(),
        "payload": payload,
    }


def _pretraining_replay(
    protocol: JEPATransferProtocol, seed: int
) -> tuple[str, np.ndarray, np.ndarray]:
    root = jr.key(seed, impl="threefry2x32")
    env_key, _model_key = jr.split(jr.fold_in(root, 1))
    environment = SwitchingTwoStateMDP(
        SwitchingTwoStateConfig(phase_length=protocol.phase_length)  # type: ignore[call-arg]
    )
    state = environment.init(env_key)
    observation = environment.observe(state)
    actions = np.asarray(
        lane._base_actions(seed + 17, protocol.pretraining_steps), dtype="<i4"
    )
    observations: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    next_observations: list[np.ndarray] = []
    transition_keys: list[np.ndarray] = []
    for step, action in enumerate(actions):
        key = jr.fold_in(env_key, step)
        next_observation, reward, state = environment.step(
            state, jnp.asarray(action, dtype=jnp.int32), key
        )
        observations.append(np.asarray(observation, dtype="<f4"))
        rewards.append(np.asarray(reward, dtype="<f4"))
        next_observations.append(np.asarray(next_observation, dtype="<f4"))
        transition_keys.append(np.asarray(jr.key_data(key), dtype="<u4"))
        observation = next_observation
    replay_hash = _array_bundle_hash(
        np.stack(observations),
        actions,
        np.stack(rewards),
        np.stack(next_observations),
    )
    return replay_hash, actions, np.stack(transition_keys)


def _execution_identity(
    protocol: JEPATransferProtocol, *, seed: int, arm_id: str
) -> dict[str, object]:
    root = jr.key(seed, impl="threefry2x32")
    deployment_env_key, agent_key = jr.split(root)
    environment = SwitchingTwoStateMDP(
        SwitchingTwoStateConfig(phase_length=protocol.phase_length)  # type: ignore[call-arg]
    )
    initial_mechanism: object | None
    deployment_environment = environment.init(deployment_env_key)
    initial_components = [_tree_hash(deployment_environment)]
    if arm_id in _PRETRAINED_ARMS:
        pretrain_env_key, pretrain_model_key = jr.split(jr.fold_in(root, 1))
        initial_environment = environment.init(pretrain_env_key)
        initial_mechanism = lane._model(encoder_learning=True).init(pretrain_model_key)
        fresh_deployment = lane._model(encoder_learning=False).init(jr.fold_in(root, 2))
        initial_components.extend(
            (
                _tree_hash(initial_environment),
                _tree_hash(initial_mechanism),
                _tree_hash(fresh_deployment),
            )
        )
        replay_hash, pretrain_actions, pretrain_keys = _pretraining_replay(protocol, seed)
    else:
        pretrain_actions = np.zeros((0,), dtype="<i4")
        pretrain_keys = np.zeros((0, 2), dtype="<u4")
        replay_hash = hashlib.sha256(b"asi-jepa-transfer-no-pretraining-replay-v1").hexdigest()
        if arm_id == "no_pretraining":
            initial_mechanism = lane._model(encoder_learning=False).init(
                jr.fold_in(root, 2)
            )
        elif arm_id == "sarsa_control":
            agent = SARSAAgent(
                SARSAConfig(  # type: ignore[call-arg]
                    n_actions=2,
                    gamma=0.99,
                    epsilon_start=0.1,
                    epsilon_end=0.1,
                    epsilon_decay_steps=1,
                ),
                hidden_sizes=(),
                sparsity=0.0,
                use_layer_norm=False,
            )
            initial_mechanism = agent.init(2, agent_key)
        else:
            initial_mechanism = None
    mechanism_hash = (
        hashlib.sha256(b"asi-jepa-transfer-mechanism-absent-v1").hexdigest()
        if initial_mechanism is None
        else _tree_hash(initial_mechanism)
    )
    if arm_id not in _PRETRAINED_ARMS:
        initial_components.append(mechanism_hash)
    base_actions = np.asarray(lane._base_actions(seed, protocol.steps), dtype="<i4")
    deployment_keys = np.stack(
        [
            np.asarray(jr.key_data(jr.fold_in(deployment_env_key, step)), dtype="<u4")
            for step in range(protocol.steps)
        ]
    )
    decision_schedule = np.asarray(
        [
            step >= protocol.warmup_steps
            and step % protocol.exploration_period != 0
            for step in range(protocol.steps)
        ],
        dtype=np.bool_,
    )
    phase_schedule = np.asarray(
        [(step // protocol.phase_length) % 2 for step in range(protocol.steps)],
        dtype="<i4",
    )
    return {
        "seed": seed,
        "arm_id": arm_id,
        "prng_implementation": "threefry2x32",
        "initial_state_sha256": _array_bundle_hash(
            *(
                np.frombuffer(bytes.fromhex(value), dtype=np.uint8)
                for value in initial_components
            ),
        ),
        "replay_sha256": replay_hash,
        "schedule_sha256": _array_bundle_hash(
            pretrain_actions,
            pretrain_keys,
            base_actions,
            deployment_keys,
            decision_schedule,
            phase_schedule,
        ),
        "imported_checkpoint_sha256": None,
    }


def _executions(protocol: JEPATransferProtocol) -> list[dict[str, object]]:
    return [
        _execution_identity(protocol, seed=seed, arm_id=arm_id)
        for seed in CAMPAIGN_SEEDS
        for arm_id in FROZEN_ARM_IDS
    ]


def _aggregate(raw_arms: object) -> dict[str, object]:
    if type(raw_arms) is not list:
        raise ValueError("campaign arms must be an exact list")
    arms = cast(list[dict[str, object]], raw_arms)
    aggregate: dict[str, object] = {}
    for arm_id in FROZEN_ARM_IDS:
        receipts = [arm for arm in arms if arm.get("arm_id") == arm_id]
        if len(receipts) != len(CAMPAIGN_SEEDS):
            raise ValueError("campaign aggregate roster drifted")
        losses = [arm["mean_prediction_loss"] for arm in receipts]
        resources = [cast(dict[str, object], arm["resources"]) for arm in receipts]
        aggregate[arm_id] = {
            "mean_metrics": {
                "return_sum": math.fsum(cast(float, arm["return_sum"]) for arm in receipts)
                / len(receipts),
                "late_return_sum": math.fsum(
                    cast(float, arm["late_return_sum"]) for arm in receipts
                )
                / len(receipts),
                "mean_prediction_loss": (
                    None
                    if all(loss is None for loss in losses)
                    else math.fsum(cast(float, loss) for loss in losses) / len(losses)
                ),
            },
            "total_additive_resources": {
                name: sum(cast(int, resource[name]) for resource in resources)
                for name in (
                    "environment_steps",
                    "pretraining_examples",
                    "pretraining_updates",
                    "pretraining_encoder_updates",
                    "imported_pretraining_bytes",
                    "pretraining_replay_bytes",
                    "online_replay_bytes",
                    "encoder_queries",
                    "model_queries",
                    "control_queries",
                )
            },
            "max_per_shard_persistent_bytes": {
                name: max(cast(int, resource[name]) for resource in resources)
                for name in (
                    "encoder_bytes",
                    "persistent_mechanism_bytes",
                    "persistent_environment_bytes",
                )
            },
            "negative_results_retained": all(
                arm["negative_outcome_retained"] is True for arm in receipts
            ),
        }
    return {"arms": aggregate, "shard_count": len(arms)}


def _normalized_without_timing(payload: object) -> object:
    normalized = copy.deepcopy(payload)
    if type(normalized) is not dict:
        return normalized
    arms = cast(dict[str, Any], normalized).get("arms")
    if type(arms) is not list:
        return normalized
    for arm in arms:
        if type(arm) is not dict or type(arm.get("timing")) is not dict:
            continue
        timing = cast(dict[str, object], arm["timing"])
        for name in (
            "pretraining_ns",
            "environment_ns",
            "decision_query_ns",
            "online_update_ns",
            "control_ns",
        ):
            timing[name] = 0
    return normalized


def _campaign_from_lane(
    protocol: JEPATransferProtocol, lane_payload: object
) -> dict[str, object]:
    parsed = validate_jepa_transfer_payload(lane_payload)
    if parsed.protocol != protocol:
        raise ValueError("lane result protocol differs from the campaign workload")
    payload = parsed.to_payload()
    arms = cast(list[dict[str, object]], payload["arms"])
    result: dict[str, object] = {
        "schema": CAMPAIGN_SCHEMA,
        "status": "complete",
        "protocol": _protocol_payload(protocol),
        "roster": [
            [seed, arm_id] for seed in CAMPAIGN_SEEDS for arm_id in FROZEN_ARM_IDS
        ],
        "identity": {
            "source": _source_identity(),
            "runtime": _runtime_identity(),
            "workload": _workload_identity(protocol),
            "prng_implementation": "threefry2x32",
            "consistency_not_attestation": True,
        },
        "policy": dict(_POLICY),
        "decision": "inconclusive",
        "executions": _executions(protocol),
        "lane_result": payload,
        "aggregate": _aggregate(arms),
    }
    result["result_sha256"] = hashlib.sha256(_canonical(result)).hexdigest()
    return result


def run_jepa_transfer_campaign(
    protocol: JEPATransferProtocol | None = None,
) -> dict[str, object]:
    """Run all 35 development shards without publishing or promoting them."""
    resolved = JEPATransferProtocol() if protocol is None else protocol
    if type(resolved) is not JEPATransferProtocol:
        raise ValueError("protocol must be an exact JEPATransferProtocol")
    result = _campaign_from_lane(
        resolved, run_jepa_transfer_feasibility(resolved).to_payload()
    )
    _validate_jepa_transfer_campaign(result, protocol=resolved, reexecute=False)
    return result


def validate_jepa_transfer_campaign(
    value: object, *, protocol: JEPATransferProtocol | None = None
) -> None:
    """Validate every identity and reexecute every shard, excluding timing."""
    resolved = JEPATransferProtocol() if protocol is None else protocol
    if type(resolved) is not JEPATransferProtocol:
        raise ValueError("protocol must be an exact JEPATransferProtocol")
    _validate_jepa_transfer_campaign(value, protocol=resolved, reexecute=True)


def _validate_jepa_transfer_campaign(
    value: object, *, protocol: JEPATransferProtocol, reexecute: bool
) -> None:
    _preflight_json(value)
    root = _exact_object(
        value,
        {
            "schema",
            "status",
            "protocol",
            "roster",
            "identity",
            "policy",
            "decision",
            "executions",
            "lane_result",
            "aggregate",
            "result_sha256",
        },
        "campaign",
    )
    if root["schema"] != CAMPAIGN_SCHEMA or root["status"] != "complete":
        raise ValueError("campaign schema or status drifted")
    if not _same_json(root["protocol"], _protocol_payload(protocol)):
        raise ValueError("campaign protocol drifted")
    expected_roster = [
        [seed, arm_id] for seed in CAMPAIGN_SEEDS for arm_id in FROZEN_ARM_IDS
    ]
    if not _same_json(root["roster"], expected_roster):
        raise ValueError("campaign roster drifted")
    expected_identity = {
        "source": _source_identity(),
        "runtime": _runtime_identity(),
        "workload": _workload_identity(protocol),
        "prng_implementation": "threefry2x32",
        "consistency_not_attestation": True,
    }
    if not _same_json(root["identity"], expected_identity):
        raise ValueError("campaign source/runtime/workload identity drifted")
    if not _same_json(root["policy"], _POLICY):
        raise ValueError("campaign must remain permanently nonpromoting")
    if type(root["decision"]) is not str or root["decision"] != "inconclusive":
        raise ValueError("campaign decision must remain inconclusive")
    if not _same_json(root["executions"], _executions(protocol)):
        raise ValueError("campaign execution identity drifted")
    lane_result = validate_jepa_transfer_payload(root["lane_result"])
    if lane_result.protocol != protocol:
        raise ValueError("campaign lane protocol drifted")
    lane_payload = lane_result.to_payload()
    arms = cast(list[dict[str, object]], lane_payload["arms"])
    if not _same_json(root["aggregate"], _aggregate(arms)):
        raise ValueError("campaign aggregate drifted")
    unsigned = dict(root)
    claimed_digest = unsigned.pop("result_sha256")
    if (
        type(claimed_digest) is not str
        or claimed_digest != hashlib.sha256(_canonical(unsigned)).hexdigest()
    ):
        raise ValueError("campaign result digest drifted")
    if reexecute:
        replayed = run_jepa_transfer_feasibility(protocol).to_payload()
        if not _same_json(
            _normalized_without_timing(root["lane_result"]),
            _normalized_without_timing(replayed),
        ):
            raise ValueError("campaign result disagrees with strict current-source reexecution")


def _preflight_destination(destination: Path) -> Path:
    if type(destination) is not type(Path()):
        raise TypeError("destination must be an exact Path")
    parent = destination.parent.resolve(strict=True)
    resolved = parent / destination.name
    if not parent.is_dir() or destination.name in {"", ".", ".."}:
        raise ValueError("destination parent must be an existing directory")
    if os.path.lexists(resolved):
        raise FileExistsError(f"refusing to replace existing campaign: {resolved}")
    return resolved


def write_jepa_transfer_campaign(
    destination: Path,
    value: object,
    *,
    protocol: JEPATransferProtocol | None = None,
) -> None:
    """Replay and atomically create one result without replacing retained data."""
    resolved = _preflight_destination(destination)
    validate_jepa_transfer_campaign(value, protocol=protocol)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=resolved.parent, prefix=f".{resolved.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, resolved, follow_symlinks=False)
        published = True
        directory = os.open(resolved.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        if published:
            resolved.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def load_jepa_transfer_campaign(path: Path) -> object:
    """Load one bounded strict JSON file for subsequent replay validation."""
    if type(path) is not type(Path()) or path.is_symlink() or not path.is_file():
        raise ValueError("campaign path must be an exact non-symlink file")
    size = path.stat().st_size
    if not 1 <= size <= _MAX_RESULT_BYTES:
        raise ValueError("campaign file exceeds its byte bound")
    try:
        value = json.loads(
            path.read_text(encoding="ascii"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("campaign must be strict ASCII JSON") from error
    _preflight_json(value)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--phase-length", type=int, default=32)
    parser.add_argument("--pretraining-steps", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--exploration-period", type=int, default=4)
    args = parser.parse_args(argv)
    _preflight_destination(args.output)
    protocol = JEPATransferProtocol(
        steps=args.steps,
        phase_length=args.phase_length,
        pretraining_steps=args.pretraining_steps,
        warmup_steps=args.warmup_steps,
        exploration_period=args.exploration_period,
    )
    result = run_jepa_transfer_campaign(protocol)
    write_jepa_transfer_campaign(args.output, result, protocol=protocol)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CAMPAIGN_SCHEMA",
    "CAMPAIGN_SEEDS",
    "load_jepa_transfer_campaign",
    "run_jepa_transfer_campaign",
    "validate_jepa_transfer_campaign",
    "write_jepa_transfer_campaign",
]
