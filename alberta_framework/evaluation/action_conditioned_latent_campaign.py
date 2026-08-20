"""Create and strictly replay the nonpromoting #1575 matched campaign."""

from __future__ import annotations

import argparse
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
from typing import Final, cast

import jax
import jax.random as jr
import numpy as np

import alberta_framework.benchmarks.action_conditioned_latent as lane
from alberta_framework.benchmarks.action_conditioned_latent import (
    FROZEN_ARM_IDS,
    FROZEN_DEVELOPMENT_SEEDS,
    ActionLatentProtocol,
    run_action_conditioned_latent_lane,
    validate_action_latent_payload,
)
from alberta_framework.streams.closed_loop import (
    SwitchingTwoStateConfig,
    SwitchingTwoStateMDP,
)

CAMPAIGN_SCHEMA: Final = "asi.action_conditioned_latent.matched-development.v3"
CAMPAIGN_SEEDS: Final = FROZEN_DEVELOPMENT_SEEDS
_POLICY: Final = {
    "development_only": True,
    "scientific_promotion_allowed": False,
    "sota_claim_allowed": False,
    "negative_results_retained": True,
}
_MAX_RESULT_BYTES: Final = 16 * 1024 * 1024
_MAX_JSON_NODES: Final = 100_000


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
        path for path in sorted(package_root.rglob("*.py")) if "__pycache__" not in path.parts
    )
    digest = hashlib.sha256(b"asi-action-latent-python-tree-v1\0")
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
        "JAX_DEFAULT_MATMUL_PRECISION",
        "JAX_DEFAULT_PRNG_IMPL",
        "JAX_COMPILATION_CACHE_DIR",
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
        "schema": "asi.action_conditioned_latent.runtime.v1",
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
    digest = hashlib.sha256(b"asi-action-latent-array-bundle-v1\0")
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
    digest = hashlib.sha256(b"asi-action-latent-state-tree-v1\0")
    flattened, _structure = jax.tree_util.tree_flatten_with_path(tree)
    for path, leaf in flattened:
        try:
            array = np.asarray(leaf)
        except TypeError:
            array = np.asarray(jr.key_data(leaf))
        path_text = "/".join(str(item) for item in path)
        # MultiHeadMLPState carries process-clock telemetry. It consumes bytes
        # but is not an algorithmic initial condition, so bind its exact fields
        # to their canonical zero representation rather than wall-clock values.
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


def _protocol_payload(protocol: ActionLatentProtocol) -> dict[str, object]:
    return cast(dict[str, object], json.loads(json.dumps(dataclasses.asdict(protocol))))


def _workload_identity(protocol: ActionLatentProtocol) -> dict[str, object]:
    payload = {
        "protocol": _protocol_payload(protocol),
        "arm_ids": list(FROZEN_ARM_IDS),
        "research_pins": dict(lane.PINNED_RESEARCH),
    }
    return {
        "schema": "asi.action_conditioned_latent.workload.v1",
        "sha256": hashlib.sha256(_canonical(payload)).hexdigest(),
        "payload": payload,
    }


def _execution_identity(
    protocol: ActionLatentProtocol, *, seed: int, arm_id: str
) -> dict[str, object]:
    root = jr.key(seed, impl="threefry2x32")
    env_key, mechanism_key = jr.split(root)
    environment = SwitchingTwoStateMDP(
        SwitchingTwoStateConfig(phase_length=protocol.phase_length)  # type: ignore[call-arg]
    )
    environment_state = environment.init(env_key)
    if arm_id == "mechanism_off":
        initial_mechanism_hash = hashlib.sha256(
            b"asi-action-latent-mechanism-absent-v1"
        ).hexdigest()
    elif arm_id.startswith("latent_"):
        model = lane._latent_model(interactions=arm_id != "latent_no_interactions")
        initial_mechanism_hash = _tree_hash(model.init(mechanism_key))
    elif arm_id == "reconstruction_control":
        initial_mechanism_hash = _tree_hash(lane._reconstruction_model().init(mechanism_key))
    elif arm_id == "one_step_ftl_control":
        initial_mechanism_hash = _tree_hash(lane._one_step_ftl_model().init(mechanism_key))
    elif arm_id == "sarsa_control":
        initial_mechanism_hash = _tree_hash(lane._sarsa_agent().init(2, mechanism_key))
    else:  # pragma: no cover - caller is the frozen roster
        raise ValueError("unsupported campaign arm")
    base_actions = np.asarray(lane._base_actions(seed, protocol.steps), dtype="<i4")
    transition_keys = np.stack(
        [np.asarray(jr.key_data(jr.fold_in(env_key, step))) for step in range(protocol.steps)]
    ).astype("<u4", copy=False)
    decision_schedule = np.asarray(
        [
            step >= protocol.warmup_steps and step % protocol.exploration_period != 0
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
            np.frombuffer(bytes.fromhex(_tree_hash(environment_state)), dtype=np.uint8),
            np.frombuffer(bytes.fromhex(initial_mechanism_hash), dtype=np.uint8),
        ),
        "schedule_sha256": _array_bundle_hash(
            base_actions, transition_keys, decision_schedule, phase_schedule
        ),
    }


def _executions(protocol: ActionLatentProtocol) -> list[dict[str, object]]:
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
        losses = [arm["mean_prequential_loss"] for arm in receipts]
        aggregate[arm_id] = {
            "mean_metrics": {
                "return_sum": math.fsum(cast(float, arm["return_sum"]) for arm in receipts)
                / len(receipts),
                "late_return_sum": math.fsum(
                    cast(float, arm["late_return_sum"]) for arm in receipts
                )
                / len(receipts),
                "mean_prequential_loss": (
                    None
                    if all(loss is None for loss in losses)
                    else math.fsum(cast(float, loss) for loss in losses) / len(losses)
                ),
            },
            "total_additive_resources": {
                name: sum(cast(int, arm[name]) for arm in receipts)
                for name in (
                    "environment_steps",
                    "model_updates",
                    "agent_updates",
                    "training_queries",
                    "decision_queries",
                )
            },
            "max_per_shard_persistent_bytes": {
                name: max(cast(int, arm[name]) for arm in receipts)
                for name in (
                    "persistent_mechanism_bytes",
                    "persistent_environment_bytes",
                )
            },
            "negative_results_retained": all(
                arm["negative_outcome_retained"] is True for arm in receipts
            ),
        }
    return {"arms": aggregate, "shard_count": len(arms)}


def _campaign_from_lane(protocol: ActionLatentProtocol, lane_payload: object) -> dict[str, object]:
    parsed = validate_action_latent_payload(lane_payload)
    if parsed.protocol != protocol:
        raise ValueError("lane result protocol differs from the campaign workload")
    payload = parsed.to_payload()
    arms = cast(list[dict[str, object]], payload["arms"])
    result: dict[str, object] = {
        "schema": CAMPAIGN_SCHEMA,
        "status": "complete",
        "protocol": _protocol_payload(protocol),
        "roster": [[seed, arm_id] for seed in CAMPAIGN_SEEDS for arm_id in FROZEN_ARM_IDS],
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


def run_action_latent_campaign(
    protocol: ActionLatentProtocol | None = None,
) -> dict[str, object]:
    """Run the exact five-seed roster without publishing or promoting it."""
    resolved = ActionLatentProtocol() if protocol is None else protocol
    if type(resolved) is not ActionLatentProtocol:
        raise ValueError("protocol must be an exact ActionLatentProtocol")
    result = _campaign_from_lane(
        resolved, run_action_conditioned_latent_lane(resolved).to_payload()
    )
    _validate_action_latent_campaign(result, protocol=resolved, reexecute=False)
    return result


def validate_action_latent_campaign(
    value: object, *, protocol: ActionLatentProtocol | None = None
) -> None:
    """Validate current identities and replay every action-latent shard."""
    resolved = ActionLatentProtocol() if protocol is None else protocol
    if type(resolved) is not ActionLatentProtocol:
        raise ValueError("protocol must be an exact ActionLatentProtocol")
    _validate_action_latent_campaign(value, protocol=resolved, reexecute=True)


def _validate_action_latent_campaign(
    value: object, *, protocol: ActionLatentProtocol, reexecute: bool
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
    expected_roster = [[seed, arm_id] for seed in CAMPAIGN_SEEDS for arm_id in FROZEN_ARM_IDS]
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
    lane_result = validate_action_latent_payload(root["lane_result"])
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
        replayed = run_action_conditioned_latent_lane(protocol).to_payload()
        if not _same_json(root["lane_result"], replayed):
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


def write_action_latent_campaign(
    destination: Path,
    value: object,
    *,
    protocol: ActionLatentProtocol | None = None,
) -> None:
    """Replay and atomically create one campaign record without replacement."""
    resolved = _preflight_destination(destination)
    validate_action_latent_campaign(value, protocol=protocol)
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


def load_action_latent_campaign(path: Path) -> object:
    """Load one bounded canonical JSON record for subsequent strict validation."""
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
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--exploration-period", type=int, default=4)
    args = parser.parse_args(argv)
    _preflight_destination(args.output)
    protocol = ActionLatentProtocol(
        steps=args.steps,
        phase_length=args.phase_length,
        warmup_steps=args.warmup_steps,
        exploration_period=args.exploration_period,
    )
    result = run_action_latent_campaign(protocol)
    write_action_latent_campaign(args.output, result, protocol=protocol)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CAMPAIGN_SCHEMA",
    "CAMPAIGN_SEEDS",
    "load_action_latent_campaign",
    "run_action_latent_campaign",
    "validate_action_latent_campaign",
    "write_action_latent_campaign",
]
