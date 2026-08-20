"""Matched, permanently nonpromoting replay/frozen IPMNIST campaign."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import stat
import sys
from importlib.metadata import version
from pathlib import Path, PosixPath
from typing import Any, Final, cast

import jax
import jax.random as jr
import numpy as np

from alberta_framework.benchmarks.ipmnist_screening import (
    _screening_dataset_provenance,
    _screening_root_key,
    replay_frozen_development_result_payload,
    run_screening_config,
    screening_spec,
)
from alberta_framework.benchmarks.upgd_ipmnist import (
    IPMNISTConfig,
    build_schedule,
    init_mlp_params,
)
from alberta_framework.evaluation.prospective_publication import (
    open_directory_chain,
    publish_prepared_json_at,
)
from alberta_framework.evaluation.replay_frozen_ipmnist_nonpromoting import (
    DEVELOPMENT_SEEDS,
    PROTOCOL,
    PROTOCOL_GAPS,
    expected_resources_for_result,
    registered_arms,
    validate_matched_replay_frozen_results,
    validate_replay_frozen_result,
)

CAMPAIGN_SCHEMA: Final = "asi.replay-frozen-ipmnist.matched-campaign.v1"
PLAN_SCHEMA: Final = "asi.replay-frozen-ipmnist.matched-plan.v1"
_RNG_IMPL: Final = "threefry2x32"
_MAX_DATASET_BYTES: Final = 256 * 1024 * 1024
_MAX_RESULT_BYTES: Final = 8 * 1024 * 1024
_MAX_JSON_NODES: Final = 100_000
_MAX_JSON_STRING_BYTES: Final = 2 * 1024 * 1024
_MAX_SOURCE_FILES: Final = 1_024
_MAX_CAMPAIGN_OBSERVATIONS: Final = 40_000_000
_MAX_CAMPAIGN_MODEL_QUERIES: Final = 800_000_000
_MAX_SCHEDULE_BYTES: Final = 128 * 1024 * 1024
_CANONICAL_X_SHA256: Final = "b8078cd833f53d89828a5e28d728517be9add34076f13fe973399f1f16381313"
_CANONICAL_Y_SHA256: Final = "4f1dd9551f104f8153409e0add59f0a71568f7bad5a5f8e2274480c186fe219a"
_DATASET_SOURCE: Final = {
    "provider": "openml",
    "name": "mnist_784",
    "version": 1,
    "row_start": 0,
    "row_stop_exclusive": 60_000,
}
_DATASET_MATERIALIZATION: Final = "alberta.ipmnist.float32-neg1-pos1-int32-labels.v1"
_SOURCE_ROOT_FILES: Final = ("pyproject.toml", "uv.lock")
_POLICY: Final = {
    "development_only": True,
    "scientific_promotion_allowed": False,
    "sota_claim_allowed": False,
    "negative_outcomes_retained": True,
    "pretrained_ceiling_claim_allowed": False,
    "hillclimb_gate_evaluated": False,
}
FROZEN_CAMPAIGN_CONFIG: Final = IPMNISTConfig(n_tasks=20, task_length=5_000)
EXECUTION_AUTHORIZED: Final = False
AUTHORIZATION_TRANSITION_APPROVED: Final = False


def _require_execution_authorized() -> None:
    if AUTHORIZATION_TRANSITION_APPROVED is not True:
        raise RuntimeError("replay/frozen execution is not independently authorized")


def _canonical(value: object) -> bytes:
    encoded = json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    if len(encoded) > _MAX_RESULT_BYTES:
        raise ValueError("campaign artifact exceeds its encoded byte ceiling")
    return encoded


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _assert_plain_json(value: object) -> None:
    seen: set[int] = set()
    nodes = 0
    strings = 0
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > 14:
            raise ValueError("campaign artifact exceeds its JSON work bound")
        if item is None or type(item) is bool:
            continue
        if type(item) is int:
            if not -(1 << 63) <= item <= (1 << 63) - 1:
                raise ValueError("campaign JSON integer exceeds signed int64")
            continue
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("campaign JSON float must be finite")
            continue
        if type(item) is str:
            try:
                encoded = item.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError("campaign JSON string must be valid UTF-8") from exc
            if len(encoded) > 4_096 or b"\x00" in encoded:
                raise ValueError("campaign JSON string must be bounded UTF-8")
            strings += len(encoded)
            if strings > _MAX_JSON_STRING_BYTES:
                raise ValueError("campaign JSON string budget exceeded")
            continue
        if type(item) is list:
            identity = id(item)
            if identity in seen or len(item) > 1_024:
                raise ValueError("campaign JSON contains an alias, cycle, or oversized list")
            seen.add(identity)
            stack.extend((child, depth + 1) for child in item)
            continue
        if type(item) is dict:
            identity = id(item)
            if identity in seen or len(item) > 256:
                raise ValueError("campaign JSON contains an alias, cycle, or oversized object")
            seen.add(identity)
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError("campaign JSON keys must be exact strings")
                stack.append((key, depth + 1))
                stack.append((child, depth + 1))
            continue
        raise ValueError("campaign artifact must be exact plain JSON")


def _preflight_config(config: object) -> IPMNISTConfig:
    if type(config) is not IPMNISTConfig:
        raise ValueError("config must be an exact IPMNISTConfig")
    checked = IPMNISTConfig(**config.to_config())
    run_count = len(DEVELOPMENT_SEEDS) * len(registered_arms())
    observations = checked.n_steps * run_count
    if observations > _MAX_CAMPAIGN_OBSERVATIONS:
        raise ValueError("campaign observation plan exceeds its bound")
    total_queries = 0
    for arm in registered_arms():
        total_queries += expected_resources_for_result(
            _receipt_family(arm),
            checked.n_steps,
            checked.input_dim,
            checked.hidden1,
            checked.hidden2,
            checked.n_classes,
        )["model_queries"] * len(DEVELOPMENT_SEEDS)
    if total_queries > _MAX_CAMPAIGN_MODEL_QUERIES:
        raise ValueError("campaign model-query plan exceeds its bound")
    schedule_bytes = (
        len(DEVELOPMENT_SEEDS)
        * checked.n_tasks
        * (checked.input_dim + checked.task_length)
        * np.dtype(np.int32).itemsize
    )
    if schedule_bytes > _MAX_SCHEDULE_BYTES:
        raise ValueError("campaign schedule plan exceeds its byte bound")
    return checked


def _receipt_family(arm: str) -> str:
    if arm not in registered_arms():
        raise ValueError("arm is outside the replay/frozen campaign roster")
    return (
        "replay"
        if screening_spec(arm).mechanism == "replay_in_context"
        else {
            "randumb_random_features": "randumb",
            "ranpac_random_projection": "ranpac",
            "prol_prompt_mechanism_off": "prol",
            "prol_prompt_proxy": "prol",
        }[arm]
    )


def _validated_arrays(
    data_x: object, data_y: object, *, config: IPMNISTConfig
) -> tuple[np.ndarray, np.ndarray]:
    if type(data_x) is not np.ndarray or data_x.dtype != np.dtype(np.float32):
        raise ValueError("data_x must be an exact float32 numpy.ndarray")
    if type(data_y) is not np.ndarray or data_y.dtype != np.dtype(np.int32):
        raise ValueError("data_y must be an exact int32 numpy.ndarray")
    if (
        data_x.ndim != 2
        or data_x.shape[1] != config.input_dim
        or data_y.shape != (data_x.shape[0],)
        or data_x.shape[0] < config.task_length
    ):
        raise ValueError("dataset shapes do not match the campaign config")
    if data_x.nbytes + data_y.nbytes > _MAX_DATASET_BYTES:
        raise ValueError("dataset exceeds the campaign byte ceiling")
    x = np.array(data_x, dtype=np.float32, order="C", copy=True)
    y = np.array(data_y, dtype=np.int32, order="C", copy=True)
    if not np.all(np.isfinite(x)):
        raise ValueError("data_x must contain only finite values")
    if np.any(y < 0) or np.any(y >= config.n_classes):
        raise ValueError("data_y contains a label outside the configured classes")
    x.flags.writeable = False
    y.flags.writeable = False
    return x, y


def _require_frozen_dataset_identity(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != {"schema", "source", "materialization", "x", "y"}:
        raise ValueError("frozen dataset identity fields drifted")
    identity = cast(dict[str, object], value)
    if (
        identity["schema"] != "alberta.ipmnist_screening.dataset_provenance.v1"
        or identity["source"] != _DATASET_SOURCE
        or identity["materialization"] != _DATASET_MATERIALIZATION
        or identity["x"] != {"dtype": "<f4", "shape": [60_000, 784], "sha256": _CANONICAL_X_SHA256}
        or identity["y"] != {"dtype": "<i4", "shape": [60_000], "sha256": _CANONICAL_Y_SHA256}
    ):
        raise ValueError("dataset does not match the retained canonical identity")
    return identity


def _read_source(path: Path) -> bytes:
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 2 * 1024 * 1024:
            raise ValueError("campaign source must be a bounded regular file")
        payload = bytearray()
        while len(payload) <= before.st_size:
            chunk = os.read(descriptor, min(64 * 1024, before.st_size + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        after = os.fstat(descriptor)
        if len(payload) != before.st_size or (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError("campaign source changed during its bounded read")
        return bytes(payload)
    finally:
        os.close(descriptor)


def _source_inventory(root: Path) -> tuple[str, ...]:
    package = tuple(
        path.relative_to(root).as_posix()
        for path in sorted((root / "alberta_framework").rglob("*.py"))
    )
    inventory = (*_SOURCE_ROOT_FILES, *package)
    if not package or len(inventory) > _MAX_SOURCE_FILES or len(set(inventory)) != len(inventory):
        raise ValueError("campaign source inventory is invalid")
    return inventory


def _source_identity() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    inventory = _source_inventory(root)
    result = {
        relative: hashlib.sha256(_read_source(root / relative)).hexdigest()
        for relative in inventory
    }
    if _source_inventory(root) != inventory:
        raise ValueError("campaign source inventory changed during capture")
    return result


def _runtime_identity() -> dict[str, object]:
    build = json.dumps(
        np.__config__.CONFIG, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    names = (
        "JAX_PLATFORMS",
        "JAX_PLATFORM_NAME",
        "JAX_ENABLE_X64",
        "JAX_DEFAULT_PRNG_IMPL",
        "JAX_DEFAULT_MATMUL_PRECISION",
        "JAX_RANDOM_SEED_OFFSET",
        "JAX_NUM_CPU_DEVICES",
        "XLA_FLAGS",
    )
    environment: dict[str, str | None] = {}
    for name in names:
        value = os.environ.get(name)
        if value is not None and (len(value.encode("utf-8")) > 4_096 or "\x00" in value):
            raise ValueError("runtime environment exceeds its text bound")
        environment[name] = value
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "numpy": np.__version__,
        "numpy_build_sha256": hashlib.sha256(build).hexdigest(),
        "jax": jax.__version__,
        "jaxlib": version("jaxlib"),
        "chex": version("chex"),
        "jaxtyping": version("jaxtyping"),
        "jax_backend": jax.default_backend(),
        "jax_devices": [
            {
                "platform": device.platform,
                "device_kind": device.device_kind,
                "process_index": int(device.process_index),
                "id": int(device.id),
            }
            for device in jax.devices()
        ],
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_default_prng_impl": str(jax.config.jax_default_prng_impl),
        "jax_default_matmul_precision": str(jax.config.jax_default_matmul_precision),
        "jax_random_seed_offset": int(jax.config.jax_random_seed_offset),
        "jax_threefry_partitionable": bool(jax.config.jax_threefry_partitionable),
        "jax_numpy_dtype_promotion": str(jax.config.jax_numpy_dtype_promotion),
        "jax_numpy_rank_promotion": str(jax.config.jax_numpy_rank_promotion),
        "environment": environment,
        "agent_rng_impl": "jax.random.key(seed, impl='threefry2x32')",
    }


def _tree_identity(value: object) -> tuple[str, int]:
    paths_and_leaves, treedef = jax.tree_util.tree_flatten_with_path(value)
    digest = hashlib.sha256(str(treedef).encode("utf-8"))
    total = 0
    for path, leaf in paths_and_leaves:
        array = np.asarray(jax.device_get(leaf))
        if array.dtype.hasobject:
            raise ValueError("initial state must contain numeric arrays")
        encoded_path = jax.tree_util.keystr(path).encode("utf-8")
        digest.update(len(encoded_path).to_bytes(4, "little"))
        digest.update(encoded_path)
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes(order="C"))
        total += int(array.nbytes)
    return digest.hexdigest(), total


def _key_sha256(key: jax.Array) -> str:
    return hashlib.sha256(
        np.asarray(jr.key_data(key), dtype=np.uint32).tobytes(order="C")
    ).hexdigest()


def _seed_identity(seed: int, *, config: IPMNISTConfig, n_train: int) -> dict[str, object]:
    if type(seed) is not int or seed not in DEVELOPMENT_SEEDS:
        raise ValueError("seed must belong to the frozen development roster")
    root = _screening_root_key(seed)
    key_init, key_schedule, key_noise = jr.split(root, 3)
    params = init_mlp_params(key_init, config)
    parameter_sha, parameter_bytes = _tree_identity(params)
    schedule = build_schedule(key_schedule, config, n_train)
    schedule_sha, schedule_bytes = _tree_identity(schedule)
    initial_states: list[dict[str, object]] = []
    for arm in registered_arms():
        spec = screening_spec(arm)
        init_fn, _step_fn = spec.factory(spec.hyperparameters)
        state_sha, state_bytes = _tree_identity(init_fn(params))
        persistent_bytes = parameter_bytes + state_bytes
        expected_persistent_bytes = expected_resources_for_result(
            _receipt_family(arm),
            config.n_steps,
            config.input_dim,
            config.hidden1,
            config.hidden2,
            config.n_classes,
        )["persistent_bytes"]
        if persistent_bytes != expected_persistent_bytes:
            raise RuntimeError("initial state bytes disagree with the resource receipt")
        initial_states.append(
            {
                "arm": arm,
                "sha256": state_sha,
                "numeric_bytes": state_bytes,
                "persistent_numeric_bytes": persistent_bytes,
            }
        )
    result: dict[str, object] = {
        "seed": seed,
        "rng_impl": _RNG_IMPL,
        "root_key_data_sha256": _key_sha256(root),
        "noise_root_key_data_sha256": _key_sha256(key_noise),
        "initial_parameters_sha256": parameter_sha,
        "initial_parameter_numeric_bytes": parameter_bytes,
        "schedule_sha256": schedule_sha,
        "schedule_numeric_bytes": schedule_bytes,
        "initial_states": initial_states,
    }
    result["identity_sha256"] = _sha256(result)
    return result


def _plan(config: IPMNISTConfig) -> dict[str, object]:
    raw_official = cast(dict[str, object], PROTOCOL["official_code"])
    official = {
        name: list(value) if type(value) is tuple else value for name, value in raw_official.items()
    }
    return {
        "schema": PLAN_SCHEMA,
        "comparison_id": "asi.replay-frozen-ipmnist.current-runner.v1",
        "seeds": list(DEVELOPMENT_SEEDS),
        "arms": list(registered_arms()),
        "config": config.to_config(),
        "run_order": "seed_major_arm_minor",
        "rng_impl": _RNG_IMPL,
        "noise_mode": "step",
        "decision_rule": "inconclusive_only_no_selection",
        "execution_authorized": EXECUTION_AUTHORIZED,
        "seed_history_audit": (
            "1573001--1573005 had zero exact matches on current main or full git history "
            "when prospectively frozen on 2026-08-20; retained seeds 0--4 are excluded"
        ),
        "selected_ipmnist_configuration": config.matches_selected_publication_configuration,
        "timing_measured": False,
        "runner_timing_telemetry_discarded": True,
        "papers": list(cast(tuple[str, ...], PROTOCOL["papers"])),
        "official_code": official,
        "protocol_gaps": list(PROTOCOL_GAPS),
        "pretrained_feature_extractor_present": False,
        "external_pretraining_examples": 0,
        "external_pretraining_updates": 0,
    }


def _normalized_receipt(result: object) -> dict[str, object]:
    receipt = replay_frozen_development_result_payload(cast(Any, result), outcome="inconclusive")
    resources = cast(dict[str, object], receipt["resources"])
    resources["timing_seconds"] = 0.0
    return validate_replay_frozen_result(receipt)


def _resources(
    runs: list[dict[str, object]],
    seeds: list[dict[str, object]],
    data_x: np.ndarray,
    data_y: np.ndarray,
) -> dict[str, object]:
    child = [
        cast(dict[str, object], cast(dict[str, object], row["receipt"])["resources"])
        for row in runs
    ]
    integer_fields = sorted(name for name, value in child[0].items() if type(value) is int)
    totals = {name: sum(cast(int, resource[name]) for resource in child) for name in integer_fields}
    state_bytes = sum(
        cast(int, state["numeric_bytes"])
        for seed in seeds
        for state in cast(list[dict[str, object]], seed["initial_states"])
    )
    identity_persistent_bytes = sum(
        cast(int, state["persistent_numeric_bytes"])
        for seed in seeds
        for state in cast(list[dict[str, object]], seed["initial_states"])
    )
    if identity_persistent_bytes != totals["persistent_bytes"]:
        raise RuntimeError("campaign state identities disagree with receipt resource totals")
    return {
        "run_count": len(runs),
        "dataset_snapshot_numeric_bytes": int(data_x.nbytes + data_y.nbytes),
        "identity_schedule_derivations": len(seeds),
        "execution_schedule_derivations": len(runs),
        "schedule_numeric_bytes_across_seed_identities": sum(
            cast(int, seed["schedule_numeric_bytes"]) for seed in seeds
        ),
        "initial_parameter_numeric_bytes_across_seed_identities": sum(
            cast(int, seed["initial_parameter_numeric_bytes"]) for seed in seeds
        ),
        "initial_state_numeric_bytes_across_seed_arm_identities": state_bytes,
        "persistent_numeric_bytes_across_seed_arm_identities": identity_persistent_bytes,
        "receipt_integer_totals": totals,
        "max_arm_persistent_bytes": max(
            cast(int, resource["persistent_bytes"]) for resource in child
        ),
        "timing_seconds": 0.0,
        "timing_measured": False,
        "timing_is_telemetry_only": True,
        "peak_working_set_claimed": False,
        "preflight_scope": (
            "bounded observations model queries dataset schedules and persistent numeric state; "
            "not scalar FLOPs compiler temporaries or a timing comparison"
        ),
    }


def _build_campaign(
    data_x: np.ndarray,
    data_y: np.ndarray,
    *,
    config: IPMNISTConfig,
    dataset_identity: dict[str, object],
) -> dict[str, object]:
    source = _source_identity()
    runtime = _runtime_identity()
    seeds = [
        _seed_identity(seed, config=config, n_train=int(data_x.shape[0]))
        for seed in DEVELOPMENT_SEEDS
    ]
    identity_by_seed = {
        cast(int, item["seed"]): cast(str, item["identity_sha256"]) for item in seeds
    }
    runs: list[dict[str, object]] = []
    for seed in DEVELOPMENT_SEEDS:
        for arm in registered_arms():
            result = run_screening_config(
                data_x,
                data_y,
                screening_spec(arm),
                seed=seed,
                config=config,
                noise_mode="step",
            )
            runs.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "seed_identity_sha256": identity_by_seed[seed],
                    "receipt": _normalized_receipt(result),
                }
            )
    if _source_identity() != source:
        raise RuntimeError("campaign source changed during execution")
    if _runtime_identity() != runtime:
        raise RuntimeError("campaign runtime changed during execution")
    plan = _plan(config)
    payload: dict[str, object] = {
        "schema": CAMPAIGN_SCHEMA,
        "status": "complete",
        "plan": plan,
        "identity": {
            "dataset": dataset_identity,
            "plan_sha256": _sha256(plan),
            "source_sha256": source,
            "runtime": runtime,
            "consistency_not_attestation": True,
        },
        "policy": dict(_POLICY),
        "seed_identities": seeds,
        "runs": runs,
        "development_outcome": "inconclusive",
        "resources": _resources(runs, seeds, data_x, data_y),
    }
    payload["result_sha256"] = _sha256(payload)
    return payload


def _execute_replay_frozen_campaign(
    data_x: object,
    data_y: object,
    *,
    config: IPMNISTConfig = FROZEN_CAMPAIGN_CONFIG,
) -> dict[str, object]:
    """Execute the exact forty-shard development roster without selecting a winner."""
    checked_config = _preflight_config(config)
    _assert_plain_json(_plan(checked_config))
    x, y = _validated_arrays(data_x, data_y, config=checked_config)
    dataset = _require_frozen_dataset_identity(_screening_dataset_provenance(x, y))
    result = _build_campaign(x, y, config=checked_config, dataset_identity=dataset)
    _assert_plain_json(result)
    return result


def run_replay_frozen_campaign(
    data_x: object,
    data_y: object,
    *,
    config: IPMNISTConfig = FROZEN_CAMPAIGN_CONFIG,
) -> dict[str, object]:
    """Fail closed until a separately reviewed change authorizes this plan."""
    _require_execution_authorized()
    return _execute_replay_frozen_campaign(data_x, data_y, config=config)


def _static_preflight(value: object) -> dict[str, object]:
    _assert_plain_json(value)
    if type(value) is not dict:
        raise ValueError("campaign must be an exact object")
    root = cast(dict[str, object], value)
    expected = {
        "schema",
        "status",
        "plan",
        "identity",
        "policy",
        "seed_identities",
        "runs",
        "development_outcome",
        "resources",
        "result_sha256",
    }
    if set(root) != expected or root["schema"] != CAMPAIGN_SCHEMA or root["status"] != "complete":
        raise ValueError("campaign schema or fields drifted")
    if root["policy"] != _POLICY or root["development_outcome"] != "inconclusive":
        raise ValueError("campaign must remain permanently nonpromoting and inconclusive")
    if type(root["plan"]) is not dict or type(root["identity"]) is not dict:
        raise ValueError("campaign plan and identity must be exact objects")
    if type(root["resources"]) is not dict:
        raise ValueError("campaign resources must be an exact object")
    seeds = root["seed_identities"]
    if type(seeds) is not list or len(seeds) != len(DEVELOPMENT_SEEDS):
        raise ValueError("campaign seed identities are incomplete")
    runs = root["runs"]
    if type(runs) is not list or len(runs) != len(DEVELOPMENT_SEEDS) * len(registered_arms()):
        raise ValueError("campaign roster is incomplete")
    roster: list[tuple[object, object]] = []
    by_seed: dict[int, list[dict[str, object]]] = {seed: [] for seed in DEVELOPMENT_SEEDS}
    for row in runs:
        if type(row) is not dict or set(row) != {"seed", "arm", "seed_identity_sha256", "receipt"}:
            raise ValueError("campaign run row fields drifted")
        seed, arm = row["seed"], row["arm"]
        if type(seed) is not int or seed not in DEVELOPMENT_SEEDS:
            raise ValueError("campaign run seed drifted")
        if type(arm) is not str or arm not in registered_arms():
            raise ValueError("campaign run arm drifted")
        receipt = validate_replay_frozen_result(row["receipt"])
        resources = cast(dict[str, object], receipt["resources"])
        if (
            receipt != row["receipt"]
            or receipt["seed"] != seed
            or receipt["arm"] != arm
            or receipt["outcome"] != "inconclusive"
            or resources["timing_seconds"] != 0.0
        ):
            raise ValueError("campaign run receipt drifted")
        if type(row["seed_identity_sha256"]) is not str or len(row["seed_identity_sha256"]) != 64:
            raise ValueError("campaign run seed identity drifted")
        roster.append((seed, arm))
        by_seed[seed].append(receipt)
    expected_roster = [(seed, arm) for seed in DEVELOPMENT_SEEDS for arm in registered_arms()]
    if roster != expected_roster:
        raise ValueError("campaign run roster order drifted")
    for seed in DEVELOPMENT_SEEDS:
        validate_matched_replay_frozen_results(by_seed[seed])
    claimed = root["result_sha256"]
    unsigned = dict(root)
    del unsigned["result_sha256"]
    if type(claimed) is not str or claimed != _sha256(unsigned):
        raise ValueError("campaign result digest drifted")
    return root


def validate_replay_frozen_campaign(
    value: object,
    data_x: object,
    data_y: object,
    *,
    config: IPMNISTConfig = FROZEN_CAMPAIGN_CONFIG,
) -> dict[str, object]:
    """Fail closed unless every identity and all forty runs recompute exactly."""
    root = _static_preflight(value)
    checked_config = _preflight_config(config)
    x, y = _validated_arrays(data_x, data_y, config=checked_config)
    dataset = _require_frozen_dataset_identity(_screening_dataset_provenance(x, y))
    plan = _plan(checked_config)
    if root["plan"] != plan:
        raise ValueError("campaign plan does not match the supplied config")
    seeds = [
        _seed_identity(seed, config=checked_config, n_train=int(x.shape[0]))
        for seed in DEVELOPMENT_SEEDS
    ]
    if root["seed_identities"] != seeds:
        raise ValueError("campaign seed identities do not match the supplied inputs")
    digests = {cast(int, seed["seed"]): cast(str, seed["identity_sha256"]) for seed in seeds}
    for row in cast(list[dict[str, object]], root["runs"]):
        if row["seed_identity_sha256"] != digests[cast(int, row["seed"])]:
            raise ValueError("campaign run does not bind its exact seed identity")
    identity = {
        "dataset": dataset,
        "plan_sha256": _sha256(plan),
        "source_sha256": _source_identity(),
        "runtime": _runtime_identity(),
        "consistency_not_attestation": True,
    }
    if root["identity"] != identity:
        raise ValueError("campaign identity does not match current inputs/source/runtime")
    expected = _build_campaign(x, y, config=checked_config, dataset_identity=dataset)
    if root != expected:
        raise ValueError("campaign does not recompute exactly from the bound inputs")
    return expected


def _recompute_derived_fields_for_test(value: dict[str, object]) -> None:
    """Re-sign a hostile test fixture so it reaches strict replay."""
    unsigned = dict(value)
    unsigned.pop("result_sha256", None)
    value["result_sha256"] = _sha256(unsigned)


def retain_replay_frozen_campaign(
    value: object,
    data_x: object,
    data_y: object,
    *,
    config: IPMNISTConfig = FROZEN_CAMPAIGN_CONFIG,
    repository_root: Path,
) -> Path:
    """Validate and publish one content-named result without replacement."""
    _require_execution_authorized()
    if type(repository_root) is not PosixPath or not repository_root.is_absolute():
        raise ValueError("repository_root must be an exact absolute POSIX Path")
    if type(value) is not dict or type(value.get("result_sha256")) is not str:
        raise ValueError("campaign lacks an exact claimed digest")
    digest = cast(str, value["result_sha256"])
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("campaign claimed digest is invalid")
    segments = ("outputs", "replay_frozen", "development.v1")
    directory = open_directory_chain(repository_root, segments)
    destination = f"result.{digest}.json"
    try:

        def validate(candidate: object) -> None:
            validate_replay_frozen_campaign(candidate, data_x, data_y, config=config)

        def prepare() -> bytes:
            return _canonical(validate_replay_frozen_campaign(value, data_x, data_y, config=config))

        publish_prepared_json_at(
            directory,
            destination,
            prepare=prepare,
            validate_loaded=validate,
            max_bytes=_MAX_RESULT_BYTES,
        )
    finally:
        os.close(directory)
    return repository_root.joinpath(*segments, destination)


__all__ = [
    "CAMPAIGN_SCHEMA",
    "FROZEN_CAMPAIGN_CONFIG",
    "retain_replay_frozen_campaign",
    "run_replay_frozen_campaign",
    "validate_replay_frozen_campaign",
]
