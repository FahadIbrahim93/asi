"""Run the complete, permanently nonpromoting issue #1571 AdaLin campaign."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path
from types import MappingProxyType
from typing import Final, cast

import jax
import numpy as np

from alberta_framework.benchmarks.adalin import (
    AdaLinConfig,
    initialize_adalin_state,
    make_pmnist_schedule,
    run_adalin_development,
    validate_adalin_result,
)

RESULT_SCHEMA: Final = "asi.adalin.matched-development.v1"
DEVELOPMENT_SEEDS: Final = (15710, 15711, 15712, 15713, 15714)
ARMS: Final = ("relu_alpha_zero_mechanism_off", "adalin")
_ARM_ENABLED: Final = MappingProxyType(
    {"relu_alpha_zero_mechanism_off": False, "adalin": True}
)
CAMPAIGN_PROFILES: Final = MappingProxyType(
    {
        "contract-smoke": AdaLinConfig(
            tasks=2,
            examples_per_task=4,
            batch_size=2,
            hidden_widths=(3, 2),
            classes=2,
        ),
        "bounded-development": AdaLinConfig(
            tasks=8,
            examples_per_task=64,
            batch_size=1,
            hidden_widths=(300, 150),
            classes=10,
        ),
    }
)
_POLICY: Final = {
    "development_only": True,
    "scientific_promotion_allowed": False,
    "sota_claim_allowed": False,
    "negative_results_retained": True,
}
_MAX_BYTES: Final = 256 * 1024 * 1024
_MAX_JSON_NODES: Final = 100_000
_MAX_JSON_STRING_BYTES: Final = 16_384


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _same_json_type_and_value(actual: object, expected: object) -> bool:
    """Compare canonical JSON so booleans, integers, and floats cannot alias."""
    try:
        return _canonical(actual) == _canonical(expected)
    except (TypeError, ValueError):
        return False


def _json_preflight(value: object) -> None:
    pending: list[tuple[object, int]] = [(value, 0)]
    seen: set[int] = set()
    nodes = 0
    while pending:
        current, depth = pending.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > 18:
            raise ValueError("matched result exceeds its JSON structure bound")
        if current is None or type(current) in {bool, int}:
            continue
        if type(current) is float:
            if not math.isfinite(current):
                raise ValueError("matched result contains a non-finite float")
            continue
        if type(current) is str:
            if len(current.encode("utf-8")) > _MAX_JSON_STRING_BYTES:
                raise ValueError("matched result contains an oversized string")
            continue
        if type(current) not in {dict, list}:
            raise ValueError("matched result must contain only exact JSON values")
        identity = id(current)
        if identity in seen:
            raise ValueError("matched result contains an aliased or cyclic container")
        seen.add(identity)
        if type(current) is dict:
            mapping = cast(dict[object, object], current)
            if len(mapping) > 4096 or any(type(key) is not str for key in mapping):
                raise ValueError("matched result object exceeds its field bound")
            pending.extend((item, depth + 1) for item in mapping.values())
        else:
            sequence = cast(list[object], current)
            if len(sequence) > 4096:
                raise ValueError("matched result list exceeds its item bound")
            pending.extend((item, depth + 1) for item in sequence)


def _exact_object(value: object, fields: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict or set(cast(dict[object, object], value)) != fields:
        raise ValueError(f"{label} fields drifted")
    return cast(dict[str, object], value)


def _source_identity() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    relative_paths = (
        "pyproject.toml",
        "uv.lock",
        "alberta_framework/benchmarks/adalin.py",
        "alberta_framework/evaluation/adalin_matched_runner.py",
    )
    return {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in relative_paths
    }


def _runtime_identity() -> dict[str, object]:
    environment_names = (
        "JAX_DEFAULT_MATMUL_PRECISION",
        "JAX_DEFAULT_PRNG_IMPL",
        "JAX_ENABLE_X64",
        "JAX_NUM_CPU_DEVICES",
        "JAX_PLATFORMS",
        "JAX_PLATFORM_NAME",
        "JAX_RANDOM_SEED_OFFSET",
        "XLA_FLAGS",
    )
    return {
        "schema": "asi.adalin.runtime.v1",
        "python": list(sys.version_info[:3]),
        "python_implementation": platform.python_implementation(),
        "byteorder": sys.byteorder,
        "platform": sys.platform,
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("chex", "jax", "jaxlib", "numpy")
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


def _checked_profile(value: object) -> str:
    if type(value) is not str or value not in CAMPAIGN_PROFILES:
        raise ValueError("profile must name one registered AdaLin campaign profile")
    return value


def _validated_arrays(
    train_inputs: object,
    train_labels: object,
    test_inputs: object,
    test_labels: object,
    profile: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    config = CAMPAIGN_PROFILES[profile]
    for value, name in ((train_inputs, "train_inputs"), (test_inputs, "test_inputs")):
        if type(value) is not np.ndarray or value.dtype != np.dtype(np.float32):
            raise ValueError(f"{name} must be an exact float32 NumPy array")
        if value.ndim != 2 or not np.isfinite(value).all():
            raise ValueError(f"{name} must be a finite matrix")
    for value, name in ((train_labels, "train_labels"), (test_labels, "test_labels")):
        if type(value) is not np.ndarray or value.dtype != np.dtype(np.int32):
            raise ValueError(f"{name} must be an exact int32 NumPy array")
        if value.ndim != 1:
            raise ValueError(f"{name} must be a vector")
    train_x = cast(np.ndarray, train_inputs)
    train_y = cast(np.ndarray, train_labels)
    test_x = cast(np.ndarray, test_inputs)
    test_y = cast(np.ndarray, test_labels)
    if (
        train_x.shape[0] != config.examples_per_task
        or train_y.shape[0] != train_x.shape[0]
        or test_x.shape[0] < 1
        or test_y.shape[0] != test_x.shape[0]
        or train_x.shape[1] != test_x.shape[1]
        or train_x.shape[1] < 1
    ):
        raise ValueError("dataset shape does not match the AdaLin campaign profile")
    if sum(value.nbytes for value in (train_x, train_y, test_x, test_y)) > _MAX_BYTES:
        raise ValueError("dataset exceeds the campaign's 256 MiB byte bound")
    if (
        np.any(train_y < 0)
        or np.any(train_y >= config.classes)
        or np.any(test_y < 0)
        or np.any(test_y >= config.classes)
    ):
        raise ValueError("labels are outside the configured class range")
    return tuple(
        np.array(value, copy=True, order="C")
        for value in (train_x, train_y, test_x, test_y)
    )  # type: ignore[return-value]


def _hash_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256(b"asi-adalin-array-bundle-v1\0")
    for index, array in enumerate(arrays):
        digest.update(index.to_bytes(4, "little"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        dtype = array.dtype.str.encode("ascii")
        digest.update(len(dtype).to_bytes(4, "little"))
        digest.update(dtype)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _state_hash(state: object) -> str:
    return _hash_arrays(*(np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(state)))


def _execution_identity(
    seed: int, profile: str, input_dim: int, *, mechanism_enabled: bool
) -> dict[str, str]:
    config = CAMPAIGN_PROFILES[profile]
    schedule = make_pmnist_schedule(config, seed=seed, input_dim=input_dim)
    state = initialize_adalin_state(
        config,
        input_dim=input_dim,
        classes=config.classes,
        seed=seed,
        mechanism_enabled=mechanism_enabled,
    )
    return {
        "schedule_sha256": _hash_arrays(
            np.asarray(schedule.pixel_permutations), np.asarray(schedule.example_orders)
        ),
        "initial_state_sha256": _state_hash(state),
        "prng_implementation": "threefry2x32",
    }


def _config_payload(profile: str) -> dict[str, object]:
    return cast(dict[str, object], json.loads(json.dumps(asdict(CAMPAIGN_PROFILES[profile]))))


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    arms: dict[str, object] = {}
    additive_names = (
        "environment_data_steps",
        "observations",
        "label_queries",
        "optimizer_updates",
        "preupdate_prediction_model_queries",
        "differentiated_training_model_queries",
        "postupdate_test_model_queries",
        "model_queries",
        "preupdate_prediction_forward_calls",
        "differentiated_training_forward_calls",
        "postupdate_test_forward_calls",
        "model_forward_calls",
    )
    peak_names = (
        "initial_total_bytes",
        "final_total_bytes",
        "parameter_bytes",
        "alpha_bytes",
        "optimizer_state_bytes",
    )
    for arm in ARMS:
        results = [
            cast(dict[str, object], row["result"]) for row in rows if row["arm"] == arm
        ]
        metrics = [cast(dict[str, object], result["metrics"]) for result in results]
        resources = [cast(dict[str, object], result["resources"]) for result in results]
        states = [cast(dict[str, object], result["state"]) for result in results]
        arms[arm] = {
            "mean_metrics": {
                "asi_whole_stream_preupdate_online_accuracy": math.fsum(
                    cast(float, metric["asi_whole_stream_preupdate_online_accuracy"])
                    for metric in metrics
                )
                / len(metrics),
                "paper_current_task_test_accuracy_mean": math.fsum(
                    cast(float, metric["paper_current_task_test_accuracy_mean"])
                    for metric in metrics
                )
                / len(metrics),
                "final_alpha_l2": math.fsum(
                    cast(float, state["final_alpha_l2"]) for state in states
                )
                / len(states),
            },
            "total_additive_resources": {
                name: sum(cast(int, resource[name]) for resource in resources)
                for name in additive_names
            },
            "max_per_shard_state_bytes": {
                name: max(cast(int, state[name]) for state in states)
                for name in peak_names
            },
        }
    return {"arms": arms, "row_count": len(rows)}


def run_adalin_matched(
    train_inputs: object,
    train_labels: object,
    test_inputs: object,
    test_labels: object,
    *,
    profile: str = "bounded-development",
) -> dict[str, object]:
    """Execute all five development seeds across the exact two-arm roster."""
    checked_profile = _checked_profile(profile)
    train_x, train_y, test_x, test_y = _validated_arrays(
        train_inputs, train_labels, test_inputs, test_labels, checked_profile
    )
    rows: list[dict[str, object]] = []
    for seed in DEVELOPMENT_SEEDS:
        for arm in ARMS:
            enabled = _ARM_ENABLED[arm]
            rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "execution_identity": _execution_identity(
                        seed,
                        checked_profile,
                        train_x.shape[1],
                        mechanism_enabled=enabled,
                    ),
                    "result": run_adalin_development(
                        train_x,
                        train_y,
                        test_x,
                        test_y,
                        config=CAMPAIGN_PROFILES[checked_profile],
                        seed=seed,
                        mechanism_enabled=enabled,
                    ),
                }
            )
    campaign: dict[str, object] = {
        "schema": RESULT_SCHEMA,
        "status": "complete",
        "profile": checked_profile,
        "config": _config_payload(checked_profile),
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "arms": list(ARMS),
        "identity": {
            "dataset_sha256": _hash_arrays(train_x, train_y, test_x, test_y),
            "source_sha256": _source_identity(),
            "runtime": _runtime_identity(),
            "consistency_not_attestation": True,
        },
        "policy": dict(_POLICY),
        "decision": "inconclusive",
        "rows": rows,
        "aggregate": _aggregate(rows),
    }
    campaign["result_sha256"] = hashlib.sha256(_canonical(campaign)).hexdigest()
    _validate_adalin_matched(
        campaign,
        train_x,
        train_y,
        test_x,
        test_y,
        profile=checked_profile,
        reexecute=False,
    )
    return campaign


def validate_adalin_matched(
    value: object,
    train_inputs: object,
    train_labels: object,
    test_inputs: object,
    test_labels: object,
    *,
    profile: str = "bounded-development",
) -> None:
    """Strictly validate and replay every row, excluding timing telemetry."""
    _validate_adalin_matched(
        value,
        train_inputs,
        train_labels,
        test_inputs,
        test_labels,
        profile=profile,
        reexecute=True,
    )


def _validate_adalin_matched(
    value: object,
    train_inputs: object,
    train_labels: object,
    test_inputs: object,
    test_labels: object,
    *,
    profile: str,
    reexecute: bool,
) -> None:
    _json_preflight(value)
    root = _exact_object(
        value,
        {
            "schema",
            "status",
            "profile",
            "config",
            "development_seeds",
            "arms",
            "identity",
            "policy",
            "decision",
            "rows",
            "aggregate",
            "result_sha256",
        },
        "matched result",
    )
    if root["schema"] != RESULT_SCHEMA or root["status"] != "complete":
        raise ValueError("matched result identity drifted")
    checked_profile = _checked_profile(profile)
    train_x, train_y, test_x, test_y = _validated_arrays(
        train_inputs, train_labels, test_inputs, test_labels, checked_profile
    )
    if root["profile"] != checked_profile or not _same_json_type_and_value(
        root["config"], _config_payload(checked_profile)
    ):
        raise ValueError("matched result profile or config drifted")
    if root["development_seeds"] != list(DEVELOPMENT_SEEDS) or root["arms"] != list(ARMS):
        raise ValueError("matched result protocol roster drifted")
    expected_identity = {
        "dataset_sha256": _hash_arrays(train_x, train_y, test_x, test_y),
        "source_sha256": _source_identity(),
        "runtime": _runtime_identity(),
        "consistency_not_attestation": True,
    }
    if not _same_json_type_and_value(root["identity"], expected_identity):
        raise ValueError("matched result identity drifted")
    if not _same_json_type_and_value(root["policy"], _POLICY):
        raise ValueError("matched result must remain permanently nonpromoting")
    if root["decision"] != "inconclusive" or type(root["decision"]) is not str:
        raise ValueError("matched campaign decision must remain inconclusive")
    expected_roster = [(seed, arm) for seed in DEVELOPMENT_SEEDS for arm in ARMS]
    raw_rows = root["rows"]
    if type(raw_rows) is not list or len(raw_rows) != len(expected_roster):
        raise ValueError("matched result roster is incomplete")
    rows = cast(list[dict[str, object]], raw_rows)
    observed: list[tuple[object, object]] = []
    for row in rows:
        checked_row = _exact_object(
            row, {"seed", "arm", "execution_identity", "result"}, "matched row"
        )
        seed = checked_row["seed"]
        arm = checked_row["arm"]
        if (
            type(seed) is not int
            or seed not in DEVELOPMENT_SEEDS
            or type(arm) is not str
            or arm not in ARMS
        ):
            raise ValueError("matched row roster identity drifted")
        enabled = _ARM_ENABLED[arm]
        expected_execution = _execution_identity(
            seed,
            checked_profile,
            train_x.shape[1],
            mechanism_enabled=enabled,
        )
        if not _same_json_type_and_value(
            checked_row["execution_identity"], expected_execution
        ):
            raise ValueError("matched row execution identity drifted")
        validate_adalin_result(checked_row["result"])
        result = cast(dict[str, object], checked_row["result"])
        if result["seed"] != seed or result["arm"] != arm:
            raise ValueError("matched row result identity drifted")
        dataset = cast(dict[str, object], result["dataset"])
        provenance = cast(dict[str, object], result["provenance"])
        if dataset["sha256"] != expected_identity["dataset_sha256"]:
            raise ValueError("matched row dataset identity drifted")
        if (
            provenance["schedule_sha256"] != expected_execution["schedule_sha256"]
            or provenance["initial_state_sha256"]
            != expected_execution["initial_state_sha256"]
        ):
            raise ValueError("matched row provenance identity drifted")
        observed.append((seed, arm))
    if observed != expected_roster:
        raise ValueError("matched result roster order drifted")
    if not _same_json_type_and_value(root["aggregate"], _aggregate(rows)):
        raise ValueError("matched result aggregate drifted")
    unsigned = dict(root)
    claimed = unsigned.pop("result_sha256")
    if type(claimed) is not str or claimed != hashlib.sha256(_canonical(unsigned)).hexdigest():
        raise ValueError("matched result digest drifted")
    if reexecute:
        for row in rows:
            claimed_result = cast(dict[str, object], row["result"])
            expected_result = run_adalin_development(
                train_x,
                train_y,
                test_x,
                test_y,
                config=CAMPAIGN_PROFILES[checked_profile],
                seed=cast(int, row["seed"]),
                mechanism_enabled=_ARM_ENABLED[cast(str, row["arm"])],
            )
            expected_resources = cast(dict[str, object], expected_result["resources"])
            claimed_resources = cast(dict[str, object], claimed_result["resources"])
            expected_resources["wall_clock_seconds_telemetry"] = claimed_resources[
                "wall_clock_seconds_telemetry"
            ]
            if expected_result != claimed_result:
                raise ValueError("matched row disagrees with strict current-source reexecution")


def _preflight_destination(destination: Path) -> Path:
    if type(destination) is not type(Path()):
        raise TypeError("destination must be an exact Path")
    parent = destination.parent.resolve(strict=True)
    resolved = parent / destination.name
    if not parent.is_dir() or destination.name in {"", ".", ".."}:
        raise ValueError("destination parent must be an existing directory")
    if os.path.lexists(resolved):
        raise FileExistsError(f"refusing to replace existing result: {resolved}")
    return resolved


def write_adalin_matched(
    destination: Path,
    value: object,
    train_inputs: object,
    train_labels: object,
    test_inputs: object,
    test_labels: object,
    *,
    profile: str = "bounded-development",
) -> None:
    """Strictly replay and publish one result without replacing retained data."""
    resolved = _preflight_destination(destination)
    validate_adalin_matched(
        value,
        train_inputs,
        train_labels,
        test_inputs,
        test_labels,
        profile=profile,
    )
    parent = resolved.parent
    descriptor, temporary_name = tempfile.mkstemp(
        dir=parent, prefix=f".{resolved.name}.", suffix=".tmp"
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
        directory = os.open(parent, os.O_RDONLY)
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


def _load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_BYTES:
        raise ValueError("dataset must be a bounded non-symlink NPZ file")
    expected = {
        "train_inputs.npy",
        "train_labels.npy",
        "test_inputs.npy",
        "test_labels.npy",
    }
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if (
                len(members) != 4
                or {member.filename for member in members} != expected
                or sum(member.file_size for member in members) > _MAX_BYTES
                or any(member.is_dir() for member in members)
            ):
                raise ValueError("dataset NPZ members exceed the exact bounded contract")
    except zipfile.BadZipFile as error:
        raise ValueError("dataset must be a valid NPZ archive") from error
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {
            "train_inputs",
            "train_labels",
            "test_inputs",
            "test_labels",
        }:
            raise ValueError("dataset NPZ fields drifted")
        return tuple(
            np.array(archive[name], copy=True)
            for name in ("train_inputs", "train_labels", "test_inputs", "test_labels")
        )  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--profile", choices=tuple(CAMPAIGN_PROFILES), default="bounded-development"
    )
    args = parser.parse_args(argv)
    _preflight_destination(args.output)
    arrays = _load_dataset(args.dataset)
    result = run_adalin_matched(*arrays, profile=args.profile)
    write_adalin_matched(args.output, result, *arrays, profile=args.profile)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
