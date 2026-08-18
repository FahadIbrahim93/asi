"""Prospective, permanently nonpromoting matched BiMU development plan."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, cast

import jax
import numpy as np

from alberta_framework.benchmarks.bimu import BiMUConfig, _dataset_sha256

PLAN_SCHEMA: Final = "asi.bimu.matched-development-plan.v2"
MANIFEST_SCHEMA: Final = "asi.bimu.matched-development-execution-manifest.v1"
_MAX_JSON_NODES = 20_000
_MAX_TEXT_BYTES = 4096
_MAX_MANIFEST_BYTES = 2 * 1024 * 1024
_MAX_DATASET_BYTES = 16 * 1024 * 1024
_DIGEST = "85c681c2f5fc5c274870b30c9accb3d2a6e9eb90a4575a2bf1ccca64f58b6227"

INVALID_PRIOR_ATTEMPT: Final[Mapping[str, object]] = MappingProxyType({
    "pull_request": 1686,
    "head_commit": "86a67df39781bba77e1a2c47451f646205daee65",
    "seed": 23,
    "status": "invalid_never_merged",
    "reason": (
        "colliding RNG domains, unpinned PRNG, majority-vote inference, and an immediate-task "
        "metric mislabeled as the paper final-model late-five metric"
    ),
    "result_retained": False,
    "seed_reuse_allowed": False,
    "unmerged_result_sha256": (
        "9b11c3944379323e33ee067cf80a9f4d772a3af4080f9718cec3b6e1d1e91a23",
        "00faf161ead42d11c8daed668ba96a905ef25baf18b0c77b04bc08e4435c4fa7",
        "0f665cbddf209422456d68835f835c8302a632372e59a0e2518297a41c30a5cb",
    ),
    "unmerged_artifact_file_sha256": (
        "0e313a49c5b2e5fb3b7a4c61c6d2618815432dfc24aac30c64b16777ed1328cb",
        "7da4d6e0411546a39d431bf1d3b6c47372c7c634c372f7133f15481851132daa",
        "2d3a05db3ba2b8af50d522ba13564d82999829f340f426f0f4b1b1389607ade0",
    ),
})


def _invalid_prior_attempt_payload() -> dict[str, object]:
    """Return the exact JSON form of the immutable invalid-attempt record."""
    return {
        key: list(value) if type(value) is tuple else value
        for key, value in INVALID_PRIOR_ATTEMPT.items()
    }


def _config(*, memory_window: int | None, input_dim: int = 784, n_classes: int = 10,
            examples: int = 256) -> BiMUConfig:
    return BiMUConfig(
        input_dim=input_dim,
        hidden_units=32,
        n_classes=n_classes,
        n_tasks=5,
        train_examples_per_task=examples,
        test_examples_per_task=examples,
        train_samples=2,
        test_samples=3,
        query_samples=3,
        temperature=1.0,
        likelihood_multiplier=161.3,
        kl_multiplier=3.76,
        alpha_max=0.0023,
        memory_window=memory_window,
        gradient_scale=4.9,
        query_threshold=0.0,
    )


@dataclass(frozen=True)
class BiMUMatchedDevelopmentPlan:
    seeds: tuple[int, ...]
    arm_names: tuple[str, str]
    control_config: BiMUConfig
    candidate_config: BiMUConfig
    dataset_sha256: str
    dataset_selection: str

    def __post_init__(self) -> None:
        if type(self.seeds) is not tuple or len(self.seeds) != 3:
            raise ValueError("seeds must be one exact three-seed tuple")
        if any(type(seed) is not int or not 0 <= seed <= 2**31 - 1 for seed in self.seeds):
            raise ValueError("seeds must be exact signed-int32 nonnegative integers")
        if len(set(self.seeds)) != len(self.seeds) or 23 in self.seeds:
            raise ValueError("seeds must be distinct and must not reuse the invalid attempt")
        if type(self.arm_names) is not tuple or self.arm_names != ("memory_off", "bimu"):
            raise ValueError("arm_names must be the exact mechanism-off/candidate pair")
        if (
            type(self.control_config) is not BiMUConfig
            or type(self.candidate_config) is not BiMUConfig
        ):
            raise ValueError("arm configs must be exact BiMUConfig values")
        control = BiMUConfig(**self.control_config.__dict__)
        candidate = BiMUConfig(**self.candidate_config.__dict__)
        control_payload = control.to_protocol_payload()
        candidate_payload = candidate.to_protocol_payload()
        differences = {
            key for key in control_payload if control_payload[key] != candidate_payload[key]
        }
        if differences != {"memory_window"} or control.memory_window is not None:
            raise ValueError("matched arms may differ only by the BiMU memory window")
        if candidate.memory_window is None:
            raise ValueError("candidate memory window must be enabled")
        if type(self.dataset_sha256) is not str or len(self.dataset_sha256) != 64:
            raise ValueError("dataset_sha256 must be one lowercase SHA-256 digest")
        if any(character not in "0123456789abcdef" for character in self.dataset_sha256):
            raise ValueError("dataset_sha256 must be one lowercase SHA-256 digest")
        if type(self.dataset_selection) is not str or self.dataset_selection != (
            "OpenML mnist_784 v1 canonical train split after [-1,1] scaling; first 256 rows "
            "train and last 256 rows disjoint development test"
        ):
            raise ValueError("dataset selection must remain exact")
        object.__setattr__(self, "control_config", control)
        object.__setattr__(self, "candidate_config", candidate)


FROZEN_BIMU_MATCHED_PLAN: Final = BiMUMatchedDevelopmentPlan(
    seeds=(157001, 157002, 157003),
    arm_names=("memory_off", "bimu"),
    control_config=_config(memory_window=None),
    candidate_config=_config(memory_window=128),
    dataset_sha256=_DIGEST,
    dataset_selection=(
        "OpenML mnist_784 v1 canonical train split after [-1,1] scaling; first 256 rows "
        "train and last 256 rows disjoint development test"
    ),
)


def _test_plan(*, input_dim: int, n_classes: int, examples: int) -> BiMUMatchedDevelopmentPlan:
    data = np.arange(8 * input_dim, dtype=np.float32).reshape(8, input_dim) / 32.0
    labels = np.arange(8, dtype=np.int32) % n_classes
    return BiMUMatchedDevelopmentPlan(
        seeds=(157001, 157002, 157003),
        arm_names=("memory_off", "bimu"),
        control_config=_config(
            memory_window=None, input_dim=input_dim, n_classes=n_classes, examples=examples
        ),
        candidate_config=_config(
            memory_window=128, input_dim=input_dim, n_classes=n_classes, examples=examples
        ),
        dataset_sha256=_dataset_sha256(data[:4], labels[:4], data[4:], labels[4:]),
        dataset_selection=FROZEN_BIMU_MATCHED_PLAN.dataset_selection,
    )


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _json_preflight(value: object) -> None:
    pending = [value]
    nodes = 0
    while pending:
        current = pending.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise ValueError("manifest exceeds exact JSON node ceiling")
        if type(current) is dict:
            mapping = cast(dict[object, object], current)
            if len(mapping) > _MAX_JSON_NODES:
                raise ValueError("manifest exceeds exact JSON node ceiling")
            for key in mapping.keys():
                if type(key) is not str or len(key.encode("utf-8")) > _MAX_TEXT_BYTES:
                    raise ValueError("manifest must be an exact JSON tree")
            pending.extend(mapping.values())
        elif type(current) is list:
            if len(cast(list[object], current)) > _MAX_JSON_NODES:
                raise ValueError("manifest exceeds exact JSON node ceiling")
            pending.extend(cast(list[object], current))
        elif type(current) is str:
            if len(current.encode("utf-8")) > _MAX_TEXT_BYTES:
                raise ValueError("manifest text exceeds ceiling")
        elif type(current) is int:
            if not -(2**63) <= current <= 2**63 - 1:
                raise ValueError("manifest integer exceeds signed-int64")
        elif type(current) is float:
            if not math.isfinite(current):
                raise ValueError("manifest float must be finite")
        elif type(current) is not bool and type(current) is not type(None):
            raise ValueError("manifest must be an exact JSON tree")


def _fields(value: object, expected: tuple[str, ...], name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{name} must be an exact object")
    mapping = cast(dict[str, object], value)
    if len(mapping) != len(expected) or set(mapping) != set(expected):
        raise ValueError(f"{name} fields drifted")
    return mapping


def _plan_payload(plan: BiMUMatchedDevelopmentPlan) -> dict[str, object]:
    checked = BiMUMatchedDevelopmentPlan(**plan.__dict__)
    config = checked.candidate_config
    observations = config.n_tasks * config.train_examples_per_task
    label_queries = observations
    model_forward_queries = (
        observations * config.query_samples
        + label_queries * config.train_samples
        + 5 * config.test_examples_per_task * config.test_samples
    )
    persistent_bytes = config.trainable_scalar_count * np.dtype(np.float32).itemsize + (
        2 * np.dtype(np.uint32).itemsize
    )
    dataset_bytes = (
        config.train_examples_per_task * (config.input_dim + 1) * 4
        + config.test_examples_per_task * (config.input_dim + 1) * 4
    )
    return {
        "schema": PLAN_SCHEMA,
        "seeds": list(checked.seeds),
        "arm_names": list(checked.arm_names),
        "control_config": checked.control_config.to_protocol_payload(),
        "candidate_config": checked.candidate_config.to_protocol_payload(),
        "dataset_sha256": checked.dataset_sha256,
        "dataset_selection": checked.dataset_selection,
        "prior_invalid_attempts": [_invalid_prior_attempt_payload()],
        "matched_axes": [
            "seed", "dataset", "schedule", "observations", "label_queries",
            "optimizer_seen", "model_forward_queries", "initial_state",
        ],
        "expected_counters_per_arm": {
            "environment_steps": observations,
            "observations": observations,
            "label_queries": label_queries,
            "optimizer_seen": observations,
            "model_forward_queries": model_forward_queries,
            "optimizer_updates_rule": "reported_nonzero_gradient_subcount_at_most_label_queries",
        },
        "expected_resources_per_arm": {
            "trainable_scalar_count": config.trainable_scalar_count,
            "parameter_numeric_bytes": config.trainable_scalar_count * 4,
            "optimizer_state_numeric_bytes": 8,
            "initial_persistent_numeric_bytes": persistent_bytes,
            "final_persistent_numeric_bytes": persistent_bytes,
            "dataset_numeric_bytes": dataset_bytes,
            "timing_qualified": False,
            "aggregate_working_set_bytes_claimed": False,
            "numeric_resource_ceiling_bytes": 256 * 1024 * 1024,
        },
        "comparison_scope": {
            "paper_comparable": False,
            "development_slice": "five tasks, 256 train and 256 test examples, width 32",
            "official_configuration_also_represented_by_runner": True,
            "claim": "bounded mechanism-on versus exact memory-off development comparison",
        },
        "primary_metric": "paper_late_five_test_accuracy",
        "secondary_metric": "asi_whole_stream_online_accuracy",
    }


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _source_identity() -> dict[str, str]:
    root = _repository_root()
    paths = (
        Path("alberta_framework/benchmarks/bimu.py"),
        Path("alberta_framework/benchmarks/upgd_ipmnist.py"),
        Path("alberta_framework/evaluation/bimu_matched_nonpromoting.py"),
        Path("uv.lock"),
    )
    return {str(path): hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths}


def _runtime_identity() -> dict[str, object]:
    devices = [
        {
            "platform": device.platform,
            "device_kind": device.device_kind,
            "id": device.id,
            "process_index": device.process_index,
        }
        for device in jax.devices()
    ]
    environment_names = (
        "JAX_DEFAULT_MATMUL_PRECISION", "JAX_DEFAULT_PRNG_IMPL", "JAX_ENABLE_X64",
        "JAX_NUM_CPU_DEVICES", "JAX_PLATFORMS", "JAX_PLATFORM_NAME",
        "JAX_RANDOM_SEED_OFFSET", "XLA_FLAGS",
    )
    return {
        "schema": "asi.bimu.matched-runtime.v1",
        "python": list(sys.version_info[:3]),
        "python_implementation": platform.python_implementation(),
        "byteorder": sys.byteorder,
        "platform": sys.platform,
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("chex", "jax", "jaxlib", "numpy", "scikit-learn")
        },
        "backend": jax.default_backend(),
        "devices": devices,
        "jax_config": {
            "jax_default_matmul_precision": jax.config.jax_default_matmul_precision,
            "jax_default_prng_impl": jax.config.jax_default_prng_impl,
            "jax_disable_jit": jax.config.jax_disable_jit,
            "jax_enable_x64": jax.config.jax_enable_x64,
            "jax_numpy_dtype_promotion": jax.config.jax_numpy_dtype_promotion.value,
            "jax_numpy_rank_promotion": jax.config.jax_numpy_rank_promotion,
            "jax_random_seed_offset": jax.config.jax_random_seed_offset,
            "jax_threefry_partitionable": jax.config.jax_threefry_partitionable,
        },
        "environment": {name: os.environ.get(name) for name in environment_names},
    }


def _validated_dataset_arrays(
    train_x: object, train_y: object, test_x: object, test_y: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    plan = FROZEN_BIMU_MATCHED_PLAN
    config = plan.candidate_config
    expected = (
        (
            "train_x",
            train_x,
            np.dtype(np.float32),
            (config.train_examples_per_task, config.input_dim),
        ),
        ("train_y", train_y, np.dtype(np.int32), (config.train_examples_per_task,)),
        ("test_x", test_x, np.dtype(np.float32), (config.test_examples_per_task, config.input_dim)),
        ("test_y", test_y, np.dtype(np.int32), (config.test_examples_per_task,)),
    )
    arrays: list[np.ndarray] = []
    total_bytes = 0
    for name, value, dtype, shape in expected:
        if type(value) is not np.ndarray or value.dtype != dtype or value.shape != shape:
            raise ValueError(f"{name} does not match the frozen exact shape/dtype")
        total_bytes += value.size * value.dtype.itemsize
        if total_bytes > _MAX_DATASET_BYTES:
            raise ValueError("dataset exceeds the frozen byte ceiling")
        arrays.append(value)
    for name, features in (("train_x", arrays[0]), ("test_x", arrays[2])):
        if not np.all(np.isfinite(features)):
            raise ValueError(f"{name} must contain only finite values")
    for name, labels in (("train_y", arrays[1]), ("test_y", arrays[3])):
        if np.any(labels < 0) or np.any(labels >= config.n_classes):
            raise ValueError(f"{name} contains an out-of-range label")
    return arrays[0], arrays[1], arrays[2], arrays[3]


def build_bimu_execution_manifest(
    train_x: object, train_y: object, test_x: object, test_y: object
) -> dict[str, object]:
    arrays = _validated_dataset_arrays(train_x, train_y, test_x, test_y)
    plan = FROZEN_BIMU_MATCHED_PLAN
    digest = _dataset_sha256(
        *arrays,
    )
    if digest != plan.dataset_sha256:
        raise ValueError("dataset does not match the frozen plan")
    plan_payload = _plan_payload(plan)
    manifest: dict[str, object] = {
        "schema": MANIFEST_SCHEMA,
        "plan": plan_payload,
        "identity": {
            "source_sha256": _source_identity(),
            "runtime": _runtime_identity(),
            "plan_sha256": hashlib.sha256(_canonical(plan_payload)).hexdigest(),
            "consistency_not_attestation": True,
        },
        "policy": {
            "development_only": True,
            "scientific_promotion_allowed": False,
            "execution_authorized": False,
            "output_retained": False,
        },
    }
    manifest["manifest_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
    validate_bimu_execution_manifest(manifest, train_x, train_y, test_x, test_y)
    return manifest


def validate_bimu_execution_manifest(
    value: object, train_x: object, train_y: object, test_x: object, test_y: object
) -> None:
    _json_preflight(value)
    root = _fields(value, ("schema", "plan", "identity", "policy", "manifest_sha256"), "manifest")
    if root["schema"] != MANIFEST_SCHEMA:
        raise ValueError("manifest schema drifted")
    expected_plan = _plan_payload(FROZEN_BIMU_MATCHED_PLAN)
    if root["plan"] != expected_plan:
        raise ValueError("plan does not match the prospective frozen plan")
    arrays = _validated_dataset_arrays(train_x, train_y, test_x, test_y)
    digest = _dataset_sha256(*arrays)
    if digest != FROZEN_BIMU_MATCHED_PLAN.dataset_sha256:
        raise ValueError("dataset does not match the frozen plan")
    identity = _fields(
        root["identity"],
        ("source_sha256", "runtime", "plan_sha256", "consistency_not_attestation"),
        "identity",
    )
    expected_identity = {
        "source_sha256": _source_identity(),
        "runtime": _runtime_identity(),
        "plan_sha256": hashlib.sha256(_canonical(expected_plan)).hexdigest(),
        "consistency_not_attestation": True,
    }
    if identity != expected_identity:
        raise ValueError("execution identity drifted")
    if root["policy"] != {
        "development_only": True,
        "scientific_promotion_allowed": False,
        "execution_authorized": False,
        "output_retained": False,
    }:
        raise ValueError("policy drifted")
    unsigned = dict(root)
    claimed = unsigned.pop("manifest_sha256")
    if type(claimed) is not str or claimed != hashlib.sha256(_canonical(unsigned)).hexdigest():
        raise ValueError("manifest digest drifted")
    if len(_canonical(root)) > _MAX_MANIFEST_BYTES:
        raise ValueError("manifest exceeds byte ceiling")
