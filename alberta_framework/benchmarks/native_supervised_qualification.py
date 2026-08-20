"""Held-out qualification for ASI's native supervised CL development suite.

This module is deliberately supplied-array only.  It neither downloads assets
nor treats caller-provided asset claims as verified provenance.  The learner
never receives task identifiers or boundary signals; a separate evaluator uses
boundaries to measure every held-out task before training and after each task.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import math
import operator
import platform
import time
from pathlib import Path
from typing import SupportsIndex, cast
from urllib.parse import urlsplit

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.benchmarks import native_supervised_suite as native

SCHEMA = "asi.native_supervised_cl_qualification.v2"
MAX_TOTAL_DATASET_BYTES = 128 * 1024 * 1024
MAX_EXAMPLES_PER_PARTITION = 100_000
MAX_LOGIT_MULTIPLY_ACCUMULATES_PER_ARM = 500_000_000
MAX_CLAIM_UTF8_BYTES = 512
ASSET_VERIFICATION = "caller_asserted_not_verified"
INPUT_CONTRACT = "finite_exact_float32_values_in_[0,1]_exact_int32_labels.v1"
PARTITION_DISJOINTNESS = "reject_value_identical_only_otherwise_caller_asserted_not_verified"
NUMERIC_PAYLOAD_SCOPE = "peak_retained_canonical_arrays_excludes_allocator_and_transients"
RNG_CONTRACT = "jax_threefry2x32_fold_in_task_and_partition.v2"


def _exact_int(value: object, name: str, low: int, high: int) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an exact integer")
    result = operator.index(cast(SupportsIndex, value))
    if not low <= result <= high:
        raise ValueError(f"{name} must lie in [{low}, {high}]")
    return result


def _digest(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _bounded_string(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty exact string")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must contain valid Unicode") from exc
    if len(encoded) > MAX_CLAIM_UTF8_BYTES:
        raise ValueError(f"{name} exceeds its UTF-8 byte bound")
    return value


def _exact_identity(
    value: object, name: str, expected: tuple[tuple[str, str], ...]
) -> tuple[tuple[str, str], ...]:
    if type(value) is not tuple:
        raise ValueError(f"{name} must be an exact tuple")
    for entry in value:
        if (
            type(entry) is not tuple
            or len(entry) != 2
            or any(type(component) is not str for component in entry)
        ):
            raise ValueError(f"{name} must contain exact string pairs")
    if value != expected:
        raise ValueError(f"current {name.replace('_', ' ')} drift")
    return cast(tuple[tuple[str, str], ...], value)


def _module_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_LOADED_SOURCE_IDENTITY = (
    (
        "alberta_framework/benchmarks/native_supervised_qualification.py",
        _module_sha256(Path(__file__)),
    ),
    (
        "alberta_framework/benchmarks/native_supervised_suite.py",
        _module_sha256(Path(native.__file__)),
    ),
)


def _source_identity() -> tuple[tuple[str, str], ...]:
    current = (
        (
            "alberta_framework/benchmarks/native_supervised_qualification.py",
            _module_sha256(Path(__file__)),
        ),
        (
            "alberta_framework/benchmarks/native_supervised_suite.py",
            _module_sha256(Path(native.__file__)),
        ),
    )
    if current != _LOADED_SOURCE_IDENTITY:
        raise ValueError("source bytes changed after module import")
    return _LOADED_SOURCE_IDENTITY


def _runtime_identity() -> tuple[tuple[str, str], ...]:
    devices = jax.devices()
    if not devices:
        raise ValueError("JAX runtime exposes no devices")
    return (
        ("python", platform.python_version()),
        ("jax", jax.__version__),
        ("jaxlib", importlib.metadata.version("jaxlib")),
        ("numpy", np.__version__),
        ("backend", jax.default_backend()),
        ("device_kind", devices[0].device_kind),
        ("jax_enable_x64", str(bool(jax.config.jax_enable_x64)).lower()),
        ("jax_default_matmul_precision", str(jax.config.jax_default_matmul_precision)),
        ("jax_numpy_dtype_promotion", str(jax.config.jax_numpy_dtype_promotion)),
        ("operating_system", platform.system()),
        ("machine", platform.machine()),
    )


def _protocol_sha256(spec: native.BenchmarkSpec) -> str:
    payload = {
        "schema": SCHEMA,
        "benchmark": dataclasses.asdict(spec),
        "comparison_reference": native.AVALANCHE_REVISION,
        "arms": list(native.ARM_IDS),
        "boundary_contract": dataclasses.asdict(BOUNDARY_CONTRACT),
        "input_contract": INPUT_CONTRACT,
        "partition_disjointness": PARTITION_DISJOINTNESS,
        "numeric_payload_scope": NUMERIC_PAYLOAD_SCOPE,
        "rng_contract": RNG_CONTRACT,
        "matrix_rows": "untrained_then_after_each_task",
        "final_average_accuracy": "mean(final_row_all_tasks)",
        "average_forgetting": "mean(max(post_learning_rows)-final_per_task)",
        "forward_transfer": "mean(pre_task_accuracy-row_zero; tasks_1_through_T_minus_1)",
        "external_parity": False,
        "scientific_promotion_allowed": False,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class DatasetClaims:
    """Caller assertions about the supplied arrays, never runner verification."""

    benchmark_id: str
    authority_uri: str
    asset_manifest_sha256: str
    split_contract: str

    def __post_init__(self) -> None:
        native.benchmark_spec(self.benchmark_id)
        authority = _bounded_string(self.authority_uri, "authority_uri")
        parsed = urlsplit(authority)
        if parsed.scheme not in ("https", "http") or not parsed.netloc:
            raise ValueError("authority_uri must be an absolute HTTP(S) URL")
        if (
            parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("authority_uri must be credential-free and omit queries/fragments")
        _digest(self.asset_manifest_sha256, "asset_manifest_sha256")
        _bounded_string(self.split_contract, "split_contract")


@dataclasses.dataclass(frozen=True, slots=True)
class BoundaryContract:
    learner_receives_task_id: bool = False
    learner_receives_boundary: bool = False
    evaluator_uses_task_boundaries: bool = True
    training_paused_for_evaluation: bool = True
    single_shared_output_head: bool = True

    def __post_init__(self) -> None:
        expected = (False, False, True, True, True)
        observed = tuple(getattr(self, field.name) for field in dataclasses.fields(self))
        if any(type(value) is not bool for value in observed) or observed != expected:
            raise ValueError("the frozen learner/evaluator boundary contract changed")


BOUNDARY_CONTRACT = BoundaryContract()


@dataclasses.dataclass(frozen=True, slots=True)
class DatasetBinding:
    claims: DatasetClaims
    supplied_train_sha256: str
    supplied_test_sha256: str
    asset_verification: str = ASSET_VERIFICATION
    partition_disjointness: str = PARTITION_DISJOINTNESS
    external_parity: bool = False

    def __post_init__(self) -> None:
        if type(self.claims) is not DatasetClaims:
            raise ValueError("dataset claims must be exact DatasetClaims")
        DatasetClaims.__post_init__(self.claims)
        _digest(self.supplied_train_sha256, "supplied_train_sha256")
        _digest(self.supplied_test_sha256, "supplied_test_sha256")
        if (
            type(self.asset_verification) is not str
            or self.asset_verification != ASSET_VERIFICATION
        ):
            raise ValueError("supplied asset bytes were not verified by this runner")
        if (
            type(self.partition_disjointness) is not str
            or self.partition_disjointness != PARTITION_DISJOINTNESS
        ):
            raise ValueError("sample-level partition disjointness was not verified")
        if type(self.external_parity) is not bool or self.external_parity:
            raise ValueError("native supplied-array qualification is not external parity")


@dataclasses.dataclass(frozen=True, slots=True)
class QualificationReceipt:
    training_examples: int
    training_bytes_read: int
    evaluation_examples: int
    evaluation_bytes_read: int
    training_model_queries: int
    evaluation_model_queries: int
    parameter_updates: int
    replay_inserts: int
    replay_samples: int
    logical_compute_units: int
    persistent_bytes: int
    peak_replay_bytes: int
    dataset_bytes_hashed: int
    materialized_schedule_bytes: int
    score_vector_elements: int
    peak_numeric_payload_bytes: int
    elapsed_ns: int
    timing_telemetry_only: bool = True
    numeric_payload_scope: str = NUMERIC_PAYLOAD_SCOPE

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            if field.name in ("timing_telemetry_only", "numeric_payload_scope"):
                continue
            _exact_int(getattr(self, field.name), field.name, 0, 2**63 - 1)
        if (
            self.training_examples == 0
            or self.evaluation_examples == 0
            or self.persistent_bytes == 0
            or self.dataset_bytes_hashed == 0
        ):
            raise ValueError("training, evaluation, persistence, and hashing must be charged")
        if type(self.timing_telemetry_only) is not bool or not self.timing_telemetry_only:
            raise ValueError("timing must remain telemetry-only")
        if (
            type(self.numeric_payload_scope) is not str
            or self.numeric_payload_scope != NUMERIC_PAYLOAD_SCOPE
        ):
            raise ValueError("numeric payload scope differs from the canonical retained arrays")


def _probability(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite exact probability")
    return value


def _signed_probability(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite exact value in [-1, 1]")
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class QualifiedArmResult:
    arm_id: str
    online_accuracy: float
    accuracy_matrix: tuple[tuple[float, ...], ...]
    final_average_accuracy: float
    average_forgetting: float
    forward_transfer: float
    receipt: QualificationReceipt

    def __post_init__(self) -> None:
        if type(self.arm_id) is not str or self.arm_id not in native.ARM_IDS:
            raise ValueError("unknown arm_id")
        _probability(self.online_accuracy, "online_accuracy")
        if type(self.accuracy_matrix) is not tuple or not self.accuracy_matrix:
            raise ValueError("accuracy_matrix must be a non-empty exact tuple")
        width = len(self.accuracy_matrix[0])
        if width == 0 or len(self.accuracy_matrix) != width + 1:
            raise ValueError("accuracy_matrix must have shape (tasks + 1, tasks)")
        for row in self.accuracy_matrix:
            if type(row) is not tuple or len(row) != width:
                raise ValueError("accuracy_matrix rows have inconsistent shape")
            for value in row:
                _probability(value, "accuracy_matrix entry")
        _probability(self.final_average_accuracy, "final_average_accuracy")
        _probability(self.average_forgetting, "average_forgetting")
        _signed_probability(self.forward_transfer, "forward_transfer")
        if type(self.receipt) is not QualificationReceipt:
            raise ValueError("receipt must be an exact QualificationReceipt")
        QualificationReceipt.__post_init__(self.receipt)


@dataclasses.dataclass(frozen=True, slots=True)
class SuppliedArrayQualification:
    schema: str
    benchmark_id: str
    seed: int
    train_examples_per_task: int
    test_examples_per_task: int
    replay_capacity: int
    input_dim: int
    n_classes: int
    comparison_reference: str
    protocol_sha256: str
    dataset_binding: DatasetBinding
    train_schedule_sha256: str
    test_schedule_sha256: str
    boundary_contract: BoundaryContract
    rng_contract: str
    source_identity: tuple[tuple[str, str], ...]
    runtime_identity: tuple[tuple[str, str], ...]
    arms: tuple[QualifiedArmResult, ...]
    development_only: bool = True
    scientific_promotion_allowed: bool = False
    negative_results_must_be_retained: bool = True

    def __post_init__(self) -> None:
        if type(self.schema) is not str or self.schema != SCHEMA:
            raise ValueError("qualification schema mismatch")
        spec = native.benchmark_spec(self.benchmark_id)
        _exact_int(self.seed, "seed", 0, 2**32 - 1)
        if self.seed not in native.FROZEN_SEEDS:
            raise ValueError("seed lies outside the consumed development schedule")
        _exact_int(
            self.train_examples_per_task,
            "train_examples_per_task",
            1,
            native.MAX_EXAMPLES_PER_TASK,
        )
        _exact_int(
            self.test_examples_per_task,
            "test_examples_per_task",
            1,
            native.MAX_EXAMPLES_PER_TASK,
        )
        _exact_int(self.replay_capacity, "replay_capacity", 1, 64)
        _exact_int(self.input_dim, "input_dim", 1, native.MAX_INPUT_DIM)
        _exact_int(self.n_classes, "n_classes", 2, 100)
        if self.n_classes != spec.n_classes:
            raise ValueError("class count differs from the catalog")
        if (
            type(self.comparison_reference) is not str
            or self.comparison_reference != native.AVALANCHE_REVISION
        ):
            raise ValueError("comparison reference differs from the audited catalog pin")
        _digest(self.protocol_sha256, "protocol_sha256")
        if self.protocol_sha256 != _protocol_sha256(spec):
            raise ValueError("protocol identity differs from the frozen qualification")
        if type(self.dataset_binding) is not DatasetBinding:
            raise ValueError("dataset binding must be an exact DatasetBinding")
        DatasetBinding.__post_init__(self.dataset_binding)
        if self.dataset_binding.claims.benchmark_id != self.benchmark_id:
            raise ValueError("dataset claims differ from the benchmark")
        _digest(self.train_schedule_sha256, "train_schedule_sha256")
        _digest(self.test_schedule_sha256, "test_schedule_sha256")
        if type(self.boundary_contract) is not BoundaryContract:
            raise ValueError("boundary contract must be exact")
        BoundaryContract.__post_init__(self.boundary_contract)
        if self.boundary_contract != BOUNDARY_CONTRACT:
            raise ValueError("boundary contract differs from the frozen contract")
        if type(self.rng_contract) is not str or self.rng_contract != RNG_CONTRACT:
            raise ValueError("RNG contract differs from the frozen explicit Threefry schedule")
        _exact_identity(self.source_identity, "source_identity", _source_identity())
        _exact_identity(self.runtime_identity, "runtime_identity", _runtime_identity())
        if type(self.arms) is not tuple or any(
            type(arm) is not QualifiedArmResult for arm in self.arms
        ):
            raise ValueError("arms must contain exact QualifiedArmResult values")
        if tuple(arm.arm_id for arm in self.arms) != native.ARM_IDS:
            raise ValueError("arms differ from the frozen roster")
        expected_flags = (True, False, True)
        flags = (
            self.development_only,
            self.scientific_promotion_allowed,
            self.negative_results_must_be_retained,
        )
        if any(type(value) is not bool for value in flags) or flags != expected_flags:
            raise ValueError("qualification must remain nonpromoting and retain negatives")


def _array_nbytes(value: object, name: str) -> int:
    if type(value) is not np.ndarray:
        raise ValueError(f"{name} must be an exact ndarray")
    nbytes = int(value.nbytes)
    if nbytes <= 0 or nbytes > MAX_TOTAL_DATASET_BYTES:
        raise ValueError(f"{name} byte size exceeds the dataset bound")
    return nbytes


def _snapshot_partition(
    images: object, labels: object, n_classes: int, name: str
) -> tuple[np.ndarray, np.ndarray]:
    _array_nbytes(images, f"{name}_images")
    _array_nbytes(labels, f"{name}_labels")
    data, targets = native._validated_arrays(images, labels, n_classes)  # noqa: SLF001
    return (
        np.array(data, dtype=np.float32, order="C", copy=True),
        np.array(targets, dtype=np.int32, order="C", copy=True),
    )


def _partition_sha256(images: np.ndarray, labels: np.ndarray, split: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"asi-native-supervised-qualified-partition-v2\0")
    digest.update(split.encode("ascii"))
    for array in (images, labels):
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        dtype = "<f4" if array.dtype == np.float32 else "<i4"
        digest.update(array.astype(dtype, copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _schedule_sha256(tasks: tuple[native.TaskBatch, ...], split: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"asi-native-supervised-qualified-schedule-v2\0")
    digest.update(split.encode("ascii"))
    for task in tasks:
        digest.update(task.task_index.to_bytes(4, "little"))
        digest.update(np.asarray(task.inputs.shape, dtype="<i8").tobytes())
        digest.update(task.inputs.astype("<f4", copy=False).tobytes(order="C"))
        digest.update(np.asarray(task.labels.shape, dtype="<i8").tobytes())
        digest.update(task.labels.astype("<i4", copy=False).tobytes(order="C"))
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class _Plan:
    spec: native.BenchmarkSpec
    seed: int
    train_count: int
    test_count: int
    capacity: int
    input_dim: int
    training_examples: int
    evaluation_examples: int
    total_dataset_bytes: int


def _preflight(
    benchmark_id: object,
    train_images: object,
    train_labels: object,
    test_images: object,
    test_labels: object,
    *,
    claims: object,
    seed: object,
    train_examples_per_task: object,
    test_examples_per_task: object,
    replay_capacity: object,
) -> _Plan:
    spec = native.benchmark_spec(benchmark_id)
    if type(claims) is not DatasetClaims:
        raise ValueError("claims must be exact DatasetClaims")
    DatasetClaims.__post_init__(claims)
    if claims.benchmark_id != spec.benchmark_id:
        raise ValueError("claims benchmark differs from requested benchmark")
    sizes = tuple(
        _array_nbytes(value, name)
        for value, name in (
            (train_images, "train_images"),
            (train_labels, "train_labels"),
            (test_images, "test_images"),
            (test_labels, "test_labels"),
        )
    )
    total_bytes = sum(sizes)
    if total_bytes > MAX_TOTAL_DATASET_BYTES:
        raise ValueError("combined dataset byte size exceeds the bound")
    if type(train_images) is not np.ndarray or train_images.dtype != np.float32:
        raise ValueError("train_images must be an exact float32 ndarray")
    if type(test_images) is not np.ndarray or test_images.dtype != np.float32:
        raise ValueError("test_images must be an exact float32 ndarray")
    if train_images.ndim < 2 or test_images.ndim < 2:
        raise ValueError("image arrays must have rank at least two")
    if (
        train_images.shape[0] > MAX_EXAMPLES_PER_PARTITION
        or test_images.shape[0] > MAX_EXAMPLES_PER_PARTITION
    ):
        raise ValueError("partition example count exceeds the qualification bound")
    if train_images.shape[1:] != test_images.shape[1:]:
        raise ValueError("train and test input shapes differ")
    if spec.benchmark_id == "rotated_mnist" and (
        train_images.ndim != 3 or train_images.shape[1] != train_images.shape[2]
    ):
        raise ValueError("rotated MNIST requires square rank-2 examples")
    input_dim = int(np.prod(train_images.shape[1:], dtype=np.int64))
    if not 1 <= input_dim <= native.MAX_INPUT_DIM:
        raise ValueError("flattened input dimension exceeds the bound")
    host_seed = _exact_int(seed, "seed", 0, 2**32 - 1)
    if host_seed not in native.FROZEN_SEEDS:
        raise ValueError("seed lies outside the consumed development schedule")
    train_count = _exact_int(
        train_examples_per_task,
        "train_examples_per_task",
        1,
        native.MAX_EXAMPLES_PER_TASK,
    )
    test_count = _exact_int(
        test_examples_per_task,
        "test_examples_per_task",
        1,
        native.MAX_EXAMPLES_PER_TASK,
    )
    capacity = _exact_int(replay_capacity, "replay_capacity", 1, 64)
    training_examples = spec.n_tasks * train_count
    evaluation_examples = (spec.n_tasks + 1) * spec.n_tasks * test_count
    maximum_training_queries = training_examples + max(0, 2 * training_examples - 1)
    multiply_accumulates = (
        (maximum_training_queries + evaluation_examples) * input_dim * spec.n_classes
    )
    if multiply_accumulates > MAX_LOGIT_MULTIPLY_ACCUMULATES_PER_ARM:
        raise ValueError("planned model work exceeds the bounded qualification budget")
    _, validated_train_labels = native._validated_arrays(  # noqa: SLF001
        train_images, train_labels, spec.n_classes
    )
    _, validated_test_labels = native._validated_arrays(  # noqa: SLF001
        test_images, test_labels, spec.n_classes
    )
    if (
        np.any(train_images < 0.0)
        or np.any(train_images > 1.0)
        or np.any(test_images < 0.0)
        or np.any(test_images > 1.0)
    ):
        raise ValueError("qualification images must use the frozen [0, 1] value range")
    if (
        train_images.shape == test_images.shape
        and np.array_equal(train_images, test_images)
        and np.array_equal(validated_train_labels, validated_test_labels)
    ):
        raise ValueError("train and held-out partitions must not be value-identical")
    for class_index in range(spec.n_classes):
        if not np.any(validated_train_labels == class_index) or not np.any(
            validated_test_labels == class_index
        ):
            raise ValueError("a class has too few train or held-out examples")
    if spec.benchmark_id in ("split_mnist", "split_cifar100"):
        classes_per_task = spec.n_classes // spec.n_tasks
        for task_index in range(spec.n_tasks):
            low = task_index * classes_per_task
            train_available = int(
                np.count_nonzero(
                    (validated_train_labels >= low)
                    & (validated_train_labels < low + classes_per_task)
                )
            )
            test_available = int(
                np.count_nonzero(
                    (validated_test_labels >= low)
                    & (validated_test_labels < low + classes_per_task)
                )
            )
            if train_available < train_count or test_available < test_count:
                raise ValueError("a task has too few train or held-out examples")
    elif (
        validated_train_labels.shape[0] < train_count
        or validated_test_labels.shape[0] < test_count
    ):
        raise ValueError("a partition has too few examples for each transformed task")
    return _Plan(
        spec,
        host_seed,
        train_count,
        test_count,
        capacity,
        input_dim,
        training_examples,
        evaluation_examples,
        total_bytes,
    )


def benchmark_qualification_payload(
    benchmark_id: object,
    train_images: object,
    train_labels: object,
    test_images: object,
    test_labels: object,
    *,
    claims: object,
    seed: object,
    train_examples_per_task: object,
    test_examples_per_task: object,
    replay_capacity: object,
) -> dict[str, object]:
    """Validate supplied arrays and return bounded execution metadata only."""
    plan = _preflight(
        benchmark_id,
        train_images,
        train_labels,
        test_images,
        test_labels,
        claims=claims,
        seed=seed,
        train_examples_per_task=train_examples_per_task,
        test_examples_per_task=test_examples_per_task,
        replay_capacity=replay_capacity,
    )
    return {
        "schema": "asi.native_supervised_cl_qualification_plan.v2",
        "benchmark_id": plan.spec.benchmark_id,
        "seed": plan.seed,
        "matrix_shape": [plan.spec.n_tasks + 1, plan.spec.n_tasks],
        "training_examples_per_arm": plan.training_examples,
        "evaluation_examples_per_arm": plan.evaluation_examples,
        "input_dim": plan.input_dim,
        "comparison_reference": native.AVALANCHE_REVISION,
        "protocol_sha256": _protocol_sha256(plan.spec),
        "dataset_bytes": plan.total_dataset_bytes,
        "boundary_contract": dataclasses.asdict(BOUNDARY_CONTRACT),
        "input_contract": INPUT_CONTRACT,
        "partition_disjointness": PARTITION_DISJOINTNESS,
        "numeric_payload_scope": NUMERIC_PAYLOAD_SCOPE,
        "rng_contract": RNG_CONTRACT,
        "asset_verification": ASSET_VERIFICATION,
        "external_parity": False,
        "ready_to_execute_supplied_array_slice": True,
        "development_only": True,
        "scientific_promotion_allowed": False,
    }


def _predict_linear(x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> int:
    return int(np.argmax(x @ weights + bias))


def _predict_centroid(
    x: np.ndarray, sums: np.ndarray, counts: np.ndarray
) -> int:
    distances = np.sum((sums / np.maximum(counts[:, None], 1) - x) ** 2, axis=1)
    distances = np.where(counts > 0, distances, np.inf)
    return int(np.argmin(distances)) if np.any(counts > 0) else 0


def _matrix_metrics(
    matrix: tuple[tuple[float, ...], ...]
) -> tuple[float, float, float]:
    tasks = len(matrix[0])
    final_average = float(sum(matrix[-1]) / tasks)
    forgetting = float(
        sum(
            max(row[task] for row in matrix[task + 1 :]) - matrix[-1][task]
            for task in range(tasks)
        )
        / tasks
    )
    forward_transfer = float(
        sum(matrix[task][task] - matrix[0][task] for task in range(1, tasks))
        / max(1, tasks - 1)
    )
    return final_average, forgetting, forward_transfer


def _build_partition_tasks(
    spec: native.BenchmarkSpec,
    images: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    count: int,
    partition_discriminator: int,
) -> tuple[native.TaskBatch, ...]:
    """Build one partition with shared task transforms and split-only sampling.

    Sampling keys include the partition discriminator.  Transform keys do not,
    so IPMNIST uses the exact same pixel permutation for train and held-out test
    arrays even when the partitions have different lengths.
    """
    root = jr.key(seed, impl="threefry2x32")
    input_dim = int(np.prod(images.shape[1:], dtype=np.int64))
    tasks: list[native.TaskBatch] = []
    for task_index in range(spec.n_tasks):
        task_key = jr.fold_in(root, task_index)
        # Rotated MNIST applies every transform to one fixed partition slice,
        # matching the benchmark's same-examples-across-rotations semantics.
        sampling_key = root if spec.benchmark_id == "rotated_mnist" else task_key
        sample_key = jr.fold_in(sampling_key, partition_discriminator)
        if spec.benchmark_id in ("split_mnist", "split_cifar100"):
            classes_per_task = spec.n_classes // spec.n_tasks
            low = task_index * classes_per_task
            eligible = np.flatnonzero((labels >= low) & (labels < low + classes_per_task))
        else:
            eligible = np.arange(images.shape[0], dtype=np.int64)
        if eligible.size < count:
            raise ValueError("dataset has too few examples for a frozen task partition")
        ordering = np.asarray(jr.permutation(sample_key, eligible.size), dtype=np.int64)
        indices = eligible[ordering[:count]]
        transformed = images[indices]
        task_labels = labels[indices]
        if spec.benchmark_id == "rotated_mnist":
            transformed = np.stack(
                tuple(
                    native._rotate_nearest(image, native.ROTATIONS[task_index])  # noqa: SLF001
                    for image in transformed
                )
            )
        elif spec.benchmark_id == "ipmnist":
            transform_key = jr.fold_in(jr.fold_in(root, 0x49504D4E), task_index)
            permutation = np.asarray(jr.permutation(transform_key, input_dim), dtype=np.int64)
            transformed = transformed.reshape((count, input_dim))[:, permutation]
        flattened = np.asarray(transformed.reshape((count, input_dim)), dtype=np.float32)
        tasks.append(
            native.TaskBatch(
                task_index=task_index,
                inputs=np.ascontiguousarray(flattened),
                labels=np.asarray(task_labels, dtype=np.int32),
            )
        )
    return tuple(tasks)


def _run_arm(
    train_tasks: tuple[native.TaskBatch, ...],
    test_tasks: tuple[native.TaskBatch, ...],
    n_classes: int,
    capacity: int,
    arm_id: str,
    dataset_bytes: int,
) -> QualifiedArmResult:
    input_dim = train_tasks[0].inputs.shape[1]
    if arm_id == "running_centroid":
        weights = np.empty((0, 0), dtype=np.float32)
        bias = np.empty((0,), dtype=np.float32)
        sums = np.zeros((n_classes, input_dim), dtype=np.float32)
        counts = np.zeros((n_classes,), dtype=np.int32)
    else:
        weights = np.zeros((input_dim, n_classes), dtype=np.float32)
        bias = np.zeros((n_classes,), dtype=np.float32)
        sums = np.empty((0, 0), dtype=np.float32)
        counts = np.empty((0,), dtype=np.int32)
    replay: list[tuple[np.ndarray, int]] = []
    training_queries = evaluation_queries = updates = inserts = samples = correct = 0
    real_seen = 0
    start = time.perf_counter_ns()

    def predict(x: np.ndarray) -> int:
        if arm_id == "running_centroid":
            return _predict_centroid(x, sums, counts)
        return _predict_linear(x, weights, bias)

    def linear_update(x: np.ndarray, label: int) -> None:
        nonlocal weights, bias, training_queries, updates
        training_queries += 1
        next_weights, next_bias = native._sgd_step(  # noqa: SLF001
            jnp.asarray(weights), jnp.asarray(bias), jnp.asarray(x), jnp.asarray(label)
        )
        weights = np.asarray(next_weights, dtype=np.float32)
        bias = np.asarray(next_bias, dtype=np.float32)
        updates += 1

    def evaluate() -> tuple[float, ...]:
        nonlocal evaluation_queries
        row: list[float] = []
        for task in test_tasks:
            task_correct = 0
            for x, raw_label in zip(task.inputs, task.labels, strict=True):
                task_correct += int(predict(x) == int(raw_label))
                evaluation_queries += 1
            row.append(float(task_correct / task.inputs.shape[0]))
        return tuple(row)

    matrix: list[tuple[float, ...]] = [evaluate()]
    for task in train_tasks:
        for x, raw_label in zip(task.inputs, task.labels, strict=True):
            label = int(raw_label)
            correct += int(predict(x) == label)
            training_queries += 1
            if arm_id in ("online_sgd", "replay_sgd"):
                linear_update(x, label)
                if arm_id == "replay_sgd":
                    if replay:
                        replay_x, replay_label = replay[(updates + label) % len(replay)]
                        linear_update(replay_x, replay_label)
                        samples += 1
                    replay.append((x.copy(), label))
                    if len(replay) > capacity:
                        replay.pop(0)
                    inserts += 1
                elif real_seen > 0:
                    linear_update(x, label)
            elif arm_id == "running_centroid":
                sums[label] += x
                counts[label] += 1
                updates += 1
            real_seen += 1
        matrix.append(evaluate())

    elapsed = time.perf_counter_ns() - start
    frozen_matrix = tuple(matrix)
    final_average, forgetting, forward_transfer = _matrix_metrics(frozen_matrix)
    training_examples = sum(task.inputs.shape[0] for task in train_tasks)
    evaluation_examples = len(matrix) * sum(task.inputs.shape[0] for task in test_tasks)
    bytes_per_example = input_dim * 4 + 4
    persistent = n_classes * input_dim * 4 + n_classes * 4
    peak_replay = (
        min(capacity, training_examples) * bytes_per_example if arm_id == "replay_sgd" else 0
    )
    logical = (
        training_examples
        + training_queries
        + evaluation_queries
        + updates
        + inserts
        + samples
    )
    materialized_schedule_bytes = (
        training_examples + sum(task.inputs.shape[0] for task in test_tasks)
    ) * bytes_per_example
    matrix_numeric_bytes = len(matrix) * len(test_tasks) * 8
    peak_numeric_payload_bytes = (
        dataset_bytes
        + materialized_schedule_bytes
        + persistent
        + peak_replay
        + matrix_numeric_bytes
    )
    return QualifiedArmResult(
        arm_id=arm_id,
        online_accuracy=float(correct / training_examples),
        accuracy_matrix=frozen_matrix,
        final_average_accuracy=final_average,
        average_forgetting=forgetting,
        forward_transfer=forward_transfer,
        receipt=QualificationReceipt(
            training_examples=training_examples,
            training_bytes_read=training_examples * bytes_per_example,
            evaluation_examples=evaluation_examples,
            evaluation_bytes_read=evaluation_examples * bytes_per_example,
            training_model_queries=training_queries,
            evaluation_model_queries=evaluation_queries,
            parameter_updates=updates,
            replay_inserts=inserts,
            replay_samples=samples,
            logical_compute_units=logical,
            persistent_bytes=persistent,
            peak_replay_bytes=peak_replay,
            dataset_bytes_hashed=dataset_bytes,
            materialized_schedule_bytes=materialized_schedule_bytes,
            score_vector_elements=(training_queries + evaluation_queries) * n_classes,
            peak_numeric_payload_bytes=peak_numeric_payload_bytes,
            elapsed_ns=elapsed,
        ),
    )


def build_heldout_task_schedules(
    benchmark_id: object,
    train_images: object,
    train_labels: object,
    test_images: object,
    test_labels: object,
    *,
    claims: object,
    seed: object,
    train_examples_per_task: object,
    test_examples_per_task: object,
    replay_capacity: object = 1,
) -> tuple[tuple[native.TaskBatch, ...], tuple[native.TaskBatch, ...]]:
    """Build deterministic train/test tasks without running any learner."""
    plan = _preflight(
        benchmark_id,
        train_images,
        train_labels,
        test_images,
        test_labels,
        claims=claims,
        seed=seed,
        train_examples_per_task=train_examples_per_task,
        test_examples_per_task=test_examples_per_task,
        replay_capacity=replay_capacity,
    )
    train_data, train_targets = _snapshot_partition(
        train_images, train_labels, plan.spec.n_classes, "train"
    )
    test_data, test_targets = _snapshot_partition(
        test_images, test_labels, plan.spec.n_classes, "test"
    )
    return (
        _build_partition_tasks(
            plan.spec,
            train_data,
            train_targets,
            seed=plan.seed,
            count=plan.train_count,
            partition_discriminator=0,
        ),
        _build_partition_tasks(
            plan.spec,
            test_data,
            test_targets,
            seed=plan.seed,
            count=plan.test_count,
            partition_discriminator=1,
        ),
    )


def _validate_structure(value: object) -> SuppliedArrayQualification:
    if type(value) is not SuppliedArrayQualification:
        raise ValueError("qualification must be an exact SuppliedArrayQualification")
    SuppliedArrayQualification.__post_init__(value)
    spec = native.benchmark_spec(value.benchmark_id)
    tasks = spec.n_tasks
    training_examples = tasks * value.train_examples_per_task
    evaluation_examples = (tasks + 1) * tasks * value.test_examples_per_task
    bytes_per_example = value.input_dim * 4 + 4
    for arm in value.arms:
        QualifiedArmResult.__post_init__(arm)
        if len(arm.accuracy_matrix) != tasks + 1 or any(
            len(row) != tasks for row in arm.accuracy_matrix
        ):
            raise ValueError("accuracy matrix differs from the benchmark task count")
        metrics = _matrix_metrics(arm.accuracy_matrix)
        if metrics != (
            arm.final_average_accuracy,
            arm.average_forgetting,
            arm.forward_transfer,
        ):
            raise ValueError("metric summary does not replay from the accuracy matrix")
        receipt = arm.receipt
        expected_updates = 0 if arm.arm_id == "frozen_no_learning" else training_examples
        expected_samples = training_examples - 1 if arm.arm_id == "replay_sgd" else 0
        if arm.arm_id in ("online_sgd", "replay_sgd"):
            expected_updates += training_examples - 1
        expected_training_queries = (
            training_examples + expected_updates
            if arm.arm_id in ("online_sgd", "replay_sgd")
            else training_examples
        )
        expected_inserts = training_examples if arm.arm_id == "replay_sgd" else 0
        expected_peak = (
            min(value.replay_capacity, training_examples) * bytes_per_example
            if expected_inserts
            else 0
        )
        expected_persistent = value.n_classes * value.input_dim * 4 + value.n_classes * 4
        heldout_schedule_examples = tasks * value.test_examples_per_task
        expected_schedule_bytes = (
            training_examples + heldout_schedule_examples
        ) * bytes_per_example
        expected_score_elements = (
            expected_training_queries + evaluation_examples
        ) * value.n_classes
        matrix_numeric_bytes = (tasks + 1) * tasks * 8
        expected_peak_numeric = (
            receipt.dataset_bytes_hashed
            + expected_schedule_bytes
            + expected_persistent
            + expected_peak
            + matrix_numeric_bytes
        )
        expected_logical = (
            training_examples
            + expected_training_queries
            + evaluation_examples
            + expected_updates
            + expected_inserts
            + expected_samples
        )
        observed = (
            receipt.training_examples,
            receipt.training_bytes_read,
            receipt.evaluation_examples,
            receipt.evaluation_bytes_read,
            receipt.training_model_queries,
            receipt.evaluation_model_queries,
            receipt.parameter_updates,
            receipt.replay_inserts,
            receipt.replay_samples,
            receipt.logical_compute_units,
            receipt.persistent_bytes,
            receipt.peak_replay_bytes,
            receipt.materialized_schedule_bytes,
            receipt.score_vector_elements,
            receipt.peak_numeric_payload_bytes,
        )
        expected = (
            training_examples,
            training_examples * bytes_per_example,
            evaluation_examples,
            evaluation_examples * bytes_per_example,
            expected_training_queries,
            evaluation_examples,
            expected_updates,
            expected_inserts,
            expected_samples,
            expected_logical,
            expected_persistent,
            expected_peak,
            expected_schedule_bytes,
            expected_score_elements,
            expected_peak_numeric,
        )
        if observed != expected:
            raise ValueError("resource receipt differs from the frozen execution contract")
    return value


def _execute(
    plan: _Plan,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    claims: DatasetClaims,
    expected: SuppliedArrayQualification | None = None,
) -> SuppliedArrayQualification:
    binding = DatasetBinding(
        claims=claims,
        supplied_train_sha256=_partition_sha256(train_images, train_labels, "train"),
        supplied_test_sha256=_partition_sha256(test_images, test_labels, "test"),
    )
    if expected is not None:
        if binding != expected.dataset_binding:
            raise ValueError("supplied partition identity differs from the qualification")
        if any(
            arm.receipt.dataset_bytes_hashed != plan.total_dataset_bytes
            for arm in expected.arms
        ):
            raise ValueError("dataset-byte receipt differs from the supplied partitions")
    train_tasks = _build_partition_tasks(
        plan.spec,
        train_images,
        train_labels,
        seed=plan.seed,
        count=plan.train_count,
        partition_discriminator=0,
    )
    test_tasks = _build_partition_tasks(
        plan.spec,
        test_images,
        test_labels,
        seed=plan.seed,
        count=plan.test_count,
        partition_discriminator=1,
    )
    train_schedule_sha256 = _schedule_sha256(train_tasks, "train")
    test_schedule_sha256 = _schedule_sha256(test_tasks, "test")
    if expected is not None and (
        train_schedule_sha256 != expected.train_schedule_sha256
        or test_schedule_sha256 != expected.test_schedule_sha256
    ):
        raise ValueError("task schedule differs from the qualification")
    result = SuppliedArrayQualification(
        schema=SCHEMA,
        benchmark_id=plan.spec.benchmark_id,
        seed=plan.seed,
        train_examples_per_task=plan.train_count,
        test_examples_per_task=plan.test_count,
        replay_capacity=plan.capacity,
        input_dim=plan.input_dim,
        n_classes=plan.spec.n_classes,
        comparison_reference=native.AVALANCHE_REVISION,
        protocol_sha256=_protocol_sha256(plan.spec),
        dataset_binding=binding,
        train_schedule_sha256=train_schedule_sha256,
        test_schedule_sha256=test_schedule_sha256,
        boundary_contract=BOUNDARY_CONTRACT,
        rng_contract=RNG_CONTRACT,
        source_identity=_source_identity(),
        runtime_identity=_runtime_identity(),
        arms=tuple(
            _run_arm(
                train_tasks,
                test_tasks,
                plan.spec.n_classes,
                plan.capacity,
                arm_id,
                plan.total_dataset_bytes,
            )
            for arm_id in native.ARM_IDS
        ),
    )
    return _validate_structure(result)


def run_supplied_array_qualification(
    benchmark_id: object,
    train_images: object,
    train_labels: object,
    test_images: object,
    test_labels: object,
    *,
    claims: object,
    seed: object,
    train_examples_per_task: object = 1,
    test_examples_per_task: object = 1,
    replay_capacity: object = 16,
) -> SuppliedArrayQualification:
    """Run one bounded, permanently nonpromoting held-out shard."""
    plan = _preflight(
        benchmark_id,
        train_images,
        train_labels,
        test_images,
        test_labels,
        claims=claims,
        seed=seed,
        train_examples_per_task=train_examples_per_task,
        test_examples_per_task=test_examples_per_task,
        replay_capacity=replay_capacity,
    )
    train_data, train_targets = _snapshot_partition(
        train_images, train_labels, plan.spec.n_classes, "train"
    )
    test_data, test_targets = _snapshot_partition(
        test_images, test_labels, plan.spec.n_classes, "test"
    )
    return _execute(
        plan,
        train_data,
        train_targets,
        test_data,
        test_targets,
        cast(DatasetClaims, claims),
    )


def _without_timing(value: SuppliedArrayQualification) -> SuppliedArrayQualification:
    return dataclasses.replace(
        value,
        arms=tuple(
            dataclasses.replace(
                arm, receipt=dataclasses.replace(arm.receipt, elapsed_ns=0)
            )
            for arm in value.arms
        ),
    )


def validate_supplied_array_qualification(
    value: object,
    train_images: object,
    train_labels: object,
    test_images: object,
    test_labels: object,
    *,
    claims: object,
) -> SuppliedArrayQualification:
    """Replay a result from caller-held arrays and current source/runtime."""
    qualified = _validate_structure(value)
    if type(claims) is not DatasetClaims or claims != qualified.dataset_binding.claims:
        raise ValueError("caller-held claims differ from the qualification")
    plan = _preflight(
        qualified.benchmark_id,
        train_images,
        train_labels,
        test_images,
        test_labels,
        claims=claims,
        seed=qualified.seed,
        train_examples_per_task=qualified.train_examples_per_task,
        test_examples_per_task=qualified.test_examples_per_task,
        replay_capacity=qualified.replay_capacity,
    )
    train_data, train_targets = _snapshot_partition(
        train_images, train_labels, plan.spec.n_classes, "train"
    )
    test_data, test_targets = _snapshot_partition(
        test_images, test_labels, plan.spec.n_classes, "test"
    )
    replay = _execute(
        plan,
        train_data,
        train_targets,
        test_data,
        test_targets,
        claims,
        expected=qualified,
    )
    if _without_timing(qualified) != _without_timing(replay):
        raise ValueError("qualification does not replay from the dataset and current execution")
    return qualified


__all__ = [
    "ASSET_VERIFICATION",
    "BOUNDARY_CONTRACT",
    "INPUT_CONTRACT",
    "MAX_LOGIT_MULTIPLY_ACCUMULATES_PER_ARM",
    "MAX_EXAMPLES_PER_PARTITION",
    "MAX_TOTAL_DATASET_BYTES",
    "NUMERIC_PAYLOAD_SCOPE",
    "PARTITION_DISJOINTNESS",
    "RNG_CONTRACT",
    "SCHEMA",
    "BoundaryContract",
    "DatasetBinding",
    "DatasetClaims",
    "QualificationReceipt",
    "QualifiedArmResult",
    "SuppliedArrayQualification",
    "benchmark_qualification_payload",
    "build_heldout_task_schedules",
    "run_supplied_array_qualification",
    "validate_supplied_array_qualification",
]
