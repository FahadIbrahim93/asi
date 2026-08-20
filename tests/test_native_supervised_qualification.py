from __future__ import annotations

import dataclasses

import jax
import numpy as np
import pytest

from alberta_framework.benchmarks import native_supervised_qualification as qualification
from alberta_framework.benchmarks.native_supervised_qualification import (
    BOUNDARY_CONTRACT,
    INPUT_CONTRACT,
    MAX_TOTAL_DATASET_BYTES,
    NUMERIC_PAYLOAD_SCOPE,
    PARTITION_DISJOINTNESS,
    DatasetClaims,
    benchmark_qualification_payload,
    build_heldout_task_schedules,
    run_supplied_array_qualification,
    validate_supplied_array_qualification,
)

pytestmark = pytest.mark.integration


def _fixture(
    n_classes: int = 10, shape: tuple[int, ...] = (4, 4)
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_labels = np.repeat(np.arange(n_classes, dtype=np.int32), 3)
    test_labels = np.repeat(np.arange(n_classes, dtype=np.int32), 2)
    width = int(np.prod(shape))
    train = np.arange(train_labels.size * width, dtype=np.float32).reshape(
        (train_labels.size, *shape)
    )
    test = np.arange(test_labels.size * width, dtype=np.float32).reshape(
        (test_labels.size, *shape)
    )
    return (train % 19) / 19.0, train_labels, (test % 23) / 23.0, test_labels


def _claims(benchmark_id: str = "split_mnist") -> DatasetClaims:
    return DatasetClaims(
        benchmark_id=benchmark_id,
        authority_uri="https://example.invalid/caller-held-test-fixture",
        asset_manifest_sha256="a" * 64,
        split_contract="caller asserts canonical train/test split; fixture only",
    )


def _run() -> tuple[object, tuple[np.ndarray, ...], DatasetClaims]:
    arrays = _fixture()
    claims = _claims()
    result = run_supplied_array_qualification(
        "split_mnist",
        *arrays,
        claims=claims,
        seed=15780,
        train_examples_per_task=2,
        test_examples_per_task=1,
        replay_capacity=3,
    )
    return result, arrays, claims


def test_full_held_out_matrix_metrics_and_boundary_contract() -> None:
    result, arrays, claims = _run()
    assert result.boundary_contract == BOUNDARY_CONTRACT
    assert not result.boundary_contract.learner_receives_task_id
    assert not result.boundary_contract.learner_receives_boundary
    assert result.boundary_contract.evaluator_uses_task_boundaries
    assert result.boundary_contract.training_paused_for_evaluation
    assert result.dataset_binding.asset_verification == "caller_asserted_not_verified"
    assert result.dataset_binding.partition_disjointness == PARTITION_DISJOINTNESS
    assert not result.dataset_binding.external_parity
    for arm in result.arms:
        assert len(arm.accuracy_matrix) == 6
        assert all(len(row) == 5 for row in arm.accuracy_matrix)
        assert arm.final_average_accuracy == pytest.approx(
            sum(arm.accuracy_matrix[-1]) / 5
        )
        expected_forgetting = sum(
            max(row[task] for row in arm.accuracy_matrix[task + 1 :])
            - arm.accuracy_matrix[-1][task]
            for task in range(5)
        ) / 5
        assert arm.average_forgetting == pytest.approx(expected_forgetting)
        expected_fwt = sum(
            arm.accuracy_matrix[task][task] - arm.accuracy_matrix[0][task]
            for task in range(1, 5)
        ) / 4
        assert arm.forward_transfer == pytest.approx(expected_fwt)
    assert validate_supplied_array_qualification(result, *arrays, claims=claims) == result


def test_receipts_separate_training_and_held_out_evaluation() -> None:
    result, _, _ = _run()
    for arm in result.arms:
        receipt = arm.receipt
        assert receipt.training_examples == 10
        assert receipt.evaluation_examples == 30
        assert receipt.evaluation_model_queries == 30
        assert receipt.dataset_bytes_hashed > 0
        assert receipt.materialized_schedule_bytes > 0
        assert receipt.score_vector_elements == (
            receipt.training_model_queries + receipt.evaluation_model_queries
        ) * result.n_classes
        assert receipt.peak_numeric_payload_bytes > receipt.persistent_bytes
        assert receipt.numeric_payload_scope == NUMERIC_PAYLOAD_SCOPE
        assert receipt.persistent_bytes > 0
        assert receipt.timing_telemetry_only
    online, replay, centroid, frozen = result.arms
    assert online.receipt.parameter_updates == replay.receipt.parameter_updates == 19
    assert replay.receipt.replay_inserts == 10
    assert replay.receipt.replay_samples == 9
    assert centroid.receipt.parameter_updates == 10
    assert frozen.receipt.parameter_updates == 0


def test_strict_validator_reruns_and_rejects_metric_resource_and_data_forgery() -> None:
    result, arrays, claims = _run()
    arm = result.arms[0]
    forged_metric = dataclasses.replace(
        result,
        arms=(dataclasses.replace(arm, final_average_accuracy=0.125), *result.arms[1:]),
    )
    with pytest.raises(ValueError, match="metric|replay"):
        validate_supplied_array_qualification(forged_metric, *arrays, claims=claims)
    forged_receipt = dataclasses.replace(arm.receipt, evaluation_examples=29)
    forged_resource = dataclasses.replace(
        result,
        arms=(dataclasses.replace(arm, receipt=forged_receipt), *result.arms[1:]),
    )
    with pytest.raises(ValueError, match="resource|evaluation"):
        validate_supplied_array_qualification(forged_resource, *arrays, claims=claims)
    changed = arrays[0].copy()
    changed[0, 0, 0] += np.float32(0.25)
    with pytest.raises(ValueError, match="dataset|partition|replay"):
        validate_supplied_array_qualification(
            result, changed, *arrays[1:], claims=claims
        )


def test_validator_rejects_wrong_array_identity_before_running_an_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, arrays, claims = _run()
    changed = arrays[0].copy()
    changed[0, 0, 0] += np.float32(0.25)

    def unexpected_arm(*args: object, **kwargs: object) -> object:
        raise AssertionError("array identity must fail before learner execution")

    monkeypatch.setattr(qualification, "_run_arm", unexpected_arm)
    with pytest.raises(ValueError, match="dataset|partition"):
        validate_supplied_array_qualification(
            result, changed, *arrays[1:], claims=claims
        )
    with pytest.raises(ValueError, match="schedule"):
        validate_supplied_array_qualification(
            dataclasses.replace(result, train_schedule_sha256="0" * 64),
            *arrays,
            claims=claims,
        )
    first_arm = result.arms[0]
    forged_receipt = dataclasses.replace(
        first_arm.receipt,
        dataset_bytes_hashed=first_arm.receipt.dataset_bytes_hashed + 1,
        peak_numeric_payload_bytes=first_arm.receipt.peak_numeric_payload_bytes + 1,
    )
    with pytest.raises(ValueError, match="dataset-byte"):
        validate_supplied_array_qualification(
            dataclasses.replace(
                result,
                arms=(
                    dataclasses.replace(first_arm, receipt=forged_receipt),
                    *result.arms[1:],
                ),
            ),
            *arrays,
            claims=claims,
        )


def test_claims_and_current_source_runtime_are_load_bearing() -> None:
    result, arrays, claims = _run()
    with pytest.raises(ValueError, match="claims"):
        validate_supplied_array_qualification(
            result, *arrays, claims=dataclasses.replace(claims, asset_manifest_sha256="b" * 64)
        )
    with pytest.raises(ValueError, match="source"):
        validate_supplied_array_qualification(
            dataclasses.replace(
                result,
                source_identity=(
                    (result.source_identity[0][0], "0" * 64),
                    *result.source_identity[1:],
                ),
            ),
            *arrays,
            claims=claims,
        )
    with jax.default_matmul_precision("highest"):
        if dict(result.runtime_identity)["jax_default_matmul_precision"] != "highest":
            with pytest.raises(ValueError, match="runtime"):
                validate_supplied_array_qualification(
                    result,
                    *arrays,
                    claims=claims,
                )
    with pytest.raises(ValueError, match="runtime"):
        validate_supplied_array_qualification(
            dataclasses.replace(
                result,
                runtime_identity=(
                    (result.runtime_identity[0][0], "forged"),
                    *result.runtime_identity[1:],
                ),
            ),
            *arrays,
            claims=claims,
        )


def test_execution_rejects_source_bytes_changed_after_module_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrays = _fixture()
    monkeypatch.setattr(qualification, "_module_sha256", lambda _: "f" * 64)
    with pytest.raises(ValueError, match="source.*import|source.*changed"):
        run_supplied_array_qualification(
            "split_mnist",
            *arrays,
            claims=_claims(),
            seed=15780,
            train_examples_per_task=1,
            test_examples_per_task=1,
            replay_capacity=1,
        )


@pytest.mark.parametrize("benchmark_id", ("rotated_mnist", "split_cifar100", "ipmnist"))
def test_all_other_catalog_lanes_build_bounded_held_out_schedules(
    benchmark_id: str,
) -> None:
    n_classes = 100 if benchmark_id == "split_cifar100" else 10
    shape = (2, 2, 3) if n_classes == 100 else (4, 4)
    arrays = _fixture(n_classes, shape)
    # Metadata planning is cheap and proves that every catalog lane has an
    # exact matrix/resource contract without executing a costly full shard.
    payload = benchmark_qualification_payload(
        benchmark_id,
        *arrays,
        claims=_claims(benchmark_id),
        seed=15780,
        train_examples_per_task=1,
        test_examples_per_task=1,
        replay_capacity=1,
    )
    tasks = 20 if benchmark_id == "split_cifar100" else (200 if benchmark_id == "ipmnist" else 5)
    assert payload["matrix_shape"] == [tasks + 1, tasks]
    assert payload["ready_to_execute_supplied_array_slice"] is True
    assert payload["external_parity"] is False
    assert payload["input_contract"] == INPUT_CONTRACT
    assert payload["partition_disjointness"] == PARTITION_DISJOINTNESS
    assert payload["numeric_payload_scope"] == NUMERIC_PAYLOAD_SCOPE


@pytest.mark.parametrize("benchmark_id", ("rotated_mnist", "split_cifar100", "ipmnist"))
def test_all_other_catalog_lanes_execute_the_full_matrix(benchmark_id: str) -> None:
    n_classes = 100 if benchmark_id == "split_cifar100" else 10
    shape = (2, 2, 3) if n_classes == 100 else (4, 4)
    arrays = _fixture(n_classes, shape)
    result = run_supplied_array_qualification(
        benchmark_id,
        *arrays,
        claims=_claims(benchmark_id),
        seed=15780,
        train_examples_per_task=1,
        test_examples_per_task=1,
        replay_capacity=1,
    )
    tasks = 20 if benchmark_id == "split_cifar100" else (200 if benchmark_id == "ipmnist" else 5)
    assert all(len(arm.accuracy_matrix) == tasks + 1 for arm in result.arms)
    assert all(arm.receipt.evaluation_examples == (tasks + 1) * tasks for arm in result.arms)


def test_preflight_rejects_unsafe_work_before_building_tasks() -> None:
    arrays = _fixture()
    huge = np.broadcast_to(arrays[0][:1], (MAX_TOTAL_DATASET_BYTES, 4, 4))
    with pytest.raises(ValueError, match="byte|bound"):
        benchmark_qualification_payload(
            "split_mnist", huge, arrays[1], arrays[2], arrays[3],
            claims=_claims(), seed=15780, train_examples_per_task=1,
            test_examples_per_task=1, replay_capacity=1,
        )


def test_exact_types_nonfinite_labels_and_absent_assets_fail_closed() -> None:
    train, train_labels, test, test_labels = _fixture()
    with pytest.raises(ValueError):
        benchmark_qualification_payload(
            "split_mnist", train.astype(np.float64), train_labels, test, test_labels,
            claims=_claims(), seed=15780, train_examples_per_task=1,
            test_examples_per_task=1, replay_capacity=1,
        )
    with pytest.raises(ValueError):
        benchmark_qualification_payload(
            "split_mnist", train, train_labels, test, test_labels,
            claims=_claims(), seed=True, train_examples_per_task=1,
            test_examples_per_task=1, replay_capacity=1,
        )
    with pytest.raises(ValueError, match="claims"):
        benchmark_qualification_payload(
            "split_mnist", train, train_labels, test, test_labels,
            claims=None, seed=15780, train_examples_per_task=1,
            test_examples_per_task=1, replay_capacity=1,
        )
    missing_class = train_labels.copy()
    missing_class[missing_class == 9] = 8
    with pytest.raises(ValueError, match="too few"):
        benchmark_qualification_payload(
            "split_mnist", train, missing_class, test, test_labels,
            claims=_claims(), seed=15780, train_examples_per_task=1,
            test_examples_per_task=1, replay_capacity=1,
        )
    out_of_range = train.copy()
    out_of_range[0, 0, 0] = np.finfo(np.float32).max
    with pytest.raises(ValueError, match=r"range|\[0, 1\]"):
        benchmark_qualification_payload(
            "split_mnist", out_of_range, train_labels, test, test_labels,
            claims=_claims(), seed=15780, train_examples_per_task=1,
            test_examples_per_task=1, replay_capacity=1,
        )


def test_result_identity_fields_require_exact_scalar_and_tuple_types() -> None:
    result, _, _ = _run()
    string_subclass = type("StringSubclass", (str,), {})
    for changes in (
        {"schema": string_subclass(result.schema)},
        {"comparison_reference": string_subclass(result.comparison_reference)},
        {"rng_contract": string_subclass(result.rng_contract)},
        {"seed": np.int64(result.seed)},
        {"n_classes": np.int64(result.n_classes)},
        {"source_identity": list(result.source_identity)},
        {"runtime_identity": list(result.runtime_identity)},
    ):
        with pytest.raises(ValueError):
            validate_supplied_array_qualification(
                dataclasses.replace(result, **changes),
                *_fixture(),
                claims=_claims(),
            )


def test_value_identical_train_and_evaluation_partitions_fail_closed() -> None:
    labels = np.arange(10, dtype=np.int32)
    images = np.arange(160, dtype=np.float32).reshape(10, 4, 4) / np.float32(159.0)
    with pytest.raises(ValueError, match="distinct|identical|held-out"):
        benchmark_qualification_payload(
            "ipmnist",
            images,
            labels,
            images.copy(),
            labels.copy(),
            claims=_claims("ipmnist"),
            seed=15780,
            train_examples_per_task=1,
            test_examples_per_task=1,
            replay_capacity=1,
        )


def test_ipmnist_train_and_test_share_transform_and_ignore_ambient_prng() -> None:
    labels = np.arange(10, dtype=np.int32)
    row = np.arange(16, dtype=np.float32).reshape(1, 4, 4) / np.float32(15.0)
    images = np.repeat(row, 10, axis=0)
    arguments = (
        "ipmnist",
        images,
        labels,
        np.concatenate((images, images[:1]), axis=0),
        np.concatenate((labels, labels[:1]), axis=0),
    )
    keywords = {
        "claims": _claims("ipmnist"),
        "seed": 15780,
        "train_examples_per_task": 1,
        "test_examples_per_task": 1,
        "replay_capacity": 1,
    }
    baseline = build_heldout_task_schedules(*arguments, **keywords)
    with jax.default_prng_impl("rbg"):
        ambient_rbg = build_heldout_task_schedules(*arguments, **keywords)
    for baseline_partition, rbg_partition in zip(baseline, ambient_rbg, strict=True):
        for expected, observed in zip(baseline_partition, rbg_partition, strict=True):
            np.testing.assert_array_equal(expected.inputs, observed.inputs)
    train_tasks, test_tasks = baseline
    for train_task, test_task in zip(train_tasks, test_tasks, strict=True):
        np.testing.assert_array_equal(train_task.inputs, test_task.inputs)


def test_rotated_mnist_uses_one_fixed_example_slice_across_tasks() -> None:
    labels = np.arange(10, dtype=np.int32)
    images = (
        np.broadcast_to(labels[:, None, None], (10, 4, 4)).astype(np.float32).copy()
        / np.float32(9.0)
    )
    arrays = (
        images,
        labels,
        np.concatenate((images, images[:1]), axis=0),
        np.concatenate((labels, labels[:1]), axis=0),
    )
    train_tasks, test_tasks = build_heldout_task_schedules(
        "rotated_mnist",
        *arrays,
        claims=_claims("rotated_mnist"),
        seed=15780,
        train_examples_per_task=2,
        test_examples_per_task=2,
    )
    for partition in (train_tasks, test_tasks):
        for task in partition[1:]:
            np.testing.assert_array_equal(task.labels, partition[0].labels)
        np.testing.assert_array_equal(partition[4].inputs, partition[0].inputs)
