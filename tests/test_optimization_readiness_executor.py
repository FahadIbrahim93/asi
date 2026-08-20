from __future__ import annotations

import tracemalloc
from copy import deepcopy

import numpy as np
import pytest

import alberta_framework.evaluation.optimization_readiness_executor as executor_module
from alberta_framework.evaluation.optimization_readiness_executor import (
    FROZEN_EXECUTION_PLAN,
    execute_optimization_readiness,
    validate_optimization_readiness_execution,
)


def _task() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, 10_000, dtype=np.float64)
    inputs = np.column_stack((axis, np.square(axis)))
    labels = 0.75 * axis - 0.25 * np.square(axis)
    checkpoint = np.asarray([0.1, -0.1], dtype=np.float64)
    return inputs, labels, checkpoint


def test_executor_derives_and_validator_recomputes_real_measurements() -> None:
    inputs, labels, checkpoint = _task()
    artifact = execute_optimization_readiness(
        validation_inputs=inputs,
        validation_labels=labels,
        checkpoint_parameters=checkpoint,
        seed=7,
        task_id="bounded-linear-regression-v1",
        checkpoint_id="checkpoint-0007",
    )

    assert artifact["schema"] == "asi.optimization-readiness.execution.v1"
    assert artifact["policy"] == {
        "development_only": True,
        "scientific_promotion_allowed": False,
        "timing_is_telemetry_only": True,
    }
    plan = artifact["plan"]
    assert isinstance(plan, dict)
    assert plan == {**FROZEN_EXECUTION_PLAN, "future_gain_steps": [1, 10, 100]}
    metrics = artifact["metrics"]
    assert isinstance(metrics, dict)
    assert set(metrics["future_relative_loss_reduction"]) == {"1", "10", "100"}
    assert metrics["optimization_readiness"] > 0.0
    assert metrics["gradient_norm"] > 0.0
    assert metrics["parameter_norm"] > 0.0
    assert 0 <= metrics["representation_energy_rank_0_99"] <= 2
    assert 0 <= metrics["curvature_energy_rank_0_99"] <= 2
    sampling = artifact["sampling"]
    assert isinstance(sampling, dict)
    assert sampling["diagnostic_gradient_count"] == 128
    assert validate_optimization_readiness_execution(
        artifact,
        validation_inputs=inputs,
        validation_labels=labels,
        checkpoint_parameters=checkpoint,
    ) == artifact


def test_validator_rejects_forged_metric_by_reexecution() -> None:
    inputs, labels, checkpoint = _task()
    artifact = execute_optimization_readiness(
        validation_inputs=inputs,
        validation_labels=labels,
        checkpoint_parameters=checkpoint,
        seed=11,
        task_id="task",
        checkpoint_id="checkpoint",
    )
    forged = deepcopy(artifact)
    metrics = forged["metrics"]
    assert isinstance(metrics, dict)
    metrics["gradient_norm"] = float(metrics["gradient_norm"]) + 1.0
    with pytest.raises(ValueError, match="recompute exactly"):
        validate_optimization_readiness_execution(
            forged,
            validation_inputs=inputs,
            validation_labels=labels,
            checkpoint_parameters=checkpoint,
        )


@pytest.mark.parametrize("section", ["sampling", "resources", "identity"])
def test_validator_rejects_forged_execution_provenance(section: str) -> None:
    inputs, labels, checkpoint = _task()
    artifact = execute_optimization_readiness(
        validation_inputs=inputs,
        validation_labels=labels,
        checkpoint_parameters=checkpoint,
        seed=17,
        task_id="task",
        checkpoint_id="checkpoint",
    )
    forged = deepcopy(artifact)
    nested = forged[section]
    assert isinstance(nested, dict)
    if section == "sampling":
        nested["schedule_sha256"] = "0" * 64
    elif section == "resources":
        nested["model_queries"] = int(nested["model_queries"]) - 1
    else:
        runtime = nested["runtime"]
        assert isinstance(runtime, dict)
        runtime["numpy"] = "forged"
    with pytest.raises(ValueError, match="recompute exactly"):
        validate_optimization_readiness_execution(
            forged,
            validation_inputs=inputs,
            validation_labels=labels,
            checkpoint_parameters=checkpoint,
        )


def test_validator_rejects_hostile_nested_runtime_types_without_hooks() -> None:
    class HostileDict(dict[str, object]):
        def __iter__(self):  # type: ignore[no-untyped-def]
            raise AssertionError("hostile iteration hook must not run")

    inputs, labels, checkpoint = _task()
    artifact = execute_optimization_readiness(
        validation_inputs=inputs,
        validation_labels=labels,
        checkpoint_parameters=checkpoint,
        seed=3,
        task_id="task",
        checkpoint_id="checkpoint",
    )
    artifact["metrics"] = HostileDict()
    with pytest.raises(ValueError, match="plain JSON"):
        validate_optimization_readiness_execution(
            artifact,
            validation_inputs=inputs,
            validation_labels=labels,
            checkpoint_parameters=checkpoint,
        )


def test_validator_rejects_shallow_schema_before_recursive_payload_walk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(value: object) -> None:
        del value
        raise AssertionError("recursive validation must follow the exact shallow schema")

    monkeypatch.setattr(executor_module, "_assert_plain_json", forbidden)
    with pytest.raises(ValueError, match="unexpected keys"):
        validate_optimization_readiness_execution(
            {"unexpected": []},
            validation_inputs=None,
            validation_labels=None,
            checkpoint_parameters=None,
        )


def test_validator_rejects_huge_exact_root_before_materializing_keys() -> None:
    payload = {f"unexpected-{index}": None for index in range(10_000)}
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="unexpected keys"):
            validate_optimization_readiness_execution(
                payload,
                validation_inputs=None,
                validation_labels=None,
                checkpoint_parameters=None,
            )
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak_bytes < 64 * 1024


def test_plain_json_gate_rejects_aliased_or_cyclic_containers() -> None:
    child: list[object] = [0]
    with pytest.raises(ValueError, match="aliased or cyclic"):
        executor_module._assert_plain_json([child, child])

    cycle: list[object] = []
    cycle.append(cycle)
    with pytest.raises(ValueError, match="aliased or cyclic"):
        executor_module._assert_plain_json(cycle)


def test_plain_json_gate_has_aggregate_node_and_string_byte_limits() -> None:
    too_many_nodes = [list(range(512)) for _ in range(9)]
    with pytest.raises(ValueError, match="aggregate node limit"):
        executor_module._assert_plain_json(too_many_nodes)

    too_many_string_bytes = [str(index) + ("x" * 4_090) for index in range(17)]
    with pytest.raises(ValueError, match="aggregate string-byte limit"):
        executor_module._assert_plain_json(too_many_string_bytes)


def test_validator_rejects_different_dataset_and_checkpoint() -> None:
    inputs, labels, checkpoint = _task()
    artifact = execute_optimization_readiness(
        validation_inputs=inputs,
        validation_labels=labels,
        checkpoint_parameters=checkpoint,
        seed=1,
        task_id="task",
        checkpoint_id="checkpoint",
    )
    changed_labels = labels.copy()
    changed_labels[0] += 1.0
    with pytest.raises(ValueError, match="dataset identity"):
        validate_optimization_readiness_execution(
            artifact,
            validation_inputs=inputs,
            validation_labels=changed_labels,
            checkpoint_parameters=checkpoint,
        )
    changed_checkpoint = checkpoint.copy()
    changed_checkpoint[0] += 1.0
    with pytest.raises(ValueError, match="checkpoint identity"):
        validate_optimization_readiness_execution(
            artifact,
            validation_inputs=inputs,
            validation_labels=labels,
            checkpoint_parameters=changed_checkpoint,
        )


def test_executor_preflights_metadata_before_numeric_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = np.broadcast_to(np.zeros(1, dtype=np.float64), (10_000, 65))
    labels = np.zeros(10_000, dtype=np.float64)
    checkpoint = np.zeros(65, dtype=np.float64)

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("hashing/scanning must follow metadata preflight")

    monkeypatch.setattr("hashlib.sha256", forbidden)
    with pytest.raises(ValueError, match="parameter count"):
        execute_optimization_readiness(
            validation_inputs=inputs,
            validation_labels=labels,
            checkpoint_parameters=checkpoint,
            seed=0,
            task_id="task",
            checkpoint_id="checkpoint",
        )


def test_executor_peak_charges_both_terminal_loss_arrays_and_scopes_work_units() -> None:
    inputs, labels, checkpoint = _task()
    _, _, _, resources = executor_module._preflight_arrays(inputs, labels, checkpoint)
    parameters = checkpoint.size
    total_updates = 128 * (1 + 10 + 100)
    schedule_items = 4 * (128 + total_updates)
    caller_bytes = inputs.nbytes + labels.nbytes + checkpoint.nbytes
    nonterminal_bytes = (
        caller_bytes
        + caller_bytes
        + schedule_items * np.dtype(np.int64).itemsize
        + schedule_items * np.dtype(np.int32).itemsize
        + 128 * parameters * np.dtype(np.float64).itemsize
        + 128
        * (parameters + 2 * 4 * parameters + 2 * 4)
        * np.dtype(np.float64).itemsize
        + (parameters * parameters + 2 * parameters) * np.dtype(np.float64).itemsize
        + inputs.nbytes
        + 17 * inputs.size
        + 24 * parameters * parameters
        + 8 * max(inputs.shape)
        + 96 * parameters
    )
    terminal_loss_bytes = 2 * 10_000 * 128 * np.dtype(np.float64).itemsize
    assert resources["peak_working_set_bytes"] == nonterminal_bytes + terminal_loss_bytes
    assert "scalar_work" not in resources
    assert resources["preflight_work_units"] > 0


@pytest.mark.parametrize("field", ["validation_inputs", "validation_labels", "checkpoint"])
def test_executor_rejects_nonfinite_model_inputs(field: str) -> None:
    inputs, labels, checkpoint = _task()
    if field == "validation_inputs":
        inputs[0, 0] = np.nan
    elif field == "validation_labels":
        labels[0] = np.inf
    else:
        checkpoint[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        execute_optimization_readiness(
            validation_inputs=inputs,
            validation_labels=labels,
            checkpoint_parameters=checkpoint,
            seed=0,
            task_id="task",
            checkpoint_id="checkpoint",
        )
