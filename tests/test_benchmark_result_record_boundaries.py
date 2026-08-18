"""Exact host-record boundaries for development benchmark results."""

from __future__ import annotations

import numpy as np
import pytest

from alberta_framework.benchmarks.ipmnist_screening import ScreeningRunResult
from alberta_framework.benchmarks.micro_continual import (
    MicroStream,
    MicroTaskConfig,
    micro_arm_spec,
)
from alberta_framework.benchmarks.upgd_ipmnist import IPMNISTConfig, IPMNISTRunResult
from alberta_framework.benchmarks.upgd_label_emnist import (
    LabelEMNISTConfig,
    LabelEMNISTRunResult,
)


class _DictSubclass(dict[str, float]):
    """A mutable identity that must not cross an exact record boundary."""


class _HostileStr(str):
    calls = 0

    def __hash__(self) -> int:
        type(self).calls += 1
        raise AssertionError("hostile hash must not execute")

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile equality must not execute")


def _ipmnist_result(**overrides: object) -> IPMNISTRunResult:
    config = IPMNISTConfig()
    shape = (1, config.n_tasks)
    payload: dict[str, object] = {
        "learner": "adamw",
        "hyperparameters": {"step_size": 0.001},
        "seeds": (0,),
        "config": config,
        "per_task_accuracy": np.zeros(shape, dtype=np.float32),
        "per_task_loss": np.zeros(shape, dtype=np.float32),
        "per_task_plasticity": np.zeros(shape, dtype=np.float32),
        "average_online_accuracy": np.zeros((1,), dtype=np.float32),
        "wall_clock_seconds": 0.0,
    }
    payload.update(overrides)
    return IPMNISTRunResult(**payload)  # type: ignore[arg-type]


def _label_result(**overrides: object) -> LabelEMNISTRunResult:
    config = LabelEMNISTConfig()
    shape = (1, config.n_tasks)
    payload: dict[str, object] = {
        "learner": "adamw",
        "hyperparameters": {"step_size": 0.001},
        "seeds": (0,),
        "config": config,
        "per_task_accuracy": np.zeros(shape, dtype=np.float64),
        "per_task_loss": np.zeros(shape, dtype=np.float64),
        "per_task_plasticity": np.zeros(shape, dtype=np.float64),
        "average_online_accuracy": np.zeros((1,), dtype=np.float64),
        "wall_clock_seconds": 0.0,
    }
    payload.update(overrides)
    return LabelEMNISTRunResult(**payload)  # type: ignore[arg-type]


def test_ipmnist_result_rejects_container_shape_domain_and_debug_residuals() -> None:
    with pytest.raises(TypeError, match="hyperparameters must be a dict"):
        _ipmnist_result(hyperparameters=_DictSubclass(step_size=0.001))
    with pytest.raises(ValueError, match="run-result shape"):
        _ipmnist_result(per_task_accuracy=np.zeros((1, 1), dtype=np.float32))
    bad = np.zeros((1, IPMNISTConfig().n_tasks), dtype=np.float32)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _ipmnist_result(per_task_loss=bad)
    with pytest.raises(ValueError, match="all present or all absent"):
        _ipmnist_result(
            per_step_accuracy=np.zeros(
                (1, IPMNISTConfig().n_tasks, IPMNISTConfig().task_length),
                dtype=np.float32,
            )
        )


def test_label_result_rejects_dtype_and_metric_domain_residuals() -> None:
    shape = (1, LabelEMNISTConfig().n_tasks)
    with pytest.raises(ValueError, match="run-result shape"):
        _label_result(per_task_accuracy=np.zeros(shape, dtype=np.int32))
    invalid = np.zeros(shape, dtype=np.float64)
    invalid[0, 0] = 1.01
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _label_result(per_task_plasticity=invalid)


def test_screening_result_enforces_exact_curve_and_noise_contract() -> None:
    config = IPMNISTConfig()
    curve = np.zeros((config.n_tasks,), dtype=np.float64)
    with pytest.raises(ValueError, match="noise_pool_steps"):
        ScreeningRunResult(
            config_name="candidate",
            base_learner="adamw",
            hyperparameters={},
            seed=0,
            config=config,
            per_task_accuracy=curve,
            per_task_loss=curve,
            per_task_plasticity=curve,
            wall_clock_seconds=0.0,
            noise_mode="pool",
            noise_pool_steps=None,
        )
    with pytest.raises(ValueError, match="one float64 value per task"):
        ScreeningRunResult(
            config_name="candidate",
            base_learner="adamw",
            hyperparameters={},
            seed=0,
            config=config,
            per_task_accuracy=np.zeros((1, config.n_tasks), dtype=np.float64),
            per_task_loss=curve,
            per_task_plasticity=curve,
            wall_clock_seconds=0.0,
        )


def test_micro_stream_is_exact_bounded_and_detached() -> None:
    config = MicroTaskConfig(
        name="tiny",
        kind="input_permutation",
        role="search",
        input_dim=64,
        n_classes=10,
        n_tasks=1,
        task_length=1,
        hidden1=1,
        hidden2=1,
        crop=False,
    )
    source = np.zeros((1, 64), dtype=np.float32)
    stream = MicroStream(
        xs=source,
        ys=np.zeros((1,), dtype=np.int32),
        example_indices=np.zeros((1,), dtype=np.int32),
        config=config,
        seed=0,
    )
    source[0, 0] = 1.0
    assert stream.xs[0, 0] == 0.0
    assert not stream.xs.flags.writeable
    with pytest.raises(ValueError, match="stream exceeds the step budget"):
        MicroTaskConfig(
            name="oversized",
            kind="input_permutation",
            role="search",
            input_dim=64,
            n_classes=10,
            n_tasks=4_000,
            task_length=4_000,
            hidden1=1,
            hidden2=1,
            crop=False,
        )


def test_micro_arm_lookup_gates_identity_before_mapping_hooks() -> None:
    _HostileStr.calls = 0
    with pytest.raises(TypeError, match="exact string"):
        micro_arm_spec(_HostileStr("upgd_w_control"))
    assert _HostileStr.calls == 0
