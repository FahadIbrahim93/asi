from __future__ import annotations

from dataclasses import replace

import jax
import numpy as np
import pytest

import alberta_framework.benchmarks.ipmnist_gradual_family as gradual_family
from alberta_framework.benchmarks.ipmnist_gradual_family import (
    GRADUAL_FAMILY_PROTOCOL,
    GradualMicroPhaseConfig,
    run_gradual_micro_phase_family,
    validate_gradual_micro_phase_result,
)


def _tiny() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, GradualMicroPhaseConfig]:
    old_x = np.asarray([[-1.0, 1.0], [1.0, -1.0]], dtype=np.float32)
    old_y = np.asarray([0, 1], dtype=np.int32)
    new_x = old_x.copy()
    new_y = np.asarray([1, 0], dtype=np.int32)
    config = GradualMicroPhaseConfig(
        transition_intervals=2,
        phase_examples=2,
        input_dim=2,
        hidden1=2,
        hidden2=2,
        n_classes=2,
    )
    return old_x, old_y, new_x, new_y, config


def test_micro_phase_family_runs_all_additive_matched_arms() -> None:
    old_x, old_y, new_x, new_y, config = _tiny()
    result = run_gradual_micro_phase_family(
        old_x,
        old_y,
        new_x,
        new_y,
        learner_name="adamw_control",
        seed=1_569_101,
        config=config,
    )

    assert result.schema == "asi.ipmnist.gradual-family.micro-phase-result.v1"
    assert result.arm_names == ("abrupt", "output_interpolation", "task_sampling")
    assert result.parent_input_result_used is False
    assert result.development_only is True
    assert result.scientific_promotion_allowed is False
    assert result.execution_attestation is False
    assert result.phase_alpha_numerators == (0, 1, 2)
    assert result.phase_alpha_denominator == 2
    np.testing.assert_array_equal(result.task_sampling_new_counts, [0, 1, 2])
    assert result.training_loss_sums.shape == (3, 3)
    assert result.new_task_eval_correct_counts.shape == (3, 3)
    assert result.new_task_eval_loss_sums.shape == (3, 3)
    assert np.all(np.isfinite(result.training_loss_sums))
    assert np.all(np.isfinite(result.new_task_eval_loss_sums))
    assert result.soft_target_updates_per_arm == (0, 6, 0)
    assert result.observations_per_arm == 12
    assert result.updates_per_arm == 6
    assert result.data_steps_per_arm == 12
    assert result.environment_steps_per_arm == 0
    assert result.model_queries_per_arm == 18
    assert len(result.dataset_sha256) == 2
    assert result.task_sampling_sha256.startswith("sha256:")
    assert result.source_sha256 == gradual_family._source_identity()
    assert result.runtime_identity == gradual_family._runtime_identity()
    assert {
        "jaxlib_version",
        "jax_devices",
        "jax_enable_x64",
        "jax_default_matmul_precision",
        "jax_random_seed_offset",
    }.issubset(dict(result.runtime_identity))


def test_micro_phase_family_uses_soft_targets_for_real_gradient_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_x, old_y, new_x, new_y, config = _tiny()
    calls: list[tuple[int, int, float]] = []
    original = gradual_family.output_interpolation

    def observed(old_label: int, new_label: int, alpha: float, *, n_classes: int):  # type: ignore[no-untyped-def]
        calls.append((old_label, new_label, alpha))
        return original(old_label, new_label, alpha, n_classes=n_classes)

    monkeypatch.setattr(gradual_family, "output_interpolation", observed)
    result = run_gradual_micro_phase_family(
        old_x,
        old_y,
        new_x,
        new_y,
        learner_name="adamw_control",
        seed=7,
        config=config,
    )
    assert len(calls) == config.phase_count * config.phase_examples
    assert {call[2] for call in calls} == {0.0, 0.5, 1.0}
    assert result.soft_target_updates_per_arm[1] == 6


def test_output_interpolation_uses_row_aligned_inputs_and_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_x = np.eye(4, dtype=np.float32)
    old_y = np.asarray([0, 1, 2, 3], dtype=np.int32)
    new_y = np.asarray([1, 2, 3, 0], dtype=np.int32)
    config = GradualMicroPhaseConfig(
        transition_intervals=1,
        phase_examples=4,
        input_dim=4,
        hidden1=2,
        hidden2=2,
        n_classes=4,
    )
    calls: list[tuple[int, int]] = []
    original = gradual_family.output_interpolation

    def observed(old_label: int, new_label: int, alpha: float, *, n_classes: int):  # type: ignore[no-untyped-def]
        calls.append((old_label, new_label))
        return original(old_label, new_label, alpha, n_classes=n_classes)

    monkeypatch.setattr(gradual_family, "output_interpolation", observed)
    run_gradual_micro_phase_family(
        shared_x,
        old_y,
        shared_x.copy(),
        new_y,
        learner_name="adamw_control",
        seed=37,
        config=config,
    )
    assert set(calls) == set(zip(old_y.tolist(), new_y.tolist(), strict=True))


def test_output_interpolation_rejects_noncorresponding_inputs_before_learner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_x, old_y, new_x, new_y, config = _tiny()
    new_x[0, 0] += 0.25

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("learner construction must follow correspondence validation")

    monkeypatch.setattr(gradual_family, "_make_adamw_learner", forbidden)
    with pytest.raises(ValueError, match="row-aligned identical inputs"):
        run_gradual_micro_phase_family(
            old_x,
            old_y,
            new_x,
            new_y,
            learner_name="adamw_control",
            seed=41,
            config=config,
        )


def test_micro_phase_family_exact_sampling_counts_include_nondivisible_phases() -> None:
    old_x = np.arange(15, dtype=np.float32).reshape(5, 3)
    new_x = old_x.copy()
    old_y = np.asarray([0, 1, 2, 0, 1], dtype=np.int32)
    new_y = np.asarray([2, 1, 0, 2, 1], dtype=np.int32)
    config = GradualMicroPhaseConfig(
        transition_intervals=3,
        phase_examples=5,
        input_dim=3,
        hidden1=2,
        hidden2=2,
        n_classes=3,
    )
    result = run_gradual_micro_phase_family(
        old_x,
        old_y,
        new_x,
        new_y,
        learner_name="adamw_control",
        seed=11,
        config=config,
    )
    np.testing.assert_array_equal(result.task_sampling_new_counts, [0, 1, 3, 5])


def test_micro_phase_family_is_independent_of_ambient_jax_prng_default() -> None:
    old_x, old_y, new_x, new_y, config = _tiny()
    with jax.default_prng_impl("threefry2x32"):
        first = run_gradual_micro_phase_family(
            old_x, old_y, new_x, new_y, learner_name="adamw_control", seed=19, config=config
        )
    with jax.default_prng_impl("rbg"):
        second = run_gradual_micro_phase_family(
            old_x, old_y, new_x, new_y, learner_name="adamw_control", seed=19, config=config
        )
    np.testing.assert_array_equal(first.training_loss_sums, second.training_loss_sums)
    np.testing.assert_array_equal(
        first.new_task_eval_correct_counts, second.new_task_eval_correct_counts
    )
    assert first.task_sampling_sha256 == second.task_sampling_sha256


def test_micro_phase_result_snapshots_arrays_and_revalidates_current_identity() -> None:
    old_x, old_y, new_x, new_y, config = _tiny()
    result = run_gradual_micro_phase_family(
        old_x, old_y, new_x, new_y, learner_name="adamw_control", seed=23, config=config
    )
    assert not result.training_loss_sums.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        result.training_loss_sums[0, 0] = 1.0
    with pytest.raises(ValueError, match="current source identity"):
        replace(result, source_sha256=(("forged.py", "0" * 64),))
    with pytest.raises(ValueError, match="current runtime identity"):
        replace(result, runtime_identity=(("python_version", "forged"),))


def test_micro_phase_validator_recomputes_dataset_and_realized_schedule() -> None:
    old_x, old_y, new_x, new_y, config = _tiny()
    result = run_gradual_micro_phase_family(
        old_x, old_y, new_x, new_y, learner_name="adamw_control", seed=29, config=config
    )
    validate_gradual_micro_phase_result(result, old_x, old_y, new_x, new_y)

    changed = old_x.copy()
    changed[0, 0] += 0.25
    with pytest.raises(ValueError, match="dataset content identity"):
        validate_gradual_micro_phase_result(result, changed, old_y, new_x, new_y)

    forged_schedule = replace(result, task_sampling_sha256="sha256:" + "0" * 64)
    with pytest.raises(ValueError, match="schedule identity"):
        validate_gradual_micro_phase_result(forged_schedule, old_x, old_y, new_x, new_y)


def test_micro_phase_result_rejects_forged_resource_and_promotion_receipts() -> None:
    old_x, old_y, new_x, new_y, config = _tiny()
    result = run_gradual_micro_phase_family(
        old_x, old_y, new_x, new_y, learner_name="adamw_control", seed=31, config=config
    )
    forged_bytes = result.persistent_numeric_bytes.copy()
    forged_bytes[0] += 4
    with pytest.raises(ValueError, match="complete exact receipt"):
        replace(result, persistent_numeric_bytes=forged_bytes)
    with pytest.raises(ValueError, match="nonpromoting"):
        replace(result, scientific_promotion_allowed=True)


def test_micro_phase_family_preflights_resources_before_learner_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = GradualMicroPhaseConfig(
        transition_intervals=1,
        phase_examples=2,
        input_dim=100_000,
        hidden1=1_000,
        hidden2=2,
        n_classes=2,
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("learner construction must happen after the resource preflight")

    monkeypatch.setattr(gradual_family, "_make_adamw_learner", forbidden)
    with pytest.raises(ValueError, match="working set"):
        run_gradual_micro_phase_family(
            np.zeros((2, 100_000), dtype=np.float32),
            np.zeros(2, dtype=np.int32),
            np.zeros((2, 100_000), dtype=np.float32),
            np.zeros(2, dtype=np.int32),
            learner_name="adamw_control",
            seed=1,
            config=config,
        )


def test_micro_phase_validator_preflights_working_set_before_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_x, old_y, new_x, new_y, _ = _tiny()
    config = GradualMicroPhaseConfig(
        transition_intervals=1,
        phase_examples=1,
        input_dim=1,
        hidden1=1,
        hidden2=1,
        n_classes=2,
    )
    base = run_gradual_micro_phase_family(
        old_x[:, :1],
        old_y,
        old_x[:, :1].copy(),
        new_y,
        learner_name="adamw_control",
        seed=43,
        config=config,
    )
    rows = 12_000_000
    persistent = rows * 2 * (config.input_dim + 1) * 4
    persistent += config.phase_count * config.phase_examples * (4 + 4 + 1)
    persistent += config.parameter_count * 16 + 6 * 5 * 4
    forged = replace(
        base,
        dataset_rows=(rows, rows),
        dataset_sha256=("sha256:" + "0" * 64, "sha256:" + "1" * 64),
        task_sampling_sha256="sha256:" + "2" * 64,
        persistent_numeric_bytes=np.full(3, persistent, dtype=np.int64),
    )
    huge_x = np.broadcast_to(np.asarray([0.0], dtype=np.float32), (rows, 1))
    huge_y = np.broadcast_to(np.asarray([0], dtype=np.int32), (rows,))

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("materialization and scheduling must follow the resource preflight")

    monkeypatch.setattr(gradual_family, "validated_ipmnist_data", forbidden)
    monkeypatch.setattr(gradual_family, "_realized_schedule", forbidden)
    with pytest.raises(ValueError, match="working set"):
        validate_gradual_micro_phase_result(forged, huge_x, huge_y, huge_x, huge_y)


def test_gradual_family_protocol_states_paper_and_asi_boundaries() -> None:
    assert GRADUAL_FAMILY_PROTOCOL["paper_revision"] == "arXiv:2602.09234v2"
    assert GRADUAL_FAMILY_PROTOCOL["paper_full_dataset_training"] is False
    assert GRADUAL_FAMILY_PROTOCOL["output_interpolation_loss"] == "soft_target_cross_entropy"
    assert GRADUAL_FAMILY_PROTOCOL["task_sampling_count"] == "floor(alpha * phase_examples)"
    assert GRADUAL_FAMILY_PROTOCOL["retained_input_result_recast"] is False
    assert GRADUAL_FAMILY_PROTOCOL["adaptation_gaps"]
    assert GRADUAL_FAMILY_PROTOCOL["development_only"] is True
    assert GRADUAL_FAMILY_PROTOCOL["scientific_promotion_allowed"] is False
