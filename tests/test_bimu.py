from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.benchmarks.bimu import (
    BIMU_PAPER_CONFIG,
    BIMU_PROTOCOL,
    RESULT_SCHEMA,
    BiMUConfig,
    BiMUState,
    _apply_gradient,
    bimu_update,
    bimu_update_transaction,
    build_bimu_matched_report,
    build_task_schedule,
    canonical_bimu_result_bytes,
    concrete_binary_weights,
    late_window_mean,
    posterior_probability,
    posterior_probability_transaction,
    retain_bimu_matched_report,
    retain_bimu_result,
    run_bimu_development,
    sample_binary_weights,
    validate_bimu_result,
)


def _tiny_config(**overrides: object) -> BiMUConfig:
    values: dict[str, object] = {
        "input_dim": 4,
        "hidden_units": 3,
        "n_classes": 2,
        "n_tasks": 5,
        "train_examples_per_task": 4,
        "test_examples_per_task": 2,
        "train_samples": 2,
        "test_samples": 3,
        "query_samples": 3,
        "temperature": 1.0,
        "likelihood_multiplier": 2.0,
        "kl_multiplier": 0.5,
        "alpha_max": 0.1,
        "memory_window": 7,
        "gradient_scale": 1.5,
        "query_threshold": 0.0,
    }
    values.update(overrides)
    return BiMUConfig(**values)


def _tiny_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x = np.asarray(
        [
            [2.0, 1.0, -1.0, -2.0],
            [-2.0, -1.0, 1.0, 2.0],
            [1.5, -0.5, 0.5, -1.5],
            [-1.5, 0.5, -0.5, 1.5],
        ],
        dtype=np.float32,
    )
    train_y = np.asarray([0, 1, 0, 1], dtype=np.int32)
    test_x = np.asarray(
        [[1.0, 0.5, -0.5, -1.0], [-1.0, -0.5, 0.5, 1.0]], dtype=np.float32
    )
    test_y = np.asarray([0, 1], dtype=np.int32)
    return train_x, train_y, test_x, test_y


def test_protocol_pins_official_source_and_paper_configuration() -> None:
    assert BIMU_PROTOCOL["paper_revision"] == "arXiv:2605.30198v1"
    assert BIMU_PROTOCOL["official_code_commit"] == (
        "1b8a1a1fb892fbe89401390b3ff9611d7f3a5168"
    )
    assert BIMU_PAPER_CONFIG.n_tasks == 1000
    assert BIMU_PAPER_CONFIG.train_examples_per_task == 60_000
    assert BIMU_PAPER_CONFIG.hidden_units == 100
    assert BIMU_PAPER_CONFIG.train_samples == 5
    assert BIMU_PAPER_CONFIG.test_samples == 5
    assert BIMU_PAPER_CONFIG.likelihood_multiplier == pytest.approx(161.3)
    assert BIMU_PAPER_CONFIG.kl_multiplier == pytest.approx(3.76)
    assert BIMU_PAPER_CONFIG.alpha_max == pytest.approx(0.0023)
    assert BIMU_PAPER_CONFIG.memory_window == 700
    assert BIMU_PAPER_CONFIG.gradient_scale == pytest.approx(4.9)
    assert BIMU_PAPER_CONFIG.matches_paper_configuration
    with pytest.raises(TypeError):
        BIMU_PROTOCOL["development_only"] = False  # type: ignore[index]


def test_equation_update_is_jittable_and_matches_scaled_official_rule() -> None:
    state = jnp.array([0.0, 1.0], dtype=jnp.float32)
    prior = jnp.zeros(2, dtype=jnp.float32)
    gradient = jnp.array([2.0, -0.5], dtype=jnp.float32)
    update = jax.jit(
        lambda s, g, p: bimu_update(
            s,
            g,
            p,
            memory_window=10,
            alpha_max=1.0,
            likelihood_multiplier=2.0,
            kl_multiplier=3.0,
            gradient_scale=4.0,
        )
    )
    updated = update(state, gradient, prior)
    scaled_gradient = 2.0 * gradient
    uncertainty = 3.0 * (1.0 - jnp.tanh(state) ** 2)
    reciprocal = uncertainty + 2.0 * jnp.tanh(state) * scaled_gradient + 1.0
    reciprocal += 2.0 * jnp.abs(scaled_gradient)
    expected = state - (4.0 * scaled_gradient + (state - prior) * uncertainty / 10) / reciprocal
    np.testing.assert_allclose(updated, expected, rtol=1e-6)


def test_binary_and_concrete_samples_are_explicit_and_reproducible() -> None:
    natural = jnp.asarray([-2.0, 0.0, 2.0], dtype=jnp.float32)
    key = jax.random.key(17)
    binary = sample_binary_weights(natural, key)
    np.testing.assert_array_equal(binary, sample_binary_weights(natural, key))
    assert set(np.asarray(binary).tolist()) <= {-1.0, 1.0}
    concrete = concrete_binary_weights(natural, key, temperature=0.7)
    np.testing.assert_allclose(concrete, concrete_binary_weights(natural, key, temperature=0.7))
    assert bool(jnp.all(jnp.abs(concrete) < 1.0))
    derivative = jax.grad(
        lambda x: concrete_binary_weights(x, key, temperature=0.7).sum()
    )(natural)
    assert bool(jnp.all(jnp.isfinite(derivative)))
    with pytest.raises(ValueError, match="temperature"):
        concrete_binary_weights(natural, key, temperature=0.0)
    with pytest.raises(ValueError, match="Threefry"):
        sample_binary_weights(natural, np.zeros(2, dtype=np.uint32))  # type: ignore[arg-type]


def test_mechanism_off_removes_only_controlled_forgetting() -> None:
    state = jnp.asarray([0.5, -0.25], dtype=jnp.float32)
    gradient = jnp.asarray([0.2, -0.1], dtype=jnp.float32)
    result = bimu_update(
        state,
        gradient,
        jnp.zeros_like(state),
        memory_window=None,
        alpha_max=0.5,
        likelihood_multiplier=1.0,
        kl_multiplier=1.0,
        gradient_scale=1.0,
    )
    uncertainty = 1.0 - jnp.tanh(state) ** 2
    eta = 1.0 / (
        uncertainty + 2.0 * jnp.tanh(state) * gradient + 2.0 + 2.0 * jnp.abs(gradient)
    )
    np.testing.assert_allclose(result, state - eta * gradient, rtol=1e-6)


def test_equation_kernel_does_not_hide_official_zero_gradient_gate() -> None:
    state = jnp.asarray([0.5], dtype=jnp.float32)
    equation_result = bimu_update(
        state,
        jnp.zeros_like(state),
        jnp.zeros_like(state),
        memory_window=5,
        alpha_max=0.5,
    )
    assert not np.array_equal(equation_result, state)
    model_state = BiMUState(
        input_hidden=state.reshape(1, 1),
        hidden_output=state.reshape(1, 1),
    )
    gradient = BiMUState(
        input_hidden=jnp.zeros((1, 1), dtype=jnp.float32),
        hidden_output=jnp.zeros((1, 1), dtype=jnp.float32),
    )
    gated = _apply_gradient(model_state, gradient, _tiny_config())
    np.testing.assert_array_equal(gated.input_hidden, model_state.input_hidden)
    np.testing.assert_array_equal(gated.hidden_output, model_state.hidden_output)


def test_task_schedule_is_deterministic_and_task_private() -> None:
    config = _tiny_config()
    first = build_task_schedule(config, seed=9)
    second = build_task_schedule(config, seed=9)
    assert first == second
    assert first != build_task_schedule(config, seed=10)
    assert len(first) == config.n_tasks
    assert all(sorted(permutation) == list(range(config.input_dim)) for permutation in first)
    assert config.learner_observes_task_boundary is False


def test_public_config_consumers_revalidate_frozen_instances() -> None:
    config = _tiny_config()
    object.__setattr__(config, "n_tasks", 2**100)
    with pytest.raises(ValueError, match="n_tasks"):
        build_task_schedule(config, seed=1)
    with pytest.raises(ValueError, match="n_tasks"):
        run_bimu_development(*_tiny_data(), config=config, seed=1)
    with pytest.raises(ValueError, match="sample ceiling"):
        _tiny_config(train_samples=129)
    with pytest.raises(ValueError, match="runtime ceiling"):
        run_bimu_development(*_tiny_data(), config=BIMU_PAPER_CONFIG, seed=1)


def test_tiny_runner_is_end_to_end_strict_and_keeps_metrics_separate() -> None:
    config = _tiny_config()
    payload = run_bimu_development(*_tiny_data(), config=config, seed=23)
    validate_bimu_result(payload)
    metrics = payload["metrics"]
    counters = payload["counters"]
    resources = payload["resources"]
    assert metrics["paper_late_five_test_accuracy"] == pytest.approx(
        np.mean(metrics["per_task_test_accuracy"][-5:])
    )
    assert "asi_whole_stream_online_accuracy" in metrics
    assert counters["environment_steps"] == 20
    assert counters["observations"] == 20
    assert counters["label_queries"] == 20
    assert counters["optimizer_updates"] == 20
    assert counters["model_forward_queries"] == 20 * (3 + 2) + 5 * 2 * 3
    assert counters["online_correct"] <= counters["observations"]
    assert len(counters["per_task_test_correct"]) == config.n_tasks
    assert resources["parameter_numeric_bytes"] == (4 * 3 + 3 * 2) * 4
    assert resources["optimizer_state_numeric_bytes"] == 8
    assert resources["initial_persistent_numeric_bytes"] == (4 * 3 + 3 * 2) * 4 + 8
    assert resources["final_persistent_numeric_bytes"] == resources[
        "initial_persistent_numeric_bytes"
    ]
    assert payload["comparison"]["paper_comparable"] is False
    assert payload["evidence_policy"]["scientific_promotion_allowed"] is False


def test_result_canonical_retention_is_exclusive_and_strict(tmp_path: Path) -> None:
    payload = run_bimu_development(*_tiny_data(), config=_tiny_config(), seed=23)
    encoded = canonical_bimu_result_bytes(payload)
    destination = retain_bimu_result(payload, repository_root=tmp_path)
    assert destination.parent == tmp_path / "outputs/bimu/development.v1"
    assert destination.read_bytes() == encoded
    with pytest.raises(FileExistsError):
        retain_bimu_result(payload, repository_root=tmp_path)


def test_matched_report_keeps_paper_and_online_outcomes_separate(tmp_path: Path) -> None:
    candidate = run_bimu_development(*_tiny_data(), config=_tiny_config(), seed=23)
    control = run_bimu_development(
        *_tiny_data(), config=_tiny_config(memory_window=None), seed=23
    )
    report = build_bimu_matched_report(candidate, control)
    assert report["metric_deltas"] == {
        "paper_late_five_test_accuracy": pytest.approx(0.3),
        "paper_late_five_outcome": "improved",
        "asi_whole_stream_online_accuracy": pytest.approx(-0.1),
        "asi_whole_stream_online_outcome": "worse",
    }
    destination = retain_bimu_matched_report(report, repository_root=tmp_path)
    assert destination.name.startswith("matched.")


def test_runner_replays_schedule_metrics_and_state_from_same_seed() -> None:
    config = _tiny_config()
    first = run_bimu_development(*_tiny_data(), config=config, seed=4)
    second = run_bimu_development(*_tiny_data(), config=config, seed=4)
    assert first["schedule_sha256"] == second["schedule_sha256"]
    assert first["final_state_sha256"] == second["final_state_sha256"]
    assert first["metrics"] == second["metrics"]
    assert first["counters"] == second["counters"]


def test_runner_rejects_finite_overflow_before_transaction_commit() -> None:
    train_x, train_y, test_x, test_y = _tiny_data()
    train_x = np.full_like(train_x, np.finfo(np.float32).max)
    with pytest.raises(ValueError, match="transaction"):
        run_bimu_development(
            train_x, train_y, test_x, test_y, config=_tiny_config(), seed=4
        )


def test_no_query_schedule_performs_no_updates() -> None:
    config = _tiny_config(query_threshold=1.0, memory_window=None)
    payload = run_bimu_development(*_tiny_data(), config=config, seed=5)
    validate_bimu_result(payload)
    assert payload["counters"]["label_queries"] == 0
    assert payload["counters"]["optimizer_updates"] == 0
    assert payload["resources"]["state_changed"] is False


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("evidence_policy", "scientific_promotion_allowed"), True),
        (("counters", "optimizer_updates"), 999),
        (("resources", "final_persistent_numeric_bytes"), 1),
        (("metrics", "paper_late_five_test_accuracy"), 0.123456),
        (("counters", "online_correct"), 999),
        (("counters", "per_task_test_correct"), [0, 0, 0, 0, 0]),
    ],
)
def test_validator_fails_closed_on_policy_accounting_and_metric_drift(
    path: tuple[str, str], value: object
) -> None:
    payload = run_bimu_development(*_tiny_data(), config=_tiny_config(), seed=11)
    corrupted = deepcopy(payload)
    corrupted[path[0]][path[1]] = value
    with pytest.raises(ValueError):
        validate_bimu_result(corrupted)


def test_validator_rejects_unknown_fields() -> None:
    payload = run_bimu_development(*_tiny_data(), config=_tiny_config(), seed=11)
    payload["claim"] = "sota"
    with pytest.raises(ValueError, match="fields"):
        validate_bimu_result(payload)


def test_validator_rejects_hostile_string_key_without_hooks() -> None:
    class HostileKey(str):
        calls = 0

        def __hash__(self) -> int:
            self.calls += 1
            return super().__hash__()

        def __eq__(self, other: object) -> bool:
            self.calls += 1
            return super().__eq__(other)

    key = HostileKey("schema")
    payload = {key: RESULT_SCHEMA}
    key.calls = 0
    with pytest.raises(ValueError, match="keys"):
        validate_bimu_result(payload)
    assert key.calls == 0


def test_late_window_metric_remains_separate_from_whole_stream() -> None:
    assert late_window_mean([0.1, 0.2, 0.8, 0.9], window=2) == pytest.approx(0.85)
    np.testing.assert_allclose(posterior_probability(jnp.asarray([0.0])), [0.5])


def test_transactions_are_outer_jit_safe_and_fail_closed() -> None:
    update = jax.jit(
        lambda state, gradient, prior: bimu_update(
            state,
            gradient,
            prior,
            memory_window=10,
            alpha_max=1.0,
            likelihood_multiplier=2.0,
            kl_multiplier=3.0,
            gradient_scale=4.0,
        )
    )
    assert bool(jnp.all(jnp.isfinite(update(jnp.zeros(2), jnp.ones(2), jnp.zeros(2)))))
    transact = jax.jit(
        lambda state, gradient: bimu_update_transaction(
            state,
            gradient,
            jnp.zeros(2),
            memory_window=10,
            alpha_max=1.0,
            likelihood_multiplier=2.0,
            kl_multiplier=3.0,
            gradient_scale=4.0,
        )
    )
    for state, gradient in (
        (jnp.asarray([jnp.nan, 0.0]), jnp.ones(2)),
        (jnp.zeros(2), jnp.asarray([jnp.inf, 0.0])),
    ):
        safe, valid = transact(state, gradient)
        assert bool(jnp.all(jnp.isfinite(safe)))
        assert not bool(valid)


def test_primitives_reject_hostile_array_protocol_objects_without_calling_them() -> None:
    class Hostile:
        calls = 0

        def __array__(self) -> np.ndarray:
            self.calls += 1
            raise AssertionError("must not run")

    hostile = Hostile()
    with pytest.raises(ValueError, match="exact NumPy or JAX"):
        posterior_probability(hostile)
    assert hostile.calls == 0

    data = _tiny_data()
    with pytest.raises(ValueError, match="exact NumPy arrays"):
        run_bimu_development(hostile, data[1], data[2], data[3], config=_tiny_config(), seed=1)
    assert hostile.calls == 0


def test_float32_overflow_is_invalid_not_laundered() -> None:
    maximum = jnp.finfo(jnp.float32).max
    transact = jax.jit(
        lambda gradient: bimu_update_transaction(
            jnp.zeros(2), gradient, jnp.zeros(2), memory_window=10, alpha_max=1.0
        )
    )
    safe, valid = transact(jnp.full((2,), maximum))
    assert bool(jnp.all(jnp.isfinite(safe)))
    assert not bool(valid)
    posterior, posterior_valid = jax.jit(posterior_probability_transaction)(
        jnp.asarray([jnp.inf])
    )
    np.testing.assert_array_equal(posterior, [0.5])
    assert not bool(posterior_valid)
    finite_posterior, finite_valid = jax.jit(posterior_probability_transaction)(
        jnp.asarray([maximum])
    )
    np.testing.assert_array_equal(finite_posterior, [0.5])
    assert not bool(finite_valid)
