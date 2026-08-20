"""Development-only calibrated partial-reset IPMNIST slice."""

from __future__ import annotations

import copy

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.benchmarks.ipmnist_screening import (
    CPR_OFFICIAL_CODE_REVISION,
    CPR_PAPER_REVISION,
    IPMNISTConfig,
    _make_cpr_ipmnist_learner,
    _make_sgd_ema_norm_learner,
    partial_reset_development_record,
    run_screening_config,
    screening_spec,
    validate_partial_reset_development_record,
)
from alberta_framework.benchmarks.upgd_ipmnist import init_mlp_params

SMALL = IPMNISTConfig(
    n_tasks=2, task_length=4, input_dim=6, hidden1=5, hidden2=4, n_classes=3
)
ARMS = (
    "cpr_ipmnist",
    "cpr_hard_reset",
    "cpr_l2_init",
    "cpr_utility_free",
    "cpr_off",
)


def _data() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.arange(72, dtype=np.float32).reshape(12, 6) / 71.0,
        np.arange(12, dtype=np.int32) % 3,
    )


def _tree_bytes(tree: object) -> int:
    return sum(int(value.nbytes) for value in jax.tree_util.tree_leaves(tree))


def test_reference_and_official_code_revision_are_pinned() -> None:
    assert CPR_PAPER_REVISION == "arXiv:2607.24996v1"
    assert CPR_OFFICIAL_CODE_REVISION == (
        "LucMc/continual-learning@6fc2af34783159f5dda50c6915dda32c2d443604"
    )
    assert all(screening_spec(name).mechanism == "calibrated_partial_reset" for name in ARMS)


def test_all_arms_have_matched_peak_state_bytes() -> None:
    params = init_mlp_params(jr.key(1), SMALL)
    state_bytes = []
    for name in ARMS:
        spec = screening_spec(name)
        init_fn, _ = spec.factory(spec.hyperparameters)
        state_bytes.append(_tree_bytes(init_fn(params)))
    assert len(set(state_bytes)) == 1
    expected_peak = _tree_bytes(params) + state_bytes[0]
    x, y = _data()
    result = run_screening_config(x, y, screening_spec(ARMS[0]), seed=1, config=SMALL)
    receipt = partial_reset_development_record(result)
    assert receipt["resources"]["peak_numeric_bytes"] == expected_peak
    assert receipt["resources"]["persistent_bytes"] == expected_peak


def test_mechanism_off_matches_normalized_sgd_params_metrics_and_norm_under_jit() -> None:
    spec = screening_spec("cpr_off")
    init_fn, step_fn = spec.factory(spec.hyperparameters)
    control_hp = {
        "step_size": spec.hyperparameters["step_size"],
        "weight_decay": 0.0,
        "norm_decay": spec.hyperparameters["norm_decay"],
        "norm_epsilon": spec.hyperparameters["norm_epsilon"],
    }
    control_init, control_step = _make_sgd_ema_norm_learner(control_hp)
    params = init_mlp_params(jr.key(2), SMALL)
    x = jr.normal(jr.key(3), (SMALL.input_dim,))
    y = jnp.asarray(1, dtype=jnp.int32)
    ours = jax.jit(step_fn)(params, init_fn(params), x, y, jr.key(4))
    control = jax.jit(control_step)(params, control_init(params), x, y, jr.key(99))
    for name in params:
        np.testing.assert_array_equal(ours[0][name], control[0][name])
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(jnp.array_equal, ours[1].norm, control[1].norm)
    )
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(jnp.array_equal, ours[2], control[2])
    )


@pytest.mark.parametrize("name", ARMS)
def test_all_arms_are_finite_and_jittable(name: str) -> None:
    spec = screening_spec(name)
    init_fn, step_fn = spec.factory(spec.hyperparameters)
    params = init_mlp_params(jr.key(5), SMALL)
    result = jax.jit(step_fn)(
        params,
        init_fn(params),
        jr.normal(jr.key(6), (SMALL.input_dim,)),
        jnp.asarray(2, dtype=jnp.int32),
        jr.key(7),
    )
    assert all(
        bool(jnp.all(jnp.isfinite(value))) for value in jax.tree_util.tree_leaves(result)
    )


@pytest.mark.parametrize(
    "name", ("cpr_ipmnist", "cpr_hard_reset", "cpr_l2_init", "cpr_utility_free")
)
def test_each_reduction_engages_relative_to_off(name: str) -> None:
    params = init_mlp_params(jr.key(8), SMALL)
    x = jr.normal(jr.key(9), (SMALL.input_dim,))
    y = jnp.asarray(0, dtype=jnp.int32)
    hp = dict(screening_spec(name).hyperparameters)
    hp["reset_frequency"] = 1.0
    init_fn, step_fn = _make_cpr_ipmnist_learner(hp)
    off_hp = dict(screening_spec("cpr_off").hyperparameters)
    off_hp["reset_frequency"] = 1.0
    off_init, off_step = _make_cpr_ipmnist_learner(off_hp)
    ours_params, ours_state, _ = step_fn(params, init_fn(params), x, y, jr.key(10))
    off_params, off_state, _ = off_step(params, off_init(params), x, y, jr.key(10))
    ours_params, _, _ = step_fn(ours_params, ours_state, x, y, jr.key(11))
    off_params, _, _ = off_step(off_params, off_state, x, y, jr.key(11))
    assert any(
        not np.array_equal(np.asarray(ours_params[key]), np.asarray(off_params[key]))
        for key in params
    )


def test_end_to_end_record_is_strict_nonpromoting_and_resource_matched() -> None:
    x, y = _data()
    records = []
    for name in ARMS:
        result = run_screening_config(x, y, screening_spec(name), seed=13, config=SMALL)
        record = partial_reset_development_record(result)
        assert validate_partial_reset_development_record(record) == record
        records.append(record)
    assert len({record["resources"]["peak_numeric_bytes"] for record in records}) == 1
    assert all(record["resources"]["observations"] == 8 for record in records)
    assert all(record["resources"]["updates"] == 8 for record in records)
    assert all(record["resources"]["model_queries"] == 16 for record in records)
    assert all(record["policy"]["scientific_promotion_allowed"] is False for record in records)

    hostile = copy.deepcopy(records[0])
    hostile["resources"]["peak_numeric_bytes"] -= 4
    with pytest.raises(ValueError, match="resource receipt"):
        validate_partial_reset_development_record(hostile)
    hostile = copy.deepcopy(records[0])
    hostile["policy"]["scientific_promotion_allowed"] = True
    with pytest.raises(ValueError, match="permanently nonpromoting"):
        validate_partial_reset_development_record(hostile)
    hostile = copy.deepcopy(records[0])
    hostile["references"]["official_code"] = "moving-main"
    with pytest.raises(ValueError, match="frozen protocol"):
        validate_partial_reset_development_record(hostile)

    class HostileDict(dict[str, object]):
        pass

    with pytest.raises(ValueError, match="exact object"):
        validate_partial_reset_development_record(HostileDict(records[0]))

    hostile = copy.deepcopy(records[0])
    hostile["metrics"]["per_task_accuracy"] = [0.5] * 1_000_000
    with pytest.raises(ValueError, match="invalid partial-reset result fields"):
        validate_partial_reset_development_record(hostile)

    hostile = copy.deepcopy(records[0])
    hostile["resources"]["timing_telemetry_seconds"] = True
    with pytest.raises(ValueError, match="invalid partial-reset result fields"):
        validate_partial_reset_development_record(hostile)
