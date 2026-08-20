from __future__ import annotations

import copy

import numpy as np
import pytest

import alberta_framework.evaluation.optimization_readiness_entk as entk_module
from alberta_framework.evaluation.optimization_readiness_entk import (
    MLPCheckpoint,
    execute_entk_readiness,
    validate_entk_readiness,
)

pytestmark = pytest.mark.unit


def _case() -> tuple[np.ndarray, np.ndarray, MLPCheckpoint]:
    x = np.asarray(
        [[-1.0, 0.5], [-0.5, 1.0], [0.25, -0.75], [0.75, 0.25], [1.0, -0.5]],
        dtype=np.float64,
    )
    y = np.asarray([-0.75, -0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    checkpoint = MLPCheckpoint(
        input_weights=np.asarray([[0.4, -0.3, 0.2], [-0.1, 0.5, 0.25]], dtype=np.float64),
        hidden_bias=np.asarray([0.1, -0.2, 0.05], dtype=np.float64),
        output_weights=np.asarray([0.6, -0.4, 0.3], dtype=np.float64),
        output_bias=np.asarray(0.02, dtype=np.float64),
    )
    return x, y, checkpoint


def test_entk_executor_is_model_bound_and_strictly_replayable() -> None:
    x, y, checkpoint = _case()
    result = execute_entk_readiness(
        x,
        y,
        checkpoint,
        task_id="tiny-regression",
        checkpoint_id="mlp-0001",
    )
    checked = validate_entk_readiness(result, x, y, checkpoint)
    assert checked == result
    assert result["protocol"]["rng"] == "none_full_batch_deterministic"
    assert result["protocol"]["relu_zero_derivative"] == 0.0
    assert result["policy"] == {
        "development_only": True,
        "scientific_promotion_allowed": False,
        "timing_is_measured": False,
    }
    metrics = result["metrics"]
    assert metrics["representation_energy_rank_99"] >= 1
    assert metrics["entk_energy_rank_99"] >= 1
    assert metrics["future_loss_gain"]["1"] > 0.0
    assert set(metrics["future_loss_gain"]) == {"1", "10", "100"}
    resources = result["resources"]
    assert resources["optimizer_updates"] == 111
    assert resources["model_queries"] == 115 * x.shape[0]
    parameters = resources["parameter_count"]
    representation_work = x.shape[0] * 3 * min(x.shape[0], 3)
    jacobian_work = x.shape[0] * parameters * min(x.shape[0], parameters)
    assert resources["svd_work_units"] == representation_work + jacobian_work
    assert resources["rollout_work_units"] == 114 * x.shape[0] * parameters
    assert resources["jacobian_bytes"] > resources["representation_bytes"]


def test_entk_validator_rejects_forgery_and_different_inputs() -> None:
    x, y, checkpoint = _case()
    result = execute_entk_readiness(x, y, checkpoint, task_id="task", checkpoint_id="ckpt")
    forged = copy.deepcopy(result)
    forged["metrics"]["future_loss_gain"]["100"] = 1.0
    with pytest.raises(ValueError, match="replay"):
        validate_entk_readiness(forged, x, y, checkpoint)
    changed = x.copy()
    changed[0, 0] += 0.01
    with pytest.raises(ValueError, match="dataset identity"):
        validate_entk_readiness(result, changed, y, checkpoint)

    integer_as_float = copy.deepcopy(result)
    integer_as_float["resources"]["hidden_units"] = 3.0
    with pytest.raises(ValueError, match="replay"):
        validate_entk_readiness(integer_as_float, x, y, checkpoint)

    boolean_as_integer = copy.deepcopy(result)
    boolean_as_integer["policy"]["development_only"] = 1
    with pytest.raises(ValueError, match="replay"):
        validate_entk_readiness(boolean_as_integer, x, y, checkpoint)


def test_entk_preflight_rejects_before_expensive_linear_algebra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, y, checkpoint = _case()
    oversized_x = np.zeros((257, 2), dtype=np.float64)
    oversized_y = np.zeros(257, dtype=np.float64)
    monkeypatch.setattr(np.linalg, "svd", lambda *args, **kwargs: pytest.fail("SVD reached"))
    with pytest.raises(ValueError, match="observations"):
        execute_entk_readiness(
            oversized_x,
            oversized_y,
            checkpoint,
            task_id="task",
            checkpoint_id="ckpt",
        )

    # At this shape each SVD is individually below the limit, but their complete
    # logical work exceeds it. Rejection must still precede snapshots and LAPACK.
    combined_x = np.zeros((214, 32), dtype=np.float64)
    combined_y = np.zeros(214, dtype=np.float64)
    combined_checkpoint = MLPCheckpoint(
        input_weights=np.zeros((32, 64), dtype=np.float64),
        hidden_bias=np.zeros(64, dtype=np.float64),
        output_weights=np.zeros(64, dtype=np.float64),
        output_bias=np.asarray(0.0, dtype=np.float64),
    )
    monkeypatch.setattr(entk_module, "_snapshot", lambda value: pytest.fail("snapshot reached"))
    with pytest.raises(ValueError, match="SVD work"):
        execute_entk_readiness(
            combined_x,
            combined_y,
            combined_checkpoint,
            task_id="task",
            checkpoint_id="ckpt",
        )


def test_entk_rejects_checkpoint_and_json_type_confusion() -> None:
    x, y, checkpoint = _case()
    bad = MLPCheckpoint(
        input_weights=checkpoint.input_weights,
        hidden_bias=checkpoint.hidden_bias,
        output_weights=checkpoint.output_weights,
        output_bias=np.asarray([0.0], dtype=np.float64),
    )
    with pytest.raises(ValueError, match="output_bias"):
        execute_entk_readiness(x, y, bad, task_id="task", checkpoint_id="ckpt")

    result = execute_entk_readiness(x, y, checkpoint, task_id="task", checkpoint_id="ckpt")
    forged = copy.deepcopy(result)
    forged["resources"]["observations"] = True
    with pytest.raises(ValueError):
        validate_entk_readiness(forged, x, y, checkpoint)


def test_entk_energy_rank_is_scale_stable_and_uses_zero_relu_derivative() -> None:
    assert entk_module._energy_rank(np.diag([9e307, 1e307])) == 2
    assert entk_module._energy_rank(np.full((2, 2), 9e307, dtype=np.float64)) == 1

    x = np.asarray([[1.0], [-1.0]], dtype=np.float64)
    y = np.zeros(2, dtype=np.float64)
    checkpoint = MLPCheckpoint(
        input_weights=np.asarray([[1.0]], dtype=np.float64),
        hidden_bias=np.asarray([-1.0], dtype=np.float64),
        output_weights=np.asarray([4.0], dtype=np.float64),
        output_bias=np.asarray(0.0, dtype=np.float64),
    )
    result = execute_entk_readiness(x, y, checkpoint, task_id="zeros", checkpoint_id="zero")
    # Both preactivations are non-positive. Only the scalar output bias has a derivative.
    assert result["metrics"]["entk_feature_frobenius_norm"] == pytest.approx(np.sqrt(2.0))


def test_entk_json_preflight_bounds_keys_and_integers() -> None:
    with pytest.raises(ValueError, match="string-byte bound"):
        entk_module._json_preflight({"x" * (entk_module.MAX_JSON_STRING_BYTES + 1): None})
    with pytest.raises(ValueError, match="bounded integer"):
        entk_module._json_preflight(1 << 64)
