from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from alberta_framework.evaluation.optimization_readiness import (
    OPTIMIZATION_READINESS_PROTOCOL,
    energy_rank,
    estimate_optimization_readiness,
    validate_development_result,
    validate_matched_development_results,
)


def test_readiness_matches_paper_empirical_equations() -> None:
    gradients = np.asarray([[1.0, 0.0], [3.0, 0.0]], dtype=np.float64)
    report = estimate_optimization_readiness(
        loss=2.0,
        full_validation_gradient=np.asarray([1.0, 0.0]),
        batch_gradients=gradients,
    )
    assert report.gradient_squared_norm == pytest.approx(1.0)
    assert report.expected_batch_gradient_squared_norm == pytest.approx(5.0)
    assert report.gradient_strength == pytest.approx(0.5)
    assert report.gradient_reliability == pytest.approx(0.2)
    assert report.optimization_readiness == pytest.approx(0.1)
    assert report.gradient_norm == pytest.approx(1.0)


def test_zero_and_mechanism_off_reductions() -> None:
    zero = estimate_optimization_readiness(
        loss=0.0,
        full_validation_gradient=np.zeros(3, dtype=np.float64),
        batch_gradients=np.zeros((2, 3), dtype=np.float64),
    )
    assert zero.optimization_readiness == 0.0
    strength_only = estimate_optimization_readiness(
        loss=2.0,
        full_validation_gradient=np.asarray([2.0]),
        batch_gradients=np.asarray([[1.0], [3.0]]),
        include_reliability=False,
    )
    assert strength_only.optimization_readiness == strength_only.gradient_strength


def test_readiness_fails_closed_on_overflowing_finite_inputs() -> None:
    with pytest.raises(ValueError, match="finite float64"):
        estimate_optimization_readiness(
            loss=1.0,
            full_validation_gradient=np.asarray([1e308]),
            batch_gradients=np.asarray([[1e308], [1e308]]),
        )


def test_readiness_rejects_mismatched_parameter_counts() -> None:
    with pytest.raises(ValueError, match="parameter axis"):
        estimate_optimization_readiness(
            loss=1.0,
            full_validation_gradient=np.ones(3),
            batch_gradients=np.ones((2, 2)),
        )


def test_energy_rank_baseline_and_exact_threshold() -> None:
    matrix = np.diag([3.0, 1.0])
    assert energy_rank(matrix, threshold=0.9) == 1
    assert energy_rank(matrix, threshold=0.99) == 2
    assert energy_rank(np.zeros((2, 2)), threshold=0.99) == 0
    rounding_case = np.random.default_rng(7).normal(size=(10, 10))
    assert energy_rank(rounding_case, threshold=1.0) == 10


@pytest.mark.parametrize("loss", [-1.0, float("nan"), True])
def test_readiness_rejects_invalid_loss(loss: object) -> None:
    with pytest.raises(ValueError, match="loss"):
        estimate_optimization_readiness(  # type: ignore[arg-type]
            loss=loss,
            full_validation_gradient=np.ones(2),
            batch_gradients=np.ones((2, 2)),
        )


def test_readiness_rejects_mismatched_parameter_axes() -> None:
    with pytest.raises(ValueError, match="parameter axis"):
        estimate_optimization_readiness(
            loss=1.0,
            full_validation_gradient=np.ones(3),
            batch_gradients=np.ones((2, 2)),
        )


def test_readiness_rejects_nonrepresentable_squared_norms() -> None:
    with pytest.raises(ValueError, match="finite float64"):
        estimate_optimization_readiness(
            loss=1.0,
            full_validation_gradient=np.asarray([1e308]),
            batch_gradients=np.asarray([[1e308]]),
        )


def test_energy_rank_is_scale_stable() -> None:
    assert energy_rank(np.diag([9e307, 1e307]), threshold=0.99) == 2


def test_protocol_is_prospective_and_nonpromoting() -> None:
    assert OPTIMIZATION_READINESS_PROTOCOL["paper_revision"] == "arXiv:2605.09044v1"
    assert OPTIMIZATION_READINESS_PROTOCOL["population_gradient_estimator"] == (
        "full_validation_set_gradient"
    )
    assert OPTIMIZATION_READINESS_PROTOCOL["reference_reliability_batch_count"] == 128
    assert OPTIMIZATION_READINESS_PROTOCOL["reference_reliability_batch_size"] == 4
    assert OPTIMIZATION_READINESS_PROTOCOL["execution_protocol_required"] is True
    assert OPTIMIZATION_READINESS_PROTOCOL["diagnostics"] == (
        "optimization_readiness",
        "gradient_norm",
        "representation_energy_rank_0.99",
        "curvature_energy_rank_0.99",
        "parameter_norm",
    )


def _result_payload(*, arm_id: str = "candidate") -> dict[str, object]:
    return {
        "schema": "asi.optimization-readiness.development-result.v1",
        "comparison_id": "or-dev-001",
        "arm_id": arm_id,
        "protocol": {
            "schema": "asi.optimization-readiness.protocol.v1",
            "seed": 7,
            "checkpoint": "checkpoint-0042",
            "task": "ipmnist-permutation-3",
            "updates": 100,
            "observations": 10_000,
            "mini_batch_size": 4,
            "diagnostic_batch_count": 128,
            "allowed_boundary_information": ["task_start"],
            "allowed_task_information": ["labels_for_current_validation_task"],
        },
        "resources": {
            "schema": "asi.optimization-readiness.resources.v1",
            "persistent_bytes": 4096,
            "environment_steps": 0,
            "data_steps": 10_000,
            "model_queries": 10_128,
            "timing_seconds": 1.25,
            "timing_is_telemetry_only": True,
        },
        "metrics": {
            "optimization_readiness": 0.25,
            "gradient_norm": 0.5,
            "representation_energy_rank_0_99": 8,
            "curvature_energy_rank_0_99": 5,
            "parameter_norm": 12.0,
            "future_relative_loss_reduction": -0.1,
        },
        "outcome": "rejected",
        "outcome_retained": True,
        "development_only": True,
        "scientific_promotion_allowed": False,
    }


def test_strict_result_and_resource_receipt_validation() -> None:
    receipt = validate_development_result(_result_payload())
    assert receipt.outcome == "rejected"
    assert receipt.resources.data_steps == 10_000
    assert receipt.protocol.allowed_boundary_information == ("task_start",)

    payload = _result_payload()
    resources = payload["resources"]
    assert isinstance(resources, dict)
    resources["unaccounted_gpu_queries"] = 1
    with pytest.raises(ValueError, match="resources keys must be exactly"):
        validate_development_result(payload)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("resources", "persistent_bytes"), -1, "persistent_bytes"),
        (("resources", "timing_seconds"), float("nan"), "timing_seconds"),
        (("resources", "timing_is_telemetry_only"), False, "telemetry"),
        (("scientific_promotion_allowed",), True, "scientific_promotion_allowed"),
        (("outcome_retained",), False, "outcome_retained"),
    ],
)
def test_result_validator_fails_closed(
    path: tuple[str, ...], value: object, match: str
) -> None:
    payload = _result_payload()
    target = payload
    for key in path[:-1]:
        nested = target[key]
        assert isinstance(nested, dict)
        target = nested
    target[path[-1]] = value
    with pytest.raises(ValueError, match=match):
        validate_development_result(payload)


def test_matched_result_validation_enforces_all_axes() -> None:
    candidate = _result_payload()
    control = _result_payload(arm_id="control")
    receipts = validate_matched_development_results([candidate, control])
    assert len(receipts) == 2

    mismatched = deepcopy(control)
    protocol = mismatched["protocol"]
    assert isinstance(protocol, dict)
    protocol["observations"] = 9_999
    with pytest.raises(ValueError, match="matched protocol axes"):
        validate_matched_development_results([candidate, mismatched])


def test_protocol_records_estimator_differences_and_nonpromotion() -> None:
    assert OPTIMIZATION_READINESS_PROTOCOL["paper_revision"] == "arXiv:2605.09044v1"
    assert (
        OPTIMIZATION_READINESS_PROTOCOL["official_code_revision"]
        == "none-cited-in-arxiv-v1-as-of-2026-08-17"
    )
    assert (
        OPTIMIZATION_READINESS_PROTOCOL["estimator"]
        == "appendix-c.1-full-gradient-plus-independent-mini-batches"
    )
    assert OPTIMIZATION_READINESS_PROTOCOL["asi_protocol_differences"]
    assert OPTIMIZATION_READINESS_PROTOCOL["development_only"] is True
    assert OPTIMIZATION_READINESS_PROTOCOL["scientific_promotion_allowed"] is False
    assert OPTIMIZATION_READINESS_PROTOCOL["completed_result_exists"] is False
    assert OPTIMIZATION_READINESS_PROTOCOL["outcome_retention_required"] is True
