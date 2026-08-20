"""Qualification contract for genuine external frozen-feature extractors."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from alberta_framework.benchmarks.pretrained_feature_qualification import (
    OFFICIAL_METHODS,
    CheckpointIdentity,
    DatasetIdentity,
    ExtractorCost,
    ExtractorRequest,
    PretrainingCost,
    QualificationPlan,
    RuntimeIdentity,
    SourceIdentity,
    blocker_manifest,
    qualify_frozen_extractor,
)

pytestmark = pytest.mark.unit

H = "a" * 64


def _plan(method: str = "ranpac") -> QualificationPlan:
    spec = OFFICIAL_METHODS[method]
    pretraining = PretrainingCost(1_000, 100, 1_000, 8_000, 10_000, 2_000, 4096)
    if method == "randumb":
        pretraining = PretrainingCost(0, 0, 0, 0, 0, 0, 4096)
    return QualificationPlan(
        schema="asi.pretrained_feature_qualification.plan.v1",
        method=method,
        source=SourceIdentity(
            spec.repository,
            spec.commit,
            H,
            H,
            tuple((path, H) for path in spec.required_files),
        ),
        checkpoint=CheckpointIdentity(method, spec.artifact_role, H, 4096, H, H, 8, True),
        evaluation_dataset=DatasetIdentity("openml", "mnist_784", "1", H, 70_000),
        pretraining_dataset=DatasetIdentity(
            "none" if method == "randumb" else "provider",
            "random-initialization" if method == "randumb" else "pretraining-corpus",
            "none" if method == "randumb" else "v1",
            H,
            0 if method == "randumb" else 1_000,
        ),
        runtime=RuntimeIdentity("sha256:" + H, H, "3.9.18", "1.13.1", "0.14.1", True),
        pretraining_cost=pretraining,
    )


class Provider:
    def extract(self, request: ExtractorRequest, inputs: np.ndarray) -> np.ndarray:
        assert request.method == "ranpac"
        return np.ones((inputs.shape[0], request.output_dimension), dtype=np.float32)


def test_catalog_uses_exact_official_sources_and_distinguishes_random_artifact() -> None:
    assert OFFICIAL_METHODS["randumb"].commit == "14a51ee0c045bff642f6ffbfe481efa4d49a3033"
    assert OFFICIAL_METHODS["ranpac"].commit == "cf4b301d18b0c27db030f4371b72b768005ae58a"
    assert OFFICIAL_METHODS["prol"].commit == "bfff8418a4f603a24ae578f1e108bfac89af1e18"
    assert OFFICIAL_METHODS["randumb"].artifact_role == "frozen_random_initialization"
    assert OFFICIAL_METHODS["ranpac"].artifact_role == "pretrained_checkpoint"


def test_real_provider_interface_charges_resources_and_emits_nonpromoting_receipt() -> None:
    inputs = np.zeros((2, 224, 224, 3), dtype=np.float32)
    cost = ExtractorCost(2, 2, inputs.nbytes, 64, 100_000, 8192, 4096)
    receipt = qualify_frozen_extractor(_plan(), Provider(), inputs, cost)
    assert receipt.feature_shape == (2, 8)
    assert receipt.feature_dtype == "float32"
    assert receipt.cost == cost
    assert receipt.ceiling_claim_allowed is False
    assert receipt.feature_sha256 == receipt.revalidate(inputs)
    with pytest.raises(ValueError, match="cannot authorize"):
        replace(receipt, ceiling_claim_allowed=True)


def test_pretrained_methods_cannot_hide_zero_pretraining_cost() -> None:
    plan = _plan()
    with pytest.raises(ValueError, match="pretrained methods require positive"):
        replace(plan, pretraining_cost=PretrainingCost(0, 0, 0, 0, 0, 0, 4096))


def test_plan_requires_every_official_source_file_identity() -> None:
    plan = _plan()
    assert len(OFFICIAL_METHODS[plan.method].required_files) > 1
    with pytest.raises(ValueError, match="every required official file"):
        replace(
            plan,
            source=replace(
                plan.source,
                required_file_sha256=plan.source.required_file_sha256[:1],
            ),
        )


def test_pretraining_examples_must_match_the_attested_dataset() -> None:
    plan = _plan()
    with pytest.raises(ValueError, match="examples must cover"):
        replace(plan, pretraining_cost=replace(plan.pretraining_cost, examples=1))


def test_extractor_rejects_unbounded_dimensions_before_provider_execution() -> None:
    plan = _plan()
    unbounded = replace(
        plan,
        checkpoint=replace(plan.checkpoint, output_dimension=2**31),
    )
    inputs = np.zeros((1, 224, 224, 3), dtype=np.float32)
    cost = ExtractorCost(1, 1, inputs.nbytes, 2**33, 1, 1, 4096)

    class MustNotRun:
        def extract(self, request: ExtractorRequest, inputs: np.ndarray) -> np.ndarray:
            raise AssertionError("provider must not run for an unbounded request")

    with pytest.raises(ValueError, match="bounded smoke limit"):
        qualify_frozen_extractor(unbounded, MustNotRun(), inputs, cost)


def test_randumb_must_be_random_and_still_charge_checkpoint_and_extraction() -> None:
    plan = _plan("randumb")
    assert plan.pretraining_cost.examples == 0
    with pytest.raises(ValueError, match="RanDumb"):
        replace(plan, checkpoint=replace(plan.checkpoint, artifact_role="pretrained_checkpoint"))


@pytest.mark.parametrize("bad", [True, 1.0, -1])
def test_exact_resource_integers_are_enforced(bad: object) -> None:
    with pytest.raises(ValueError, match="exact non-negative integer"):
        PretrainingCost(bad, 0, 0, 0, 0, 0, 1)  # type: ignore[arg-type]


def test_provider_output_fails_closed_on_dtype_shape_and_nonfinite() -> None:
    inputs = np.zeros((2, 224, 224, 3), dtype=np.float32)
    cost = ExtractorCost(2, 2, inputs.nbytes, 64, 100_000, 8192, 4096)

    class Bad:
        def __init__(self, value: np.ndarray) -> None:
            self.value = value

        def extract(self, request: ExtractorRequest, inputs: np.ndarray) -> np.ndarray:
            return self.value

    for value in (
        np.ones((2, 8), dtype=np.float64),
        np.ones((2, 7), dtype=np.float32),
        np.full((2, 8), np.nan, dtype=np.float32),
    ):
        with pytest.raises(ValueError):
            qualify_frozen_extractor(_plan(), Bad(value), inputs, cost)


def test_execution_cost_must_exactly_match_inputs_outputs_and_checkpoint() -> None:
    inputs = np.zeros((2, 224, 224, 3), dtype=np.float32)
    good = ExtractorCost(2, 2, inputs.nbytes, 64, 100_000, 8192, 4096)
    for cost in (
        replace(good, queries=1),
        replace(good, input_bytes=1),
        replace(good, output_bytes=1),
        replace(good, persistent_bytes=1),
    ):
        with pytest.raises(ValueError):
            qualify_frozen_extractor(_plan(), Provider(), inputs, cost)


def test_invalid_cost_fails_before_external_provider_execution() -> None:
    inputs = np.zeros((1, 224, 224, 3), dtype=np.float32)
    invalid = ExtractorCost(1, 1, inputs.nbytes, 32, 0, 1, 4096)

    class MustNotRun:
        def extract(self, request: ExtractorRequest, inputs: np.ndarray) -> np.ndarray:
            raise AssertionError("provider must not run before cost preflight")

    with pytest.raises(ValueError, match="compute and peak bytes"):
        qualify_frozen_extractor(_plan(), MustNotRun(), inputs, invalid)


def test_blocker_manifest_is_fail_closed_and_does_not_upgrade_proxies() -> None:
    manifest = blocker_manifest()
    assert manifest["schema"] == "asi.pretrained_feature_qualification.blockers.v1"
    assert manifest["ready"] is False
    assert manifest["proxy_arms_are_ceilings"] is False
    assert set(manifest["methods"]) == {"randumb", "ranpac", "prol"}
    assert "official_artifacts_not_acquired" in manifest["blockers"]
    assert "full_matched_ipmnist_ceiling_not_executed" in manifest["blockers"]
