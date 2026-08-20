"""Hostile contracts for the official V-JEPA 2-AC qualification boundary."""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest

from alberta_framework.benchmarks import vjepa_external_qualification as lane

pytestmark = pytest.mark.unit

_SHA = "1" * 64


def _source() -> lane.SourceQualification:
    return lane.SourceQualification(
        repository=lane.VJEPA_REPOSITORY,
        commit=lane.VJEPA_COMMIT,
        source_archive_sha256="2" * 64,
        source_tree_sha256="3" * 64,
        source_archive_bytes=123_456,
        required_file_sha256=tuple(
            (path, f"{index + 4:x}" * 64)
            for index, path in enumerate(lane.REQUIRED_SOURCE_FILES)
        ),
    )


def _license() -> lane.LicenseQualification:
    return lane.LicenseQualification(
        source_license_id="MIT",
        source_license_sha256="4" * 64,
        checkpoint_terms_sha256="7" * 64,
        web_video_terms_sha256="8" * 64,
        robot_video_terms_sha256="9" * 64,
        review_identity="legal-review:test-fixture:v1",
        research_evaluation_allowed=True,
        redistribution_authority=False,
    )


def _checkpoint() -> lane.CheckpointQualification:
    return lane.CheckpointQualification(
        model_id=lane.VJEPA_MODEL_ID,
        artifact_url="https://dl.example.invalid/vjepa2-ac-vitg16.pt",
        checkpoint_sha256="a" * 64,
        checkpoint_bytes=10_000,
        terms_sha256="7" * 64,
        state_manifest_sha256="b" * 64,
        architecture_sha256="c" * 64,
        preprocessing_sha256="d" * 64,
        action_conditioning_sha256="e" * 64,
        image_size=lane.IMAGE_SIZE,
        context_frames=lane.CONTEXT_FRAMES,
        target_frames=lane.TARGET_FRAMES,
        tokens_per_frame=lane.TOKENS_PER_FRAME,
        token_dimension=lane.TOKEN_DIMENSION,
        action_dimension=lane.ACTION_DIMENSION,
        ema_target=True,
        frozen=True,
    )


def _dataset(
    role: str, digit: str, *, examples: int, materialized_bytes: int
) -> lane.DatasetQualification:
    return lane.DatasetQualification(
        role=role,
        name=f"fixture-{role}",
        revision="immutable-manifest-v1",
        manifest_sha256=digit * 64,
        terms_sha256=({"web_video_pretraining": "8", "robot_action_posttraining": "9"}[role]) * 64,
        examples=examples,
        materialized_bytes=materialized_bytes,
        duration_seconds=examples * 2,
    )


def _runtime() -> lane.RuntimeQualification:
    return lane.RuntimeQualification(
        image_digest="sha256:" + "1" * 64,
        dependency_lock_sha256="2" * 64,
        python="3.12.9",
        torch="2.7.1",
        cuda="12.8",
        accelerator="NVIDIA-H100-80GB-HBM3",
        deterministic_algorithms=True,
        network_disabled=True,
    )


def _provider() -> lane.ProviderQualification:
    return lane.ProviderQualification(
        schema=lane.PROVIDER_SCHEMA,
        implementation_sha256="3" * 64,
        executable_sha256="4" * 64,
        protocol_sha256="5" * 64,
        isolation_identity="oci:test-fixture:offline",
        imports_official_source=True,
        loads_exact_checkpoint=True,
    )


def _cost() -> lane.PretrainingResources:
    return lane.PretrainingResources(
        examples=30,
        updates=40,
        model_queries=50,
        dataset_bytes=3_000,
        checkpoint_bytes=10_000,
        source_bytes=123_456,
        scalar_flops=1_000_000,
        accelerator_ns=2_000_000,
        peak_bytes=20_000,
    )


def _plan() -> lane.VJEPAQualificationPlan:
    return lane.VJEPAQualificationPlan(
        schema=lane.PLAN_SCHEMA,
        source=_source(),
        license=_license(),
        checkpoint=_checkpoint(),
        web_video=_dataset("web_video_pretraining", "f", examples=10, materialized_bytes=1_000),
        robot_video=_dataset(
            "robot_action_posttraining", "0", examples=20, materialized_bytes=2_000
        ),
        runtime=_runtime(),
        provider=_provider(),
        pretraining=_cost(),
    )


class DeterministicProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.request: lane.VisualTokenRequest | None = None

    def infer(
        self,
        request: lane.VisualTokenRequest,
        clips: np.ndarray,
        actions: np.ndarray,
        states: np.ndarray,
    ) -> lane.VisualTokenOutput:
        self.calls += 1
        self.request = request
        assert clips.dtype == np.uint8
        assert actions.dtype == np.float32
        assert states.dtype == np.float32
        batch = clips.shape[0]
        context_shape = (
            batch,
            lane.CONTEXT_FRAMES * lane.TOKENS_PER_FRAME,
            lane.TOKEN_DIMENSION,
        )
        target_shape = (
            batch,
            lane.TARGET_FRAMES * lane.TOKENS_PER_FRAME,
            lane.TOKEN_DIMENSION,
        )
        context = np.zeros(context_shape, dtype=np.float32)
        predicted = np.full(target_shape, 0.25, dtype=np.float32)
        target = np.full(target_shape, 0.5, dtype=np.float32)
        return lane.VisualTokenOutput(
            checkpoint_sha256=request.checkpoint_sha256,
            preprocessing_sha256=request.preprocessing_sha256,
            action_conditioning_sha256=request.action_conditioning_sha256,
            provider_implementation_sha256=request.provider_implementation_sha256,
            provider_executable_sha256=request.provider_executable_sha256,
            provider_protocol_sha256=request.provider_protocol_sha256,
            runtime_image_digest=request.runtime_image_digest,
            context_tokens=context,
            predicted_target_tokens=predicted,
            ema_target_tokens=target,
            resources=lane.InferenceResources(
                examples=batch,
                checkpoint_loads=1,
                checkpoint_read_bytes=_checkpoint().checkpoint_bytes,
                encoder_queries=batch * (lane.CONTEXT_FRAMES + lane.TARGET_FRAMES),
                predictor_queries=batch * lane.TARGET_FRAMES,
                ema_target_queries=batch * lane.TARGET_FRAMES,
                input_bytes=clips.nbytes + actions.nbytes + states.nbytes,
                output_bytes=context.nbytes + predicted.nbytes + target.nbytes,
                scalar_flops=1_000_000,
                accelerator_ns=2_000_000,
                peak_bytes=(
                    _checkpoint().checkpoint_bytes
                    + clips.nbytes
                    + actions.nbytes
                    + states.nbytes
                    + context.nbytes
                    + predicted.nbytes
                    + target.nbytes
                ),
                persistent_bytes=_checkpoint().checkpoint_bytes,
            ),
        )


def test_visual_checkpoint_adapter_is_real_bounded_and_nonpromoting() -> None:
    provider = DeterministicProvider()
    receipt = lane.qualify_visual_token_adapter(_plan(), provider, seed=1_577_000, batch_size=2)

    assert provider.calls == 1
    assert provider.request is not None
    assert provider.request.threefry_key_data == lane.smoke_key_data(1_577_000)
    assert receipt.schema == lane.RECEIPT_SCHEMA
    assert receipt.outcome_scope == lane.OUTCOME_SCOPE
    assert receipt.visual_adapter_executed is True
    assert receipt.official_parity_claimed is False
    assert receipt.scientific_promotion_allowed is False
    assert receipt.mean_token_mse == pytest.approx(0.0625)
    assert lane.validate_receipt_payload(receipt.to_payload(), expected_plan=_plan()) == receipt


def test_official_source_and_adapter_geometry_match_the_pinned_vjepa_tree() -> None:
    assert lane.VJEPA_SOURCE_LICENSE_ID == "MIT"
    assert lane.IMAGE_SIZE == 256
    assert lane.TOKENS_PER_FRAME == 256
    clips, actions, states = lane.build_smoke_inputs(seed=1_577_000, batch_size=2)
    assert clips.shape == (2, 4, 256, 256, 3)
    assert actions.shape == (2, 3, 7)
    assert states.shape == (2, 3, 7)


def test_smoke_inputs_are_explicit_threefry_deterministic_and_distinct() -> None:
    first = lane.build_smoke_inputs(seed=1_577_000, batch_size=2)
    second = lane.build_smoke_inputs(seed=1_577_000, batch_size=2)
    other = lane.build_smoke_inputs(seed=1_577_001, batch_size=2)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert not np.array_equal(first[0], other[0])
    assert first[0].shape == (2, lane.CONTEXT_FRAMES + lane.TARGET_FRAMES, 256, 256, 3)
    assert first[1].shape == (2, lane.CONTEXT_FRAMES, lane.ACTION_DIMENSION)
    assert first[2].shape == (2, lane.CONTEXT_FRAMES, lane.ACTION_DIMENSION)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("commit", "0" * 40, "official immutable revision"),
        ("repository", "https://example.invalid/vjepa2.git", "official immutable revision"),
        ("required_file_sha256", (("README.md", _SHA),), "required source files"),
    ],
)
def test_source_pin_and_file_closure_fail_closed(field: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        source = dataclasses.replace(_source(), **{field: value})  # type: ignore[arg-type]
        dataclasses.replace(_plan(), source=source)


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"research_evaluation_allowed": False}, "research-evaluation"),
        ({"redistribution_authority": True}, "redistribution"),
        ({"checkpoint_terms_sha256": "8" * 64}, "checkpoint terms"),
        ({"source_license_id": "CC-BY-NC-4.0"}, "source license ID"),
    ],
)
def test_license_and_terms_are_bound(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        license = dataclasses.replace(_license(), **changes)  # type: ignore[arg-type]
        dataclasses.replace(_plan(), license=license)


@pytest.mark.parametrize(
    "component,changes,message",
    [
        ("checkpoint", {"ema_target": False}, "EMA"),
        ("checkpoint", {"frozen": False}, "frozen"),
        ("checkpoint", {"token_dimension": 1}, "official adapter geometry"),
        ("runtime", {"network_disabled": False}, "network"),
        ("runtime", {"deterministic_algorithms": False}, "deterministic"),
        ("provider", {"imports_official_source": False}, "official source"),
        ("provider", {"loads_exact_checkpoint": False}, "exact checkpoint"),
    ],
)
def test_checkpoint_runtime_and_provider_contracts_fail_closed(
    component: str, changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        value = dataclasses.replace(getattr(_plan(), component), **changes)
        dataclasses.replace(_plan(), **{component: value})


def test_pretraining_costs_match_assets_exactly() -> None:
    with pytest.raises(ValueError, match="dataset bytes"):
        dataclasses.replace(_plan(), pretraining=dataclasses.replace(_cost(), dataset_bytes=2_999))
    with pytest.raises(ValueError, match="checkpoint bytes"):
        dataclasses.replace(
            _plan(), pretraining=dataclasses.replace(_cost(), checkpoint_bytes=9_999)
        )
    with pytest.raises(ValueError, match="source bytes"):
        dataclasses.replace(_plan(), pretraining=dataclasses.replace(_cost(), source_bytes=123_455))


@pytest.mark.parametrize(
    "fault", ["dtype", "shape", "nan", "echo", "runtime", "undercharge", "peak", "alias"]
)
def test_hostile_provider_outputs_are_rejected(fault: str) -> None:
    class Hostile(DeterministicProvider):
        def infer(
            self,
            request: lane.VisualTokenRequest,
            clips: np.ndarray,
            actions: np.ndarray,
            states: np.ndarray,
        ) -> lane.VisualTokenOutput:
            result = super().infer(request, clips, actions, states)
            if fault == "dtype":
                return dataclasses.replace(
                    result, context_tokens=result.context_tokens.astype(np.float64)
                )
            if fault == "shape":
                return dataclasses.replace(
                    result,
                    predicted_target_tokens=result.predicted_target_tokens[:, :-1],
                )
            if fault == "nan":
                changed = result.ema_target_tokens.copy()
                changed[0, 0, 0] = np.nan
                return dataclasses.replace(result, ema_target_tokens=changed)
            if fault == "echo":
                return dataclasses.replace(result, checkpoint_sha256="0" * 64)
            if fault == "runtime":
                return dataclasses.replace(result, runtime_image_digest="sha256:" + "f" * 64)
            if fault == "undercharge":
                return dataclasses.replace(
                    result,
                    resources=dataclasses.replace(
                        result.resources, output_bytes=result.resources.output_bytes - 1
                    ),
                )
            if fault == "peak":
                return dataclasses.replace(
                    result,
                    resources=dataclasses.replace(
                        result.resources, peak_bytes=result.resources.persistent_bytes
                    ),
                )
            return dataclasses.replace(result, ema_target_tokens=result.predicted_target_tokens)

    with pytest.raises(ValueError):
        lane.qualify_visual_token_adapter(_plan(), Hostile(), seed=1_577_000, batch_size=2)


@pytest.mark.parametrize("fault", ["request", "clips", "actions", "states"])
def test_provider_cannot_rebind_or_mutate_the_attested_request(fault: str) -> None:
    class Mutating(DeterministicProvider):
        def infer(
            self,
            request: lane.VisualTokenRequest,
            clips: np.ndarray,
            actions: np.ndarray,
            states: np.ndarray,
        ) -> lane.VisualTokenOutput:
            result = super().infer(request, clips, actions, states)
            if fault == "request":
                object.__setattr__(request, "checkpoint_sha256", "f" * 64)
                return dataclasses.replace(result, checkpoint_sha256="f" * 64)
            target = {"clips": clips, "actions": actions, "states": states}[fault]
            target.setflags(write=True)
            target.flat[0] = 0 if target.flat[0] != 0 else 1
            return result

    with pytest.raises(ValueError, match="mutated"):
        lane.qualify_visual_token_adapter(_plan(), Mutating(), seed=1_577_000, batch_size=1)


def test_receipt_parser_is_pure_strict_and_current() -> None:
    receipt = lane.qualify_visual_token_adapter(
        _plan(), DeterministicProvider(), seed=1_577_000, batch_size=1
    )
    payload = receipt.to_payload()
    original = copy.deepcopy(payload)
    assert lane.validate_receipt_payload(payload, expected_plan=_plan()) == receipt
    assert payload == original

    expanded = copy.deepcopy(payload)
    expanded["extra"] = 1
    with pytest.raises(ValueError, match="receipt fields"):
        lane.validate_receipt_payload(expanded, expected_plan=_plan())
    forged = copy.deepcopy(payload)
    assert isinstance(forged["identity"], dict)
    forged["identity"]["workload_registry_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="current tree/runtime"):
        lane.validate_receipt_payload(forged, expected_plan=_plan())
    wrong_plan = dataclasses.replace(
        _plan(), checkpoint=dataclasses.replace(_checkpoint(), checkpoint_sha256="f" * 64)
    )
    with pytest.raises(ValueError, match="plan"):
        lane.validate_receipt_payload(payload, expected_plan=wrong_plan)

    forged_resources = copy.deepcopy(payload)
    assert isinstance(forged_resources["resources"], dict)
    forged_resources["resources"]["encoder_queries"] = 1
    with pytest.raises(ValueError, match="semantic counts"):
        lane.validate_receipt_payload(forged_resources, expected_plan=_plan())

    forged_checkpoint_cost = copy.deepcopy(payload)
    assert isinstance(forged_checkpoint_cost["resources"], dict)
    forged_checkpoint_cost["resources"]["checkpoint_read_bytes"] = 9_999
    with pytest.raises(ValueError, match="checkpoint-read bytes"):
        lane.validate_receipt_payload(forged_checkpoint_cost, expected_plan=_plan())


def test_runtime_registry_mutation_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lane, "_WORKLOAD_REGISTRY", ())
    with pytest.raises(ValueError, match="literal frozen V-JEPA contract"):
        lane.blocker_manifest()


def test_nested_instances_are_revalidated_after_object_level_tampering() -> None:
    plan = _plan()
    object.__setattr__(plan.runtime, "network_disabled", False)
    with pytest.raises(ValueError, match="network"):
        plan.__post_init__()


def test_blocker_manifest_is_honest_and_static() -> None:
    report = lane.blocker_manifest()
    assert report["issue"] == 1577
    assert report["ready"] is False
    assert report["external_execution_performed"] is False
    assert report["native_proxy_is_official_vjepa"] is False
    assert report["no_imported_pretraining_ablation"] == dict(
        lane.NO_IMPORTED_PRETRAINING_ABLATION
    )
    assert isinstance(report["blockers"], list)
    assert "official_assets_not_acquired" in report["blockers"]
