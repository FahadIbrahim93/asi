"""Fail-closed external V-JEPA 2-AC visual-token qualification.

This module does not download, import, or execute V-JEPA itself.  It defines a
bounded adapter that an isolated official-code provider can implement.  Unlike
ASI's small state-space JEPA-inspired controls, the adapter consumes real
``uint8`` video clips plus action and robot-state tensors and returns context,
predicted-target, and EMA-target visual tokens bound to one exact checkpoint.

Passing this smoke is qualification evidence only.  It is not checkpoint
parity, a paper reproduction, robot evidence, or scientific evidence.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import operator
import sys
from typing import Any, Final, Protocol, SupportsIndex, cast

import jax.random as jr
import numpy as np

import alberta_framework.benchmarks.external_qualification as external_qualification_module
from alberta_framework.benchmarks.qualification_provenance import (
    QualificationIdentity,
    collect_qualification_identity,
    identity_from_payload,
    require_current_identity,
)

PLAN_SCHEMA: Final = "asi.vjepa2_ac.qualification.plan.v1"
RECEIPT_SCHEMA: Final = "asi.vjepa2_ac.visual_token_smoke.v1"
PROVIDER_SCHEMA: Final = "asi.vjepa2_ac.provider.v1"
QUALIFICATION_ISSUE: Final = 1577
QUALIFICATION_LANE_ID: Final = "jepa-transfer"
VJEPA_PAPER: Final = "arXiv:2506.09985v1"
JEPA_WM_PAPER: Final = "arXiv:2512.24497v3"
VJEPA_REPOSITORY: Final = "https://github.com/facebookresearch/vjepa2.git"
VJEPA_COMMIT: Final = "204698b45b3712590f06245fbfba32d3be539812"
JEPA_WM_REPOSITORY: Final = "https://github.com/facebookresearch/jepa-wms.git"
JEPA_WM_COMMIT: Final = "13cf1d9c7e476f53c17714d2e0f1dc239a883ce0"
VJEPA_MODEL_ID: Final = "vjepa2-ac-vitg"
VJEPA_SOURCE_LICENSE_ID: Final = "MIT"
REQUIRED_SOURCE_FILES: Final = ("LICENSE", "README.md")

# The adapter geometry is deliberately one small normalization surface, not a
# claim that every internal tensor in the official implementation has this
# shape.  An official provider must normalize its selected ViT-g/16 features to
# this declared context/target token interface.
IMAGE_SIZE: Final = 256
CONTEXT_FRAMES: Final = 3
TARGET_FRAMES: Final = 1
TOKENS_PER_FRAME: Final = 256
TOKEN_DIMENSION: Final = 1408
ACTION_DIMENSION: Final = 7
OUTCOME_SCOPE: Final = "official_vjepa2_ac_visual_token_adapter_smoke_only"
NO_IMPORTED_PRETRAINING_ABLATION: Final = (
    ("ablation_id", "asi_visual_vjepa2_ac_architecture_from_scratch"),
    ("architecture_and_adapter_geometry_matched", True),
    ("initialization_rng", "threefry2x32"),
    ("imported_checkpoint_bytes", 0),
    ("imported_pretraining_bytes", 0),
    ("imported_pretraining_examples", 0),
    ("training_data_scope", "asi_generated_or_separately_authorized_visual_stream_only"),
    ("comparison_scope", "matched_online_queries_peak_bytes_and_evaluation_workload"),
    ("pretraining_compute_matched", False),
    ("scientific_promotion_allowed", False),
)

_HEX = frozenset("0123456789abcdef")
_MAX_TEXT_BYTES = 512
_MAX_BATCH = 4
_MAX_TENSOR_BYTES = 64 * 1024 * 1024
_MAX_RESOURCE = 2**63 - 1

_WORKLOAD_REGISTRY = (
    ("action_dimension", ACTION_DIMENSION),
    ("action_dtype", "float32"),
    ("action_frames", CONTEXT_FRAMES),
    ("context_frames", CONTEXT_FRAMES),
    ("image_dtype", "uint8"),
    ("image_size", IMAGE_SIZE),
    ("max_batch", _MAX_BATCH),
    ("model_id", VJEPA_MODEL_ID),
    ("no_imported_pretraining_ablation", NO_IMPORTED_PRETRAINING_ABLATION),
    ("outcome_scope", OUTCOME_SCOPE),
    ("provider_schema", PROVIDER_SCHEMA),
    ("state_dimension", ACTION_DIMENSION),
    ("state_dtype", "float32"),
    ("state_frames", CONTEXT_FRAMES),
    ("target_frames", TARGET_FRAMES),
    ("token_dimension", TOKEN_DIMENSION),
    ("token_dtype", "float32"),
    ("tokens_per_frame", TOKENS_PER_FRAME),
)
_PAPER_REGISTRY = (
    ("jepa_wm_commit", JEPA_WM_COMMIT),
    ("jepa_wm_paper", JEPA_WM_PAPER),
    ("jepa_wm_repository", JEPA_WM_REPOSITORY),
    ("vjepa_commit", VJEPA_COMMIT),
    ("vjepa_paper", VJEPA_PAPER),
    ("vjepa_repository", VJEPA_REPOSITORY),
)


def _text(value: object, name: str, maximum: int = _MAX_TEXT_BYTES) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be non-empty exact text")
    try:
        size = len(value.encode("utf-8", errors="strict"))
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must be valid UTF-8") from error
    if size > maximum:
        raise ValueError(f"{name} exceeds its UTF-8 byte bound")
    return value


def _sha(value: object, name: str) -> str:
    result = _text(value, name, 64)
    if len(result) != 64 or any(character not in _HEX for character in result):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return result


def _integer(
    value: object, name: str, *, minimum: int = 0, maximum: int = _MAX_RESOURCE
) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an exact integer")
    result = operator.index(cast(SupportsIndex, value))
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} lies outside the qualification bound")
    return result


def _exact_bool(value: object, name: str, expected: bool) -> None:
    if type(value) is not bool or value is not expected:
        raise ValueError(f"{name} must be exactly {expected}")


def _current_identity() -> QualificationIdentity:
    _require_authoritative_plan()
    return collect_qualification_identity(
        lane_module=sys.modules[__name__],
        dependency_modules=(external_qualification_module,),
        workload_registry=_WORKLOAD_REGISTRY,
        paper_registry=_PAPER_REGISTRY,
    )


def _require_authoritative_plan() -> None:
    literal_constants = (
        "asi.vjepa2_ac.qualification.plan.v1",
        "asi.vjepa2_ac.visual_token_smoke.v1",
        "asi.vjepa2_ac.provider.v1",
        1577,
        "jepa-transfer",
        "arXiv:2506.09985v1",
        "arXiv:2512.24497v3",
        "https://github.com/facebookresearch/vjepa2.git",
        "204698b45b3712590f06245fbfba32d3be539812",
        "https://github.com/facebookresearch/jepa-wms.git",
        "13cf1d9c7e476f53c17714d2e0f1dc239a883ce0",
        "vjepa2-ac-vitg",
        "MIT",
        ("LICENSE", "README.md"),
        256,
        3,
        1,
        256,
        1408,
        7,
        "official_vjepa2_ac_visual_token_adapter_smoke_only",
        (
            ("ablation_id", "asi_visual_vjepa2_ac_architecture_from_scratch"),
            ("architecture_and_adapter_geometry_matched", True),
            ("initialization_rng", "threefry2x32"),
            ("imported_checkpoint_bytes", 0),
            ("imported_pretraining_bytes", 0),
            ("imported_pretraining_examples", 0),
            (
                "training_data_scope",
                "asi_generated_or_separately_authorized_visual_stream_only",
            ),
            (
                "comparison_scope",
                "matched_online_queries_peak_bytes_and_evaluation_workload",
            ),
            ("pretraining_compute_matched", False),
            ("scientific_promotion_allowed", False),
        ),
    )
    current_constants = (
        PLAN_SCHEMA,
        RECEIPT_SCHEMA,
        PROVIDER_SCHEMA,
        QUALIFICATION_ISSUE,
        QUALIFICATION_LANE_ID,
        VJEPA_PAPER,
        JEPA_WM_PAPER,
        VJEPA_REPOSITORY,
        VJEPA_COMMIT,
        JEPA_WM_REPOSITORY,
        JEPA_WM_COMMIT,
        VJEPA_MODEL_ID,
        VJEPA_SOURCE_LICENSE_ID,
        REQUIRED_SOURCE_FILES,
        IMAGE_SIZE,
        CONTEXT_FRAMES,
        TARGET_FRAMES,
        TOKENS_PER_FRAME,
        TOKEN_DIMENSION,
        ACTION_DIMENSION,
        OUTCOME_SCOPE,
        NO_IMPORTED_PRETRAINING_ABLATION,
    )
    literal_workload_registry = (
        ("action_dimension", 7),
        ("action_dtype", "float32"),
        ("action_frames", 3),
        ("context_frames", 3),
        ("image_dtype", "uint8"),
        ("image_size", 256),
        ("max_batch", 4),
        ("model_id", "vjepa2-ac-vitg"),
        ("no_imported_pretraining_ablation", literal_constants[-1]),
        ("outcome_scope", "official_vjepa2_ac_visual_token_adapter_smoke_only"),
        ("provider_schema", "asi.vjepa2_ac.provider.v1"),
        ("state_dimension", 7),
        ("state_dtype", "float32"),
        ("state_frames", 3),
        ("target_frames", 1),
        ("token_dimension", 1408),
        ("token_dtype", "float32"),
        ("tokens_per_frame", 256),
    )
    literal_paper_registry = (
        ("jepa_wm_commit", "13cf1d9c7e476f53c17714d2e0f1dc239a883ce0"),
        ("jepa_wm_paper", "arXiv:2512.24497v3"),
        ("jepa_wm_repository", "https://github.com/facebookresearch/jepa-wms.git"),
        ("vjepa_commit", "204698b45b3712590f06245fbfba32d3be539812"),
        ("vjepa_paper", "arXiv:2506.09985v1"),
        ("vjepa_repository", "https://github.com/facebookresearch/vjepa2.git"),
    )
    if (
        current_constants != literal_constants
        or _WORKLOAD_REGISTRY != literal_workload_registry
        or _PAPER_REGISTRY != literal_paper_registry
    ):
        raise ValueError("runtime state differs from the literal frozen V-JEPA contract")
    plan = external_qualification_module.qualification_plan(QUALIFICATION_ISSUE)
    expected = ((JEPA_WM_REPOSITORY, JEPA_WM_COMMIT), (VJEPA_REPOSITORY, VJEPA_COMMIT))
    actual = tuple((revision.repository, revision.commit) for revision in plan.code_revisions)
    if (
        plan.lane_id != QUALIFICATION_LANE_ID
        or plan.paper_revisions != (JEPA_WM_PAPER, VJEPA_PAPER)
        or actual != expected
        or "no_imported_pretraining_ablation_defined" not in plan.required_gates
    ):
        raise ValueError("V-JEPA authority differs from the external qualification plan")


@dataclasses.dataclass(frozen=True, slots=True)
class SourceQualification:
    repository: str
    commit: str
    source_archive_sha256: str
    source_tree_sha256: str
    source_archive_bytes: int
    required_file_sha256: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _text(self.repository, "source repository")
        if len(self.commit) != 40 or any(character not in _HEX for character in self.commit):
            raise ValueError("source commit must be a full lowercase Git commit ID")
        _sha(self.source_archive_sha256, "source archive SHA-256")
        _sha(self.source_tree_sha256, "source tree SHA-256")
        _integer(self.source_archive_bytes, "source archive bytes", minimum=1)
        if type(self.required_file_sha256) is not tuple:
            raise ValueError("required source file identities must be an exact tuple")
        paths: list[str] = []
        for entry in self.required_file_sha256:
            if type(entry) is not tuple or len(entry) != 2:
                raise ValueError("required source file identities must be exact pairs")
            paths.append(_text(entry[0], "required source path", 256))
            _sha(entry[1], "required source file SHA-256")
        if tuple(paths) != REQUIRED_SOURCE_FILES:
            raise ValueError("required source files differ from the frozen closure")


@dataclasses.dataclass(frozen=True, slots=True)
class LicenseQualification:
    source_license_id: str
    source_license_sha256: str
    checkpoint_terms_sha256: str
    web_video_terms_sha256: str
    robot_video_terms_sha256: str
    review_identity: str
    research_evaluation_allowed: bool
    redistribution_authority: bool

    def __post_init__(self) -> None:
        _text(self.source_license_id, "source license ID", 128)
        if self.source_license_id != VJEPA_SOURCE_LICENSE_ID:
            raise ValueError("source license ID differs from the pinned source license")
        for name in (
            "source_license_sha256",
            "checkpoint_terms_sha256",
            "web_video_terms_sha256",
            "robot_video_terms_sha256",
        ):
            _sha(getattr(self, name), name)
        _text(self.review_identity, "license review identity")
        _exact_bool(
            self.research_evaluation_allowed,
            "research-evaluation license review",
            True,
        )
        _exact_bool(self.redistribution_authority, "redistribution authority", False)


@dataclasses.dataclass(frozen=True, slots=True)
class CheckpointQualification:
    model_id: str
    artifact_url: str
    checkpoint_sha256: str
    checkpoint_bytes: int
    terms_sha256: str
    state_manifest_sha256: str
    architecture_sha256: str
    preprocessing_sha256: str
    action_conditioning_sha256: str
    image_size: int
    context_frames: int
    target_frames: int
    tokens_per_frame: int
    token_dimension: int
    action_dimension: int
    ema_target: bool
    frozen: bool

    def __post_init__(self) -> None:
        _text(self.model_id, "checkpoint model ID", 128)
        url = _text(self.artifact_url, "checkpoint artifact URL")
        if not url.startswith("https://") or "@" in url.split("/", 3)[2]:
            raise ValueError("checkpoint artifact URL must be credential-free HTTPS")
        for name in (
            "checkpoint_sha256",
            "terms_sha256",
            "state_manifest_sha256",
            "architecture_sha256",
            "preprocessing_sha256",
            "action_conditioning_sha256",
        ):
            _sha(getattr(self, name), name)
        _integer(self.checkpoint_bytes, "checkpoint bytes", minimum=1)
        for name in (
            "image_size",
            "context_frames",
            "target_frames",
            "tokens_per_frame",
            "token_dimension",
            "action_dimension",
        ):
            _integer(getattr(self, name), name, minimum=1, maximum=1 << 20)
        _exact_bool(self.ema_target, "EMA target encoder", True)
        _exact_bool(self.frozen, "frozen checkpoint", True)


@dataclasses.dataclass(frozen=True, slots=True)
class DatasetQualification:
    role: str
    name: str
    revision: str
    manifest_sha256: str
    terms_sha256: str
    examples: int
    materialized_bytes: int
    duration_seconds: int

    def __post_init__(self) -> None:
        if self.role not in {"web_video_pretraining", "robot_action_posttraining"}:
            raise ValueError("dataset role is not supported")
        _text(self.name, "dataset name")
        _text(self.revision, "dataset revision")
        _sha(self.manifest_sha256, "dataset manifest SHA-256")
        _sha(self.terms_sha256, "dataset terms SHA-256")
        _integer(self.examples, "dataset examples", minimum=1)
        _integer(self.materialized_bytes, "dataset bytes", minimum=1)
        _integer(self.duration_seconds, "dataset duration seconds", minimum=1)


@dataclasses.dataclass(frozen=True, slots=True)
class RuntimeQualification:
    image_digest: str
    dependency_lock_sha256: str
    python: str
    torch: str
    cuda: str
    accelerator: str
    deterministic_algorithms: bool
    network_disabled: bool

    def __post_init__(self) -> None:
        image = _text(self.image_digest, "runtime image digest", 128)
        if not image.startswith("sha256:"):
            raise ValueError("runtime image digest must use sha256")
        _sha(image[7:], "runtime image digest")
        _sha(self.dependency_lock_sha256, "dependency lock SHA-256")
        for name in ("python", "torch", "cuda", "accelerator"):
            _text(getattr(self, name), name, 128)
        _exact_bool(self.deterministic_algorithms, "deterministic algorithms", True)
        _exact_bool(self.network_disabled, "network-disabled runtime", True)


@dataclasses.dataclass(frozen=True, slots=True)
class ProviderQualification:
    schema: str
    implementation_sha256: str
    executable_sha256: str
    protocol_sha256: str
    isolation_identity: str
    imports_official_source: bool
    loads_exact_checkpoint: bool

    def __post_init__(self) -> None:
        if self.schema != PROVIDER_SCHEMA:
            raise ValueError("unsupported provider schema")
        for name in ("implementation_sha256", "executable_sha256", "protocol_sha256"):
            _sha(getattr(self, name), name)
        _text(self.isolation_identity, "provider isolation identity")
        _exact_bool(self.imports_official_source, "provider imports official source", True)
        _exact_bool(self.loads_exact_checkpoint, "provider loads exact checkpoint", True)


@dataclasses.dataclass(frozen=True, slots=True)
class PretrainingResources:
    examples: int
    updates: int
    model_queries: int
    dataset_bytes: int
    checkpoint_bytes: int
    source_bytes: int
    scalar_flops: int
    accelerator_ns: int
    peak_bytes: int

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            _integer(getattr(self, field.name), field.name, minimum=1)


@dataclasses.dataclass(frozen=True, slots=True)
class VJEPAQualificationPlan:
    schema: str
    source: SourceQualification
    license: LicenseQualification
    checkpoint: CheckpointQualification
    web_video: DatasetQualification
    robot_video: DatasetQualification
    runtime: RuntimeQualification
    provider: ProviderQualification
    pretraining: PretrainingResources

    def __post_init__(self) -> None:
        if self.schema != PLAN_SCHEMA:
            raise ValueError("unsupported V-JEPA qualification plan schema")
        for value, cls, name in (
            (self.source, SourceQualification, "source"),
            (self.license, LicenseQualification, "license"),
            (self.checkpoint, CheckpointQualification, "checkpoint"),
            (self.web_video, DatasetQualification, "web_video"),
            (self.robot_video, DatasetQualification, "robot_video"),
            (self.runtime, RuntimeQualification, "runtime"),
            (self.provider, ProviderQualification, "provider"),
            (self.pretraining, PretrainingResources, "pretraining"),
        ):
            if type(value) is not cls:
                raise ValueError(f"{name} must have its exact qualification type")
            value.__post_init__()
        if (self.source.repository, self.source.commit) != (VJEPA_REPOSITORY, VJEPA_COMMIT):
            raise ValueError("source does not match the official immutable revision")
        required_hashes = dict(self.source.required_file_sha256)
        if required_hashes["LICENSE"] != self.license.source_license_sha256:
            raise ValueError("source license hash differs from the qualified source closure")
        checkpoint = self.checkpoint
        geometry = (
            checkpoint.model_id,
            checkpoint.image_size,
            checkpoint.context_frames,
            checkpoint.target_frames,
            checkpoint.tokens_per_frame,
            checkpoint.token_dimension,
            checkpoint.action_dimension,
        )
        if geometry != (
            VJEPA_MODEL_ID,
            IMAGE_SIZE,
            CONTEXT_FRAMES,
            TARGET_FRAMES,
            TOKENS_PER_FRAME,
            TOKEN_DIMENSION,
            ACTION_DIMENSION,
        ):
            raise ValueError("checkpoint differs from the official adapter geometry")
        if checkpoint.terms_sha256 != self.license.checkpoint_terms_sha256:
            raise ValueError("checkpoint terms differ from the license qualification")
        if self.web_video.role != "web_video_pretraining":
            raise ValueError("web-video dataset has the wrong role")
        if self.robot_video.role != "robot_action_posttraining":
            raise ValueError("robot-video dataset has the wrong role")
        if self.web_video.terms_sha256 != self.license.web_video_terms_sha256:
            raise ValueError("web-video terms differ from the license qualification")
        if self.robot_video.terms_sha256 != self.license.robot_video_terms_sha256:
            raise ValueError("robot-video terms differ from the license qualification")
        if self.web_video.manifest_sha256 == self.robot_video.manifest_sha256:
            raise ValueError("pretraining and robot-video manifests must be distinct")
        resources = self.pretraining
        if resources.dataset_bytes != (
            self.web_video.materialized_bytes + self.robot_video.materialized_bytes
        ):
            raise ValueError("pretraining dataset bytes are not charged exactly")
        if resources.checkpoint_bytes != checkpoint.checkpoint_bytes:
            raise ValueError("pretraining checkpoint bytes are not charged exactly")
        if resources.source_bytes != self.source.source_archive_bytes:
            raise ValueError("pretraining source bytes are not charged exactly")
        if resources.examples < self.web_video.examples + self.robot_video.examples:
            raise ValueError("pretraining examples do not cover both dataset manifests")
        if resources.peak_bytes < checkpoint.checkpoint_bytes:
            raise ValueError("pretraining peak bytes cannot exclude the checkpoint")


@dataclasses.dataclass(frozen=True, slots=True)
class VisualTokenRequest:
    schema: str
    model_id: str
    checkpoint_sha256: str
    preprocessing_sha256: str
    action_conditioning_sha256: str
    provider_implementation_sha256: str
    provider_executable_sha256: str
    provider_protocol_sha256: str
    runtime_image_digest: str
    clip_shape: tuple[int, int, int, int, int]
    action_shape: tuple[int, int, int]
    state_shape: tuple[int, int, int]
    context_token_shape: tuple[int, int, int]
    target_token_shape: tuple[int, int, int]
    threefry_key_data: tuple[int, int]

    def __post_init__(self) -> None:
        if self.schema != PROVIDER_SCHEMA or self.model_id != VJEPA_MODEL_ID:
            raise ValueError("visual-token request schema or model is unsupported")
        for name in (
            "checkpoint_sha256",
            "preprocessing_sha256",
            "action_conditioning_sha256",
            "provider_implementation_sha256",
            "provider_executable_sha256",
            "provider_protocol_sha256",
        ):
            _sha(getattr(self, name), name)
        image = _text(self.runtime_image_digest, "request runtime image digest", 128)
        if not image.startswith("sha256:"):
            raise ValueError("request runtime image digest must use sha256")
        _sha(image[7:], "request runtime image digest")
        for name, value, rank in (
            ("clip_shape", self.clip_shape, 5),
            ("action_shape", self.action_shape, 3),
            ("state_shape", self.state_shape, 3),
            ("context_token_shape", self.context_token_shape, 3),
            ("target_token_shape", self.target_token_shape, 3),
        ):
            if type(value) is not tuple or len(value) != rank:
                raise ValueError(f"{name} must be an exact rank-{rank} shape")
            for dimension in value:
                _integer(dimension, f"{name} dimension", minimum=1, maximum=1 << 20)
        if type(self.threefry_key_data) is not tuple or len(self.threefry_key_data) != 2:
            raise ValueError("Threefry key data must be an exact pair")
        for word in self.threefry_key_data:
            _integer(word, "Threefry key word", maximum=2**32 - 1)


@dataclasses.dataclass(frozen=True, slots=True)
class InferenceResources:
    examples: int
    checkpoint_loads: int
    checkpoint_read_bytes: int
    encoder_queries: int
    predictor_queries: int
    ema_target_queries: int
    input_bytes: int
    output_bytes: int
    scalar_flops: int
    accelerator_ns: int
    peak_bytes: int
    persistent_bytes: int

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            _integer(getattr(self, field.name), field.name, minimum=1)


@dataclasses.dataclass(frozen=True, slots=True)
class VisualTokenOutput:
    checkpoint_sha256: str
    preprocessing_sha256: str
    action_conditioning_sha256: str
    provider_implementation_sha256: str
    provider_executable_sha256: str
    provider_protocol_sha256: str
    runtime_image_digest: str
    context_tokens: np.ndarray[Any, Any]
    predicted_target_tokens: np.ndarray[Any, Any]
    ema_target_tokens: np.ndarray[Any, Any]
    resources: InferenceResources


class VJEPAVisualTokenProvider(Protocol):
    """Boundary implemented in a separately qualified official-code runtime."""

    def infer(
        self,
        request: VisualTokenRequest,
        clips: np.ndarray[Any, Any],
        actions: np.ndarray[Any, Any],
        states: np.ndarray[Any, Any],
    ) -> VisualTokenOutput: ...


@dataclasses.dataclass(frozen=True, slots=True)
class QualificationReceipt:
    schema: str
    plan_sha256: str
    seed: int
    batch_size: int
    threefry_key_data: tuple[int, int]
    clip_sha256: str
    action_sha256: str
    state_sha256: str
    context_sha256: str
    prediction_sha256: str
    ema_target_sha256: str
    context_shape: tuple[int, int, int]
    target_shape: tuple[int, int, int]
    token_dtype: str
    mean_token_mse: float
    mean_token_cosine: float
    resources: InferenceResources
    identity: QualificationIdentity
    outcome_scope: str = OUTCOME_SCOPE
    visual_adapter_executed: bool = True
    official_parity_claimed: bool = False
    scientific_promotion_allowed: bool = False

    def __post_init__(self) -> None:
        if self.schema != RECEIPT_SCHEMA:
            raise ValueError("unsupported V-JEPA receipt schema")
        for name in (
            "plan_sha256",
            "clip_sha256",
            "action_sha256",
            "state_sha256",
            "context_sha256",
            "prediction_sha256",
            "ema_target_sha256",
        ):
            _sha(getattr(self, name), name)
        _integer(self.seed, "seed", maximum=2**32 - 1)
        _integer(self.batch_size, "batch size", minimum=1, maximum=_MAX_BATCH)
        if type(self.threefry_key_data) is not tuple or len(self.threefry_key_data) != 2:
            raise ValueError("receipt Threefry key data must be an exact pair")
        if self.context_shape != (
            self.batch_size,
            CONTEXT_FRAMES * TOKENS_PER_FRAME,
            TOKEN_DIMENSION,
        ):
            raise ValueError("receipt context-token shape differs from the contract")
        if self.target_shape != (
            self.batch_size,
            TARGET_FRAMES * TOKENS_PER_FRAME,
            TOKEN_DIMENSION,
        ):
            raise ValueError("receipt target-token shape differs from the contract")
        if self.token_dtype != "float32":
            raise ValueError("receipt token dtype must be float32")
        for name in ("mean_token_mse", "mean_token_cosine"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"{name} must be an exact finite float")
        if not -1.0 <= self.mean_token_cosine <= 1.0:
            raise ValueError("mean token cosine lies outside [-1, 1]")
        if self.mean_token_mse < 0:
            raise ValueError("mean token MSE must be nonnegative")
        if type(self.resources) is not InferenceResources:
            raise ValueError("resources must be an exact inference receipt")
        self.resources.__post_init__()
        expected_input_bytes = (
            self.batch_size
            * (CONTEXT_FRAMES + TARGET_FRAMES)
            * IMAGE_SIZE
            * IMAGE_SIZE
            * 3
            + 2 * self.batch_size * CONTEXT_FRAMES * ACTION_DIMENSION * 4
        )
        expected_output_bytes = (
            math.prod(self.context_shape) + 2 * math.prod(self.target_shape)
        ) * 4
        expected_semantics = (
            self.resources.examples,
            self.resources.checkpoint_loads,
            self.resources.encoder_queries,
            self.resources.predictor_queries,
            self.resources.ema_target_queries,
            self.resources.input_bytes,
            self.resources.output_bytes,
        )
        required_semantics = (
            self.batch_size,
            1,
            self.batch_size * (CONTEXT_FRAMES + TARGET_FRAMES),
            self.batch_size * TARGET_FRAMES,
            self.batch_size * TARGET_FRAMES,
            expected_input_bytes,
            expected_output_bytes,
        )
        if expected_semantics != required_semantics:
            raise ValueError("receipt semantic counts or tensor bytes differ from its shapes")
        minimum_peak = (
            self.resources.persistent_bytes
            + self.resources.input_bytes
            + self.resources.output_bytes
        )
        if self.resources.peak_bytes < minimum_peak:
            raise ValueError("receipt peak bytes omit resident input, output, or checkpoint bytes")
        if type(self.identity) is not QualificationIdentity:
            raise ValueError("identity must be an exact qualification identity")
        self.identity.__post_init__()
        if self.outcome_scope != OUTCOME_SCOPE:
            raise ValueError("receipt outcome scope is too broad")
        _exact_bool(self.visual_adapter_executed, "visual adapter execution", True)
        _exact_bool(self.official_parity_claimed, "official parity claim", False)
        _exact_bool(self.scientific_promotion_allowed, "scientific promotion", False)

    def to_payload(self) -> dict[str, object]:
        payload = dataclasses.asdict(self)
        payload["threefry_key_data"] = list(self.threefry_key_data)
        payload["context_shape"] = list(self.context_shape)
        payload["target_shape"] = list(self.target_shape)
        payload["identity"] = self.identity.to_payload()
        return cast(dict[str, object], payload)


def _canonical_plan_sha256(plan: VJEPAQualificationPlan) -> str:
    raw = json.dumps(
        dataclasses.asdict(plan), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _array_sha256(array: np.ndarray[Any, Any]) -> str:
    canonical = np.ascontiguousarray(array)
    header = f"{canonical.dtype.str}|{canonical.shape}|".encode("ascii")
    return hashlib.sha256(header + canonical.tobytes(order="C")).hexdigest()


def smoke_key_data(seed: int) -> tuple[int, int]:
    """Return the exact Threefry root words used by the bounded smoke."""

    _integer(seed, "seed", maximum=2**32 - 1)
    key = jr.fold_in(jr.key(seed, impl="threefry2x32"), QUALIFICATION_ISSUE)
    words = np.asarray(jr.key_data(key), dtype=np.uint32)
    return (int(words[0]), int(words[1]))


def build_smoke_inputs(
    *, seed: int, batch_size: int
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Create bounded visual/action/state inputs from one explicit Threefry root."""

    _integer(seed, "seed", maximum=2**32 - 1)
    batch = _integer(batch_size, "batch size", minimum=1, maximum=_MAX_BATCH)
    root = jr.fold_in(jr.key(seed, impl="threefry2x32"), QUALIFICATION_ISSUE)
    clip_key, action_key, state_key = jr.split(root, 3)
    clip_shape = (batch, CONTEXT_FRAMES + TARGET_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3)
    action_shape = (batch, CONTEXT_FRAMES, ACTION_DIMENSION)
    clips = np.asarray(jr.randint(clip_key, clip_shape, 0, 256, dtype=np.uint8))
    actions = np.asarray(
        jr.uniform(action_key, action_shape, minval=-1.0, maxval=1.0, dtype=np.float32)
    )
    states = np.asarray(
        jr.uniform(state_key, action_shape, minval=-1.0, maxval=1.0, dtype=np.float32)
    )
    if clips.nbytes + actions.nbytes + states.nbytes > _MAX_TENSOR_BYTES:
        raise ValueError("smoke input tensors exceed the byte bound")
    return clips, actions, states


def _request(plan: VJEPAQualificationPlan, seed: int, batch_size: int) -> VisualTokenRequest:
    batch = _integer(batch_size, "batch size", minimum=1, maximum=_MAX_BATCH)
    return VisualTokenRequest(
        schema=PROVIDER_SCHEMA,
        model_id=VJEPA_MODEL_ID,
        checkpoint_sha256=plan.checkpoint.checkpoint_sha256,
        preprocessing_sha256=plan.checkpoint.preprocessing_sha256,
        action_conditioning_sha256=plan.checkpoint.action_conditioning_sha256,
        provider_implementation_sha256=plan.provider.implementation_sha256,
        provider_executable_sha256=plan.provider.executable_sha256,
        provider_protocol_sha256=plan.provider.protocol_sha256,
        runtime_image_digest=plan.runtime.image_digest,
        clip_shape=(batch, CONTEXT_FRAMES + TARGET_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3),
        action_shape=(batch, CONTEXT_FRAMES, ACTION_DIMENSION),
        state_shape=(batch, CONTEXT_FRAMES, ACTION_DIMENSION),
        context_token_shape=(batch, CONTEXT_FRAMES * TOKENS_PER_FRAME, TOKEN_DIMENSION),
        target_token_shape=(batch, TARGET_FRAMES * TOKENS_PER_FRAME, TOKEN_DIMENSION),
        threefry_key_data=smoke_key_data(seed),
    )


def _validate_output(
    output: object,
    *,
    plan: VJEPAQualificationPlan,
    request: VisualTokenRequest,
    clips: np.ndarray[Any, Any],
    actions: np.ndarray[Any, Any],
    states: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    if type(output) is not VisualTokenOutput:
        raise ValueError("provider output must have the exact output type")
    for name, expected_echo in (
        ("checkpoint_sha256", request.checkpoint_sha256),
        ("preprocessing_sha256", request.preprocessing_sha256),
        ("action_conditioning_sha256", request.action_conditioning_sha256),
        ("provider_implementation_sha256", request.provider_implementation_sha256),
        ("provider_executable_sha256", request.provider_executable_sha256),
        ("provider_protocol_sha256", request.provider_protocol_sha256),
        ("runtime_image_digest", request.runtime_image_digest),
    ):
        if getattr(output, name) != expected_echo:
            raise ValueError(f"provider {name} echo differs from the request")
    arrays = (
        output.context_tokens,
        output.predicted_target_tokens,
        output.ema_target_tokens,
    )
    shapes = (request.context_token_shape, request.target_token_shape, request.target_token_shape)
    for name, array, shape in zip(
        ("context", "predicted target", "EMA target"), arrays, shapes, strict=True
    ):
        if type(array) is not np.ndarray or array.dtype != np.float32:
            raise ValueError(f"provider {name} tokens must be an exact float32 ndarray")
        if array.shape != shape:
            raise ValueError(f"provider {name} token shape differs from the request")
        if array.nbytes > _MAX_TENSOR_BYTES or not np.isfinite(array).all():
            raise ValueError(f"provider {name} tokens are invalid or exceed the byte bound")
    if any(
        np.shares_memory(arrays[left], arrays[right])
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise ValueError("provider token tensors must not alias")
    if type(output.resources) is not InferenceResources:
        raise ValueError("provider resources must have the exact receipt type")
    output.resources.__post_init__()
    resources = output.resources
    batch = request.clip_shape[0]
    expected_output_bytes = sum(array.nbytes for array in arrays)
    charged = (
        resources.examples,
        resources.checkpoint_loads,
        resources.checkpoint_read_bytes,
        resources.encoder_queries,
        resources.predictor_queries,
        resources.ema_target_queries,
        resources.input_bytes,
        resources.output_bytes,
    )
    actual = (
        batch,
        1,
        plan.checkpoint.checkpoint_bytes,
        batch * (CONTEXT_FRAMES + TARGET_FRAMES),
        batch * TARGET_FRAMES,
        batch * TARGET_FRAMES,
        clips.nbytes + actions.nbytes + states.nbytes,
        expected_output_bytes,
    )
    if charged != actual:
        raise ValueError("provider semantic counts or tensor bytes are not charged exactly")
    if resources.persistent_bytes < plan.checkpoint.checkpoint_bytes:
        raise ValueError("provider persistent bytes omit the checkpoint")
    minimum_peak_bytes = resources.persistent_bytes + resources.input_bytes + resources.output_bytes
    if resources.peak_bytes < minimum_peak_bytes:
        raise ValueError("provider peak bytes omit resident input, output, or checkpoint bytes")
    return arrays


def qualify_visual_token_adapter(
    plan: VJEPAQualificationPlan,
    provider: VJEPAVisualTokenProvider,
    *,
    seed: int,
    batch_size: int,
) -> QualificationReceipt:
    """Execute one bounded provider call and return a nonpromoting receipt."""

    if type(plan) is not VJEPAQualificationPlan:
        raise ValueError("plan must have the exact V-JEPA qualification type")
    plan.__post_init__()
    identity = _current_identity()
    plan_sha256 = _canonical_plan_sha256(plan)
    clips, actions, states = build_smoke_inputs(seed=seed, batch_size=batch_size)
    request = _request(plan, seed, batch_size)
    request_snapshot = dataclasses.replace(request)
    provider_clips = np.array(clips, copy=True)
    provider_actions = np.array(actions, copy=True)
    provider_states = np.array(states, copy=True)
    provider_input_hashes = tuple(
        _array_sha256(array) for array in (provider_clips, provider_actions, provider_states)
    )
    provider_clips.setflags(write=False)
    provider_actions.setflags(write=False)
    provider_states.setflags(write=False)
    output = provider.infer(request, provider_clips, provider_actions, provider_states)
    if request != request_snapshot:
        raise ValueError("provider mutated the attested request")
    if provider_input_hashes != tuple(
        _array_sha256(array) for array in (provider_clips, provider_actions, provider_states)
    ):
        raise ValueError("provider mutated the attested smoke inputs")
    plan.__post_init__()
    if _canonical_plan_sha256(plan) != plan_sha256:
        raise ValueError("provider mutated the attested qualification plan")
    context, prediction, target = _validate_output(
        output,
        plan=plan,
        request=request,
        clips=clips,
        actions=actions,
        states=states,
    )
    difference = prediction.astype(np.float64) - target.astype(np.float64)
    mean_mse = float(np.mean(np.square(difference), dtype=np.float64))
    predicted_flat = prediction.astype(np.float64).reshape(prediction.shape[0], -1)
    target_flat = target.astype(np.float64).reshape(target.shape[0], -1)
    denominator = np.linalg.norm(predicted_flat, axis=1) * np.linalg.norm(target_flat, axis=1)
    if np.any(denominator == 0):
        raise ValueError("provider tokens cannot produce a zero-norm cosine")
    mean_cosine = float(np.mean(np.sum(predicted_flat * target_flat, axis=1) / denominator))
    receipt = QualificationReceipt(
        schema=RECEIPT_SCHEMA,
        plan_sha256=plan_sha256,
        seed=seed,
        batch_size=batch_size,
        threefry_key_data=request.threefry_key_data,
        clip_sha256=_array_sha256(clips),
        action_sha256=_array_sha256(actions),
        state_sha256=_array_sha256(states),
        context_sha256=_array_sha256(context),
        prediction_sha256=_array_sha256(prediction),
        ema_target_sha256=_array_sha256(target),
        context_shape=request.context_token_shape,
        target_shape=request.target_token_shape,
        token_dtype="float32",
        mean_token_mse=mean_mse,
        mean_token_cosine=mean_cosine,
        resources=output.resources,
        identity=identity,
    )
    receipt.__post_init__()
    return receipt


def _exact_dict(value: object, fields: set[str], name: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{name} fields differ from the schema")
    return dict(value)


def validate_receipt_payload(
    payload: object, *, expected_plan: VJEPAQualificationPlan
) -> QualificationReceipt:
    """Strictly parse a receipt and bind it to the current tree and exact plan."""

    if type(expected_plan) is not VJEPAQualificationPlan:
        raise ValueError("expected plan must have the exact V-JEPA qualification type")
    expected_plan.__post_init__()
    raw = _exact_dict(
        payload,
        {field.name for field in dataclasses.fields(QualificationReceipt)},
        "receipt",
    )
    resources_raw = _exact_dict(
        raw["resources"],
        {field.name for field in dataclasses.fields(InferenceResources)},
        "resources",
    )
    identity = identity_from_payload(raw["identity"])
    for name in ("threefry_key_data", "context_shape", "target_shape"):
        value = raw[name]
        if type(value) is not list or any(type(item) is not int for item in value):
            raise ValueError(f"receipt {name} must be an exact integer list")
        raw[name] = tuple(value)
    resources = InferenceResources(
        **cast(dict[str, Any], resources_raw)
    )
    receipt = QualificationReceipt(
        schema=cast(str, raw["schema"]),
        plan_sha256=cast(str, raw["plan_sha256"]),
        seed=cast(int, raw["seed"]),
        batch_size=cast(int, raw["batch_size"]),
        threefry_key_data=cast(tuple[int, int], raw["threefry_key_data"]),
        clip_sha256=cast(str, raw["clip_sha256"]),
        action_sha256=cast(str, raw["action_sha256"]),
        state_sha256=cast(str, raw["state_sha256"]),
        context_sha256=cast(str, raw["context_sha256"]),
        prediction_sha256=cast(str, raw["prediction_sha256"]),
        ema_target_sha256=cast(str, raw["ema_target_sha256"]),
        context_shape=cast(tuple[int, int, int], raw["context_shape"]),
        target_shape=cast(tuple[int, int, int], raw["target_shape"]),
        token_dtype=cast(str, raw["token_dtype"]),
        mean_token_mse=cast(float, raw["mean_token_mse"]),
        mean_token_cosine=cast(float, raw["mean_token_cosine"]),
        resources=resources,
        identity=identity,
        outcome_scope=cast(str, raw["outcome_scope"]),
        visual_adapter_executed=cast(bool, raw["visual_adapter_executed"]),
        official_parity_claimed=cast(bool, raw["official_parity_claimed"]),
        scientific_promotion_allowed=cast(bool, raw["scientific_promotion_allowed"]),
    )
    if receipt.plan_sha256 != _canonical_plan_sha256(expected_plan):
        raise ValueError("receipt does not match the expected qualification plan")
    if receipt.resources.checkpoint_read_bytes != expected_plan.checkpoint.checkpoint_bytes:
        raise ValueError("receipt checkpoint-read bytes differ from the expected plan")
    if receipt.resources.persistent_bytes < expected_plan.checkpoint.checkpoint_bytes:
        raise ValueError("receipt persistent bytes omit the expected checkpoint")
    if receipt.threefry_key_data != smoke_key_data(receipt.seed):
        raise ValueError("receipt Threefry root differs from its seed")
    require_current_identity(receipt.identity, _current_identity())
    return receipt


def blocker_manifest() -> dict[str, object]:
    """Return static unresolved gates without probing the host or network."""

    _require_authoritative_plan()
    return {
        "schema": "asi.vjepa2_ac.qualification.blockers.v1",
        "issue": QUALIFICATION_ISSUE,
        "ready": False,
        "external_execution_performed": False,
        "native_proxy_is_official_vjepa": False,
        "source": {"repository": VJEPA_REPOSITORY, "commit": VJEPA_COMMIT},
        "no_imported_pretraining_ablation": dict(NO_IMPORTED_PRETRAINING_ABLATION),
        "blockers": [
            "official_assets_not_acquired",
            "source_and_license_closure_not_verified",
            "checkpoint_and_terms_not_verified",
            "web_video_and_robot_video_provenance_not_attested",
            "pretraining_resources_not_attested",
            "isolated_runtime_and_provider_not_built",
            "official_visual_token_adapter_not_executed",
            "jepa_wm_matched_planner_comparison_not_executed",
            "no_imported_pretraining_asi_visual_ablation_not_executed",
            "robot_safety_and_hardware_gates_not_qualified",
            "external_execution_not_authorized",
        ],
        "claim": "provider contract only; no official parity or scientific promotion",
    }


def main() -> int:
    print(json.dumps(blocker_manifest(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ACTION_DIMENSION",
    "CONTEXT_FRAMES",
    "CheckpointQualification",
    "DatasetQualification",
    "IMAGE_SIZE",
    "InferenceResources",
    "JEPA_WM_COMMIT",
    "JEPA_WM_REPOSITORY",
    "LicenseQualification",
    "NO_IMPORTED_PRETRAINING_ABLATION",
    "OUTCOME_SCOPE",
    "PLAN_SCHEMA",
    "PROVIDER_SCHEMA",
    "PretrainingResources",
    "ProviderQualification",
    "QualificationReceipt",
    "RECEIPT_SCHEMA",
    "REQUIRED_SOURCE_FILES",
    "RuntimeQualification",
    "SourceQualification",
    "TARGET_FRAMES",
    "TOKEN_DIMENSION",
    "TOKENS_PER_FRAME",
    "VJEPAQualificationPlan",
    "VJEPAVisualTokenProvider",
    "VJEPA_COMMIT",
    "VJEPA_MODEL_ID",
    "VJEPA_REPOSITORY",
    "VJEPA_SOURCE_LICENSE_ID",
    "VisualTokenOutput",
    "VisualTokenRequest",
    "blocker_manifest",
    "build_smoke_inputs",
    "qualify_visual_token_adapter",
    "smoke_key_data",
    "validate_receipt_payload",
]
