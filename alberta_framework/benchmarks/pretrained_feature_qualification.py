"""Fail-closed qualification boundary for external frozen-feature ceilings.

This module does not download or execute official code.  It defines the exact
caller-supplied identities and provider boundary required before RanDumb,
RanPAC, or PROL features can enter a separately authorized matched campaign.
The local replay-frozen arms remain mechanism proxies, never paper ceilings.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

_HEX = frozenset("0123456789abcdef")
_MAX_SMOKE_TENSOR_BYTES: Final = 256 * 1024 * 1024


def _text(value: object, name: str, limit: int = 512) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty exact string")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must contain valid Unicode") from exc
    if len(encoded) > limit:
        raise ValueError(f"{name} exceeds its UTF-8 byte limit")
    return value


def _sha(value: object, name: str) -> str:
    value = _text(value, name, 64)
    if len(value) != 64 or any(character not in _HEX for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _count(value: object, name: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be an exact {qualifier} integer")
    return value


@dataclass(frozen=True, slots=True)
class MethodSpec:
    method: str
    paper_revision: str
    repository: str
    commit: str
    license_id: str
    artifact_role: str
    required_files: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.method, "method", 32)
        _text(self.paper_revision, "paper_revision", 64)
        repository = _text(self.repository, "repository")
        if not repository.startswith("https://github.com/") or not repository.endswith(".git"):
            raise ValueError("repository must be a credential-free GitHub HTTPS clone URL")
        if len(self.commit) != 40 or any(character not in _HEX for character in self.commit):
            raise ValueError("commit must be a full lowercase Git commit ID")
        if self.artifact_role not in {"frozen_random_initialization", "pretrained_checkpoint"}:
            raise ValueError("unsupported artifact_role")
        if type(self.required_files) is not tuple or not self.required_files:
            raise ValueError("required_files must be a non-empty exact tuple")
        for value in self.required_files:
            _text(value, "required_files entry", 128)


OFFICIAL_METHODS: Final[Mapping[str, MethodSpec]] = MappingProxyType({
    "randumb": MethodSpec(
        "randumb", "arXiv:2402.08823v3", "https://github.com/drimpossible/RanDumb.git",
        "14a51ee0c045bff642f6ffbfe481efa4d49a3033", "GPL-3.0-only",
        "frozen_random_initialization", ("src", "get_feats.py"),
    ),
    "ranpac": MethodSpec(
        "ranpac", "arXiv:2307.02251v3",
        "https://github.com/McDonnell-Research-Lab/RanPAC.git",
        "cf4b301d18b0c27db030f4371b72b768005ae58a", "MIT", "pretrained_checkpoint",
        ("RanPAC.py", "inc_net.py"),
    ),
    "prol": MethodSpec(
        "prol", "arXiv:2507.12305v1", "https://github.com/anwarmaxsum/PROL.git",
        "bfff8418a4f603a24ae578f1e108bfac89af1e18", "MIT", "pretrained_checkpoint",
        ("main_prol.py", "engine_prol.py"),
    ),
})


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    repository: str
    commit: str
    source_archive_sha256: str
    source_tree_sha256: str
    required_file_sha256: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _text(self.repository, "repository")
        if len(self.commit) != 40 or any(character not in _HEX for character in self.commit):
            raise ValueError("commit must be a full lowercase Git commit ID")
        _sha(self.source_archive_sha256, "source_archive_sha256")
        _sha(self.source_tree_sha256, "source_tree_sha256")
        if type(self.required_file_sha256) is not tuple or not self.required_file_sha256:
            raise ValueError("required_file_sha256 must be a non-empty exact tuple")
        paths: list[str] = []
        for entry in self.required_file_sha256:
            if type(entry) is not tuple or len(entry) != 2:
                raise ValueError("required file identities must be exact pairs")
            paths.append(_text(entry[0], "required file path", 128))
            _sha(entry[1], "required file SHA-256")
        if len(paths) != len(set(paths)):
            raise ValueError("required file paths must be unique")


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    method: str
    artifact_role: str
    checkpoint_sha256: str
    checkpoint_bytes: int
    architecture_sha256: str
    preprocessing_sha256: str
    output_dimension: int
    frozen: bool

    def __post_init__(self) -> None:
        _text(self.method, "method", 32)
        _text(self.artifact_role, "artifact_role", 64)
        _sha(self.checkpoint_sha256, "checkpoint_sha256")
        _count(self.checkpoint_bytes, "checkpoint_bytes", positive=True)
        _sha(self.architecture_sha256, "architecture_sha256")
        _sha(self.preprocessing_sha256, "preprocessing_sha256")
        _count(self.output_dimension, "output_dimension", positive=True)
        if type(self.frozen) is not bool or not self.frozen:
            raise ValueError("extractor checkpoint must be frozen")


@dataclass(frozen=True, slots=True)
class DatasetIdentity:
    provider: str
    name: str
    revision: str
    materialization_sha256: str
    examples: int

    def __post_init__(self) -> None:
        _text(self.provider, "dataset provider", 128)
        _text(self.name, "dataset name", 128)
        _text(self.revision, "dataset revision", 128)
        _sha(self.materialization_sha256, "dataset materialization_sha256")
        _count(self.examples, "dataset examples")


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    image_digest: str
    dependency_lock_sha256: str
    python: str
    torch: str
    torchvision: str
    network_disabled: bool

    def __post_init__(self) -> None:
        image = _text(self.image_digest, "image_digest", 128)
        if not image.startswith("sha256:"):
            raise ValueError("image_digest must use sha256")
        _sha(image[7:], "image digest")
        _sha(self.dependency_lock_sha256, "dependency_lock_sha256")
        for name in ("python", "torch", "torchvision"):
            _text(getattr(self, name), name, 64)
        if type(self.network_disabled) is not bool or not self.network_disabled:
            raise ValueError("qualified runtime must disable network access")


@dataclass(frozen=True, slots=True)
class PretrainingCost:
    examples: int
    updates: int
    model_queries: int
    scalar_flops: int
    accelerator_ns: int
    peak_bytes: int
    checkpoint_bytes: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _count(getattr(self, name), name)
        _count(self.checkpoint_bytes, "checkpoint_bytes", positive=True)


@dataclass(frozen=True, slots=True)
class ExtractorCost:
    examples: int
    queries: int
    input_bytes: int
    output_bytes: int
    scalar_flops: int
    peak_bytes: int
    persistent_bytes: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _count(getattr(self, name), name, positive=name in {"examples", "queries"})


@dataclass(frozen=True, slots=True)
class QualificationPlan:
    schema: str
    method: str
    source: SourceIdentity
    checkpoint: CheckpointIdentity
    evaluation_dataset: DatasetIdentity
    pretraining_dataset: DatasetIdentity
    runtime: RuntimeIdentity
    pretraining_cost: PretrainingCost

    def __post_init__(self) -> None:
        if self.schema != "asi.pretrained_feature_qualification.plan.v1":
            raise ValueError("unsupported qualification plan schema")
        spec = OFFICIAL_METHODS.get(self.method)
        if spec is None:
            raise ValueError("method must be an exact official method identifier")
        for value, cls, name in (
            (self.source, SourceIdentity, "source"),
            (self.checkpoint, CheckpointIdentity, "checkpoint"),
            (self.evaluation_dataset, DatasetIdentity, "evaluation_dataset"),
            (self.pretraining_dataset, DatasetIdentity, "pretraining_dataset"),
            (self.runtime, RuntimeIdentity, "runtime"),
            (self.pretraining_cost, PretrainingCost, "pretraining_cost"),
        ):
            if type(value) is not cls:
                raise ValueError(f"{name} must have its exact identity type")
            value.__post_init__()
        if (self.source.repository, self.source.commit) != (spec.repository, spec.commit):
            raise ValueError("source does not match the official immutable revision")
        supplied_paths = {path for path, _ in self.source.required_file_sha256}
        if supplied_paths != set(spec.required_files):
            raise ValueError("source identity must cover every required official file exactly")
        checkpoint = self.checkpoint
        if (checkpoint.method, checkpoint.artifact_role) != (self.method, spec.artifact_role):
            if self.method == "randumb":
                raise ValueError("RanDumb must identify a frozen random initialization artifact")
            raise ValueError("checkpoint method or artifact role mismatch")
        if checkpoint.checkpoint_bytes != self.pretraining_cost.checkpoint_bytes:
            raise ValueError("checkpoint bytes must be charged exactly")
        counts = self.pretraining_cost
        if spec.artifact_role == "pretrained_checkpoint":
            if min(counts.examples, counts.updates, counts.model_queries, counts.scalar_flops) <= 0:
                raise ValueError("pretrained methods require positive pretraining costs")
            if self.pretraining_dataset.examples <= 0:
                raise ValueError("pretrained methods require a non-empty pretraining dataset")
            if counts.examples < self.pretraining_dataset.examples:
                raise ValueError("pretraining cost examples must cover the attested dataset")
        elif any((counts.examples, counts.updates, counts.model_queries, counts.scalar_flops)):
            raise ValueError("RanDumb random initialization cannot claim imported pretraining")


@dataclass(frozen=True, slots=True)
class ExtractorRequest:
    method: str
    checkpoint_sha256: str
    preprocessing_sha256: str
    output_dimension: int


@runtime_checkable
class FrozenFeatureProvider(Protocol):
    """Adapter implemented by a future isolated official-code extractor."""

    def extract(
        self, request: ExtractorRequest, inputs: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Return one float32 feature row for each caller-supplied input row."""


@dataclass(frozen=True, slots=True)
class QualificationReceipt:
    schema: str
    method: str
    plan_sha256: str
    input_sha256: str
    feature_sha256: str
    feature_shape: tuple[int, int]
    feature_dtype: str
    cost: ExtractorCost
    ceiling_claim_allowed: bool = False

    def __post_init__(self) -> None:
        if self.schema != "asi.pretrained_feature_qualification.receipt.v1":
            raise ValueError("unsupported qualification receipt schema")
        if self.method not in OFFICIAL_METHODS:
            raise ValueError("receipt method is not official")
        for name in ("plan_sha256", "input_sha256", "feature_sha256"):
            _sha(getattr(self, name), name)
        if (
            type(self.feature_shape) is not tuple
            or len(self.feature_shape) != 2
            or any(type(value) is not int or value <= 0 for value in self.feature_shape)
        ):
            raise ValueError("feature_shape must be an exact positive pair")
        if self.feature_dtype != "float32":
            raise ValueError("receipt feature_dtype must be float32")
        if type(self.cost) is not ExtractorCost:
            raise ValueError("receipt cost must be an exact ExtractorCost")
        self.cost.__post_init__()
        if type(self.ceiling_claim_allowed) is not bool or self.ceiling_claim_allowed:
            raise ValueError("qualification receipts cannot authorize a ceiling claim")

    def revalidate(self, inputs: NDArray[np.float32]) -> str:
        self.__post_init__()
        if type(inputs) is not np.ndarray or inputs.dtype != np.float32:
            raise ValueError("inputs must be an exact float32 ndarray")
        if _array_sha(inputs) != self.input_sha256:
            raise ValueError("inputs do not match the qualification receipt")
        return self.feature_sha256


def _array_sha(array: np.ndarray[tuple[int, ...], np.dtype[np.generic]]) -> str:
    canonical = np.ascontiguousarray(array)
    header = f"{canonical.dtype.str}|{canonical.shape}|".encode()
    return hashlib.sha256(header + canonical.tobytes(order="C")).hexdigest()


def _plan_sha(plan: QualificationPlan) -> str:
    payload = repr(plan).encode("utf-8", errors="strict")
    return hashlib.sha256(payload).hexdigest()


def qualify_frozen_extractor(
    plan: QualificationPlan,
    provider: FrozenFeatureProvider,
    inputs: NDArray[np.float32],
    cost: ExtractorCost,
) -> QualificationReceipt:
    """Validate one bounded provider call and return a nonpromoting receipt."""

    plan.__post_init__()
    cost.__post_init__()
    if type(inputs) is not np.ndarray or inputs.dtype != np.float32 or inputs.ndim != 4:
        raise ValueError("inputs must be an exact rank-four float32 ndarray")
    if inputs.shape[0] <= 0 or tuple(inputs.shape[1:]) != (224, 224, 3):
        raise ValueError("inputs must have shape (N, 224, 224, 3)")
    expected_output_bytes = (
        inputs.shape[0] * plan.checkpoint.output_dimension * np.dtype(np.float32).itemsize
    )
    if inputs.nbytes > _MAX_SMOKE_TENSOR_BYTES or expected_output_bytes > _MAX_SMOKE_TENSOR_BYTES:
        raise ValueError("extractor tensors exceed the bounded smoke limit")
    if not np.isfinite(inputs).all():
        raise ValueError("inputs must be finite")
    if cost.examples != inputs.shape[0] or cost.queries != inputs.shape[0]:
        raise ValueError("extractor examples and queries must match the batch")
    if cost.input_bytes != inputs.nbytes or cost.output_bytes != expected_output_bytes:
        raise ValueError("extractor tensor bytes must be charged exactly")
    if cost.scalar_flops <= 0 or cost.peak_bytes <= 0:
        raise ValueError("extractor compute and peak bytes must be positive")
    if cost.persistent_bytes < plan.checkpoint.checkpoint_bytes:
        raise ValueError("persistent bytes must include the checkpoint")
    request = ExtractorRequest(
        plan.method,
        plan.checkpoint.checkpoint_sha256,
        plan.checkpoint.preprocessing_sha256,
        plan.checkpoint.output_dimension,
    )
    features = provider.extract(request, np.array(inputs, copy=True))
    expected_shape = (inputs.shape[0], plan.checkpoint.output_dimension)
    if type(features) is not np.ndarray or features.dtype != np.float32:
        raise ValueError("provider features must be an exact float32 ndarray")
    if features.shape != expected_shape:
        raise ValueError("provider feature shape does not match the request")
    if not np.isfinite(features).all():
        raise ValueError("provider features must be finite")
    return QualificationReceipt(
        "asi.pretrained_feature_qualification.receipt.v1",
        plan.method,
        _plan_sha(plan),
        _array_sha(inputs),
        _array_sha(features),
        expected_shape,
        "float32",
        cost,
        False,
    )


def blocker_manifest() -> dict[str, object]:
    """Return static readiness blockers; never probes the network or host."""

    return {
        "schema": "asi.pretrained_feature_qualification.blockers.v1",
        "issue": 1573,
        "ready": False,
        "proxy_arms_are_ceilings": False,
        "methods": {
            name: {"repository": spec.repository, "commit": spec.commit}
            for name, spec in OFFICIAL_METHODS.items()
        },
        "blockers": [
            "official_artifacts_not_acquired",
            "source_file_and_license_hashes_not_verified",
            "checkpoint_or_random_initialization_not_verified",
            "pretraining_dataset_identity_and_cost_not_attested",
            "isolated_runtime_not_built",
            "official_extractor_adapter_parity_not_verified",
            "full_matched_ipmnist_ceiling_not_executed",
            "external_execution_not_authorized",
        ],
        "claim": "qualification metadata only; local replay-frozen arms remain proxies",
    }


def main() -> int:
    print(json.dumps(blocker_manifest(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
