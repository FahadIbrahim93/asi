"""Bounded, nonexecuting qualification contracts for the CLEAR benchmark.

This module never downloads data or trains a model.  It binds a caller-held
local dataset receipt to a reviewed protocol and computes the exact workload
accounting a future runner must reproduce.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

SCHEMA = "asi.clear.qualification.v1"
RUNNER_INPUT_SCHEMA = "asi.clear.runner-input.v1"
ACQUISITION_REVIEW_SCHEMA = "asi.clear.acquisition-review.v1"
RIGHTS_REVIEW_SCHEMA = "asi.clear.rights-storage-review.v1"
SPLIT_REVIEW_SCHEMA = "asi.clear.split-review.v1"
PAPER_REVISION = "arXiv:2201.06289v3"
CURATION_COMMIT = "620cab4a7d99921fde73b67b53879470533cb39a"
REFERENCE_COMMIT = "75d5d2e7d412a787e0decf0417a4868c56691252"
AVALANCHE_COMMIT = "eb075be393e1f458b2c352514ff6c17b5a2c0f4e"
DATASET_NAME = "clear100"
BUCKETS = tuple(range(1, 11))
YEARS = tuple(range(2005, 2015))
DEV_SEEDS = (0, 1, 2, 3, 4)
MAX_MANIFEST_BYTES = 1 << 20
MAX_ARCHIVES = 8
MAX_SAMPLES_PER_BUCKET = 10_000_000
MAX_RESULT_BYTES = 1 << 20
MAX_INDEX_BYTES = 1 << 30
MAX_INDEX_LINE_BYTES = 4096
CLASS_COUNT = 100
_INT64_MAX = (1 << 63) - 1
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_CURRENT_SOURCE_PATHS = ("alberta_framework/benchmarks/clear_qualification.py",)
_MAX_CURRENT_SOURCE_BYTES = 16 << 20


class ClearQualificationError(ValueError):
    """A CLEAR setup or result record failed closed."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _exact_int(value: object, label: str, *, minimum: int = 0, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ClearQualificationError(f"{label} must be an exact integer in range")
    return value


def _exact_str(value: object, label: str, *, maximum: int = 512) -> str:
    if type(value) is not str or not value:
        raise ClearQualificationError(f"{label} must be a bounded non-empty exact string")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ClearQualificationError(f"{label} must contain valid Unicode") from exc
    if len(encoded) > maximum:
        raise ClearQualificationError(f"{label} must be a bounded non-empty exact string")
    return value


def _sha256(value: object, label: str) -> str:
    text = _exact_str(value, label, maximum=64)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ClearQualificationError(f"{label} must be a lowercase SHA-256")
    return text


def _object(value: object, label: str) -> Mapping[str, object]:
    if type(value) is not dict:
        raise ClearQualificationError(f"{label} must be an exact JSON object")
    return value


def _keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ClearQualificationError(f"{label} fields do not match the schema")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ClearQualificationError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ClearQualificationError(f"non-finite JSON number is forbidden: {value}")


@dataclass(frozen=True, slots=True)
class ArchiveIdentity:
    role: str
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ClearDatasetReceipt:
    archives: tuple[ArchiveIdentity, ...]
    samples_per_bucket: tuple[int, ...]
    archive_bytes: int
    sample_count: int
    dataset_sha256: str


@dataclass(frozen=True, slots=True)
class ClearRunnerInputReceipt:
    dataset_sha256: str
    runner_input_sha256: str
    train_samples_per_bucket: tuple[int, ...]
    evaluation_samples_per_bucket: tuple[int, ...]
    train_sample_count: int
    evaluation_sample_count: int
    class_count: int
    sample_bytes: int
    index_bytes: int
    review_bytes: int
    training_observations: int
    optimizer_updates: int
    model_queries: int
    data_samples_read: int
    current_source_sha256: tuple[tuple[str, str], ...]
    runtime_identity: tuple[tuple[str, str], ...]
    external_reviews_authenticated: bool
    provider_snapshot_bytes_verified: bool
    redistribution_authorized: bool
    execution_authorized: bool

    def __post_init__(self) -> None:
        _sha256(self.dataset_sha256, "runner receipt dataset sha256")
        _sha256(self.runner_input_sha256, "runner receipt identity sha256")
        expected_source = tuple(sorted(current_source_identity().items()))
        expected_runtime = tuple(sorted(runtime_identity().items()))
        if (
            type(self.current_source_sha256) is not tuple
            or self.current_source_sha256 != expected_source
            or type(self.runtime_identity) is not tuple
            or self.runtime_identity != expected_runtime
        ):
            raise ClearQualificationError("runner receipt current identity drift")
        if (
            type(self.train_samples_per_bucket) is not tuple
            or type(self.evaluation_samples_per_bucket) is not tuple
            or len(self.train_samples_per_bucket) != len(BUCKETS)
            or len(self.evaluation_samples_per_bucket) != len(BUCKETS)
        ):
            raise ClearQualificationError("runner receipt split accounting is invalid")
        train_counts = tuple(
            _exact_int(value, "runner train count", minimum=CLASS_COUNT,
                       maximum=MAX_SAMPLES_PER_BUCKET)
            for value in self.train_samples_per_bucket
        )
        evaluation_counts = tuple(
            _exact_int(value, "runner evaluation count", minimum=CLASS_COUNT,
                       maximum=MAX_SAMPLES_PER_BUCKET)
            for value in self.evaluation_samples_per_bucket
        )
        expected_train = sum(train_counts)
        expected_evaluation = sum(evaluation_counts)
        expected_updates = sum(math.ceil(count / 256) * 100 for count in train_counts)
        expected_queries = expected_evaluation * len(BUCKETS)
        expected_training = expected_train * 100
        expected_reads = expected_training + expected_queries
        exact_resources = (
            ("train_sample_count", self.train_sample_count, expected_train),
            ("evaluation_sample_count", self.evaluation_sample_count, expected_evaluation),
            ("class_count", self.class_count, CLASS_COUNT),
            ("training_observations", self.training_observations, expected_training),
            ("optimizer_updates", self.optimizer_updates, expected_updates),
            ("model_queries", self.model_queries, expected_queries),
            ("data_samples_read", self.data_samples_read, expected_reads),
        )
        for name, value, expected in exact_resources:
            if _exact_int(value, f"runner receipt {name}", maximum=_INT64_MAX) != expected:
                raise ClearQualificationError("runner receipt resource accounting is invalid")
        _exact_int(self.sample_bytes, "runner receipt sample bytes", minimum=1,
                   maximum=_INT64_MAX)
        _exact_int(self.index_bytes, "runner receipt index bytes", minimum=1,
                   maximum=MAX_INDEX_BYTES)
        _exact_int(self.review_bytes, "runner receipt review bytes", minimum=1,
                   maximum=3 * MAX_MANIFEST_BYTES)
        if (
            self.external_reviews_authenticated is not False
            or self.provider_snapshot_bytes_verified is not False
            or self.redistribution_authorized is not False
            or self.execution_authorized is not False
        ):
            raise ClearQualificationError("runner receipt may not invent external authority")


def _validate_dataset_receipt(receipt: object) -> ClearDatasetReceipt:
    if type(receipt) is not ClearDatasetReceipt:
        raise TypeError("dataset_receipt must be an exact ClearDatasetReceipt")
    if type(receipt.archives) is not tuple or not 1 <= len(receipt.archives) <= MAX_ARCHIVES:
        raise ClearQualificationError("dataset receipt archives are invalid")
    archives: list[dict[str, object]] = []
    archive_sizes: list[int] = []
    seen_roles: set[str] = set()
    seen_paths: set[str] = set()
    for archive in receipt.archives:
        if type(archive) is not ArchiveIdentity:
            raise ClearQualificationError("dataset receipt archive identity is invalid")
        role = _exact_str(archive.role, "dataset receipt archive role", maximum=32)
        path = _safe_relative_path(archive.path)
        size = _exact_int(archive.size_bytes, "dataset receipt archive size", maximum=1 << 50)
        digest = _sha256(archive.sha256, "dataset receipt archive sha256")
        if role in seen_roles or path in seen_paths:
            raise ClearQualificationError("dataset receipt archive identities must be unique")
        seen_roles.add(role)
        seen_paths.add(path)
        archives.append({"role": role, "path": path, "size_bytes": size, "sha256": digest})
        archive_sizes.append(size)
    if (
        type(receipt.samples_per_bucket) is not tuple
        or len(receipt.samples_per_bucket) != len(BUCKETS)
    ):
        raise ClearQualificationError("dataset receipt sample counts are invalid")
    samples = tuple(
        _exact_int(value, "dataset receipt sample count", minimum=1, maximum=MAX_SAMPLES_PER_BUCKET)
        for value in receipt.samples_per_bucket
    )
    if (
        type(receipt.archive_bytes) is not int
        or receipt.archive_bytes != sum(archive_sizes)
        or type(receipt.sample_count) is not int
        or receipt.sample_count != sum(samples)
    ):
        raise ClearQualificationError("dataset receipt accounting is invalid")
    identity = {
        "dataset": DATASET_NAME,
        "protocol": "streaming-near-future",
        "buckets": BUCKETS,
        "years": YEARS,
        "samples_per_bucket": samples,
        "archives": archives,
    }
    if receipt.dataset_sha256 != hashlib.sha256(_canonical(identity)).hexdigest():
        raise ClearQualificationError("dataset receipt identity is invalid")
    return receipt


def _decode(raw: bytes, *, limit: int, label: str) -> Mapping[str, object]:
    if type(raw) is not bytes:
        raise TypeError("raw JSON input must be exact bytes")
    if len(raw) > limit:
        raise ClearQualificationError(f"{label} exceeds its byte limit")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ClearQualificationError(f"{label} is not valid JSON") from exc
    return _object(value, label)


def load_dataset_manifest(path: Path) -> bytes:
    """Read one local manifest without following links or exceeding the byte cap."""
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    try:
        before = path.lstat()
    except OSError as exc:
        raise ClearQualificationError("dataset manifest metadata is unavailable") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ClearQualificationError("dataset manifest must be a regular non-symlink file")
    if before.st_nlink != 1:
        raise ClearQualificationError("dataset manifest must not have a hard-link alias")
    if before.st_size > MAX_MANIFEST_BYTES:
        raise ClearQualificationError("dataset manifest exceeds its byte limit")

    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or _stat_identity(opened) != _stat_identity(before)
        ):
            raise ClearQualificationError("dataset manifest changed before its bounded read")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            raw = stream.read(MAX_MANIFEST_BYTES + 1)
            after = os.fstat(stream.fileno())
        current = path.stat(follow_symlinks=False)
    except ClearQualificationError:
        raise
    except OSError as exc:
        raise ClearQualificationError("dataset manifest could not be read safely") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)

    if len(raw) > MAX_MANIFEST_BYTES:
        raise ClearQualificationError("dataset manifest exceeds its byte limit")
    if (
        len(raw) != opened.st_size
        or _stat_identity(after) != _stat_identity(opened)
        or _stat_identity(current) != _stat_identity(opened)
    ):
        raise ClearQualificationError("dataset manifest changed during its bounded read")
    return raw


def _safe_relative_path(value: object) -> str:
    text = _exact_str(value, "archive path", maximum=256)
    path = PurePosixPath(text)
    if not path.parts or path.is_absolute() or ".." in path.parts or path.as_posix() != text:
        raise ClearQualificationError("archive path must be canonical and relative")
    return text


def _verified_root(root: Path) -> Path:
    """Bind one caller-selected root without following the final path component."""
    if not isinstance(root, Path):
        raise TypeError("root must be a Path")
    descriptor = -1
    try:
        before = root.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
            raise ClearQualificationError("dataset root must be a real directory")
        descriptor = os.open(
            root,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if _directory_identity(opened) != _directory_identity(before):
            raise ClearQualificationError("dataset root changed before verification")
        return root.resolve(strict=True)
    except ClearQualificationError:
        raise
    except OSError as exc:
        raise ClearQualificationError("dataset root is unavailable") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _https_locator(value: object) -> str:
    text = _exact_str(value, "provider locator", maximum=1024)
    try:
        parsed = urlsplit(text)
        port = parsed.port
    except ValueError as exc:
        raise ClearQualificationError("provider locator is not a valid URL") from exc
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or any(char.isspace() for char in text)
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise ClearQualificationError("provider locator must be a bounded HTTPS locator")
    return text


def verify_dataset_manifest(raw: bytes, *, root: Path) -> ClearDatasetReceipt:
    """Verify bounded local archive identities; never fetch or extract them."""
    if type(raw) is not bytes or not isinstance(root, Path):
        raise TypeError("raw must be bytes and root must be a Path")
    payload = _decode(raw, limit=MAX_MANIFEST_BYTES, label="dataset manifest")
    _keys(
        payload,
        {
            "schema_version",
            "dataset",
            "protocol",
            "buckets",
            "years",
            "samples_per_bucket",
            "archives",
            "provider_archive_checksums_published",
        },
        "dataset manifest",
    )
    if payload["schema_version"] != SCHEMA or payload["dataset"] != DATASET_NAME:
        raise ClearQualificationError("dataset identity drift")
    if payload["protocol"] != "streaming-near-future":
        raise ClearQualificationError("only the streaming protocol is qualified")
    if payload["buckets"] != list(BUCKETS) or payload["years"] != list(YEARS):
        raise ClearQualificationError("temporal bucket identity drift")
    if payload["provider_archive_checksums_published"] is not False:
        raise ClearQualificationError("provider checksum disclosure must not be invented")
    samples_value = payload["samples_per_bucket"]
    if type(samples_value) is not list or len(samples_value) != len(BUCKETS):
        raise ClearQualificationError("samples_per_bucket must cover every labeled bucket")
    samples = tuple(
        _exact_int(value, "bucket sample count", minimum=1, maximum=MAX_SAMPLES_PER_BUCKET)
        for value in samples_value
    )
    archive_values = payload["archives"]
    if type(archive_values) is not list or not 1 <= len(archive_values) <= MAX_ARCHIVES:
        raise ClearQualificationError("archives must be a bounded non-empty exact list")
    root_resolved = _verified_root(root)
    archives: list[ArchiveIdentity] = []
    seen_paths: set[str] = set()
    seen_archive_file_ids: set[tuple[int, int]] = set()
    for index, value in enumerate(archive_values):
        item = _object(value, f"archive {index}")
        _keys(item, {"role", "path", "size_bytes", "sha256"}, f"archive {index}")
        role = _exact_str(item["role"], "archive role", maximum=32)
        path_text = _safe_relative_path(item["path"])
        if path_text in seen_paths:
            raise ClearQualificationError("archive paths must be unique")
        seen_paths.add(path_text)
        size = _exact_int(item["size_bytes"], "archive size", maximum=1 << 50)
        digest = _sha256(item["sha256"], "archive sha256")
        if (
            _hash_below_root(
                root_resolved,
                path_text,
                expected_size=size,
                label="archive",
                seen_file_ids=seen_archive_file_ids,
            )
            != digest
        ):
            raise ClearQualificationError("archive SHA-256 does not match the manifest")
        archives.append(ArchiveIdentity(role, path_text, size, digest))
    if len({archive.role for archive in archives}) != len(archives):
        raise ClearQualificationError("archive roles must be unique")
    identity = {
        "dataset": DATASET_NAME,
        "protocol": "streaming-near-future",
        "buckets": BUCKETS,
        "years": YEARS,
        "samples_per_bucket": samples,
        "archives": [asdict(archive) for archive in archives],
    }
    return ClearDatasetReceipt(
        tuple(archives), samples, sum(item.size_bytes for item in archives), sum(samples),
        hashlib.sha256(_canonical(identity)).hexdigest(),
    )


def _file_identity(value: object, *, label: str) -> ArchiveIdentity:
    item = _object(value, label)
    _keys(item, {"path", "size_bytes", "sha256"}, label)
    return ArchiveIdentity(
        "",
        _safe_relative_path(item["path"]),
        _exact_int(item["size_bytes"], f"{label} size", maximum=1 << 50),
        _sha256(item["sha256"], f"{label} sha256"),
    )


@dataclass(slots=True)
class _BoundFile:
    descriptor: int
    root: Path
    root_identity: tuple[int, ...]
    directory_descriptors: list[int]
    directory_bindings: list[tuple[int, str, tuple[int, ...]]]
    parent_descriptor: int
    filename: str

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1
        for descriptor in reversed(self.directory_descriptors):
            os.close(descriptor)
        self.directory_descriptors.clear()


def _directory_identity(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_mode)


def _open_below_root(root: Path, relative: str, *, label: str) -> _BoundFile:
    """Open one regular file while retaining its entire no-follow path walk."""
    parts = PurePosixPath(relative).parts
    directory_descriptors: list[int] = []
    descriptor = -1
    try:
        root_before = root.lstat()
        if stat.S_ISLNK(root_before.st_mode) or not stat.S_ISDIR(root_before.st_mode):
            raise ClearQualificationError("dataset root must be a real directory")
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        current = os.open(root, directory_flags)
        directory_descriptors.append(current)
        root_opened = os.fstat(current)
        root_identity = _directory_identity(root_opened)
        if root_identity != _directory_identity(root_before):
            raise ClearQualificationError("dataset root changed before its bounded read")
        bindings: list[tuple[int, str, tuple[int, ...]]] = []
        for part in parts[:-1]:
            before = os.stat(part, dir_fd=current, follow_symlinks=False)
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise ClearQualificationError(
                    f"{label} is unavailable below the dataset root: link or non-directory"
                )
            following = os.open(part, directory_flags, dir_fd=current)
            opened = os.fstat(following)
            identity = _directory_identity(opened)
            if identity != _directory_identity(before):
                os.close(following)
                raise ClearQualificationError(
                    f"{label} directory changed before its bounded read"
                )
            bindings.append((current, part, identity))
            directory_descriptors.append(following)
            current = following
        filename = parts[-1]
        before = os.stat(filename, dir_fd=current, follow_symlinks=False)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise ClearQualificationError(
                f"{label} must be a regular file and not a symlink"
            )
        descriptor = os.open(
            filename,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=current,
        )
        opened = os.fstat(descriptor)
        if _stat_identity(opened) != _stat_identity(before):
            raise ClearQualificationError(f"{label} changed before its bounded read")
        return _BoundFile(
            descriptor,
            root,
            root_identity,
            directory_descriptors,
            bindings,
            current,
            filename,
        )
    except ClearQualificationError:
        if descriptor >= 0:
            os.close(descriptor)
        for directory_descriptor in reversed(directory_descriptors):
            os.close(directory_descriptor)
        raise
    except OSError as exc:
        if descriptor >= 0:
            os.close(descriptor)
        for directory_descriptor in reversed(directory_descriptors):
            os.close(directory_descriptor)
        raise ClearQualificationError(f"{label} is unavailable below the dataset root") from exc


def _revalidate_bound_file(
    bound: _BoundFile, opened: os.stat_result, *, label: str
) -> None:
    try:
        current = os.stat(
            bound.filename,
            dir_fd=bound.parent_descriptor,
            follow_symlinks=False,
        )
        if _stat_identity(current) != _stat_identity(opened):
            raise ClearQualificationError(f"{label} changed during its bounded read")
        for parent, component, identity in reversed(bound.directory_bindings):
            current_directory = os.stat(
                component, dir_fd=parent, follow_symlinks=False
            )
            if _directory_identity(current_directory) != identity:
                raise ClearQualificationError(
                    f"{label} directory changed during its bounded read"
                )
        if _directory_identity(bound.root.lstat()) != bound.root_identity:
            raise ClearQualificationError("dataset root changed during its bounded read")
    except ClearQualificationError:
        raise
    except OSError as exc:
        raise ClearQualificationError(f"{label} path changed during its bounded read") from exc


def _hash_below_root(
    root: Path,
    relative: str,
    *,
    expected_size: int,
    label: str,
    seen_file_ids: set[tuple[int, int]] | None = None,
) -> str:
    bound = _open_below_root(root, relative, label=label)
    try:
        opened = os.fstat(bound.descriptor)
        if opened.st_nlink != 1:
            raise ClearQualificationError(f"{label} files must not be hard-link aliases")
        file_id = (opened.st_dev, opened.st_ino)
        if seen_file_ids is not None:
            if file_id in seen_file_ids:
                raise ClearQualificationError(f"{label} files must not be hard-link aliases")
            seen_file_ids.add(file_id)
        if opened.st_size != expected_size:
            raise ClearQualificationError(f"{label} size does not match")
        digest = hashlib.sha256()
        remaining = expected_size
        while remaining:
            chunk = os.read(bound.descriptor, min(remaining, 1 << 20))
            if not chunk:
                raise ClearQualificationError(f"{label} ended during its bounded read")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(bound.descriptor, 1):
            raise ClearQualificationError(f"{label} grew during its bounded read")
        if _stat_identity(os.fstat(bound.descriptor)) != _stat_identity(opened):
            raise ClearQualificationError(f"{label} changed during its bounded read")
        _revalidate_bound_file(bound, opened, label=label)
        return digest.hexdigest()
    except ClearQualificationError:
        raise
    except OSError as exc:
        raise ClearQualificationError(f"{label} could not be read safely") from exc
    finally:
        bound.close()


def _read_below_root(
    root: Path, relative: str, *, expected_size: int, limit: int, label: str
) -> bytes:
    if expected_size > limit:
        raise ClearQualificationError(f"{label} exceeds its byte limit")
    bound = _open_below_root(root, relative, label=label)
    try:
        opened = os.fstat(bound.descriptor)
        if opened.st_nlink != 1:
            raise ClearQualificationError(f"{label} files must not be hard-link aliases")
        if opened.st_size != expected_size:
            raise ClearQualificationError(f"{label} size does not match")
        with os.fdopen(bound.descriptor, "rb") as stream:
            bound.descriptor = -1
            raw = stream.read(limit + 1)
            after = os.fstat(stream.fileno())
        if len(raw) != expected_size or len(raw) > limit:
            raise ClearQualificationError(f"{label} bounded read does not match")
        if _stat_identity(after) != _stat_identity(opened):
            raise ClearQualificationError(f"{label} changed during its bounded read")
        _revalidate_bound_file(bound, opened, label=label)
        return raw
    except ClearQualificationError:
        raise
    except OSError as exc:
        raise ClearQualificationError(f"{label} could not be read safely") from exc
    finally:
        bound.close()


def _verify_review_documents(
    value: object,
    *,
    root: Path,
    acquisition: Mapping[str, object],
    rights: Mapping[str, object],
    dataset_sha256: str,
    splits: list[Mapping[str, object]],
) -> tuple[int, list[dict[str, object]]]:
    if type(value) is not list or len(value) != 3:
        raise ClearQualificationError("review documents must contain exactly three records")
    expected_roles = {"acquisition-review", "rights-storage-review", "split-review"}
    seen_roles: set[str] = set()
    seen_paths: set[str] = set()
    total_bytes = 0
    identities: list[dict[str, object]] = []
    for index, raw_item in enumerate(value):
        item = _object(raw_item, f"review document {index}")
        _keys(item, {"role", "path", "size_bytes", "sha256"}, f"review document {index}")
        role = _exact_str(item["role"], "review role", maximum=32)
        if role not in expected_roles or role in seen_roles:
            raise ClearQualificationError("review document roles must match the required reviews")
        identity = _file_identity(
            {name: item[name] for name in ("path", "size_bytes", "sha256")},
            label=f"review document {index}",
        )
        if identity.path in seen_paths:
            raise ClearQualificationError("review document paths must be unique")
        review_raw = _read_below_root(
            root,
            identity.path,
            expected_size=identity.size_bytes,
            limit=MAX_MANIFEST_BYTES,
            label="review document",
        )
        if hashlib.sha256(review_raw).hexdigest() != identity.sha256:
            raise ClearQualificationError("review document SHA-256 does not match")
        review = _decode(review_raw, limit=MAX_MANIFEST_BYTES, label=role)
        authentication = "external-review-record-not-authenticated-by-asi"
        if role == "acquisition-review":
            _keys(
                review,
                {
                    "schema_version",
                    "decision",
                    "reviewer",
                    "provider_locator",
                    "provider_snapshot_sha256",
                    "archives",
                    "authentication",
                },
                role,
            )
            _exact_str(review["reviewer"], "acquisition reviewer", maximum=256)
            if (
                review["schema_version"] != ACQUISITION_REVIEW_SCHEMA
                or review["decision"] != "accepted-for-local-development"
                or review["provider_locator"] != acquisition["provider_locator"]
                or review["provider_snapshot_sha256"]
                != acquisition["provider_snapshot_sha256"]
                or review["archives"] != acquisition["archives"]
                or review["authentication"] != authentication
            ):
                raise ClearQualificationError("acquisition review semantics do not match")
        elif role == "rights-storage-review":
            _keys(
                review,
                {
                    "schema_version",
                    "decision",
                    "reviewer",
                    "reviewed_scopes",
                    "storage_approval_id",
                    "authentication",
                },
                role,
            )
            _exact_str(review["reviewer"], "rights reviewer", maximum=256)
            _exact_str(review["storage_approval_id"], "storage approval id", maximum=256)
            if (
                review["schema_version"] != RIGHTS_REVIEW_SCHEMA
                or review["decision"] != rights["decision"]
                or review["reviewed_scopes"]
                != [
                    "yfcc-terms",
                    "flickr-asset-terms",
                    "takedown-process",
                    "approved-storage",
                ]
                or review["authentication"] != authentication
            ):
                raise ClearQualificationError("rights and storage review semantics do not match")
        else:
            _keys(
                review,
                {
                    "schema_version",
                    "decision",
                    "reviewer",
                    "dataset_sha256",
                    "protocol",
                    "buckets",
                    "years",
                    "splits",
                    "authentication",
                },
                role,
            )
            _exact_str(review["reviewer"], "split reviewer", maximum=256)
            if (
                review["schema_version"] != SPLIT_REVIEW_SCHEMA
                or review["decision"] != "accepted-prepared-streaming-splits"
                or review["dataset_sha256"] != dataset_sha256
                or review["protocol"] != "streaming-near-future"
                or review["buckets"] != list(BUCKETS)
                or review["years"] != list(YEARS)
                or review["splits"] != splits
                or review["authentication"] != authentication
            ):
                raise ClearQualificationError("prepared split review semantics do not match")
        seen_roles.add(role)
        seen_paths.add(identity.path)
        total_bytes += identity.size_bytes
        identities.append(
            {
                "role": role,
                "path": identity.path,
                "size_bytes": identity.size_bytes,
                "sha256": identity.sha256,
            }
        )
    return total_bytes, identities


def _verify_acquisition(
    value: object, *, receipt: ClearDatasetReceipt
) -> Mapping[str, object]:
    acquisition = _object(value, "acquisition")
    _keys(
        acquisition,
        {
            "mode",
            "provider_locator",
            "provider_snapshot_sha256",
            "provider_checksums_published",
            "archives",
        },
        "acquisition",
    )
    if acquisition["mode"] != "independently-reviewed-acquisition":
        raise ClearQualificationError("acquisition mode is not independently reviewed")
    locator = _https_locator(acquisition["provider_locator"])
    snapshot = _sha256(acquisition["provider_snapshot_sha256"], "provider snapshot sha256")
    if acquisition["provider_checksums_published"] is not False:
        raise ClearQualificationError("provider checksum disclosure contradicts qualification")
    archive_values = acquisition["archives"]
    if type(archive_values) is not list:
        raise ClearQualificationError("acquisition archives must be an exact list")
    expected = [asdict(archive) for archive in receipt.archives]
    if archive_values != expected:
        raise ClearQualificationError("acquisition archives do not match the dataset receipt")
    return {
        "mode": "independently-reviewed-acquisition",
        "provider_locator": locator,
        "provider_snapshot_sha256": snapshot,
        "provider_checksums_published": False,
        "archives": expected,
    }


def _verify_rights_and_storage(value: object) -> Mapping[str, object]:
    review = _object(value, "rights and storage")
    expected = {
        "decision",
        "yfcc_terms_reviewed",
        "flickr_asset_terms_reviewed",
        "takedown_process_documented",
        "storage_approved",
        "authentication",
    }
    _keys(review, expected, "rights and storage")
    if (
        review["decision"] != "approved-local-development-only"
        or review["authentication"] != "external-review-record-not-authenticated-by-asi"
        or any(review[name] is not True for name in expected - {"decision", "authentication"})
    ):
        raise ClearQualificationError("rights and storage review is not approved and complete")
    return dict(review)


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _verify_split_index(
    identity_value: object,
    *,
    root: Path,
    label: str,
    seen_ids: set[str],
    seen_paths: set[str],
    seen_file_ids: set[tuple[int, int]],
    seen_index_file_ids: set[tuple[int, int]],
    seen_sample_digests: set[str],
) -> tuple[int, int, set[int]]:
    identity_raw = _object(identity_value, label)
    _keys(identity_raw, {"path", "size_bytes", "sha256", "sample_count"}, label)
    identity = _file_identity(
        {name: identity_raw[name] for name in ("path", "size_bytes", "sha256")}, label=label
    )
    if identity.size_bytes > MAX_INDEX_BYTES:
        raise ClearQualificationError(f"{label} exceeds its byte limit")
    expected_count = _exact_int(
        identity_raw["sample_count"],
        f"{label} sample count",
        minimum=CLASS_COUNT,
        maximum=MAX_SAMPLES_PER_BUCKET,
    )
    bound = _open_below_root(root, identity.path, label=label)
    count = 0
    sample_bytes = 0
    classes: set[int] = set()
    digest = hashlib.sha256()
    try:
        opened = os.fstat(bound.descriptor)
        if opened.st_nlink != 1:
            raise ClearQualificationError("split index files must not be hard-link aliases")
        index_file_id = (opened.st_dev, opened.st_ino)
        if index_file_id in seen_index_file_ids:
            raise ClearQualificationError("split index files must not be hard-link aliases")
        seen_index_file_ids.add(index_file_id)
        if opened.st_size != identity.size_bytes:
            raise ClearQualificationError(f"{label} size does not match")
        with os.fdopen(bound.descriptor, "rb") as stream:
            bound.descriptor = -1
            while True:
                line = stream.readline(MAX_INDEX_LINE_BYTES + 1)
                if not line:
                    break
                digest.update(line)
                if len(line) > MAX_INDEX_LINE_BYTES or not line.endswith(b"\n"):
                    raise ClearQualificationError(f"{label} contains an oversized or partial line")
                record = _decode(line, limit=MAX_INDEX_LINE_BYTES, label=f"{label} record")
                _keys(
                    record,
                    {"sample_id", "path", "size_bytes", "sha256", "class_index"},
                    f"{label} record",
                )
                if line != _canonical(record) + b"\n":
                    raise ClearQualificationError(f"{label} records must use canonical JSON")
                sample_id = _exact_str(record["sample_id"], "sample id", maximum=256)
                sample_path = _safe_relative_path(record["path"])
                if sample_id in seen_ids or sample_path in seen_paths:
                    raise ClearQualificationError("sample paths and IDs must be globally unique")
                size = _exact_int(record["size_bytes"], "sample size", minimum=1, maximum=1 << 40)
                sample_digest = _sha256(record["sha256"], "sample sha256")
                if sample_digest in seen_sample_digests:
                    raise ClearQualificationError(
                        "sample content identities must be globally unique"
                    )
                class_index = _exact_int(
                    record["class_index"], "class index", maximum=CLASS_COUNT - 1
                )
                if (
                    _hash_below_root(
                        root,
                        sample_path,
                        expected_size=size,
                        label="sample",
                        seen_file_ids=seen_file_ids,
                    )
                    != sample_digest
                ):
                    raise ClearQualificationError("sample SHA-256 does not match the index")
                seen_ids.add(sample_id)
                seen_paths.add(sample_path)
                seen_sample_digests.add(sample_digest)
                classes.add(class_index)
                sample_bytes += size
                count += 1
                if count > expected_count:
                    raise ClearQualificationError(f"{label} contains more samples than declared")
            after = os.fstat(stream.fileno())
        if _stat_identity(after) != _stat_identity(opened):
            raise ClearQualificationError(f"{label} changed during its bounded read")
        _revalidate_bound_file(bound, opened, label=label)
    except ClearQualificationError:
        raise
    except OSError as exc:
        raise ClearQualificationError(f"{label} could not be read safely") from exc
    finally:
        bound.close()
    if count != expected_count:
        raise ClearQualificationError(f"{label} sample count does not match")
    if digest.hexdigest() != identity.sha256:
        raise ClearQualificationError(f"{label} SHA-256 does not match")
    if classes != set(range(CLASS_COUNT)):
        raise ClearQualificationError(f"{label} must cover all 100 classes")
    return count, sample_bytes, classes


def verify_runner_input_manifest(
    raw: bytes,
    *,
    root: Path,
    dataset_receipt: ClearDatasetReceipt,
) -> ClearRunnerInputReceipt:
    """Bind reviewed acquisition and exact prepared samples without authorizing execution."""
    if type(raw) is not bytes or not isinstance(root, Path):
        raise TypeError("raw must be bytes and root must be a Path")
    dataset_receipt = _validate_dataset_receipt(dataset_receipt)
    payload = _decode(raw, limit=MAX_MANIFEST_BYTES, label="runner input manifest")
    _keys(
        payload,
        {
            "schema_version",
            "dataset_sha256",
            "protocol",
            "acquisition",
            "review_documents",
            "rights_and_storage",
            "splits",
        },
        "runner input manifest",
    )
    if payload["schema_version"] != RUNNER_INPUT_SCHEMA:
        raise ClearQualificationError("runner input schema drift")
    if payload["dataset_sha256"] != dataset_receipt.dataset_sha256:
        raise ClearQualificationError("runner input dataset identity drift")
    if payload["protocol"] != "streaming-near-future":
        raise ClearQualificationError("runner input protocol drift")
    root_resolved = _verified_root(root)
    acquisition = _verify_acquisition(payload["acquisition"], receipt=dataset_receipt)
    rights = _verify_rights_and_storage(payload["rights_and_storage"])
    split_values = payload["splits"]
    if type(split_values) is not list or len(split_values) != len(BUCKETS):
        raise ClearQualificationError("runner inputs must cover every temporal bucket")
    seen_index_paths: set[str] = set()
    seen_sample_ids: set[str] = set()
    seen_sample_paths: set[str] = set()
    seen_sample_digests: set[str] = set()
    seen_sample_file_ids: set[tuple[int, int]] = set()
    seen_index_file_ids: set[tuple[int, int]] = set()
    train_counts: list[int] = []
    evaluation_counts: list[int] = []
    sample_bytes = 0
    index_bytes = 0
    canonical_splits: list[Mapping[str, object]] = []
    for offset, raw_split in enumerate(split_values):
        split = _object(raw_split, f"split {offset}")
        _keys(split, {"bucket", "year", "train_index", "evaluation_index"}, f"split {offset}")
        if split["bucket"] != BUCKETS[offset] or split["year"] != YEARS[offset]:
            raise ClearQualificationError("runner input temporal split identity drift")
        for name in ("train_index", "evaluation_index"):
            index = _object(split[name], f"split {offset} {name}")
            path_text = _safe_relative_path(index.get("path"))
            if path_text in seen_index_paths:
                raise ClearQualificationError("split index paths must be unique")
            seen_index_paths.add(path_text)
            index_bytes += _exact_int(
                index.get("size_bytes"), "index size", maximum=MAX_INDEX_BYTES
            )
            if index_bytes > MAX_INDEX_BYTES:
                raise ClearQualificationError("aggregate index bytes exceed the byte limit")
        train_count, train_bytes, _ = _verify_split_index(
            split["train_index"],
            root=root_resolved,
            label=f"bucket {BUCKETS[offset]} train index",
            seen_ids=seen_sample_ids,
            seen_paths=seen_sample_paths,
            seen_file_ids=seen_sample_file_ids,
            seen_index_file_ids=seen_index_file_ids,
            seen_sample_digests=seen_sample_digests,
        )
        evaluation_count, evaluation_bytes, _ = _verify_split_index(
            split["evaluation_index"],
            root=root_resolved,
            label=f"bucket {BUCKETS[offset]} evaluation index",
            seen_ids=seen_sample_ids,
            seen_paths=seen_sample_paths,
            seen_file_ids=seen_sample_file_ids,
            seen_index_file_ids=seen_index_file_ids,
            seen_sample_digests=seen_sample_digests,
        )
        train_counts.append(train_count)
        evaluation_counts.append(evaluation_count)
        sample_bytes += train_bytes + evaluation_bytes
        if sample_bytes > _INT64_MAX:
            raise ClearQualificationError("aggregate sample bytes exceed signed-int64 capacity")
        canonical_splits.append(dict(split))
    if tuple(train_counts) != dataset_receipt.samples_per_bucket:
        raise ClearQualificationError(
            "prepared training counts do not match the qualified dataset receipt"
        )
    review_bytes, reviews = _verify_review_documents(
        payload["review_documents"],
        root=root_resolved,
        acquisition=acquisition,
        rights=rights,
        dataset_sha256=dataset_receipt.dataset_sha256,
        splits=canonical_splits,
    )
    source_identity = tuple(sorted(current_source_identity().items()))
    runtime = tuple(sorted(runtime_identity().items()))
    identity = {
        "schema_version": RUNNER_INPUT_SCHEMA,
        "dataset_sha256": dataset_receipt.dataset_sha256,
        "protocol": "streaming-near-future",
        "acquisition": acquisition,
        "review_documents": reviews,
        "rights_and_storage": rights,
        "splits": canonical_splits,
        "current_source_sha256": source_identity,
        "runtime_identity": runtime,
        "external_reviews_authenticated": False,
        "provider_snapshot_bytes_verified": False,
        "redistribution_authorized": False,
        "execution_authorized": False,
    }
    return ClearRunnerInputReceipt(
        dataset_sha256=dataset_receipt.dataset_sha256,
        runner_input_sha256=hashlib.sha256(_canonical(identity)).hexdigest(),
        train_samples_per_bucket=tuple(train_counts),
        evaluation_samples_per_bucket=tuple(evaluation_counts),
        train_sample_count=sum(train_counts),
        evaluation_sample_count=sum(evaluation_counts),
        class_count=CLASS_COUNT,
        sample_bytes=sample_bytes,
        index_bytes=index_bytes,
        review_bytes=review_bytes,
        training_observations=sum(train_counts) * 100,
        optimizer_updates=sum(math.ceil(count / 256) * 100 for count in train_counts),
        model_queries=sum(evaluation_counts) * len(BUCKETS),
        data_samples_read=sum(train_counts) * 100
        + sum(evaluation_counts) * len(BUCKETS),
        current_source_sha256=source_identity,
        runtime_identity=runtime,
        external_reviews_authenticated=False,
        provider_snapshot_bytes_verified=False,
        redistribution_authorized=False,
        execution_authorized=False,
    )


def runtime_identity() -> Mapping[str, str]:
    packages: dict[str, str] = {}
    for name in ("jax", "numpy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "absent"
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        **packages,
    }


def current_source_identity() -> Mapping[str, str]:
    """Bind the exact current qualification implementation used by a plan."""
    identities: dict[str, str] = {}
    for relative in _CURRENT_SOURCE_PATHS:
        path = _REPOSITORY_ROOT / relative
        descriptor: int | None = None
        try:
            before = path.stat(follow_symlinks=False)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size > _MAX_CURRENT_SOURCE_BYTES
            ):
                raise ClearQualificationError("current qualification source is not bounded")
            descriptor = os.open(
                path,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            )
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or _stat_identity(opened) != _stat_identity(before)
            ):
                raise ClearQualificationError("current qualification source changed before read")
            with os.fdopen(descriptor, "rb") as stream:
                descriptor = None
                raw = stream.read(_MAX_CURRENT_SOURCE_BYTES + 1)
                after = os.fstat(stream.fileno())
            if (
                len(raw) != opened.st_size
                or len(raw) > _MAX_CURRENT_SOURCE_BYTES
                or _stat_identity(after) != _stat_identity(opened)
                or _stat_identity(path.stat(follow_symlinks=False))
                != _stat_identity(opened)
            ):
                raise ClearQualificationError("current qualification source changed during read")
        except OSError as exc:
            raise ClearQualificationError("current qualification source is unavailable") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
        identities[relative] = hashlib.sha256(raw).hexdigest()
    return identities


def execution_config(*, mechanism_enabled: bool) -> Mapping[str, object]:
    if type(mechanism_enabled) is not bool:
        raise TypeError("mechanism_enabled must be an exact bool")
    base: dict[str, object] = {
        "dataset": DATASET_NAME,
        "protocol": "streaming-near-future",
        "model": "resnet18-from-scratch",
        "batch_size": 256,
        "epochs_per_bucket": 100,
        "optimizer": {"name": "sgd", "learning_rate": 0.01, "momentum": 0.9, "weight_decay": 1e-5},
        "scheduler": {"name": "step", "step_size_epochs": 30, "gamma": 0.1},
        "image_size": 224,
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }
    if mechanism_enabled:
        base["candidate_mechanism"] = "unimplemented-placeholder"
    return base


def qualification_plan(receipt: ClearDatasetReceipt) -> Mapping[str, object]:
    """Create a matched plan.  It has no execution-authority field by design."""
    receipt = _validate_dataset_receipt(receipt)
    control = execution_config(mechanism_enabled=False)
    mechanism_off = execution_config(mechanism_enabled=False)
    if control != mechanism_off:
        raise AssertionError("mechanism-off must reduce exactly to the control")
    epochs = 100
    batch = 256
    updates = sum(math.ceil(count / batch) * epochs for count in receipt.samples_per_bucket)
    train_observations = receipt.sample_count * epochs
    # The full 10x10 matrix is evaluated after each training bucket.
    model_queries = receipt.sample_count * len(BUCKETS)
    axes = [
        {"seed": seed, "arm": arm}
        for seed in DEV_SEEDS
        for arm in ("control", "mechanism-off")
    ]
    return {
        "schema_version": SCHEMA,
        "classification": "development-only-permanently-nonpromoting",
        "paper_revision": PAPER_REVISION,
        "source_revisions": {
            "curation": CURATION_COMMIT,
            "reference_runner": REFERENCE_COMMIT,
            "avalanche": AVALANCHE_COMMIT,
        },
        "current_source_sha256": current_source_identity(),
        "dataset_sha256": receipt.dataset_sha256,
        "runtime": runtime_identity(),
        "axes": axes,
        "control_config": control,
        "mechanism_off_config": mechanism_off,
        "metrics": [
            "accuracy",
            "in_domain",
            "next_domain",
            "forward_transfer",
            "backward_transfer",
        ],
        "resource_budget_per_axis": {
            "archive_bytes": receipt.archive_bytes,
            "training_observations": train_observations,
            "data_samples_read": train_observations + model_queries,
            "optimizer_updates": updates,
            "model_queries": model_queries,
            "environment_steps": 0,
            "timing": "telemetry-only",
            "persistent_bytes": "runner-receipt-required",
        },
        "negative_retention_required": True,
        "promotion_authorized": False,
        "execution_authorized": False,
    }


def _metric_values(matrix: list[list[float]]) -> Mapping[str, float]:
    rows = len(matrix)
    diagonal = [matrix[index][index] for index in range(rows)]
    lower = [matrix[i][j] for i in range(rows) for j in range(i)]
    upper = [matrix[i][j] for i in range(rows) for j in range(i + 1, rows)]
    seen = [matrix[i][j] for i in range(rows) for j in range(i + 1)]
    next_domain = [matrix[index][index + 1] for index in range(rows - 1)]
    return {
        "accuracy": sum(seen) / len(seen),
        "in_domain": sum(diagonal) / len(diagonal),
        "next_domain": sum(next_domain) / len(next_domain),
        "forward_transfer": sum(upper) / len(upper),
        "backward_transfer": sum(lower) / len(lower),
    }


def _validate_plan_resource_budget(value: object) -> Mapping[str, object]:
    budget = _object(value, "expected plan resource budget")
    _keys(
        budget,
        {
            "archive_bytes",
            "training_observations",
            "data_samples_read",
            "optimizer_updates",
            "model_queries",
            "environment_steps",
            "timing",
            "persistent_bytes",
        },
        "expected plan resource budget",
    )
    if (
        budget["timing"] != "telemetry-only"
        or budget["persistent_bytes"] != "runner-receipt-required"
    ):
        raise ClearQualificationError("expected resource policy differs from the frozen plan")
    _exact_int(
        budget["archive_bytes"],
        "expected archive_bytes",
        maximum=MAX_ARCHIVES * (1 << 50),
    )
    training_observations = _exact_int(
        budget["training_observations"],
        "expected training_observations",
        minimum=100 * len(BUCKETS),
        maximum=100 * MAX_SAMPLES_PER_BUCKET * len(BUCKETS),
    )
    if training_observations % 100:
        raise ClearQualificationError("training_observations do not encode 100 exact epochs")
    sample_count = training_observations // 100
    model_queries = _exact_int(
        budget["model_queries"], "expected model_queries", maximum=(1 << 63) - 1
    )
    data_samples_read = _exact_int(
        budget["data_samples_read"], "expected data_samples_read", maximum=(1 << 63) - 1
    )
    optimizer_updates = _exact_int(
        budget["optimizer_updates"], "expected optimizer_updates", maximum=(1 << 63) - 1
    )
    minimum_batches = max(len(BUCKETS), math.ceil(sample_count / 256))
    maximum_batches = math.ceil((sample_count - len(BUCKETS) + 1) / 256) + len(BUCKETS) - 1
    if (
        model_queries != sample_count * len(BUCKETS)
        or data_samples_read != training_observations + model_queries
        or optimizer_updates % 100
        or not minimum_batches * 100 <= optimizer_updates <= maximum_batches * 100
        or budget["environment_steps"] != 0
    ):
        raise ClearQualificationError("expected resource budget is not derivable from CLEAR")
    return budget


def validate_result(
    raw: bytes,
    *,
    expected_plan: Mapping[str, object],
) -> Mapping[str, object]:
    """Validate the narrow result envelope; scores remain uninterpreted development data."""
    plan = _object(expected_plan, "expected CLEAR plan")
    _keys(
        plan,
        {
            "schema_version",
            "classification",
            "paper_revision",
            "source_revisions",
            "current_source_sha256",
            "dataset_sha256",
            "runtime",
            "axes",
            "control_config",
            "mechanism_off_config",
            "metrics",
            "resource_budget_per_axis",
            "negative_retention_required",
            "promotion_authorized",
            "execution_authorized",
        },
        "expected CLEAR plan",
    )
    if (
        plan["schema_version"] != SCHEMA
        or plan["classification"] != "development-only-permanently-nonpromoting"
        or plan["paper_revision"] != PAPER_REVISION
        or plan["source_revisions"]
        != {
            "curation": CURATION_COMMIT,
            "reference_runner": REFERENCE_COMMIT,
            "avalanche": AVALANCHE_COMMIT,
        }
        or plan["current_source_sha256"] != current_source_identity()
        or plan["runtime"] != runtime_identity()
        or plan["axes"]
        != [
            {"seed": seed, "arm": arm}
            for seed in DEV_SEEDS
            for arm in ("control", "mechanism-off")
        ]
        or plan["control_config"] != execution_config(mechanism_enabled=False)
        or plan["mechanism_off_config"] != execution_config(mechanism_enabled=False)
        or plan["metrics"]
        != [
            "accuracy",
            "in_domain",
            "next_domain",
            "forward_transfer",
            "backward_transfer",
        ]
        or plan["negative_retention_required"] is not True
        or plan["promotion_authorized"] is not False
        or plan["execution_authorized"] is not False
    ):
        raise ClearQualificationError("expected plan violates the frozen qualification policy")
    _sha256(plan["dataset_sha256"], "expected dataset_sha256")
    plan_digest = hashlib.sha256(_canonical(plan)).hexdigest()
    expected_resource_budget = _validate_plan_resource_budget(plan["resource_budget_per_axis"])
    payload = _decode(raw, limit=MAX_RESULT_BYTES, label="CLEAR result")
    _keys(
        payload,
        {
            "schema_version",
            "plan_sha256",
            "status",
            "promotion_authorized",
            "negative_retained",
            "accuracy_matrix",
            "metrics",
            "resource_receipts",
        },
        "CLEAR result",
    )
    if payload["schema_version"] != SCHEMA or payload["plan_sha256"] != plan_digest:
        raise ClearQualificationError("result provenance drift")
    if payload["status"] not in ("completed-development", "negative-development"):
        raise ClearQualificationError("result status is not allowed")
    if payload["promotion_authorized"] is not False or payload["negative_retained"] is not True:
        raise ClearQualificationError("result violates nonpromotion or negative retention")
    matrix_value = payload["accuracy_matrix"]
    if type(matrix_value) is not list or len(matrix_value) != len(BUCKETS):
        raise ClearQualificationError("accuracy matrix must be an exact 10x10 list")
    matrix: list[list[float]] = []
    for row_value in matrix_value:
        if type(row_value) is not list or len(row_value) != len(BUCKETS):
            raise ClearQualificationError("accuracy matrix must be an exact 10x10 list")
        row: list[float] = []
        for score in row_value:
            if type(score) is not float or not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ClearQualificationError("accuracy entries must be finite exact floats")
            row.append(score)
        matrix.append(row)
    metric_value = _object(payload["metrics"], "CLEAR metrics")
    expected_metrics = _metric_values(matrix)
    _keys(metric_value, set(expected_metrics), "CLEAR metrics")
    if any(type(value) is not float or not math.isfinite(value) for value in metric_value.values()):
        raise ClearQualificationError("CLEAR metrics must be finite exact floats")
    if metric_value != expected_metrics:
        raise ClearQualificationError("CLEAR metrics do not replay from the accuracy matrix")
    resources = _object(payload["resource_receipts"], "resource receipts")
    _keys(
        resources,
        {
            "persistent_bytes",
            "archive_bytes",
            "training_observations",
            "data_samples_read",
            "optimizer_updates",
            "model_queries",
            "environment_steps",
            "wall_seconds_telemetry",
        },
        "resource receipts",
    )
    for name in resources:
        maximum = (1 << 63) - 1 if name != "wall_seconds_telemetry" else (1 << 53)
        _exact_int(resources[name], name, maximum=maximum)
    for name in (
        "archive_bytes",
        "training_observations",
        "data_samples_read",
        "optimizer_updates",
        "model_queries",
        "environment_steps",
    ):
        expected = _exact_int(
            expected_resource_budget[name],
            f"expected {name}",
            maximum=(1 << 63) - 1,
        )
        if resources[name] != expected:
            raise ClearQualificationError(f"{name} does not match the frozen plan")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--runner-input-manifest", type=Path)
    args = parser.parse_args(argv)
    raw = load_dataset_manifest(args.manifest)
    receipt = verify_dataset_manifest(raw, root=args.dataset_root)
    plan = qualification_plan(receipt)
    if args.runner_input_manifest is None:
        output: object = plan
    else:
        runner_raw = load_dataset_manifest(args.runner_input_manifest)
        runner_receipt = verify_runner_input_manifest(
            runner_raw, root=args.dataset_root, dataset_receipt=receipt
        )
        output = {
            "qualification_plan": plan,
            "runner_input_receipt": asdict(runner_receipt),
            "execution_authorized": False,
        }
    print(_canonical(output).decode())
    return 0


if __name__ == "__main__":
    sys.exit(main())
