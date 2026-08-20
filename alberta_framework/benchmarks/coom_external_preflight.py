"""Verify a caller-produced COOM execution receipt without running COOM.

This boundary reads only caller-selected local files.  It does not clone COOM,
download Doom assets, import ViZDoom, or launch an external process.  A passing
receipt establishes local byte consistency and a repeated-trace contract; it
does not authenticate the upstream tree, authorize execution, or create a
benchmark result.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import cast

import alberta_framework.benchmarks.coom_qualification as coom_qualification_module
import alberta_framework.benchmarks.external_qualification as external_qualification_module
from alberta_framework.benchmarks.coom_qualification import (
    CD8_TASKS,
    CO8_TASKS,
    COOM_COMMIT,
    COOM_REPOSITORY,
)
from alberta_framework.benchmarks.external_qualification import qualification_plan
from alberta_framework.benchmarks.qualification_provenance import (
    QualificationIdentity,
    collect_qualification_identity,
)

COOM_EXTERNAL_PREFLIGHT_SCHEMA = "asi.coom_external_preflight.local.v1"
MAX_MANIFEST_BYTES = 1 << 20
MAX_FILE_BYTES = 1 << 40
MAX_FILES = 64
_MAX_TEXT_BYTES = 512
_INT64_MAX = (1 << 63) - 1
_SOURCE_ROLES = (
    "source-archive",
    "license",
    "package-metadata",
    "environment-registration",
    "sequence-definition",
)
_ASSET_ROLES = ("scenario-config", "scenario-asset")
_PACKAGE_NAMES = ("vizdoom", "opencv-python", "scipy", "gymnasium")
_WORKLOAD_REGISTRY = (
    ("asset_roles", _ASSET_ROLES),
    ("cd8_tasks", CD8_TASKS),
    ("co8_tasks", CO8_TASKS),
    ("manifest_limit", MAX_MANIFEST_BYTES),
    ("source_roles", _SOURCE_ROLES),
)
_PAPER_REGISTRY = (
    ("commit", COOM_COMMIT),
    ("repository", COOM_REPOSITORY),
    ("schema", COOM_EXTERNAL_PREFLIGHT_SCHEMA),
)


class COOMExternalPreflightError(ValueError):
    """A local COOM execution receipt failed closed."""


@dataclasses.dataclass(frozen=True, slots=True)
class COOMExternalPreflightReceipt:
    """Non-authorizing verification result for one local receipt."""

    schema_version: str
    manifest_sha256: str
    source_repository: str
    source_commit: str
    source_git_tree_oid: str
    source_archive_sha256: str
    runtime_identity_sha256: str
    assets_identity_sha256: str
    config_sha256: str
    workload_identity_sha256: str
    trace_sha256: str
    sequence: str
    local_files_verified: int
    local_bytes_verified: int
    source_identity_authenticated: bool
    license_review_authenticated: bool
    runtime_identity_authenticated: bool
    asset_semantics_authenticated: bool
    deterministic_trace_pair_verified: bool
    trace_execution_authenticated: bool
    external_runtime_executed_by_caller: bool
    execution_authorized: bool
    promotion_authorized: bool
    benchmark_result_claimed: bool
    remaining_qualification_gates: tuple[str, ...]
    qualification_identity: QualificationIdentity

    def __post_init__(self) -> None:
        if self.schema_version != COOM_EXTERNAL_PREFLIGHT_SCHEMA:
            raise COOMExternalPreflightError("preflight receipt schema drift")
        for name in (
            "manifest_sha256",
            "source_archive_sha256",
            "runtime_identity_sha256",
            "assets_identity_sha256",
            "config_sha256",
            "workload_identity_sha256",
            "trace_sha256",
        ):
            _sha256(getattr(self, name), name)
        _hex(self.source_git_tree_oid, "source_git_tree_oid", 40)
        if (
            self.source_repository != COOM_REPOSITORY
            or self.source_commit != COOM_COMMIT
            or self.sequence not in ("CD8", "CO8")
        ):
            raise COOMExternalPreflightError("preflight receipt identity drift")
        _exact_int(self.local_files_verified, "local_files_verified", minimum=1)
        _exact_int(self.local_bytes_verified, "local_bytes_verified", minimum=1)
        if (
            self.source_identity_authenticated is not False
            or self.license_review_authenticated is not False
            or self.runtime_identity_authenticated is not False
            or self.asset_semantics_authenticated is not False
            or self.deterministic_trace_pair_verified is not True
            or self.trace_execution_authenticated is not False
            or self.external_runtime_executed_by_caller is not True
            or self.execution_authorized is not False
            or self.promotion_authorized is not False
            or self.benchmark_result_claimed is not False
        ):
            raise COOMExternalPreflightError("receipt may not invent authentication or authority")
        if self.remaining_qualification_gates != qualification_plan(1582).required_gates:
            raise COOMExternalPreflightError("qualification gates must remain open")
        if type(self.qualification_identity) is not QualificationIdentity:
            raise COOMExternalPreflightError("qualification identity must use the exact type")

    def to_payload(self) -> dict[str, object]:
        return cast(dict[str, object], json.loads(json.dumps(dataclasses.asdict(self))))


def _current_identity() -> QualificationIdentity:
    return collect_qualification_identity(
        lane_module=sys.modules[__name__],
        dependency_modules=(coom_qualification_module, external_qualification_module),
        workload_registry=_WORKLOAD_REGISTRY,
        paper_registry=_PAPER_REGISTRY,
    )


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _exact_object(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise COOMExternalPreflightError(f"{label} must be an exact JSON object")
    return cast(dict[str, object], value)


def _keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise COOMExternalPreflightError(f"{label} fields differ from the schema")


def _exact_string(value: object, label: str, *, maximum: int = _MAX_TEXT_BYTES) -> str:
    if type(value) is not str or not value:
        raise COOMExternalPreflightError(f"{label} must be a non-empty exact string")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise COOMExternalPreflightError(f"{label} must contain valid Unicode") from exc
    if len(encoded) > maximum:
        raise COOMExternalPreflightError(f"{label} exceeds its byte limit")
    return value


def _exact_int(
    value: object,
    label: str,
    *,
    minimum: int = 0,
    maximum: int = _INT64_MAX,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise COOMExternalPreflightError(f"{label} must be an exact integer in range")
    return value


def _exact_bool(value: object, label: str, expected: bool) -> None:
    if type(value) is not bool or value is not expected:
        raise COOMExternalPreflightError(f"{label} differs from the required policy")


def _hex(value: object, label: str, length: int) -> str:
    text = _exact_string(value, label, maximum=length)
    if len(text) != length or any(character not in "0123456789abcdef" for character in text):
        raise COOMExternalPreflightError(f"{label} must be lowercase hexadecimal")
    return text


def _sha256(value: object, label: str) -> str:
    return _hex(value, label, 64)


def workload_identity_sha256(
    *,
    source_archive_sha256: object,
    runtime_identity_sha256: object,
    assets_identity_sha256: object,
    config_sha256: object,
) -> str:
    """Bind a caller trace to the exact verified workload identities."""
    identity = {
        "source_archive_sha256": _sha256(
            source_archive_sha256, "workload source archive SHA-256"
        ),
        "runtime_identity_sha256": _sha256(
            runtime_identity_sha256, "workload runtime identity SHA-256"
        ),
        "assets_identity_sha256": _sha256(
            assets_identity_sha256, "workload assets identity SHA-256"
        ),
        "config_sha256": _sha256(config_sha256, "workload configuration SHA-256"),
    }
    return hashlib.sha256(_canonical(identity)).hexdigest()


def _reject_constant(value: str) -> None:
    raise COOMExternalPreflightError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise COOMExternalPreflightError("duplicate JSON object key")
        result[key] = value
    return result


def _decode(raw: bytes) -> dict[str, object]:
    if type(raw) is not bytes:
        raise TypeError("manifest must be exact bytes")
    if len(raw) > MAX_MANIFEST_BYTES:
        raise COOMExternalPreflightError("manifest exceeds its byte limit")
    try:
        decoded = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise COOMExternalPreflightError("manifest is not valid strict JSON") from exc
    return _exact_object(decoded, "manifest")


def _directory_identity(item: os.stat_result) -> tuple[int, ...]:
    return (item.st_dev, item.st_ino, item.st_mode)


def _file_stat_identity(item: os.stat_result) -> tuple[int, ...]:
    return (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )


def _open_directory_nofollow(
    path: Path, label: str
) -> tuple[int, list[int], list[tuple[int, str, tuple[int, ...]]]]:
    """Open an entire directory path without following any component link."""
    absolute = path if path.is_absolute() else Path.cwd() / path
    anchor = Path(absolute.anchor)
    components = absolute.parts[1:]
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    descriptors: list[int] = []
    bindings: list[tuple[int, str, tuple[int, ...]]] = []
    try:
        current = os.open(anchor, flags)
        descriptors.append(current)
        for component in components:
            before = os.stat(component, dir_fd=current, follow_symlinks=False)
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise COOMExternalPreflightError(
                    f"{label} path may not traverse a link or non-directory"
                )
            following = os.open(component, flags, dir_fd=current)
            opened = os.fstat(following)
            identity = _directory_identity(opened)
            if identity != _directory_identity(before):
                os.close(following)
                raise COOMExternalPreflightError(
                    f"{label} directory changed before its bounded read"
                )
            bindings.append((current, component, identity))
            descriptors.append(following)
            current = following
        return current, descriptors, bindings
    except COOMExternalPreflightError:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise
    except OSError as exc:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise COOMExternalPreflightError(f"{label} path could not be opened safely") from exc


def _verify_directory_bindings(
    bindings: list[tuple[int, str, tuple[int, ...]]], label: str
) -> None:
    for parent, component, expected in reversed(bindings):
        current = os.stat(component, dir_fd=parent, follow_symlinks=False)
        if _directory_identity(current) != expected:
            raise COOMExternalPreflightError(
                f"{label} directory changed during its bounded read"
            )


def load_preflight_manifest(path: Path) -> bytes:
    """Read a bounded regular manifest without following any component link."""
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    descriptor = -1
    directory_descriptors: list[int] = []
    try:
        absolute = path if path.is_absolute() else Path.cwd() / path
        parent_descriptor, directory_descriptors, bindings = _open_directory_nofollow(
            absolute.parent, "manifest"
        )
        filename = absolute.name
        before = os.stat(filename, dir_fd=parent_descriptor, follow_symlinks=False)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise COOMExternalPreflightError("manifest must be a regular non-symlink file")
        if before.st_nlink != 1:
            raise COOMExternalPreflightError("manifest must not have a hard-link alias")
        if before.st_size > MAX_MANIFEST_BYTES:
            raise COOMExternalPreflightError("manifest exceeds its byte limit")
        descriptor = os.open(
            filename,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _file_stat_identity(
            opened
        ) != _file_stat_identity(before):
            raise COOMExternalPreflightError("manifest changed before its bounded read")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1 << 20))
            if not chunk:
                raise COOMExternalPreflightError("manifest ended during its bounded read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise COOMExternalPreflightError("manifest exceeds its byte limit")
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        current = os.stat(filename, dir_fd=parent_descriptor, follow_symlinks=False)
        if (
            len(raw) != opened.st_size
            or _file_stat_identity(after) != _file_stat_identity(opened)
            or _file_stat_identity(current) != _file_stat_identity(opened)
        ):
            raise COOMExternalPreflightError("manifest changed during its bounded read")
        _verify_directory_bindings(bindings, "manifest")
        return raw
    except COOMExternalPreflightError:
        raise
    except OSError as exc:
        raise COOMExternalPreflightError("manifest could not be read safely") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        for directory_descriptor in reversed(directory_descriptors):
            os.close(directory_descriptor)


def _relative_path(value: object, label: str) -> str:
    text = _exact_string(value, label, maximum=256)
    path = PurePosixPath(text)
    if not path.parts or path.is_absolute() or ".." in path.parts or path.as_posix() != text:
        raise COOMExternalPreflightError(f"{label} must be canonical and relative")
    return text


def _verify_file(root: Path, raw: object, label: str) -> tuple[str, int, str]:
    value = _exact_object(raw, label)
    _keys(value, {"path", "size_bytes", "sha256"}, label)
    relative = _relative_path(value["path"], f"{label} path")
    size = _exact_int(
        value["size_bytes"], f"{label} size", minimum=1, maximum=MAX_FILE_BYTES
    )
    expected = _sha256(value["sha256"], f"{label} sha256")
    descriptor = -1
    directory_descriptors: list[int] = []
    try:
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_DIRECTORY", 0)
        )
        root_descriptor, directory_descriptors, bindings = _open_directory_nofollow(
            root, label
        )
        parent_descriptor = root_descriptor
        components = PurePosixPath(relative).parts
        for component in components[:-1]:
            component_before = os.stat(
                component, dir_fd=parent_descriptor, follow_symlinks=False
            )
            if stat.S_ISLNK(component_before.st_mode) or not stat.S_ISDIR(
                component_before.st_mode
            ):
                raise COOMExternalPreflightError(
                    f"{label} path may not traverse a link or non-directory"
                )
            child_descriptor = os.open(
                component, directory_flags, dir_fd=parent_descriptor
            )
            child_opened = os.fstat(child_descriptor)
            child_identity = _directory_identity(child_opened)
            if child_identity != _directory_identity(component_before):
                os.close(child_descriptor)
                raise COOMExternalPreflightError(
                    f"{label} directory changed before its bounded read"
                )
            bindings.append((parent_descriptor, component, child_identity))
            directory_descriptors.append(child_descriptor)
            parent_descriptor = child_descriptor

        filename = components[-1]
        before = os.stat(filename, dir_fd=parent_descriptor, follow_symlinks=False)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise COOMExternalPreflightError(f"{label} must be a regular non-symlink file")
        if before.st_nlink != 1:
            raise COOMExternalPreflightError(f"{label} must not have a hard-link alias")
        if before.st_size != size:
            raise COOMExternalPreflightError(f"{label} size differs from the manifest")
        descriptor = os.open(
            filename,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        opened = os.fstat(descriptor)

        if not stat.S_ISREG(opened.st_mode) or _file_stat_identity(
            opened
        ) != _file_stat_identity(before):
            raise COOMExternalPreflightError(f"{label} changed before its bounded read")
        digest = hashlib.sha256()
        remaining = size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1 << 20))
            if not chunk:
                raise COOMExternalPreflightError(f"{label} ended during its bounded read")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise COOMExternalPreflightError(f"{label} grew during its bounded read")
        after = os.fstat(descriptor)
        current = os.stat(filename, dir_fd=parent_descriptor, follow_symlinks=False)
        if _file_stat_identity(after) != _file_stat_identity(
            opened
        ) or _file_stat_identity(current) != _file_stat_identity(opened):
            raise COOMExternalPreflightError(f"{label} changed during its bounded read")
        _verify_directory_bindings(bindings, label)
    except COOMExternalPreflightError:
        raise
    except OSError as exc:
        raise COOMExternalPreflightError(f"{label} could not be read safely") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        for directory_descriptor in reversed(directory_descriptors):
            os.close(directory_descriptor)
    if digest.hexdigest() != expected:
        raise COOMExternalPreflightError(f"{label} SHA-256 differs from the manifest")
    return relative, size, expected


def _file_identity(value: object, root: Path, label: str) -> tuple[str, int, str]:
    return _verify_file(root, value, label)


def _verify_source(value: object, root: Path) -> tuple[str, str, int, int, str]:
    source = _exact_object(value, "source")
    _keys(
        source,
        {
            "repository",
            "commit",
            "git_tree_oid",
            "source_archive_sha256",
            "identity_authenticated",
            "files",
        },
        "source",
    )
    if source["repository"] != COOM_REPOSITORY or source["commit"] != COOM_COMMIT:
        raise COOMExternalPreflightError("source differs from the official pin")
    tree_oid = _hex(source["git_tree_oid"], "source git tree OID", 40)
    archive_sha = _sha256(source["source_archive_sha256"], "source archive SHA-256")
    _exact_bool(source["identity_authenticated"], "source authentication", False)
    files = source["files"]
    if type(files) is not list or len(files) != len(_SOURCE_ROLES):
        raise COOMExternalPreflightError("source files must cover every required role")
    seen_paths: set[str] = set()
    roles: list[str] = []
    total = 0
    license_sha = ""
    verified_archive_sha = ""
    for index, raw in enumerate(files):
        item = _exact_object(raw, f"source file {index}")
        _keys(item, {"role", "path", "size_bytes", "sha256"}, f"source file {index}")
        role = _exact_string(item["role"], f"source file {index} role", maximum=64)
        file_raw = {name: item[name] for name in ("path", "size_bytes", "sha256")}
        path, size, digest = _file_identity(file_raw, root, f"source file {index}")
        if path in seen_paths:
            raise COOMExternalPreflightError("source file paths must be unique")
        seen_paths.add(path)
        roles.append(role)
        total += size
        if role == "license":
            license_sha = digest
        elif role == "source-archive":
            verified_archive_sha = digest
    if tuple(roles) != _SOURCE_ROLES:
        raise COOMExternalPreflightError("source file roles differ from the frozen order")
    if archive_sha != verified_archive_sha:
        raise COOMExternalPreflightError("source archive SHA-256 is not bound to the local archive")
    return tree_oid, archive_sha, len(files), total, license_sha


def _verify_license(value: object, license_sha: str) -> None:
    license_value = _exact_object(value, "license")
    _keys(
        license_value,
        {
            "spdx",
            "reviewed",
            "review_authenticated",
            "redistribution_authorized",
            "file_sha256",
        },
        "license",
    )
    if license_value["spdx"] != "MIT" or license_value["file_sha256"] != license_sha:
        raise COOMExternalPreflightError("license identity differs from the local source file")
    _exact_bool(license_value["reviewed"], "license review", True)
    _exact_bool(
        license_value["review_authenticated"], "license review authentication", False
    )
    _exact_bool(
        license_value["redistribution_authorized"],
        "asset redistribution authorization",
        False,
    )


def _verify_runtime(value: object, root: Path) -> tuple[int, str]:
    runtime = _exact_object(value, "runtime")
    _keys(
        runtime,
        {
            "system",
            "machine",
            "python",
            "implementation",
            "container_image_digest",
            "lock",
            "network_disabled",
            "packages",
            "engine",
        },
        "runtime",
    )
    if runtime["system"] != "Linux" or runtime["machine"] not in ("x86_64", "aarch64"):
        raise COOMExternalPreflightError("external runtime must be an explicit Linux target")
    if runtime["implementation"] != "CPython":
        raise COOMExternalPreflightError("external runtime must use CPython")
    _exact_string(runtime["python"], "runtime Python", maximum=32)
    image = _exact_string(runtime["container_image_digest"], "container image digest", maximum=71)
    if not image.startswith("sha256:"):
        raise COOMExternalPreflightError("container image must use a SHA-256 digest")
    _sha256(image.removeprefix("sha256:"), "container image digest")
    _exact_bool(runtime["network_disabled"], "runtime network isolation", True)
    packages = _exact_object(runtime["packages"], "runtime packages")
    _keys(packages, set(_PACKAGE_NAMES), "runtime packages")
    for name in _PACKAGE_NAMES:
        _exact_string(packages[name], f"runtime package {name}", maximum=64)
    if packages["scipy"] != "1.11.4" or packages["gymnasium"] != "0.28.1":
        raise COOMExternalPreflightError("runtime packages differ from the pinned COOM stack")
    engine = _exact_object(runtime["engine"], "runtime engine")
    _keys(engine, {"name", "version", "path", "size_bytes", "sha256"}, "runtime engine")
    if engine["name"] != "ViZDoom" or engine["version"] != packages["vizdoom"]:
        raise COOMExternalPreflightError("runtime engine identity differs from ViZDoom")
    file_raw = {name: engine[name] for name in ("path", "size_bytes", "sha256")}
    _path, engine_size, _digest = _file_identity(file_raw, root, "runtime engine")
    _lock_path, lock_size, _lock_digest = _file_identity(
        runtime["lock"], root, "runtime lock"
    )
    return engine_size + lock_size, hashlib.sha256(_canonical(runtime)).hexdigest()


def _verify_assets(value: object, root: Path, tasks: tuple[str, ...]) -> tuple[int, int, str]:
    if type(value) is not list or len(value) != len(tasks) * len(_ASSET_ROLES):
        raise COOMExternalPreflightError("assets must cover both required roles for every task")
    seen_paths: set[str] = set()
    roster: list[tuple[str, str]] = []
    identities: list[dict[str, object]] = []
    total = 0
    for index, raw in enumerate(value):
        item = _exact_object(raw, f"asset {index}")
        _keys(
            item,
            {
                "role",
                "task_id",
                "path",
                "size_bytes",
                "sha256",
                "license_spdx",
                "redistribution_reviewed",
            },
            f"asset {index}",
        )
        role = _exact_string(item["role"], f"asset {index} role", maximum=64)
        task = _exact_string(item["task_id"], f"asset {index} task", maximum=128)
        if role not in _ASSET_ROLES or task not in tasks:
            raise COOMExternalPreflightError("asset task or role is outside the frozen roster")
        _exact_string(item["license_spdx"], f"asset {index} license", maximum=64)
        _exact_bool(
            item["redistribution_reviewed"],
            f"asset {index} redistribution review",
            False,
        )
        file_raw = {name: item[name] for name in ("path", "size_bytes", "sha256")}
        path, size, digest = _file_identity(file_raw, root, f"asset {index}")
        if path in seen_paths:
            raise COOMExternalPreflightError("asset paths must be unique")
        seen_paths.add(path)
        roster.append((task, role))
        total += size
        identities.append(
            {
                "task_id": task,
                "role": role,
                "path": path,
                "size_bytes": size,
                "sha256": digest,
                "license_spdx": item["license_spdx"],
                "redistribution_reviewed": False,
            }
        )
    expected = [(task, role) for task in tasks for role in _ASSET_ROLES]
    if roster != expected:
        raise COOMExternalPreflightError("asset roster differs from the frozen task/role order")
    return len(identities), total, hashlib.sha256(_canonical(identities)).hexdigest()


def _verify_config(value: object) -> tuple[tuple[str, ...], str, int]:
    config = _exact_object(value, "configuration")
    expected_fields = {
        "sequence",
        "tasks",
        "seed",
        "steps_per_task",
        "replay_capacity",
        "update_warmup_steps",
        "batch_size",
        "frame_skip",
        "frame_stack",
        "frame_height",
        "frame_width",
        "test_episodes",
        "task_boundaries_available",
        "task_id_visible",
        "previous_environment_access_during_training",
        "reset_replay_at_task_boundary",
        "reset_optimizer_at_task_boundary",
        "reset_critic_at_task_boundary",
        "action_space_sha256_by_task",
    }
    _keys(config, expected_fields, "configuration")
    sequence = config["sequence"]
    if type(sequence) is not str or sequence not in ("CD8", "CO8"):
        raise COOMExternalPreflightError("configuration sequence must be CD8 or CO8")
    tasks = CD8_TASKS if sequence == "CD8" else CO8_TASKS
    expected: dict[str, object] = {
        "tasks": list(tasks),
        "steps_per_task": 200_000,
        "replay_capacity": 50_000,
        "update_warmup_steps": 5_000,
        "batch_size": 128,
        "frame_skip": 4,
        "frame_stack": 4,
        "frame_height": 84,
        "frame_width": 84,
        "test_episodes": 3,
        "task_boundaries_available": True,
        "task_id_visible": True,
        "previous_environment_access_during_training": False,
        "reset_replay_at_task_boundary": True,
        "reset_optimizer_at_task_boundary": True,
        "reset_critic_at_task_boundary": False,
    }
    for name, required in expected.items():
        if type(config[name]) is not type(required) or config[name] != required:
            raise COOMExternalPreflightError("configuration differs from the audited defaults")
    seed = _exact_int(config["seed"], "configuration seed", maximum=9)
    actions = _exact_object(config["action_space_sha256_by_task"], "action-space identities")
    _keys(actions, set(tasks), "action-space identities")
    for task in tasks:
        _sha256(actions[task], f"action-space identity {task}")
    return tasks, hashlib.sha256(_canonical(config)).hexdigest(), seed


_TRACE_FIELDS = {
    "repetition",
    "seed",
    "task_ids",
    "environment_resets",
    "environment_steps",
    "policy_queries",
    "observation_bytes",
    "action_bytes",
    "reward_bytes",
    "terminal_bytes",
    "truncation_bytes",
    "task_id_bytes",
    "persistent_environment_bytes",
    "reset_sha256",
    "observation_sha256",
    "action_sha256",
    "reward_sha256",
    "terminal_sha256",
    "truncation_sha256",
    "task_id_sha256",
    "workload_identity_sha256",
}


def _verify_trace(
    value: object, tasks: tuple[str, ...], seed: int, workload_sha256: str
) -> str:
    if type(value) is not list or len(value) != 2:
        raise COOMExternalPreflightError("trace must contain exactly two repetitions")
    repetitions: list[dict[str, object]] = []
    for index, raw in enumerate(value):
        trace = _exact_object(raw, f"trace repetition {index}")
        _keys(trace, _TRACE_FIELDS, f"trace repetition {index}")
        if trace["repetition"] != index or type(trace["repetition"]) is not int:
            raise COOMExternalPreflightError("trace repetition indices must be exact")
        if trace["seed"] != seed or type(trace["seed"]) is not int:
            raise COOMExternalPreflightError("trace seed differs from the configuration")
        if trace["task_ids"] != list(tasks) or type(trace["task_ids"]) is not list:
            raise COOMExternalPreflightError("trace task roster differs from the configuration")
        counts = {
            "environment_resets": len(tasks),
            "environment_steps": len(tasks),
            "policy_queries": len(tasks),
            "action_bytes": 4 * len(tasks),
            "reward_bytes": 4 * len(tasks),
            "terminal_bytes": len(tasks),
            "truncation_bytes": len(tasks),
            "task_id_bytes": 4 * len(tasks),
            "observation_bytes": 2 * len(tasks) * 84 * 84 * 4,
        }
        for name, expected in counts.items():
            if _exact_int(trace[name], f"trace {name}") != expected:
                raise COOMExternalPreflightError(f"trace {name} differs from the smoke budget")
        _exact_int(
            trace["persistent_environment_bytes"],
            "trace persistent_environment_bytes",
            minimum=1,
        )
        for name in (
            "reset_sha256",
            "observation_sha256",
            "action_sha256",
            "reward_sha256",
            "terminal_sha256",
            "truncation_sha256",
            "task_id_sha256",
        ):
            _sha256(trace[name], f"trace {name}")
        if trace["workload_identity_sha256"] != workload_sha256:
            raise COOMExternalPreflightError(
                "trace workload identity differs from the verified inputs"
            )
        repetitions.append(trace)
    left = {key: value for key, value in repetitions[0].items() if key != "repetition"}
    right = {key: value for key, value in repetitions[1].items() if key != "repetition"}
    if left != right:
        raise COOMExternalPreflightError("deterministic trace repetitions do not match exactly")
    return hashlib.sha256(_canonical(repetitions)).hexdigest()


def verify_external_preflight(
    raw: bytes,
    *,
    source_root: Path,
    asset_root: Path,
    runtime_root: Path,
) -> COOMExternalPreflightReceipt:
    """Verify one local external receipt and keep every qualification gate open."""
    if not all(isinstance(root, Path) for root in (source_root, asset_root, runtime_root)):
        raise TypeError("all roots must be Path values")
    manifest = _decode(raw)
    _keys(
        manifest,
        {
            "schema_version",
            "classification",
            "source",
            "license",
            "runtime",
            "assets",
            "config",
            "trace_repetitions",
            "external_runtime_executed_by_caller",
            "negative_outcome_retained",
            "execution_authorized_by_asi",
            "promotion_authorized",
            "benchmark_result_claimed",
        },
        "manifest",
    )
    if (
        manifest["schema_version"] != COOM_EXTERNAL_PREFLIGHT_SCHEMA
        or manifest["classification"] != "local-readiness-only-nonpromoting"
    ):
        raise COOMExternalPreflightError("manifest classification or schema drift")
    _exact_bool(
        manifest["external_runtime_executed_by_caller"],
        "external caller execution receipt",
        True,
    )
    _exact_bool(manifest["negative_outcome_retained"], "negative retention", True)
    _exact_bool(manifest["execution_authorized_by_asi"], "ASI execution authority", False)
    _exact_bool(manifest["promotion_authorized"], "promotion authority", False)
    if manifest["benchmark_result_claimed"] is not False:
        raise COOMExternalPreflightError("local preflight may not make a benchmark claim")
    tree_oid, archive_sha, source_count, source_bytes, license_sha = _verify_source(
        manifest["source"], source_root
    )
    _verify_license(manifest["license"], license_sha)
    runtime_bytes, runtime_sha = _verify_runtime(manifest["runtime"], runtime_root)
    tasks, config_sha, seed = _verify_config(manifest["config"])
    asset_count, asset_bytes, asset_sha = _verify_assets(manifest["assets"], asset_root, tasks)
    workload_sha = workload_identity_sha256(
        source_archive_sha256=archive_sha,
        runtime_identity_sha256=runtime_sha,
        assets_identity_sha256=asset_sha,
        config_sha256=config_sha,
    )
    trace_sha = _verify_trace(manifest["trace_repetitions"], tasks, seed, workload_sha)
    return COOMExternalPreflightReceipt(
        schema_version=COOM_EXTERNAL_PREFLIGHT_SCHEMA,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        source_repository=COOM_REPOSITORY,
        source_commit=COOM_COMMIT,
        source_git_tree_oid=tree_oid,
        source_archive_sha256=archive_sha,
        runtime_identity_sha256=runtime_sha,
        assets_identity_sha256=asset_sha,
        config_sha256=config_sha,
        workload_identity_sha256=workload_sha,
        trace_sha256=trace_sha,
        sequence="CD8" if tasks == CD8_TASKS else "CO8",
        local_files_verified=source_count + asset_count + 2,
        local_bytes_verified=source_bytes + asset_bytes + runtime_bytes,
        source_identity_authenticated=False,
        license_review_authenticated=False,
        runtime_identity_authenticated=False,
        asset_semantics_authenticated=False,
        deterministic_trace_pair_verified=True,
        trace_execution_authenticated=False,
        external_runtime_executed_by_caller=True,
        execution_authorized=False,
        promotion_authorized=False,
        benchmark_result_claimed=False,
        remaining_qualification_gates=qualification_plan(1582).required_gates,
        qualification_identity=_current_identity(),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    args = parser.parse_args(argv)
    raw = load_preflight_manifest(args.manifest)
    receipt = verify_external_preflight(
        raw,
        source_root=args.source_root,
        asset_root=args.asset_root,
        runtime_root=args.runtime_root,
    )
    print(json.dumps(receipt.to_payload(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "COOM_EXTERNAL_PREFLIGHT_SCHEMA",
    "COOMExternalPreflightError",
    "COOMExternalPreflightReceipt",
    "MAX_MANIFEST_BYTES",
    "load_preflight_manifest",
    "main",
    "verify_external_preflight",
    "workload_identity_sha256",
]
