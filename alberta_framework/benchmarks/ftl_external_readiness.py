"""Fail-closed readiness gate for an external FTL/Continual Bench reproduction.

This module does not download, import, or execute external code.  It qualifies
an already-present checkout of the official Continual Bench repository and
reports the remaining blockers between that environment source and a paper-level
FTL Online Agent reproduction.  The report is diagnostic and permanently
nonpromoting; in particular, the ASI-native development analogue is not accepted
as a substitute for unpublished external agent code or an independently frozen
paper protocol.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import stat
import subprocess
import sys
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, cast

SCHEMA = "asi.ftl_external_readiness.v2"
OFFICIAL_REPOSITORY = "https://github.com/sail-sg/ContinualBench"
OFFICIAL_REVISION = "a4fdb3b94a07a40d76e28d3aeab0f8ca97519dad"
PAPER_PMLR_URL = "https://proceedings.mlr.press/v267/liu25p.html"
PAPER_PDF_URL = "https://proceedings.mlr.press/v267/liu25p/liu25p.pdf"
PAPER_ARXIV_ID = "2507.09177v1"
PAPER_TASK_ORDER = (
    "pick-place",
    "button-press",
    "door-open",
    "peg-unplug",
    "window-close",
    "faucet-close",
)
EXPECTED_DIRECT_DEPENDENCIES = ("glfw==2.5.0",)
REQUIRED_SOURCE_PATHS = (
    "LICENSE.txt",
    "README.md",
    "pyproject.toml",
    "continual_bench/__init__.py",
    "continual_bench/envs/__init__.py",
    "continual_bench/envs/reward_fns.py",
    "continual_bench/envs/mujoco/mujoco_env.py",
    "continual_bench/envs/mujoco/sawyer_bench.py",
    "continual_bench/envs/mujoco/sawyer_xyz_env.py",
)
RUNTIME_DISTRIBUTIONS = (
    "glfw",
    "gymnasium",
    "mbrl",
    "mujoco",
    "numpy",
    "scipy",
    "torch",
)
_ASSET_PREFIX = "continual_bench/envs/assets/"
_MAX_TRACKED_FILES = 4096
_MAX_FILE_BYTES = 64 * 1024 * 1024
_MAX_TREE_BYTES = 512 * 1024 * 1024
_MAX_JSON_NODES = 8192
_MAX_JSON_STRING_BYTES = 16 * 1024
_PERMANENT_BLOCKERS = (
    "matched_baselines_unimplemented",
    "mujoco_asset_license_audit_missing",
    "oa_reference_implementation_unpublished",
    "official_dependency_lock_incomplete",
    "paper_protocol_lock_missing",
    "resource_instrumentation_unqualified",
    "runtime_environment_unqualified",
    "scientific_protocol_unfrozen",
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha(value: object, *, name: str, length: int = 64) -> str:
    if (
        type(value) is not str
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase hexadecimal digest")
    return value


def _canonical_sha(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")
    return _sha256(encoded)


def _exact_string_tuple(value: object, *, name: str, nonempty: bool = True) -> tuple[str, ...]:
    if type(value) is not tuple or (nonempty and not value) or len(value) > 4096:
        raise ValueError(f"{name} must be an exact tuple")
    result = cast(tuple[object, ...], value)
    if any(
        type(item) is not str
        or not item
        or len(item.encode("utf-8")) > _MAX_JSON_STRING_BYTES
        for item in result
    ):
        raise ValueError(f"{name} entries must be nonempty exact strings")
    return cast(tuple[str, ...], result)


def _preflight_json(value: object) -> None:
    pending: list[tuple[object, int]] = [(value, 0)]
    seen: set[int] = set()
    nodes = 0
    while pending:
        current, depth = pending.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > 16:
            raise ValueError("readiness JSON exceeds its structural bound")
        if current is None or type(current) in {bool, int}:
            continue
        if type(current) is str:
            if len(current.encode("utf-8")) > _MAX_JSON_STRING_BYTES:
                raise ValueError("readiness JSON contains an oversized string")
            continue
        if type(current) not in {dict, list}:
            raise ValueError("readiness JSON must contain only exact JSON values")
        identity = id(current)
        if identity in seen:
            raise ValueError("readiness JSON contains an aliased or cyclic container")
        seen.add(identity)
        if type(current) is dict:
            mapping = cast(dict[object, object], current)
            if len(mapping) > 256 or any(type(key) is not str for key in mapping):
                raise ValueError("readiness JSON object exceeds its field bound")
            pending.extend((item, depth + 1) for item in mapping.values())
        else:
            sequence = cast(list[object], current)
            if len(sequence) > 4096:
                raise ValueError("readiness JSON list exceeds its item bound")
            pending.extend((item, depth + 1) for item in sequence)


@dataclasses.dataclass(frozen=True, slots=True)
class PaperProtocolReceipt:
    pmlr_url: str = PAPER_PMLR_URL
    pdf_url: str = PAPER_PDF_URL
    arxiv_id: str = PAPER_ARXIV_ID
    task_order: tuple[str, ...] = PAPER_TASK_ORDER
    episodes: int = 600
    observation_dim: int = 26
    reported_accelerators: int = 1
    reported_cpus: int = 16
    reported_wall_hours_low: int = 10
    reported_wall_hours_high: int = 15
    facts_only: bool = True
    complete_hyperparameter_lock: bool = False

    def __post_init__(self) -> None:
        for name in ("pmlr_url", "pdf_url", "arxiv_id"):
            if type(getattr(self, name)) is not str:
                raise ValueError(f"paper {name} must be an exact string")
        _exact_string_tuple(self.task_order, name="paper task_order")
        for name in (
            "episodes",
            "observation_dim",
            "reported_accelerators",
            "reported_cpus",
            "reported_wall_hours_low",
            "reported_wall_hours_high",
        ):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"paper {name} must be an exact integer")
        if type(self.facts_only) is not bool or type(self.complete_hyperparameter_lock) is not bool:
            raise ValueError("paper qualification flags must be exact booleans")
        actual = tuple(getattr(self, field.name) for field in dataclasses.fields(self))
        expected = (
            PAPER_PMLR_URL,
            PAPER_PDF_URL,
            PAPER_ARXIV_ID,
            PAPER_TASK_ORDER,
            600,
            26,
            1,
            16,
            10,
            15,
            True,
            False,
        )
        if actual != expected:
            raise ValueError("paper protocol receipt differs from the reviewed public facts")

    def to_payload(self) -> dict[str, object]:
        payload = dataclasses.asdict(self)
        payload["task_order"] = list(self.task_order)
        return payload


@dataclasses.dataclass(frozen=True, slots=True)
class RuntimeReceipt:
    python_implementation: str
    python_version: str
    system: str
    machine: str
    python_executable_sha256: str
    distributions: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for name in ("python_implementation", "python_version", "system", "machine"):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise ValueError(f"runtime {name} must be a nonempty exact string")
        _require_sha(self.python_executable_sha256, name="python_executable_sha256")
        if type(self.distributions) is not tuple or len(self.distributions) != len(
            RUNTIME_DISTRIBUTIONS
        ):
            raise ValueError("runtime distributions differ from the qualification inventory")
        expected_names = tuple(sorted(RUNTIME_DISTRIBUTIONS))
        names: list[str] = []
        for item in self.distributions:
            if (
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not str
                or type(item[1]) is not str
                or not item[1]
            ):
                raise ValueError("runtime distributions must be exact nonempty string pairs")
            names.append(item[0])
        if tuple(names) != expected_names:
            raise ValueError("runtime distribution names must be exact, unique, and sorted")

    def to_payload(self) -> dict[str, object]:
        return {
            "python_implementation": self.python_implementation,
            "python_version": self.python_version,
            "system": self.system,
            "machine": self.machine,
            "python_executable_sha256": self.python_executable_sha256,
            "distributions": [list(item) for item in self.distributions],
        }


@dataclasses.dataclass(frozen=True, slots=True)
class SourceReceipt:
    repository: str
    revision: str
    checkout_realpath: str
    head_commit_sha1: str
    head_tree_sha1: str
    object_format: str
    file_count: int
    asset_file_count: int
    total_tracked_bytes: int
    tracked_files_sha256: str
    assets_sha256: str
    pyproject_sha256: str
    license_sha256: str
    direct_dependencies: tuple[str, ...]
    checkout_clean: bool = True
    repository_origin_attested: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.repository) is not str
            or type(self.revision) is not str
            or self.repository != OFFICIAL_REPOSITORY
            or self.revision != OFFICIAL_REVISION
        ):
            raise ValueError("source receipt does not bind the official repository revision")
        if (
            type(self.checkout_realpath) is not str
            or not Path(self.checkout_realpath).is_absolute()
        ):
            raise ValueError("checkout_realpath must be an absolute path")
        _require_sha(self.head_commit_sha1, name="head_commit_sha1", length=40)
        _require_sha(self.head_tree_sha1, name="head_tree_sha1", length=40)
        if self.head_commit_sha1 != self.revision or self.object_format != "sha1":
            raise ValueError("source receipt must use the exact SHA-1 official commit")
        for name in ("file_count", "asset_file_count", "total_tracked_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")
        if self.asset_file_count > self.file_count:
            raise ValueError("asset_file_count cannot exceed file_count")
        for name in (
            "tracked_files_sha256",
            "assets_sha256",
            "pyproject_sha256",
            "license_sha256",
        ):
            _require_sha(getattr(self, name), name=name)
        if self.direct_dependencies != EXPECTED_DIRECT_DEPENDENCIES:
            raise ValueError("direct dependencies differ from the official pinned metadata")
        if type(self.checkout_clean) is not bool or not self.checkout_clean:
            raise ValueError("source checkout must be exactly clean")
        if (
            type(self.repository_origin_attested) is not bool
            or self.repository_origin_attested
        ):
            raise ValueError("local source identity is not an origin attestation")

    def to_payload(self) -> dict[str, object]:
        payload = dataclasses.asdict(self)
        payload["direct_dependencies"] = list(self.direct_dependencies)
        return payload


@dataclasses.dataclass(frozen=True, slots=True)
class ExternalReadinessReport:
    schema: str
    source: SourceReceipt | None
    runtime: RuntimeReceipt
    protocol: PaperProtocolReceipt
    blockers: tuple[str, ...]
    source_checkout_qualified: bool
    execution_ready: bool = False
    reproduction_claim_allowed: bool = False
    external_results_present: bool = False
    development_only: bool = True

    def __post_init__(self) -> None:
        if type(self.schema) is not str or self.schema != SCHEMA:
            raise ValueError("unsupported external readiness schema")
        if self.source is not None:
            if type(self.source) is not SourceReceipt:
                raise ValueError("source must be an exact SourceReceipt")
            SourceReceipt.__post_init__(self.source)
        if type(self.runtime) is not RuntimeReceipt:
            raise ValueError("runtime must be an exact RuntimeReceipt")
        RuntimeReceipt.__post_init__(self.runtime)
        if type(self.protocol) is not PaperProtocolReceipt:
            raise ValueError("protocol must be an exact PaperProtocolReceipt")
        PaperProtocolReceipt.__post_init__(self.protocol)
        blockers = _exact_string_tuple(self.blockers, name="blockers")
        if blockers != tuple(sorted(set(blockers))):
            raise ValueError("blockers must be unique and sorted")
        if not set(_PERMANENT_BLOCKERS).issubset(blockers):
            raise ValueError("report omits unresolved external reproduction blockers")
        checkout_blockers = tuple(
            blocker for blocker in blockers if blocker not in _PERMANENT_BLOCKERS
        )
        if self.source is None:
            if len(checkout_blockers) != 1 or not (
                checkout_blockers[0] == "official_checkout_missing"
                or (
                    checkout_blockers[0].startswith("official_checkout_invalid:")
                    and len(checkout_blockers[0]) > len("official_checkout_invalid:")
                )
            ):
                raise ValueError("unqualified source must retain exactly one checkout blocker")
        elif checkout_blockers:
            raise ValueError("qualified source cannot retain a checkout blocker")
        if type(self.source_checkout_qualified) is not bool or self.source_checkout_qualified != (
            self.source is not None
        ):
            raise ValueError("source qualification flag disagrees with its receipt")
        if any(
            type(flag) is not bool
            for flag in (
                self.execution_ready,
                self.reproduction_claim_allowed,
                self.external_results_present,
                self.development_only,
            )
        ):
            raise ValueError("readiness flags must be exact booleans")
        if self.execution_ready:
            raise ValueError("incomplete external gate cannot be execution-ready")
        if (
            self.reproduction_claim_allowed
            or self.external_results_present
            or not self.development_only
        ):
            raise ValueError("readiness report cannot carry a reproduction claim or result")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "source": None if self.source is None else self.source.to_payload(),
            "runtime": self.runtime.to_payload(),
            "protocol": self.protocol.to_payload(),
            "blockers": list(self.blockers),
            "source_checkout_qualified": self.source_checkout_qualified,
            "execution_ready": self.execution_ready,
            "reproduction_claim_allowed": self.reproduction_claim_allowed,
            "external_results_present": self.external_results_present,
            "development_only": self.development_only,
        }


def _git(root: Path, *arguments: str) -> bytes:
    environment = {
        name: value for name, value in os.environ.items() if not name.startswith("GIT_")
    }
    environment.update(
        {
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PAGER": "cat",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    try:
        completed = subprocess.run(
            [
                "git",
                "--no-optional-locks",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.fileMode=true",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "core.untrackedCache=false",
                "-c",
                "submodule.recurse=false",
                "-C",
                os.fspath(root),
                *arguments,
            ],
            check=False,
            capture_output=True,
            env=environment,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError(f"git inspection failed: {type(error).__name__}") from error
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip().splitlines()
        suffix = detail[-1] if detail else f"exit {completed.returncode}"
        raise ValueError(f"git inspection failed: {suffix}")
    return completed.stdout


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or value.startswith("/")
        or path.is_absolute()
        or ".." in path.parts
        or "" in path.parts
        or "\\" in value
    ):
        raise ValueError("git tree contains an unsafe path")
    return path


def _parse_tree(value: bytes) -> tuple[tuple[str, str, str, str], ...]:
    entries: list[tuple[str, str, str, str]] = []
    records = value.split(b"\0")
    if records[-1] != b"":
        raise ValueError("git tree output is not NUL terminated")
    for record in records[:-1]:
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, kind, oid = metadata.decode("ascii").split(" ")
            relative = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("git tree contains an invalid entry") from error
        _safe_relative_path(relative)
        if mode not in ("100644", "100755") or kind != "blob":
            raise ValueError("checkout contains a symlink, submodule, or unsupported tree mode")
        _require_sha(oid, name="git blob oid", length=40)
        entries.append((relative, mode, kind, oid))
    if not entries or len(entries) > _MAX_TRACKED_FILES:
        raise ValueError("tracked source file count is empty or exceeds the qualification bound")
    paths = tuple(entry[0] for entry in entries)
    if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
        raise ValueError("git tree paths are not exact, unique, and sorted")
    return tuple(entries)


def _blob_sha1(value: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(value)).encode("ascii") + b"\0" + value).hexdigest()


def _open_flags(*, directory: bool) -> int:
    required = ("O_CLOEXEC", "O_NOFOLLOW") + (("O_DIRECTORY",) if directory else ())
    if any(not hasattr(os, name) for name in required):
        raise ValueError("platform lacks fail-closed no-follow checkout inspection")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    if directory:
        flags |= os.O_DIRECTORY
    return flags


def _read_tracked_file(root_descriptor: int, relative: str, mode: str) -> bytes:
    """Read one tracked file through no-follow directory descriptors."""
    parts = PurePosixPath(relative).parts
    directory_descriptor = os.dup(root_descriptor)
    file_descriptor: int | None = None
    try:
        for component in parts[:-1]:
            next_descriptor = os.open(
                component,
                _open_flags(directory=True),
                dir_fd=directory_descriptor,
            )
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        file_descriptor = os.open(
            parts[-1],
            _open_flags(directory=False),
            dir_fd=directory_descriptor,
        )
        metadata = os.fstat(file_descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("tracked checkout entry is not a regular file")
        if bool(metadata.st_mode & 0o111) != (mode == "100755"):
            raise ValueError("tracked checkout executable mode differs from the pinned tree")
        if metadata.st_size > _MAX_FILE_BYTES:
            raise ValueError("tracked checkout file exceeds the qualification byte bound")
        with os.fdopen(file_descriptor, "rb", closefd=True) as stream:
            file_descriptor = None
            value = stream.read(_MAX_FILE_BYTES + 1)
        if len(value) != metadata.st_size:
            raise ValueError("tracked checkout file changed while it was read")
        return value
    except OSError as error:
        raise ValueError("tracked checkout path contains a symlink or unsafe component") from error
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        os.close(directory_descriptor)


def _qualify_checkout(checkout: Path, *, expected_revision: str) -> SourceReceipt:
    if not isinstance(checkout, Path):
        raise ValueError("checkout must be an exact Path")
    if not checkout.exists() or not checkout.is_dir():
        raise FileNotFoundError("official checkout is absent")
    root = checkout.resolve(strict=True)
    top = Path(_git(root, "rev-parse", "--show-toplevel").decode("utf-8").strip()).resolve()
    if top != root:
        raise ValueError("checkout path must be the exact repository root")
    object_format = _git(root, "rev-parse", "--show-object-format").decode("ascii").strip()
    if object_format != "sha1":
        raise ValueError("official pin requires a SHA-1 git object store")
    if _git(root, "for-each-ref", "--format=%(refname)", "refs/replace/"):
        raise ValueError("checkout contains forbidden Git replacement objects")
    head = _git(root, "rev-parse", "HEAD").decode("ascii").strip()
    if head != expected_revision:
        raise ValueError("checkout HEAD differs from the official pinned revision")
    if _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all"):
        raise ValueError("checkout has tracked or untracked modifications")
    if _git(root, "ls-files", "--others", "--ignored", "--exclude-standard", "-z"):
        raise ValueError("checkout has ignored untracked files")
    tree = _git(root, "rev-parse", f"{head}^{{tree}}").decode("ascii").strip()
    _require_sha(tree, name="head tree", length=40)
    entries = _parse_tree(_git(root, "ls-tree", "-r", "-z", "--full-tree", head))
    paths = {entry[0] for entry in entries}
    missing = tuple(path for path in REQUIRED_SOURCE_PATHS if path not in paths)
    if missing:
        raise ValueError("checkout lacks required source paths: " + ",".join(missing))
    asset_paths = tuple(path for path in sorted(paths) if path.startswith(_ASSET_PREFIX))
    if not asset_paths:
        raise ValueError("checkout contains no tracked Continual Bench assets")

    file_receipts: list[tuple[str, str, int, str]] = []
    asset_receipts: list[tuple[str, str, int, str]] = []
    contents: dict[str, bytes] = {}
    total = 0
    try:
        root_descriptor = os.open(root, _open_flags(directory=True))
    except OSError as error:
        raise ValueError("checkout root is not a stable real directory") from error
    root_identity = os.fstat(root_descriptor)
    try:
        for relative, mode, _kind, oid in entries:
            value = _read_tracked_file(root_descriptor, relative, mode)
            total += len(value)
            if total > _MAX_TREE_BYTES:
                raise ValueError("tracked checkout exceeds the qualification byte bound")
            if _blob_sha1(value) != oid:
                raise ValueError("worktree bytes differ from the pinned git blob")
            receipt = (relative, mode, len(value), _sha256(value))
            file_receipts.append(receipt)
            contents[relative] = value
            if relative.startswith(_ASSET_PREFIX):
                asset_receipts.append(receipt)
    finally:
        os.close(root_descriptor)

    try:
        project = tomllib.loads(contents["pyproject.toml"].decode("utf-8"))
        project_table = project["project"]
        dependencies = project_table["dependencies"]
    except (KeyError, TypeError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("official pyproject metadata is malformed") from error
    if type(project_table) is not dict or project_table.get("name") != "continual-bench":
        raise ValueError("official pyproject package name is invalid")
    if type(dependencies) is not list or any(type(item) is not str for item in dependencies):
        raise ValueError("official direct dependencies are not an exact string list")
    direct_dependencies = tuple(dependencies)
    if direct_dependencies != EXPECTED_DIRECT_DEPENDENCIES:
        raise ValueError("official direct dependency metadata differs from the reviewed pin")

    # Close the time-of-check/time-of-use window over both worktree state and commit identity.
    if _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all"):
        raise ValueError("checkout changed during qualification")
    if _git(root, "ls-files", "--others", "--ignored", "--exclude-standard", "-z"):
        raise ValueError("ignored files appeared during qualification")
    if _git(root, "rev-parse", "HEAD").decode("ascii").strip() != head:
        raise ValueError("checkout HEAD changed during qualification")
    final_root = root.lstat()
    if (
        not stat.S_ISDIR(final_root.st_mode)
        or (final_root.st_dev, final_root.st_ino)
        != (root_identity.st_dev, root_identity.st_ino)
    ):
        raise ValueError("checkout root changed during qualification")
    return SourceReceipt(
        repository=OFFICIAL_REPOSITORY,
        revision=expected_revision,
        checkout_realpath=os.fspath(root),
        head_commit_sha1=head,
        head_tree_sha1=tree,
        object_format=object_format,
        file_count=len(file_receipts),
        asset_file_count=len(asset_receipts),
        total_tracked_bytes=total,
        tracked_files_sha256=_canonical_sha(file_receipts),
        assets_sha256=_canonical_sha(asset_receipts),
        pyproject_sha256=_sha256(contents["pyproject.toml"]),
        license_sha256=_sha256(contents["LICENSE.txt"]),
        direct_dependencies=direct_dependencies,
    )


def _runtime_receipt() -> RuntimeReceipt:
    versions: list[tuple[str, str]] = []
    for distribution in sorted(RUNTIME_DISTRIBUTIONS):
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = "<missing>"
        versions.append((distribution, version))
    executable = Path(sys.executable).resolve(strict=True)
    return RuntimeReceipt(
        python_implementation=sys.implementation.name,
        python_version=platform.python_version(),
        system=platform.system(),
        machine=platform.machine(),
        python_executable_sha256=_sha256(executable.read_bytes()),
        distributions=tuple(versions),
    )


def inspect_external_readiness(checkout: str | os.PathLike[str]) -> ExternalReadinessReport:
    """Inspect a local external checkout without importing, executing, or changing it."""
    if isinstance(checkout, bool) or not isinstance(checkout, (str, os.PathLike)):
        raise ValueError("checkout must be a filesystem path")
    path = Path(checkout)
    source: SourceReceipt | None = None
    blockers = list(_PERMANENT_BLOCKERS)
    if not path.exists():
        blockers.append("official_checkout_missing")
    else:
        try:
            source = _qualify_checkout(path, expected_revision=OFFICIAL_REVISION)
        except (FileNotFoundError, ValueError) as error:
            blockers.append(f"official_checkout_invalid:{error}")
    report = ExternalReadinessReport(
        schema=SCHEMA,
        source=source,
        runtime=_runtime_receipt(),
        protocol=PaperProtocolReceipt(),
        blockers=tuple(sorted(blockers)),
        source_checkout_qualified=source is not None,
    )
    return validate_report(report)


def _exact_fields(value: dict[object, object], cls: Any, *, name: str) -> None:
    expected = {field.name for field in dataclasses.fields(cls)}
    keys = tuple(value.keys())
    if any(type(key) is not str for key in keys) or set(keys) != expected:
        raise ValueError(f"{name} must contain the exact fields")


def _report_from_payload(value: dict[object, object]) -> ExternalReadinessReport:
    _exact_fields(value, ExternalReadinessReport, name="report payload")
    raw_source = value["source"]
    source = None
    if raw_source is not None:
        if type(raw_source) is not dict:
            raise ValueError("source payload must be an exact mapping or null")
        _exact_fields(raw_source, SourceReceipt, name="source payload")
        source_values = dict(raw_source)
        dependencies = source_values["direct_dependencies"]
        if type(dependencies) is not list or any(type(item) is not str for item in dependencies):
            raise ValueError("source direct_dependencies must be an exact string list")
        source_values["direct_dependencies"] = tuple(dependencies)
        source = SourceReceipt(**source_values)

    raw_runtime = value["runtime"]
    if type(raw_runtime) is not dict:
        raise ValueError("runtime payload must be an exact mapping")
    _exact_fields(raw_runtime, RuntimeReceipt, name="runtime payload")
    runtime_values = dict(raw_runtime)
    raw_distributions = runtime_values["distributions"]
    if type(raw_distributions) is not list or any(
        type(item) is not list
        or len(item) != 2
        or type(item[0]) is not str
        or type(item[1]) is not str
        for item in raw_distributions
    ):
        raise ValueError("runtime distributions must be exact string-pair lists")
    runtime_values["distributions"] = tuple((item[0], item[1]) for item in raw_distributions)
    runtime = RuntimeReceipt(**runtime_values)

    raw_protocol = value["protocol"]
    if type(raw_protocol) is not dict:
        raise ValueError("protocol payload must be an exact mapping")
    _exact_fields(raw_protocol, PaperProtocolReceipt, name="protocol payload")
    protocol_values = dict(raw_protocol)
    raw_tasks = protocol_values["task_order"]
    if type(raw_tasks) is not list or any(type(item) is not str for item in raw_tasks):
        raise ValueError("protocol task_order must be an exact string list")
    protocol_values["task_order"] = tuple(raw_tasks)
    protocol = PaperProtocolReceipt(**protocol_values)

    raw_blockers = value["blockers"]
    if type(raw_blockers) is not list or any(type(item) is not str for item in raw_blockers):
        raise ValueError("blockers must be an exact string list")
    return ExternalReadinessReport(
        schema=cast(str, value["schema"]),
        source=source,
        runtime=runtime,
        protocol=protocol,
        blockers=tuple(raw_blockers),
        source_checkout_qualified=cast(bool, value["source_checkout_qualified"]),
        execution_ready=cast(bool, value["execution_ready"]),
        reproduction_claim_allowed=cast(bool, value["reproduction_claim_allowed"]),
        external_results_present=cast(bool, value["external_results_present"]),
        development_only=cast(bool, value["development_only"]),
    )


def validate_report(value: object) -> ExternalReadinessReport:
    """Validate an exact readiness dataclass or JSON-shaped payload."""
    if type(value) is dict:
        _preflight_json(value)
        value = _report_from_payload(value)
    elif isinstance(value, Mapping):
        raise ValueError("report payload must be an exact dict")
    if type(value) is not ExternalReadinessReport:
        raise ValueError("report must be an exact ExternalReadinessReport")
    ExternalReadinessReport.__post_init__(value)
    if value.runtime != _runtime_receipt():
        raise ValueError("report runtime differs from the current runtime")
    if value.source is not None:
        current_source = _qualify_checkout(
            Path(value.source.checkout_realpath), expected_revision=OFFICIAL_REVISION
        )
        if value.source != current_source:
            raise ValueError("source receipt differs from the current qualified checkout")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect an existing official Continual Bench checkout without executing it."
    )
    parser.add_argument("--checkout", required=True, type=Path)
    arguments = parser.parse_args(argv)
    report = inspect_external_readiness(arguments.checkout)
    print(
        json.dumps(
            report.to_payload(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    )
    return 0 if report.execution_ready else 2


__all__ = [
    "EXPECTED_DIRECT_DEPENDENCIES",
    "OFFICIAL_REPOSITORY",
    "OFFICIAL_REVISION",
    "PAPER_TASK_ORDER",
    "REQUIRED_SOURCE_PATHS",
    "SCHEMA",
    "ExternalReadinessReport",
    "PaperProtocolReceipt",
    "RuntimeReceipt",
    "SourceReceipt",
    "inspect_external_readiness",
    "main",
    "validate_report",
]


if __name__ == "__main__":  # pragma: no cover - console entrypoint
    raise SystemExit(main())
