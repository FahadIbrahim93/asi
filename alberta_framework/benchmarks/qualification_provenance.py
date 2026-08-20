"""Current-tree identity helpers for nonpromoting qualification smokes."""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import math
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import jax
import numpy as np

_MAX_TREE_DEPTH = 64
_MAX_TREE_NODES = 50_000
_MAX_CONTAINER_ITEMS = 10_000
_MAX_CUMULATIVE_UTF8_BYTES = 1 << 20
_MIN_INTEGER = -(1 << 63)
_MAX_INTEGER = (1 << 63) - 1


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def preflight_qualification_tree(value: object) -> None:
    """Admit one bounded exact primitive tree without dispatching subclass hooks."""

    stack: list[tuple[object, int]] = [(value, 0)]
    seen_containers: set[int] = set()
    nodes = 0
    utf8_bytes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_TREE_NODES or depth > _MAX_TREE_DEPTH:
            raise ValueError("qualification tree exceeds its node or depth limit")
        actual_type = type(item)
        if item is None or actual_type is bool:
            continue
        if actual_type is int:
            if not _MIN_INTEGER <= cast(int, item) <= _MAX_INTEGER:
                raise ValueError("qualification tree integer lies outside signed 64-bit bounds")
            continue
        if actual_type is float:
            if not math.isfinite(cast(float, item)):
                raise ValueError("qualification tree contains a non-finite float")
            continue
        if actual_type is str:
            utf8_bytes += len(cast(str, item).encode("utf-8", errors="strict"))
        elif actual_type is dict:
            identity = id(item)
            if identity in seen_containers:
                raise ValueError("qualification tree contains a cycle or container alias")
            seen_containers.add(identity)
            mapping = cast(dict[object, object], item)
            keys = tuple(mapping.keys())
            if len(keys) > _MAX_CONTAINER_ITEMS:
                raise ValueError("qualification object exceeds its item limit")
            if any(type(key) is not str for key in keys):
                raise ValueError("qualification object keys must be exact strings")
            nodes += len(keys)
            if nodes > _MAX_TREE_NODES:
                raise ValueError("qualification tree exceeds its node or depth limit")
            for key in cast(tuple[str, ...], keys):
                utf8_bytes += len(key.encode("utf-8", errors="strict"))
                stack.append((mapping[key], depth + 1))
        elif actual_type is list or actual_type is tuple:
            if actual_type is list:
                identity = id(item)
                if identity in seen_containers:
                    raise ValueError("qualification tree contains a cycle or container alias")
                seen_containers.add(identity)
            sequence = cast(Sequence[object], item)
            if len(sequence) > _MAX_CONTAINER_ITEMS:
                raise ValueError("qualification sequence exceeds its item limit")
            stack.extend((child, depth + 1) for child in sequence)
        else:
            raise ValueError("qualification tree must use exact primitive containers and scalars")
        if utf8_bytes > _MAX_CUMULATIVE_UTF8_BYTES:
            raise ValueError("qualification tree exceeds its cumulative UTF-8 limit")


def exact_qualification_object(
    value: object, expected: Sequence[str], *, name: str
) -> dict[str, object]:
    """Require exact string keys before any hash-based schema comparison."""

    if type(value) is not dict:
        raise ValueError(f"{name} fields differ from the schema")
    mapping = cast(dict[object, object], value)
    keys = tuple(mapping.keys())
    expected_keys = tuple(expected)
    if any(type(key) is not str for key in keys) or any(
        type(key) is not str for key in expected_keys
    ):
        raise ValueError(f"{name} keys must be exact strings")
    if len(keys) != len(expected_keys) or frozenset(keys) != frozenset(expected_keys):
        raise ValueError(f"{name} fields differ from the schema")
    return cast(dict[str, object], value)


def _canonical_admitted(value: object) -> object:
    if type(value) is dict:
        mapping = cast(dict[str, object], value)
        return {key: _canonical_admitted(mapping[key]) for key in sorted(mapping)}
    if type(value) in (tuple, list):
        return [_canonical_admitted(item) for item in cast(Sequence[object], value)]
    if type(value) in (str, int, float, bool) or value is None:
        return value
    raise AssertionError("preflight admitted an unsupported qualification value")


def _registry_sha256(value: object) -> str:
    preflight_qualification_tree(value)
    raw = json.dumps(
        _canonical_admitted(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return _sha256_bytes(raw)


def _module_sha256(module: ModuleType) -> str:
    source = inspect.getsourcefile(module)
    if source is None:
        raise RuntimeError(f"cannot locate source for {module.__name__}")
    return _sha256_bytes(Path(source).read_bytes())


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class QualificationIdentity:
    lane_source_sha256: str
    dependency_source_sha256: tuple[tuple[str, str], ...]
    runtime_identity: tuple[tuple[str, str], ...]
    dependency_versions: tuple[tuple[str, str], ...]
    workload_registry_sha256: str
    paper_registry_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.lane_source_sha256, "lane_source_sha256")
        _require_sha256(self.workload_registry_sha256, "workload_registry_sha256")
        _require_sha256(self.paper_registry_sha256, "paper_registry_sha256")
        for name, entries, hashed in (
            ("dependency_source_sha256", self.dependency_source_sha256, True),
            ("runtime_identity", self.runtime_identity, False),
            ("dependency_versions", self.dependency_versions, False),
        ):
            preflight_qualification_tree(entries)
            if type(entries) is not tuple or not entries:
                raise ValueError(f"{name} must be a nonempty exact tuple")
            keys: list[str] = []
            for item in entries:
                if (
                    type(item) is not tuple
                    or len(item) != 2
                    or type(item[0]) is not str
                    or not item[0]
                    or type(item[1]) is not str
                    or not item[1]
                ):
                    raise ValueError(f"{name} entries must be exact string pairs")
                keys.append(item[0])
                if hashed:
                    _require_sha256(item[1], f"{name}.{item[0]}")
            if keys != sorted(keys) or len(keys) != len(set(keys)):
                raise ValueError(f"{name} keys must be sorted and unique")

    def to_payload(self) -> dict[str, object]:
        return {
            "lane_source_sha256": self.lane_source_sha256,
            "dependency_source_sha256": [list(item) for item in self.dependency_source_sha256],
            "runtime_identity": [list(item) for item in self.runtime_identity],
            "dependency_versions": [list(item) for item in self.dependency_versions],
            "workload_registry_sha256": self.workload_registry_sha256,
            "paper_registry_sha256": self.paper_registry_sha256,
        }


def collect_qualification_identity(
    *,
    lane_module: ModuleType,
    dependency_modules: Sequence[ModuleType],
    workload_registry: object,
    paper_registry: object,
) -> QualificationIdentity:
    modules = {module.__name__: module for module in dependency_modules}
    modules[__name__] = sys.modules[__name__]
    dependencies = tuple(
        sorted((name, _module_sha256(module)) for name, module in modules.items())
    )
    runtime = tuple(
        sorted(
            (
                ("jax_backend", jax.default_backend()),
                ("machine", platform.machine()),
                ("python_implementation", sys.implementation.name),
                ("python_version", platform.python_version()),
                ("system", platform.system()),
            )
        )
    )
    versions = tuple(sorted((("jax", jax.__version__), ("numpy", np.__version__))))
    return QualificationIdentity(
        lane_source_sha256=_module_sha256(lane_module),
        dependency_source_sha256=dependencies,
        runtime_identity=runtime,
        dependency_versions=versions,
        workload_registry_sha256=_registry_sha256(workload_registry),
        paper_registry_sha256=_registry_sha256(paper_registry),
    )


def identity_from_payload(value: object) -> QualificationIdentity:
    preflight_qualification_tree(value)
    expected = tuple(field.name for field in dataclasses.fields(QualificationIdentity))
    raw = cast(
        dict[str, Any], dict(exact_qualification_object(value, expected, name="identity payload"))
    )
    for name in ("dependency_source_sha256", "runtime_identity", "dependency_versions"):
        entries = raw[name]
        if type(entries) is not list or any(
            type(item) is not list
            or len(item) != 2
            or any(type(part) is not str for part in item)
            for item in entries
        ):
            raise ValueError(f"identity {name} must be an exact list of string pairs")
        raw[name] = tuple((item[0], item[1]) for item in entries)
    return QualificationIdentity(**raw)


def require_current_identity(
    actual: object, expected: QualificationIdentity
) -> QualificationIdentity:
    if type(actual) is not QualificationIdentity or actual != expected:
        raise ValueError("qualification identity differs from the current tree/runtime")
    return actual


__all__ = [
    "QualificationIdentity",
    "collect_qualification_identity",
    "exact_qualification_object",
    "identity_from_payload",
    "preflight_qualification_tree",
    "require_current_identity",
]
