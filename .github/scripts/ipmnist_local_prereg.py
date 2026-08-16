#!/usr/bin/env python3
"""Fail-closed local launch and result checks for frozen IPMNIST protocols.

This driver never downloads MNIST and never runs a learner.  Its mutating
commands only claim a previously absent append-only namespace and publish
exclusive JSON receipts.  Dataset materialization and benchmark execution are
separate, explicitly authorized operator steps.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Final, cast

AUTHORIZED_REPOSITORY: Final = "elizaOS/asi"
AUTHORIZED_LOGIN: Final = "lalalune"
AUTHORIZED_USER_ID: Final = 18_633_264
AUTHORIZED_ASSOCIATION: Final = "MEMBER"
BENCHMARK_PATH: Final = "alberta_framework/benchmarks/ipmnist_screening.py"
OUTPUT_ROOT: Final = Path("outputs/ipmnist_screening")
MNIST_CACHE_RELATIVE_PATH: Final = Path(
    "openml/openml.org/data/v1/download/52667/mnist_784.arff.gz"
)
MNIST_CACHE_SIZE_BYTES: Final = 15_469_256
MNIST_CACHE_SHA256: Final = (
    "fe4410d8dbb50f6db6482b187557c5cb8bccfbcec74eeb6abc47c858f4ffab78"
)
EXPECTED_PACKAGES: Final = {
    "chex": "0.1.92",
    "jax": "0.11.0",
    "jaxlib": "0.11.0",
    "numpy": "2.5.1",
    "scikit-learn": "1.9.0",
}
EXPECTED_JAX_CONFIG: Final = {
    "jax_enable_x64": False,
    "jax_default_matmul_precision": None,
    "jax_disable_jit": False,
    "jax_numpy_dtype_promotion": "standard",
    "jax_numpy_rank_promotion": "allow",
    "jax_random_seed_offset": 0,
    "jax_threefry_partitionable": True,
    "jax_default_prng_impl": "threefry2x32",
}
EXPECTED_PROCESS_ENVIRONMENT: Final = {
    "CUDA_VISIBLE_DEVICES": "",
    "JAX_DEFAULT_MATMUL_PRECISION": None,
    "JAX_ENABLE_X64": "false",
    "JAX_PLATFORM_NAME": "cpu",
    "JAX_PLATFORMS": "cpu",
    "OMP_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONOPTIMIZE": "0",
    "XLA_FLAGS": "--xla_force_host_platform_device_count=1",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
}
SCREENING_ENVIRONMENT_KEYS: Final = (
    "CUDA_VISIBLE_DEVICES",
    "JAX_DEFAULT_MATMUL_PRECISION",
    "JAX_ENABLE_X64",
    "JAX_PLATFORM_NAME",
    "JAX_PLATFORMS",
    "OMP_NUM_THREADS",
    "XLA_FLAGS",
)
EXPECTED_POLICY: Final = {
    "evidence_class": "development_screening_diagnostic",
    "development_only": True,
    "scientific_promotion_allowed": False,
}


@dataclass(frozen=True)
class LocalStage:
    key: str
    seeds: tuple[int, ...]
    n_tasks: int


@dataclass(frozen=True)
class LocalProtocol:
    key: str
    issue: int
    namespace: str
    control: str
    candidate: str
    stages: tuple[LocalStage, ...]
    max_shards: int


LOCAL_PROTOCOLS: Final = {
    "issue184": LocalProtocol(
        key="issue184",
        issue=184,
        namespace="rls_preset_ablation_r1",
        control="rls_head_resid_l1_preset005",
        candidate="rls_head_resid_l1_noreset",
        stages=(LocalStage("screen_60", (0, 1, 2), 60),),
        max_shards=6,
    ),
    "issue14-v2": LocalProtocol(
        key="issue14-v2",
        issue=14,
        namespace="rls_l2init_v2",
        control="rls_head_resid_l1_preset005",
        candidate="rls_head_resid_l1_preset005_l2init",
        stages=(
            LocalStage("screen_60", (20, 21, 22), 60),
            LocalStage("confirm_200_tuning", (20, 21, 22), 200),
            LocalStage("confirm_200_evaluation", tuple(range(23, 40)), 200),
        ),
        max_shards=46,
    ),
}


def protocol_for(key: str) -> LocalProtocol:
    try:
        return LOCAL_PROTOCOLS[key]
    except KeyError as exc:
        raise ValueError(f"unsupported local preregistration protocol: {key!r}") from exc


def _canonical_json(value: Any) -> str:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    try:
        return json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value cannot be encoded as canonical strict JSON") from exc


def _canonical_sha256(value: Any) -> str:
    if isinstance(value, bytes):
        encoded = value
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
    else:
        encoded = _canonical_json(value).encode()
    return hashlib.sha256(encoded).hexdigest()


def _receipt_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError("receipt cannot be encoded as strict JSON") from exc


def _receipt_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_receipt_bytes(payload)).hexdigest()


def _lower_hex(value: object, length: int, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be exactly {length} lowercase hexadecimal characters")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive built-in integer")
    return value


def _binding_text(
    protocol: LocalProtocol,
    *,
    source: str,
    tree: str,
    uv_lock_sha256: str,
    benchmark_blob_sha1: str,
    ref_name: str,
    runner_receipt_sha256: str,
    data_home_sha256: str,
) -> str:
    source = _lower_hex(source, 40, name="source")
    tree = _lower_hex(tree, 40, name="tree")
    uv_lock_sha256 = _lower_hex(uv_lock_sha256, 64, name="uv_lock_sha256")
    benchmark_blob_sha1 = _lower_hex(
        benchmark_blob_sha1, 40, name="benchmark_blob_sha1"
    )
    runner_receipt_sha256 = _lower_hex(
        runner_receipt_sha256, 64, name="runner_receipt_sha256"
    )
    data_home_sha256 = _lower_hex(data_home_sha256, 64, name="data_home_sha256")
    if (
        not isinstance(ref_name, str)
        or not ref_name
        or any(character.isspace() for character in ref_name)
    ):
        raise ValueError("ref_name must be non-empty and contain no whitespace")
    return (
        f"issue={protocol.issue} protocol={protocol.key} source={source} tree={tree} "
        f"uv_lock_sha256={uv_lock_sha256} benchmark_blob_sha1={benchmark_blob_sha1} "
        f"ref={ref_name} runner_receipt_sha256={runner_receipt_sha256} "
        f"data_home_sha256={data_home_sha256} "
        f"namespace={OUTPUT_ROOT.as_posix()}/{protocol.namespace} "
        f"plan_sha256={_canonical_sha256(protocol)}"
    )


def amendment_line(
    protocol: LocalProtocol,
    *,
    source: str,
    tree: str,
    uv_lock_sha256: str,
    benchmark_blob_sha1: str,
    ref_name: str,
    runner_receipt_sha256: str,
    data_home_sha256: str,
) -> str:
    binding = _binding_text(
        protocol,
        source=source,
        tree=tree,
        uv_lock_sha256=uv_lock_sha256,
        benchmark_blob_sha1=benchmark_blob_sha1,
        ref_name=ref_name,
        runner_receipt_sha256=runner_receipt_sha256,
        data_home_sha256=data_home_sha256,
    )
    return f"ASI_LOCAL_PREREG_AMENDMENT_V1 {binding} compute=uncompensated"


def authorization_line(
    protocol: LocalProtocol,
    *,
    source: str,
    tree: str,
    uv_lock_sha256: str,
    benchmark_blob_sha1: str,
    ref_name: str,
    runner_receipt_sha256: str,
    data_home_sha256: str,
    amendment_comment_id: int,
    amendment_sha256: str,
) -> str:
    binding = _binding_text(
        protocol,
        source=source,
        tree=tree,
        uv_lock_sha256=uv_lock_sha256,
        benchmark_blob_sha1=benchmark_blob_sha1,
        ref_name=ref_name,
        runner_receipt_sha256=runner_receipt_sha256,
        data_home_sha256=data_home_sha256,
    )
    amendment_comment_id = _positive_int(
        amendment_comment_id, name="amendment_comment_id"
    )
    amendment_sha256 = _lower_hex(amendment_sha256, 64, name="amendment_sha256")
    return (
        f"ASI_LOCAL_PREREG_LAUNCH_V1 {binding} "
        f"amendment_comment_id={amendment_comment_id} "
        f"amendment_sha256={amendment_sha256} protocol_approval=approved "
        "seed_budget=approved compute=authorized-uncompensated"
    )


def _finite_float(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite built-in float")
    return value


def _finite_differences(
    values: Sequence[object], *, expected: int | None = None
) -> tuple[float, ...]:
    resolved = tuple(_finite_float(value, name="per_seed_diff") for value in values)
    if not resolved or (expected is not None and len(resolved) != expected):
        requirement = "non-empty" if expected is None else f"exactly {expected} values"
        raise ValueError(f"per-seed difference coverage must contain {requirement}")
    return resolved


def classify_issue184(mean_diff: object, per_seed_diff: Sequence[object]) -> str:
    mean = _finite_float(mean_diff, name="mean_diff")
    differences = _finite_differences(per_seed_diff, expected=3)
    if mean > 0.002 and all(value > 0.0 for value in differences):
        return "no_reset_win"
    if mean < -0.002 and all(value < 0.0 for value in differences):
        return "reset_load_bearing"
    if abs(mean) <= 0.001 and all(abs(value) <= 0.0015 for value in differences):
        return "practical_equivalence"
    return "inconclusive"


def l2init_gate_passes(mean_diff: object, per_seed_diff: Sequence[object]) -> bool:
    mean = _finite_float(mean_diff, name="mean_diff")
    differences = _finite_differences(per_seed_diff)
    return mean > 0.002 and all(value > 0.0 for value in differences)


def _parse_cpuset(value: object) -> tuple[int, ...]:
    if not isinstance(value, str) or not value or re.fullmatch(r"[0-9,-]+", value) is None:
        raise ValueError("cpuset must use canonical comma-separated CPU IDs or ranges")
    cpus: list[int] = []
    for component in value.split(","):
        if "-" in component:
            pieces = component.split("-")
            if len(pieces) != 2:
                raise ValueError("cpuset contains an invalid range")
            start_text, stop_text = pieces
            if (
                not start_text
                or not stop_text
                or (start_text.startswith("0") and start_text != "0")
                or (stop_text.startswith("0") and stop_text != "0")
            ):
                raise ValueError("cpuset ranges must use canonical integers")
            start, stop = int(start_text), int(stop_text)
            if stop <= start:
                raise ValueError("cpuset ranges must be strictly increasing")
            cpus.extend(range(start, stop + 1))
        else:
            if not component or (component.startswith("0") and component != "0"):
                raise ValueError("cpuset must use canonical integers")
            cpus.append(int(component))
    if not cpus or cpus != sorted(set(cpus)):
        raise ValueError("cpuset CPU IDs must be unique and strictly increasing")
    return tuple(cpus)


def _parse_utc(value: object, *, label: str) -> dt.datetime:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} timestamp is missing or invalid")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError(f"{label} timestamp is not valid ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise RuntimeError(f"{label} timestamp must identify UTC")
    return parsed


def _parse_github_utc(value: object, *, label: str) -> dt.datetime:
    if not isinstance(value, str) or re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", value
    ) is None:
        raise RuntimeError(f"{label} timestamp is not canonical GitHub UTC")
    return _parse_utc(value, label=label)


def _owner_matches(comment: object, *, body: str) -> bool:
    return (
        isinstance(comment, dict)
        and comment.get("body") == body
        and isinstance(comment.get("user"), dict)
        and comment["user"].get("login") == AUTHORIZED_LOGIN
        and type(comment["user"].get("id")) is int
        and comment["user"].get("id") == AUTHORIZED_USER_ID
        and comment.get("author_association") == AUTHORIZED_ASSOCIATION
    )


def _comment_receipt(
    comment: Mapping[str, Any], *, issue: int, label: str
) -> tuple[int, str, dt.datetime]:
    try:
        comment_id = _positive_int(comment.get("id"), name=f"{label}_comment_id")
    except ValueError as exc:
        raise RuntimeError(f"{label} comment ID is missing or invalid") from exc
    created_at = comment.get("created_at")
    updated_at = comment.get("updated_at")
    if created_at != updated_at:
        raise RuntimeError(f"{label} comment must be exact and never edited")
    timestamp = _parse_github_utc(created_at, label=label)
    expected_url = (
        f"https://github.com/{AUTHORIZED_REPOSITORY}/issues/{issue}"
        f"#issuecomment-{comment_id}"
    )
    if comment.get("html_url") != expected_url:
        raise RuntimeError(f"{label} comment URL is not the canonical GitHub issue record")
    return comment_id, cast(str, created_at), timestamp


def verify_owner_records(
    protocol: LocalProtocol,
    *,
    comments: Sequence[object],
    launch_time: dt.datetime,
    source: str,
    tree: str,
    uv_lock_sha256: str,
    benchmark_blob_sha1: str,
    ref_name: str,
    runner_receipt_sha256: str,
    data_home_sha256: str,
) -> dict[str, Any]:
    if launch_time.tzinfo is None or launch_time.utcoffset() != dt.timedelta(0):
        raise ValueError("launch_time must identify UTC")
    amendment = amendment_line(
        protocol,
        source=source,
        tree=tree,
        uv_lock_sha256=uv_lock_sha256,
        benchmark_blob_sha1=benchmark_blob_sha1,
        ref_name=ref_name,
        runner_receipt_sha256=runner_receipt_sha256,
        data_home_sha256=data_home_sha256,
    )
    amendment_matches = [
        cast(dict[str, Any], comment)
        for comment in comments
        if _owner_matches(comment, body=amendment)
    ]
    if len(amendment_matches) != 1:
        raise RuntimeError(
            "expected exactly one standalone owner amendment comment; "
            f"found {len(amendment_matches)}"
        )
    amendment_comment = amendment_matches[0]
    amendment_id, amendment_created, amendment_time = _comment_receipt(
        amendment_comment, issue=protocol.issue, label="amendment"
    )
    amendment_sha256 = _canonical_sha256(amendment)
    authorization = authorization_line(
        protocol,
        source=source,
        tree=tree,
        uv_lock_sha256=uv_lock_sha256,
        benchmark_blob_sha1=benchmark_blob_sha1,
        ref_name=ref_name,
        runner_receipt_sha256=runner_receipt_sha256,
        data_home_sha256=data_home_sha256,
        amendment_comment_id=amendment_id,
        amendment_sha256=amendment_sha256,
    )
    authorization_matches = [
        cast(dict[str, Any], comment)
        for comment in comments
        if _owner_matches(comment, body=authorization)
    ]
    if len(authorization_matches) != 1:
        raise RuntimeError(
            "expected exactly one standalone owner authorization comment; "
            f"found {len(authorization_matches)}"
        )
    authorization_comment = authorization_matches[0]
    authorization_id, authorization_created, authorization_time = _comment_receipt(
        authorization_comment, issue=protocol.issue, label="authorization"
    )
    if amendment_time >= authorization_time:
        raise RuntimeError("amendment must be durable before authorization")
    if authorization_time >= launch_time:
        raise RuntimeError("authorization must be durable before local launch")
    return {
        "amendment_comment_id": amendment_id,
        "amendment_comment_url": amendment_comment["html_url"],
        "amendment_created_at": amendment_created,
        "amendment_updated_at": amendment_created,
        "amendment_line": amendment,
        "amendment_sha256": amendment_sha256,
        "authorization_comment_id": authorization_id,
        "authorization_comment_url": authorization_comment["html_url"],
        "authorization_created_at": authorization_created,
        "authorization_updated_at": authorization_created,
        "authorization_line": authorization,
        "authorization_sha256": _canonical_sha256(authorization),
    }


def _claim_namespace(path: Path) -> None:
    target = Path(path)
    if os.path.lexists(target):
        raise FileExistsError(f"append-only namespace is already occupied: {target}")
    try:
        os.mkdir(target)
    except FileExistsError as exc:
        raise FileExistsError(
            f"append-only namespace is already occupied: {target}"
        ) from exc


def _strict_json_bytes(path: Path) -> tuple[dict[str, Any], bytes]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{path}: duplicate JSON key {key!r}")
            value[key] = item
        return value

    def reject_constant(value: str) -> Any:
        raise ValueError(f"{path}: non-finite JSON constant {value!r}")

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"{path}: non-finite JSON number {value!r}")
        return parsed

    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"{path}: could not read one UTF-8 JSON artifact") from exc
    payload = json.loads(
        text,
        object_pairs_hook=pairs_hook,
        parse_constant=reject_constant,
        parse_float=finite_float,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected one JSON object")
    return payload, raw


def _strict_json(path: Path) -> dict[str, Any]:
    payload, _raw = _strict_json_bytes(path)
    return payload


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = _receipt_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _git(root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ("git", "-C", str(root), *args),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise RuntimeError("Git is required for local preregistration") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip()
        raise RuntimeError(f"Git command failed: {detail or args}")
    return completed.stdout.strip()


def _screening_module(root: Path) -> ModuleType:
    import alberta_framework.benchmarks.ipmnist_screening as screening

    module_file = getattr(screening, "__file__", None)
    expected = root / BENCHMARK_PATH
    if (
        not isinstance(module_file, str)
        or expected.is_symlink()
        or not expected.is_file()
        or Path(module_file).resolve(strict=True) != expected
    ):
        raise RuntimeError("IPMNIST screening module was not imported from the bound root")
    return screening


def _github_json(path: str, *, token: str) -> Any:
    request = urllib.request.Request(
        f"https://api.github.com{path}",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "asi-ipmnist-local-prereg-v1",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, urllib.error.HTTPError) as exc:
        raise RuntimeError(f"GitHub API request failed for {path}: {exc}") from exc


def _github_pages(path: str, *, token: str) -> list[Any]:
    separator = "&" if "?" in path else "?"
    values: list[Any] = []
    for page in range(1, 101):
        payload = _github_json(f"{path}{separator}per_page=100&page={page}", token=token)
        if not isinstance(payload, list):
            raise RuntimeError(f"GitHub API pagination expected a list for {path}")
        values.extend(payload)
        if len(payload) < 100:
            return values
    raise RuntimeError(f"GitHub API pagination exceeded 10,000 records for {path}")


def _require_exact_keys(value: object, expected: set[str], *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be one JSON object")
    if set(value) != expected:
        raise ValueError(
            f"{context} key mismatch; missing={sorted(expected - set(value))}, "
            f"unexpected={sorted(set(value) - expected)}"
        )
    return cast(dict[str, Any], value)


def _validate_runner_receipt(
    payload: object, *, expected_cpuset: str
) -> dict[str, Any]:
    receipt = _require_exact_keys(
        payload,
        {
            "schema",
            "runner",
            "platform",
            "cpu",
            "python",
            "python_optimization_level",
            "packages",
            "jax",
            "process_environment",
            "screening_environment",
            "cache_contract",
        },
        context="runner receipt",
    )
    if receipt["schema"] != "asi.ipmnist_local_prereg.runner.v1":
        raise ValueError("runner receipt schema is unsupported")
    if receipt["runner"] != "local-beast-linux-x86_64-cpu":
        raise ValueError("runner receipt must identify the local BEAST CPU host")
    host = _require_exact_keys(
        receipt["platform"], {"system", "release", "machine"}, context="runner platform"
    )
    if (
        host["system"] != "Linux"
        or host["machine"] != "x86_64"
        or not isinstance(host["release"], str)
        or not host["release"]
    ):
        raise ValueError(f"runner platform must be Linux x86_64, got {host}")
    cpu = _require_exact_keys(
        receipt["cpu"],
        {"model", "requested_cpuset", "effective_cpuset"},
        context="runner CPU",
    )
    if not isinstance(cpu["model"], str) or not cpu["model"].strip():
        raise ValueError("runner CPU model must be non-empty")
    expected_cpus = _parse_cpuset(expected_cpuset)
    if cpu["requested_cpuset"] != expected_cpuset:
        raise ValueError("runner requested cpuset differs from the launch binding")
    effective = cpu["effective_cpuset"]
    if (
        not isinstance(effective, list)
        or any(type(value) is not int or value < 0 for value in effective)
        or tuple(effective) != expected_cpus
    ):
        raise ValueError("runner effective cpuset differs from the requested canonical set")
    python = _require_exact_keys(
        receipt["python"], {"implementation", "version"}, context="runner Python"
    )
    if (
        python["implementation"] != "CPython"
        or not isinstance(python["version"], str)
        or not re.fullmatch(r"3\.12\.[0-9]+", python["version"])
    ):
        raise ValueError(f"runner must use CPython 3.12.x, got {python}")
    if (
        type(receipt["python_optimization_level"]) is not int
        or receipt["python_optimization_level"] != 0
    ):
        raise ValueError("runner must use Python optimization level zero")
    packages = _require_exact_keys(
        receipt["packages"], set(EXPECTED_PACKAGES), context="runner packages"
    )
    if packages != EXPECTED_PACKAGES:
        raise ValueError(f"runner package versions differ from the frozen lock: {packages}")
    jax_binding = _require_exact_keys(
        receipt["jax"], {"backend", "devices", "config"}, context="runner JAX"
    )
    devices = jax_binding["devices"]
    if jax_binding["backend"] != "cpu" or not isinstance(devices, list) or len(devices) != 1:
        raise ValueError("runner must expose exactly one JAX CPU device")
    device = _require_exact_keys(
        devices[0],
        {"id", "platform", "device_kind", "process_index"},
        context="runner JAX device",
    )
    if (
        type(device["id"]) is not int
        or device["id"] < 0
        or device["platform"] != "cpu"
        or not isinstance(device["device_kind"], str)
        or not device["device_kind"]
        or type(device["process_index"]) is not int
        or device["process_index"] < 0
    ):
        raise ValueError(f"runner JAX device is invalid: {device}")
    config = _require_exact_keys(
        jax_binding["config"], set(EXPECTED_JAX_CONFIG), context="runner JAX config"
    )
    if _canonical_json(config) != _canonical_json(EXPECTED_JAX_CONFIG):
        raise ValueError(f"runner JAX config differs from the frozen contract: {config}")
    process = _require_exact_keys(
        receipt["process_environment"],
        set(EXPECTED_PROCESS_ENVIRONMENT),
        context="runner process environment",
    )
    if process != EXPECTED_PROCESS_ENVIRONMENT:
        raise ValueError(f"runner process environment differs from the contract: {process}")
    screening = _require_exact_keys(
        receipt["screening_environment"],
        {"schema", "python", "platform", "packages", "jax", "process_environment"},
        context="runner screening environment",
    )
    expected_screening = {
        "schema": "alberta.ipmnist_screening.runtime.v1",
        "python": python,
        "platform": host,
        "packages": packages,
        "jax": jax_binding,
        "process_environment": {
            name: process[name] for name in SCREENING_ENVIRONMENT_KEYS
        },
    }
    if screening != expected_screening:
        raise ValueError("runner screening environment is not an exact derived binding")
    cache = _require_exact_keys(
        receipt["cache_contract"],
        {"relative_path", "size_bytes", "sha256"},
        context="runner cache contract",
    )
    if cache != {
        "relative_path": MNIST_CACHE_RELATIVE_PATH.as_posix(),
        "size_bytes": MNIST_CACHE_SIZE_BYTES,
        "sha256": MNIST_CACHE_SHA256,
    }:
        raise ValueError("runner cache contract differs from canonical MNIST")
    return receipt


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name") and ":" in line:
                model = line.split(":", 1)[1].strip()
                if model:
                    return model
    except OSError:
        pass
    model = platform.processor().strip()
    if not model:
        raise RuntimeError("local runner CPU model is unavailable")
    return model


def capture_runner_receipt(cpuset: str) -> dict[str, Any]:
    expected_cpus = _parse_cpuset(cpuset)
    if not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("local runner cannot report its effective CPU affinity")
    effective_cpus = tuple(sorted(os.sched_getaffinity(0)))
    if effective_cpus != expected_cpus:
        raise RuntimeError(
            f"effective CPU affinity {effective_cpus} differs from requested {expected_cpus}; "
            "invoke this command under the exact taskset"
        )
    try:
        import jax

        packages = {
            name: importlib.metadata.version(name) for name in EXPECTED_PACKAGES
        }
        devices = [
            {
                "id": int(device.id),
                "platform": str(device.platform),
                "device_kind": str(device.device_kind),
                "process_index": int(device.process_index),
            }
            for device in jax.devices()
        ]
        jax_binding = {
            "backend": jax.default_backend(),
            "devices": devices,
            "config": {
                "jax_enable_x64": bool(jax.config.jax_enable_x64),
                "jax_default_matmul_precision": (
                    None
                    if jax.config.jax_default_matmul_precision is None
                    else str(jax.config.jax_default_matmul_precision)
                ),
                "jax_disable_jit": bool(jax.config.jax_disable_jit),
                "jax_numpy_dtype_promotion": str(jax.config.jax_numpy_dtype_promotion),
                "jax_numpy_rank_promotion": str(jax.config.jax_numpy_rank_promotion),
                "jax_random_seed_offset": int(jax.config.jax_random_seed_offset),
                "jax_threefry_partitionable": bool(
                    jax.config.jax_threefry_partitionable
                ),
                "jax_default_prng_impl": str(jax.config.jax_default_prng_impl),
            },
        }
    except Exception as exc:
        raise RuntimeError("local runner JAX identity is unavailable") from exc
    python = {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
    }
    host = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    }
    process = {name: os.environ.get(name) for name in EXPECTED_PROCESS_ENVIRONMENT}
    screening = {
        "schema": "alberta.ipmnist_screening.runtime.v1",
        "python": python,
        "platform": host,
        "packages": packages,
        "jax": jax_binding,
        "process_environment": {
            name: process[name] for name in SCREENING_ENVIRONMENT_KEYS
        },
    }
    receipt = {
        "schema": "asi.ipmnist_local_prereg.runner.v1",
        "runner": "local-beast-linux-x86_64-cpu",
        "platform": host,
        "cpu": {
            "model": _cpu_model(),
            "requested_cpuset": cpuset,
            "effective_cpuset": list(effective_cpus),
        },
        "python": python,
        "python_optimization_level": sys.flags.optimize,
        "packages": packages,
        "jax": jax_binding,
        "process_environment": process,
        "screening_environment": screening,
        "cache_contract": {
            "relative_path": MNIST_CACHE_RELATIVE_PATH.as_posix(),
            "size_bytes": MNIST_CACHE_SIZE_BYTES,
            "sha256": MNIST_CACHE_SHA256,
        },
    }
    return _validate_runner_receipt(receipt, expected_cpuset=cpuset)


def _validate_repository_identity(payload: object) -> dict[str, Any]:
    identity = _require_exact_keys(
        payload,
        {
            "schema",
            "source",
            "tree",
            "uv_lock_sha256",
            "benchmark_blob_sha1",
            "source_provenance",
        },
        context="repository identity",
    )
    if identity["schema"] != "asi.ipmnist_local_prereg.source.v1":
        raise ValueError("repository identity schema is unsupported")
    source = _lower_hex(identity["source"], 40, name="source")
    tree = _lower_hex(identity["tree"], 40, name="tree")
    uv_lock = _lower_hex(identity["uv_lock_sha256"], 64, name="uv_lock_sha256")
    _lower_hex(identity["benchmark_blob_sha1"], 40, name="benchmark_blob_sha1")
    provenance = _require_exact_keys(
        identity["source_provenance"],
        {
            "schema",
            "git_commit",
            "git_tree",
            "git_object_format",
            "relevant_source_scope",
            "relevant_source_file_count",
            "relevant_source_sha256",
            "uv_lock_sha256",
            "worktree_clean",
        },
        context="repository source provenance",
    )
    if (
        provenance["schema"] != "alberta.ipmnist_screening.source_provenance.v1"
        or provenance["git_commit"] != source
        or provenance["git_tree"] != tree
        or provenance["git_object_format"] != "sha1"
        or provenance["uv_lock_sha256"] != uv_lock
        or provenance["worktree_clean"] is not True
        or type(provenance["relevant_source_file_count"]) is not int
        or provenance["relevant_source_file_count"] <= 0
    ):
        raise ValueError("repository source provenance differs from the launch identity")
    _lower_hex(
        provenance["relevant_source_sha256"], 64, name="relevant_source_sha256"
    )
    return identity


def capture_repository_identity(root: Path) -> dict[str, Any]:
    if Path(root).is_symlink():
        raise RuntimeError("repository root must not be a symlink")
    try:
        resolved = Path(root).resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("repository root is unavailable") from exc
    top_level = Path(_git(resolved, "rev-parse", "--show-toplevel")).resolve(strict=True)
    if top_level != resolved:
        raise RuntimeError("root must be the exact Git repository top level")
    screening = _screening_module(resolved)
    provenance = screening._screening_source_provenance(resolved)
    source = _git(resolved, "rev-parse", "--verify", "HEAD")
    tree = _git(resolved, "rev-parse", "--verify", "HEAD^{tree}")
    benchmark_blob = _git(resolved, "rev-parse", f"HEAD:{BENCHMARK_PATH}")
    identity = {
        "schema": "asi.ipmnist_local_prereg.source.v1",
        "source": source,
        "tree": tree,
        "uv_lock_sha256": provenance["uv_lock_sha256"],
        "benchmark_blob_sha1": benchmark_blob,
        "source_provenance": provenance,
    }
    return _validate_repository_identity(identity)


def _validate_remote_tag(payload: object, *, ref_name: str, source: str) -> None:
    if not isinstance(payload, dict):
        raise RuntimeError("remote tag lookup returned a non-object")
    tag_object = payload.get("object")
    if (
        payload.get("ref") != f"refs/tags/{ref_name}"
        or not isinstance(tag_object, dict)
        or tag_object.get("type") != "commit"
    ):
        raise RuntimeError("launch ref must be one exact lightweight remote tag")
    if tag_object.get("sha") != source:
        raise RuntimeError("remote launch tag does not point to the authorized source")


def _safe_data_home(path: Path) -> tuple[str, str]:
    absolute = os.path.abspath(os.fspath(path))
    return absolute, _canonical_sha256(absolute)


def claim_local_launch(
    *,
    protocol_key: str,
    root: Path,
    repository: str,
    ref_name: str,
    cpuset: str,
    data_home: Path,
    token: str,
    launch_time: dt.datetime | None = None,
    runner_receipt: Mapping[str, Any] | None = None,
    repository_identity: Mapping[str, Any] | None = None,
    comments: Sequence[object] | None = None,
    tag_payload: object | None = None,
) -> Path:
    if repository != AUTHORIZED_REPOSITORY:
        raise RuntimeError(f"repository must be exactly {AUTHORIZED_REPOSITORY}")
    protocol = protocol_for(protocol_key)
    if Path(root).is_symlink():
        raise RuntimeError("repository root must not be a symlink")
    try:
        resolved_root = Path(root).resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("repository root is unavailable") from exc
    identity = _validate_repository_identity(
        capture_repository_identity(resolved_root)
        if repository_identity is None
        else repository_identity
    )
    runner = _validate_runner_receipt(
        capture_runner_receipt(cpuset) if runner_receipt is None else runner_receipt,
        expected_cpuset=cpuset,
    )
    runner_sha256 = _receipt_sha256(runner)
    data_home_text, data_home_sha256 = _safe_data_home(data_home)
    source = cast(str, identity["source"])
    if tag_payload is None:
        encoded = urllib.parse.quote(ref_name, safe="")
        tag_payload = _github_json(
            f"/repos/{repository}/git/ref/tags/{encoded}", token=token
        )
    _validate_remote_tag(tag_payload, ref_name=ref_name, source=source)
    if comments is None:
        comments = _github_pages(
            f"/repos/{repository}/issues/{protocol.issue}/comments", token=token
        )
    launch_timestamp = dt.datetime.now(dt.UTC) if launch_time is None else launch_time
    authorization = verify_owner_records(
        protocol,
        comments=comments,
        launch_time=launch_timestamp,
        source=source,
        tree=cast(str, identity["tree"]),
        uv_lock_sha256=cast(str, identity["uv_lock_sha256"]),
        benchmark_blob_sha1=cast(str, identity["benchmark_blob_sha1"]),
        ref_name=ref_name,
        runner_receipt_sha256=runner_sha256,
        data_home_sha256=data_home_sha256,
    )
    cache_contract = cast(dict[str, Any], runner["cache_contract"])
    cache_path = Path(data_home_text) / cast(str, cache_contract["relative_path"])
    if os.path.lexists(cache_path):
        raise FileExistsError(
            f"canonical cache must be absent before namespace claim: {cache_path}"
        )
    output_parent = resolved_root / OUTPUT_ROOT
    if (
        not output_parent.is_dir()
        or output_parent.is_symlink()
        or output_parent.resolve(strict=True) != output_parent
    ):
        raise RuntimeError("local preregistration output parent must be a real directory")
    namespace = output_parent / protocol.namespace
    _claim_namespace(namespace)
    _write_json_exclusive(namespace / "runner.v1.json", runner)
    launch_payload = {
        "schema": "asi.ipmnist_local_prereg.launch.v1",
        "protocol_key": protocol.key,
        "protocol": asdict(protocol),
        "plan_sha256": _canonical_sha256(protocol),
        "repository": identity,
        "ref_name": ref_name,
        "data_home": data_home_text,
        "data_home_sha256": data_home_sha256,
        "cache_contract": cache_contract,
        "runner_receipt": "runner.v1.json",
        "runner_receipt_sha256": runner_sha256,
        "authorization": authorization,
        "launch_created_at": launch_timestamp.isoformat().replace("+00:00", "Z"),
        "dataset_accessed": False,
        "rerun_allowed": False,
    }
    _write_json_exclusive(namespace / "launch.v1.json", launch_payload)
    return namespace


def _hash_file(path: Path) -> tuple[int, str]:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError("cache file changed while it was hashed")
    return before.st_size, digest.hexdigest()


def record_cache_receipt(
    *,
    protocol_key: str,
    root: Path,
    data_home: Path,
    expected_relative_path: Path = MNIST_CACHE_RELATIVE_PATH,
    expected_size: int = MNIST_CACHE_SIZE_BYTES,
    expected_sha256: str = MNIST_CACHE_SHA256,
) -> dict[str, Any]:
    protocol = protocol_for(protocol_key)
    if Path(root).is_symlink():
        raise RuntimeError("repository root must not be a symlink")
    resolved_root = Path(root).resolve(strict=True)
    namespace = resolved_root / OUTPUT_ROOT / protocol.namespace
    for directory, label in (
        (resolved_root / "outputs", "outputs root"),
        (resolved_root / OUTPUT_ROOT, "IPMNIST output root"),
        (namespace, "claimed local preregistration namespace"),
    ):
        try:
            _require_real_directory(directory, label=label)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
    receipt_path = namespace / "cache.v1.json"
    if os.path.lexists(receipt_path):
        raise FileExistsError(f"cache receipt is already occupied: {receipt_path}")
    launch, launch_raw = _strict_json_bytes(namespace / "launch.v1.json")
    if (
        launch.get("schema") != "asi.ipmnist_local_prereg.launch.v1"
        or launch.get("protocol_key") != protocol.key
    ):
        raise ValueError("launch receipt does not bind this local protocol")
    data_home_text, data_home_sha256 = _safe_data_home(data_home)
    if (
        launch.get("data_home") != data_home_text
        or launch.get("data_home_sha256") != data_home_sha256
    ):
        raise ValueError("cache data home differs from the authorized launch")
    expected_sha256 = _lower_hex(expected_sha256, 64, name="expected_cache_sha256")
    if type(expected_size) is not int or expected_size <= 0:
        raise ValueError("expected cache size must be a positive built-in integer")
    expected_contract = {
        "relative_path": expected_relative_path.as_posix(),
        "size_bytes": expected_size,
        "sha256": expected_sha256,
    }
    if launch.get("cache_contract") != expected_contract:
        raise ValueError("cache contract differs from the authorized launch")
    cache_path = Path(data_home_text) / expected_relative_path
    if cache_path.is_symlink() or not cache_path.is_file():
        raise ValueError("canonical MNIST cache must be one non-symlink regular file")
    size, sha256 = _hash_file(cache_path)
    if size != expected_size or sha256 != expected_sha256:
        raise ValueError(
            "canonical MNIST cache bytes differ from the frozen size/hash contract"
        )
    receipt = {
        "schema": "asi.ipmnist_local_prereg.cache.v1",
        "protocol_key": protocol.key,
        "data_home": data_home_text,
        "data_home_sha256": data_home_sha256,
        "relative_path": expected_relative_path.as_posix(),
        "size_bytes": size,
        "sha256": sha256,
        "launch_receipt_sha256": hashlib.sha256(launch_raw).hexdigest(),
        "checked_at": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
    }
    _write_json_exclusive(receipt_path, receipt)
    return receipt


def _validate_authorization_receipt(
    payload: object,
    *,
    protocol: LocalProtocol,
    identity: Mapping[str, Any],
    ref_name: str,
    runner_receipt_sha256: str,
    data_home_sha256: str,
    launch_time: dt.datetime,
) -> dict[str, Any]:
    receipt = _require_exact_keys(
        payload,
        {
            "amendment_comment_id",
            "amendment_comment_url",
            "amendment_created_at",
            "amendment_updated_at",
            "amendment_line",
            "amendment_sha256",
            "authorization_comment_id",
            "authorization_comment_url",
            "authorization_created_at",
            "authorization_updated_at",
            "authorization_line",
            "authorization_sha256",
        },
        context="launch authorization receipt",
    )
    amendment_id = _positive_int(
        receipt["amendment_comment_id"], name="amendment_comment_id"
    )
    authorization_id = _positive_int(
        receipt["authorization_comment_id"], name="authorization_comment_id"
    )
    for name in ("amendment_comment_url", "authorization_comment_url"):
        if not isinstance(receipt[name], str) or not receipt[name]:
            raise ValueError(f"launch authorization receipt {name} must be non-empty")
    if receipt["amendment_created_at"] != receipt["amendment_updated_at"]:
        raise ValueError("persisted amendment receipt records an edited comment")
    if receipt["authorization_created_at"] != receipt["authorization_updated_at"]:
        raise ValueError("persisted authorization receipt records an edited comment")
    expected_amendment_url = (
        f"https://github.com/{AUTHORIZED_REPOSITORY}/issues/{protocol.issue}"
        f"#issuecomment-{amendment_id}"
    )
    expected_authorization_url = (
        f"https://github.com/{AUTHORIZED_REPOSITORY}/issues/{protocol.issue}"
        f"#issuecomment-{authorization_id}"
    )
    if (
        receipt["amendment_comment_url"] != expected_amendment_url
        or receipt["authorization_comment_url"] != expected_authorization_url
    ):
        raise ValueError("persisted owner comment URLs are not canonical GitHub records")
    amendment_time = _parse_github_utc(
        receipt["amendment_created_at"], label="amendment"
    )
    authorization_time = _parse_github_utc(
        receipt["authorization_created_at"], label="authorization"
    )
    if not amendment_time < authorization_time < launch_time:
        raise ValueError("persisted amendment, authorization, and launch order is invalid")
    binding = {
        "source": cast(str, identity["source"]),
        "tree": cast(str, identity["tree"]),
        "uv_lock_sha256": cast(str, identity["uv_lock_sha256"]),
        "benchmark_blob_sha1": cast(str, identity["benchmark_blob_sha1"]),
        "ref_name": ref_name,
        "runner_receipt_sha256": runner_receipt_sha256,
        "data_home_sha256": data_home_sha256,
    }
    expected_amendment = amendment_line(protocol, **binding)
    expected_amendment_sha256 = _canonical_sha256(expected_amendment)
    expected_authorization = authorization_line(
        protocol,
        amendment_comment_id=amendment_id,
        amendment_sha256=expected_amendment_sha256,
        **binding,
    )
    if (
        receipt["amendment_line"] != expected_amendment
        or receipt["amendment_sha256"] != expected_amendment_sha256
        or receipt["authorization_line"] != expected_authorization
        or receipt["authorization_sha256"] != _canonical_sha256(expected_authorization)
    ):
        raise ValueError("persisted owner comments differ from the exact launch binding")
    if amendment_id == authorization_id:
        raise ValueError("amendment and authorization must be distinct GitHub comments")
    return receipt


def _validate_launch_receipt(
    payload: object,
    *,
    protocol: LocalProtocol,
    identity: Mapping[str, Any],
    runner: Mapping[str, Any],
    runner_raw: bytes,
) -> dict[str, Any]:
    launch = _require_exact_keys(
        payload,
        {
            "schema",
            "protocol_key",
            "protocol",
            "plan_sha256",
            "repository",
            "ref_name",
            "data_home",
            "data_home_sha256",
            "cache_contract",
            "runner_receipt",
            "runner_receipt_sha256",
            "authorization",
            "launch_created_at",
            "dataset_accessed",
            "rerun_allowed",
        },
        context="launch receipt",
    )
    if (
        launch["schema"] != "asi.ipmnist_local_prereg.launch.v1"
        or launch["protocol_key"] != protocol.key
        or _canonical_json(launch["protocol"]) != _canonical_json(asdict(protocol))
        or launch["plan_sha256"] != _canonical_sha256(protocol)
        or launch["repository"] != identity
        or launch["runner_receipt"] != "runner.v1.json"
        or launch["runner_receipt_sha256"] != hashlib.sha256(runner_raw).hexdigest()
        or launch["cache_contract"] != runner["cache_contract"]
        or launch["dataset_accessed"] is not False
        or launch["rerun_allowed"] is not False
    ):
        raise ValueError(
            "launch receipt differs from the frozen protocol/source or exact runner "
            "receipt SHA-256"
        )
    ref_name = launch["ref_name"]
    if not isinstance(ref_name, str) or not ref_name or any(c.isspace() for c in ref_name):
        raise ValueError("launch receipt ref_name is invalid")
    data_home = launch["data_home"]
    if (
        not isinstance(data_home, str)
        or not data_home
        or launch["data_home_sha256"] != _canonical_sha256(data_home)
    ):
        raise ValueError("launch receipt data home binding is invalid")
    launch_time = _parse_utc(launch["launch_created_at"], label="launch")
    _validate_authorization_receipt(
        launch["authorization"],
        protocol=protocol,
        identity=identity,
        ref_name=ref_name,
        runner_receipt_sha256=cast(str, launch["runner_receipt_sha256"]),
        data_home_sha256=cast(str, launch["data_home_sha256"]),
        launch_time=launch_time,
    )
    return launch


def _validate_cache_receipt(
    payload: object,
    *,
    protocol: LocalProtocol,
    launch: Mapping[str, Any],
    launch_raw: bytes,
    verify_file: bool,
) -> dict[str, Any]:
    receipt = _require_exact_keys(
        payload,
        {
            "schema",
            "protocol_key",
            "data_home",
            "data_home_sha256",
            "relative_path",
            "size_bytes",
            "sha256",
            "launch_receipt_sha256",
            "checked_at",
        },
        context="cache receipt",
    )
    contract = cast(dict[str, Any], launch["cache_contract"])
    if (
        receipt["schema"] != "asi.ipmnist_local_prereg.cache.v1"
        or receipt["protocol_key"] != protocol.key
        or receipt["data_home"] != launch["data_home"]
        or receipt["data_home_sha256"] != launch["data_home_sha256"]
        or receipt["relative_path"] != contract["relative_path"]
        or receipt["size_bytes"] != contract["size_bytes"]
        or receipt["sha256"] != contract["sha256"]
        or receipt["launch_receipt_sha256"] != hashlib.sha256(launch_raw).hexdigest()
    ):
        raise ValueError("cache receipt differs from the authorized launch contract")
    checked_at = _parse_utc(receipt["checked_at"], label="cache receipt")
    if checked_at <= _parse_utc(launch["launch_created_at"], label="launch"):
        raise ValueError("cache receipt must be created after the namespace launch")
    if verify_file:
        cache_path = Path(cast(str, receipt["data_home"])) / cast(
            str, receipt["relative_path"]
        )
        if cache_path.is_symlink() or not cache_path.is_file():
            raise ValueError("cache receipt target is unavailable or is a symlink")
        size, sha256 = _hash_file(cache_path)
        if size != receipt["size_bytes"] or sha256 != receipt["sha256"]:
            raise ValueError("cache bytes changed after the independent receipt")
    return receipt


def _reject_symlinks(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"result namespace is unavailable or is a symlink: {root}")
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in (*directory_names, *file_names):
            path = base / name
            if path.is_symlink():
                raise ValueError(f"result namespace contains a symlink: {path}")


def _require_real_directory(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be one existing non-symlink directory")
    try:
        if path.resolve(strict=True) != path:
            raise ValueError(f"{label} must not traverse a symlink")
    except OSError as exc:
        raise ValueError(f"{label} is unavailable") from exc


def _expected_config(n_tasks: int) -> dict[str, int]:
    return {
        "n_tasks": n_tasks,
        "task_length": 5_000,
        "input_dim": 784,
        "hidden1": 300,
        "hidden2": 150,
        "n_classes": 10,
    }


def _expected_manifest(
    paths: Sequence[Path], shards: Sequence[Mapping[str, Any]], *, root: Path
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path, shard in zip(paths, shards, strict=True):
        raw = path.read_bytes()
        try:
            relative = path.resolve(strict=True).relative_to(root).as_posix()
        except (OSError, ValueError) as exc:
            raise ValueError("shard manifest input escaped the repository root") from exc
        entries.append(
            {
                "path": relative,
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "config_name": shard["config_name"],
                "seed": shard["seed"],
            }
        )
    return sorted(
        entries,
        key=lambda entry: (
            cast(str, entry["config_name"]),
            cast(int, entry["seed"]),
            cast(str, entry["path"]),
        ),
    )


def _validate_summary_reconstruction(
    *,
    summary: Mapping[str, Any],
    recomputed: Mapping[str, Any],
    expected_manifest: Sequence[Mapping[str, Any]],
) -> None:
    created_unix = summary.get("created_unix")
    if type(created_unix) is not float or not math.isfinite(created_unix) or created_unix < 0:
        raise ValueError("summary created_unix must be a finite non-negative float")
    if _canonical_json(summary.get("shard_manifest")) != _canonical_json(expected_manifest):
        raise ValueError("summary shard manifest does not bind exact shard paths and bytes")
    normalized = {**recomputed, "shard_manifest": list(expected_manifest)}
    stored_derivation = {key: value for key, value in summary.items() if key != "created_unix"}
    fresh_derivation = {key: value for key, value in normalized.items() if key != "created_unix"}
    if _canonical_json(stored_derivation) != _canonical_json(fresh_derivation):
        raise ValueError("summary derivation does not match exact reconstruction")


def _validate_collection(
    *,
    root: Path,
    protocol: LocalProtocol,
    seeds: tuple[int, ...],
    n_tasks: int,
    paths: Sequence[Path],
    summary_path: Path,
    identity: Mapping[str, Any],
    runner: Mapping[str, Any],
) -> dict[str, Any]:
    import alberta_framework.benchmarks.ipmnist_screening as screening

    expected_pairs = {
        (arm, seed)
        for arm in (protocol.control, protocol.candidate)
        for seed in seeds
    }
    expected_names = {
        f"{arm}_seed{seed}.json" for arm, seed in expected_pairs
    }
    if len(paths) != len(expected_pairs) or {path.name for path in paths} != expected_names:
        raise ValueError("shard filename coverage is not exact")
    shards = [screening.load_shard(path) for path in paths]
    observed_pairs: set[tuple[str, int]] = set()
    first_dataset: Mapping[str, Any] | None = None
    shard_created_times: list[float] = []
    for path, shard in zip(paths, shards, strict=True):
        expected_name = f"{shard['config_name']}_seed{shard['seed']}.json"
        if path.name != expected_name:
            raise ValueError(
                f"shard filename/payload identity mismatch: {path.name} contains {expected_name}"
            )
        pair = (cast(str, shard["config_name"]), cast(int, shard["seed"]))
        if pair in observed_pairs:
            raise ValueError("duplicate shard payload arm/seed identity")
        observed_pairs.add(pair)
        shard_created_times.append(
            _finite_float(shard.get("created_unix"), name="shard created_unix")
        )
        if (
            shard["schema"] != screening.SHARD_SCHEMA
            or shard["evidence_policy"] != EXPECTED_POLICY
            or shard["config"] != _expected_config(n_tasks)
            or shard["noise_mode"] != "step"
            or shard["noise_pool_steps"] is not None
        ):
            raise ValueError(f"shard protocol contract mismatch: {path}")
        if shard["source_provenance"] != identity["source_provenance"]:
            raise ValueError(f"shard source provenance mismatch: {path}")
        if shard["environment"] != runner["screening_environment"]:
            raise ValueError(f"shard runtime environment mismatch: {path}")
        dataset = cast(Mapping[str, Any], shard["dataset_provenance"])
        if first_dataset is None:
            first_dataset = dataset
        elif dataset != first_dataset:
            raise ValueError("shards do not share exact dataset provenance")
    if observed_pairs != expected_pairs:
        raise ValueError("shard payload arm/seed coverage is not exact")
    summary, summary_raw = _strict_json_bytes(summary_path)
    _require_exact_keys(
        summary,
        {
            "schema",
            "evidence_policy",
            "created_unix",
            "protocol_config",
            "environment",
            "noise_mode",
            "noise_pool_steps",
            "control_name",
            "confirmation_threshold",
            "slope_window",
            "n_shards",
            "results",
            "source_provenance",
            "dataset_provenance",
            "shard_manifest",
        },
        context=str(summary_path),
    )
    recomputed = screening.merge_shards(
        paths,
        control_name=protocol.control,
        slope_window=15,
    )
    expected_manifest = _expected_manifest(paths, shards, root=root)
    _validate_summary_reconstruction(
        summary=summary,
        recomputed=recomputed,
        expected_manifest=expected_manifest,
    )
    summary_created_unix = _finite_float(
        summary.get("created_unix"), name="summary created_unix"
    )
    if max(shard_created_times) >= summary_created_unix:
        raise ValueError("stage summary must be created strictly after every input shard")
    if (
        summary["schema"] != screening.SUMMARY_SCHEMA
        or summary["evidence_policy"] != EXPECTED_POLICY
        or summary["protocol_config"] != _expected_config(n_tasks)
        or summary["environment"] != runner["screening_environment"]
        or summary["noise_mode"] != "step"
        or summary["noise_pool_steps"] is not None
        or summary["control_name"] != protocol.control
        or summary["n_shards"] != len(expected_pairs)
        or summary["source_provenance"] != identity["source_provenance"]
        or summary["dataset_provenance"] != first_dataset
    ):
        raise ValueError("summary protocol, source, runtime, or dataset binding mismatch")
    results = recomputed["results"]
    if not isinstance(results, list) or len(results) != 2:
        raise ValueError("summary must contain exactly the control and candidate")
    by_name = {
        cast(str, row["config_name"]): cast(dict[str, Any], row)
        for row in results
        if isinstance(row, dict) and isinstance(row.get("config_name"), str)
    }
    if set(by_name) != {protocol.control, protocol.candidate}:
        raise ValueError("summary arm coverage is not exact")
    for arm, row in by_name.items():
        if row.get("seeds") != list(seeds) or row.get("n_seeds") != len(seeds):
            raise ValueError(f"summary seed coverage differs for {arm}")
    paired = by_name[protocol.candidate].get("paired_vs_control")
    if not isinstance(paired, dict):
        raise ValueError("candidate summary is missing its paired comparison")
    if paired.get("control") != protocol.control or paired.get("seeds") != list(seeds):
        raise ValueError("paired summary control/seed binding differs")
    raw_differences = paired.get("per_seed_diff")
    if not isinstance(raw_differences, list):
        raise ValueError("paired summary per_seed_diff must be a list")
    differences = _finite_differences(raw_differences, expected=len(seeds))
    mean_diff = _finite_float(paired.get("mean_diff"), name="mean_diff")
    stderr_diff = _finite_float(paired.get("stderr_diff"), name="stderr_diff")
    if stderr_diff < 0:
        raise ValueError("paired summary stderr_diff must be non-negative")
    return {
        "n_shards": len(expected_pairs),
        "n_seeds": len(seeds),
        "seeds": list(seeds),
        "mean_diff": mean_diff,
        "stderr_diff": stderr_diff,
        "per_seed_diff": list(differences),
        "summary": summary_path.relative_to(root).as_posix(),
        "summary_sha256": hashlib.sha256(summary_raw).hexdigest(),
        "dataset_provenance": first_dataset,
        "first_shard_created_unix": min(shard_created_times),
        "last_shard_created_unix": max(shard_created_times),
        "summary_created_unix": summary_created_unix,
    }


def _validate_stage(
    *,
    root: Path,
    namespace: Path,
    protocol: LocalProtocol,
    stage: LocalStage,
    identity: Mapping[str, Any],
    runner: Mapping[str, Any],
) -> dict[str, Any]:
    stage_root = namespace / stage.key
    if not stage_root.is_dir() or stage_root.is_symlink():
        raise ValueError(f"required stage {stage.key} is missing or invalid")
    if {path.name for path in stage_root.iterdir()} != {"shards", "summary.json"}:
        raise ValueError(f"stage {stage.key} contains missing or unexpected entries")
    shards_dir = stage_root / "shards"
    if not shards_dir.is_dir() or shards_dir.is_symlink():
        raise ValueError(f"stage {stage.key} shard directory is missing or invalid")
    paths = sorted(shards_dir.iterdir())
    return _validate_collection(
        root=root,
        protocol=protocol,
        seeds=stage.seeds,
        n_tasks=stage.n_tasks,
        paths=paths,
        summary_path=stage_root / "summary.json",
        identity=identity,
        runner=runner,
    )


def _require_stage_after(
    current: Mapping[str, Any], prior: Mapping[str, Any], *, stage: str
) -> None:
    first_shard = _finite_float(
        current.get("first_shard_created_unix"),
        name=f"{stage} first shard created_unix",
    )
    prior_summary = _finite_float(
        prior.get("summary_created_unix"), name="prior summary created_unix"
    )
    if first_shard <= prior_summary:
        raise ValueError(f"{stage} must be created after the prior gate was derived")


def _require_same_stage_dataset(
    current: Mapping[str, Any], prior: Mapping[str, Any], *, stage: str
) -> None:
    if _canonical_json(current.get("dataset_provenance")) != _canonical_json(
        prior.get("dataset_provenance")
    ):
        raise ValueError(f"{stage} dataset provenance differs from the prior stage")


def _expected_root_entries(*stages: str, include_result_claim: bool) -> set[str]:
    entries = {"runner.v1.json", "launch.v1.json", "cache.v1.json", *stages}
    if include_result_claim:
        entries.add("result-claim.v1.json")
    return entries


def _require_root_entries(
    namespace: Path, expected: set[str], *, context: str
) -> None:
    observed = {path.name for path in namespace.iterdir()}
    if observed != expected:
        raise ValueError(
            f"{context} has missing or unexpected namespace entries; "
            f"missing={sorted(expected - observed)}, unexpected={sorted(observed - expected)}"
        )


def validate_result_bundle(
    *,
    protocol_key: str,
    root: Path,
    repository_identity: Mapping[str, Any] | None = None,
    runner_receipt: Mapping[str, Any] | None = None,
    verify_cache_file: bool = True,
) -> dict[str, Any]:
    protocol = protocol_for(protocol_key)
    if Path(root).is_symlink():
        raise ValueError("repository root must not be a symlink")
    resolved_root = Path(root).resolve(strict=True)
    namespace = resolved_root / OUTPUT_ROOT / protocol.namespace
    for directory, label in (
        (resolved_root, "repository root"),
        (resolved_root / "outputs", "outputs root"),
        (resolved_root / OUTPUT_ROOT, "IPMNIST output root"),
        (namespace, "protocol namespace"),
    ):
        _require_real_directory(directory, label=label)
    _reject_symlinks(namespace)
    stored_runner, runner_raw = _strict_json_bytes(namespace / "runner.v1.json")
    cpu = cast(dict[str, Any], stored_runner.get("cpu"))
    cpuset = cpu.get("requested_cpuset") if isinstance(cpu, dict) else None
    if not isinstance(cpuset, str):
        raise ValueError("stored runner receipt has no canonical requested cpuset")
    stored_runner = _validate_runner_receipt(stored_runner, expected_cpuset=cpuset)
    current_runner = _validate_runner_receipt(
        capture_runner_receipt(cpuset) if runner_receipt is None else runner_receipt,
        expected_cpuset=cpuset,
    )
    if current_runner != stored_runner:
        raise ValueError("current result-validation runner differs from launch runner")
    stored_launch, launch_raw = _strict_json_bytes(namespace / "launch.v1.json")
    current_identity = _validate_repository_identity(
        capture_repository_identity(resolved_root)
        if repository_identity is None
        else repository_identity
    )
    launch = _validate_launch_receipt(
        stored_launch,
        protocol=protocol,
        identity=current_identity,
        runner=stored_runner,
        runner_raw=runner_raw,
    )
    stored_cache, cache_raw = _strict_json_bytes(namespace / "cache.v1.json")
    _validate_cache_receipt(
        stored_cache,
        protocol=protocol,
        launch=launch,
        launch_raw=launch_raw,
        verify_file=verify_cache_file,
    )
    cache_receipt_sha256 = hashlib.sha256(cache_raw).hexdigest()
    receipt_bindings = {
        "runner_receipt_sha256": hashlib.sha256(runner_raw).hexdigest(),
        "launch_receipt_sha256": hashlib.sha256(launch_raw).hexdigest(),
        "cache_receipt_sha256": cache_receipt_sha256,
    }
    include_claim = (namespace / "result-claim.v1.json").is_file()
    stages: list[dict[str, Any]] = []
    first = _validate_stage(
        root=resolved_root,
        namespace=namespace,
        protocol=protocol,
        stage=protocol.stages[0],
        identity=current_identity,
        runner=stored_runner,
    )
    stages.append(first)
    if protocol.key == "issue184":
        _require_root_entries(
            namespace,
            _expected_root_entries("screen_60", include_result_claim=include_claim),
            context="issue184 terminal result",
        )
        outcome = classify_issue184(first["mean_diff"], first["per_seed_diff"])
        result: dict[str, Any] = {
            "schema": "asi.ipmnist_local_prereg.result.v1",
            "protocol_key": protocol.key,
            "source": current_identity["source"],
            "tree": current_identity["tree"],
            "outcome": outcome,
            "n_shards": first["n_shards"],
            "stages": stages,
            "screen": first,
            **receipt_bindings,
        }
        return result
    if l2init_gate_passes(first["mean_diff"], first["per_seed_diff"]):
        second = _validate_stage(
            root=resolved_root,
            namespace=namespace,
            protocol=protocol,
            stage=protocol.stages[1],
            identity=current_identity,
            runner=stored_runner,
        )
        _require_stage_after(second, first, stage="confirm_200_tuning")
        _require_same_stage_dataset(second, first, stage="confirm_200_tuning")
        stages.append(second)
    else:
        _require_root_entries(
            namespace,
            _expected_root_entries("screen_60", include_result_claim=include_claim),
            context="issue14 stage-1 terminal result",
        )
        return {
            "schema": "asi.ipmnist_local_prereg.result.v1",
            "protocol_key": protocol.key,
            "source": current_identity["source"],
            "tree": current_identity["tree"],
            "outcome": "stage1_rejected",
            "n_shards": first["n_shards"],
            "stages": stages,
            **receipt_bindings,
        }
    if l2init_gate_passes(second["mean_diff"], second["per_seed_diff"]):
        evaluation = _validate_stage(
            root=resolved_root,
            namespace=namespace,
            protocol=protocol,
            stage=protocol.stages[2],
            identity=current_identity,
            runner=stored_runner,
        )
        _require_stage_after(
            evaluation, second, stage="confirm_200_evaluation"
        )
        _require_same_stage_dataset(
            evaluation, second, stage="confirm_200_evaluation"
        )
        stages.append(evaluation)
    else:
        _require_root_entries(
            namespace,
            _expected_root_entries(
                "screen_60", "confirm_200_tuning", include_result_claim=include_claim
            ),
            context="issue14 stage-2 terminal result",
        )
        return {
            "schema": "asi.ipmnist_local_prereg.result.v1",
            "protocol_key": protocol.key,
            "source": current_identity["source"],
            "tree": current_identity["tree"],
            "outcome": "stage2_rejected",
            "n_shards": sum(cast(int, stage["n_shards"]) for stage in stages),
            "stages": stages,
            **receipt_bindings,
        }
    combined_root = namespace / "confirm_200_all"
    if not combined_root.is_dir() or combined_root.is_symlink():
        raise ValueError("required stage confirm_200_all is missing or invalid")
    if {path.name for path in combined_root.iterdir()} != {"summary.json"}:
        raise ValueError("confirm_200_all contains missing or unexpected entries")
    tuning_paths = sorted((namespace / "confirm_200_tuning/shards").iterdir())
    evaluation_paths = sorted((namespace / "confirm_200_evaluation/shards").iterdir())
    combined = _validate_collection(
        root=resolved_root,
        protocol=protocol,
        seeds=tuple(range(20, 40)),
        n_tasks=200,
        paths=(*tuning_paths, *evaluation_paths),
        summary_path=combined_root / "summary.json",
        identity=current_identity,
        runner=stored_runner,
    )
    if _finite_float(
        combined.get("summary_created_unix"), name="combined summary created_unix"
    ) <= _finite_float(
        evaluation.get("summary_created_unix"), name="evaluation summary created_unix"
    ):
        raise ValueError("combined summary must be created after the stage-3 summary")
    _require_root_entries(
        namespace,
        _expected_root_entries(
            "screen_60",
            "confirm_200_tuning",
            "confirm_200_evaluation",
            "confirm_200_all",
            include_result_claim=include_claim,
        ),
        context="issue14 full terminal result",
    )
    outcome = (
        "win"
        if l2init_gate_passes(combined["mean_diff"], combined["per_seed_diff"])
        else "no_win"
    )
    return {
        "schema": "asi.ipmnist_local_prereg.result.v1",
        "protocol_key": protocol.key,
        "source": current_identity["source"],
        "tree": current_identity["tree"],
        "outcome": outcome,
        "n_shards": sum(cast(int, stage["n_shards"]) for stage in stages),
        "stages": stages,
        "evaluation": evaluation,
        "combined": combined,
        **receipt_bindings,
    }


def validate_and_publish_result(
    *,
    protocol_key: str,
    root: Path,
    repository_identity: Mapping[str, Any] | None = None,
    runner_receipt: Mapping[str, Any] | None = None,
    verify_cache_file: bool = True,
) -> dict[str, Any]:
    protocol = protocol_for(protocol_key)
    resolved_root = Path(root).resolve(strict=True)
    namespace = resolved_root / OUTPUT_ROOT / protocol.namespace
    claim_path = namespace / "result-claim.v1.json"
    claim = {
        "schema": "asi.ipmnist_local_prereg.result_claim.v1",
        "protocol_key": protocol.key,
        "claimed_at": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
        "rerun_allowed": False,
    }
    _write_json_exclusive(claim_path, claim)
    result = validate_result_bundle(
        protocol_key=protocol_key,
        root=resolved_root,
        repository_identity=repository_identity,
        runner_receipt=runner_receipt,
        verify_cache_file=verify_cache_file,
    )
    published = {**result, "result_claim_sha256": _receipt_sha256(claim)}
    _write_json_exclusive(namespace / "result.v1.json", published)
    return published


def _github_token() -> str:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is required for GitHub preflight reads")
    return token


def _validated_repository(value: object) -> str:
    if value != AUTHORIZED_REPOSITORY:
        raise RuntimeError(f"repository must be exactly {AUTHORIZED_REPOSITORY}")
    return AUTHORIZED_REPOSITORY


def _require_prelaunch_absence(
    *,
    root: Path,
    protocol: LocalProtocol,
    data_home: Path,
    runner: Mapping[str, Any],
) -> None:
    output_parent = root / OUTPUT_ROOT
    if (
        not output_parent.is_dir()
        or output_parent.is_symlink()
        or output_parent.resolve(strict=True) != output_parent
    ):
        raise RuntimeError("local preregistration output parent must be a real directory")
    namespace = output_parent / protocol.namespace
    if os.path.lexists(namespace):
        raise FileExistsError(f"append-only namespace is already occupied: {namespace}")
    cache_contract = cast(Mapping[str, Any], runner["cache_contract"])
    data_home_text, _data_home_sha256 = _safe_data_home(data_home)
    cache_path = Path(data_home_text) / cast(str, cache_contract["relative_path"])
    if os.path.lexists(cache_path):
        raise FileExistsError(
            f"canonical cache must be absent before local authorization: {cache_path}"
        )


def _cli_binding(
    args: argparse.Namespace,
) -> tuple[
    LocalProtocol,
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
    str,
]:
    token = _github_token()
    protocol = protocol_for(cast(str, args.protocol))
    root = Path(args.root).resolve(strict=True)
    repository = _validated_repository(args.repository)
    identity = capture_repository_identity(root)
    runner = capture_runner_receipt(cast(str, args.cpuset))
    data_home_text, data_home_sha256 = _safe_data_home(Path(args.data_home))
    _require_prelaunch_absence(
        root=root,
        protocol=protocol,
        data_home=Path(data_home_text),
        runner=runner,
    )
    ref_name = cast(str, args.ref_name)
    encoded_ref = urllib.parse.quote(ref_name, safe="")
    tag = _github_json(
        f"/repos/{repository}/git/ref/tags/{encoded_ref}", token=token
    )
    _validate_remote_tag(tag, ref_name=ref_name, source=cast(str, identity["source"]))
    binding = {
        "source": cast(str, identity["source"]),
        "tree": cast(str, identity["tree"]),
        "uv_lock_sha256": cast(str, identity["uv_lock_sha256"]),
        "benchmark_blob_sha1": cast(str, identity["benchmark_blob_sha1"]),
        "ref_name": ref_name,
        "runner_receipt_sha256": _receipt_sha256(runner),
        "data_home_sha256": data_home_sha256,
    }
    return protocol, identity, runner, binding, token


def _amendment_command(args: argparse.Namespace) -> int:
    protocol, _identity, _runner, binding, token = _cli_binding(args)
    expected = amendment_line(protocol, **binding)
    repository = _validated_repository(args.repository)
    comments = _github_pages(
        f"/repos/{repository}/issues/{protocol.issue}/comments", token=token
    )
    existing = [comment for comment in comments if _owner_matches(comment, body=expected)]
    if existing:
        raise RuntimeError(
            "the exact owner amendment already exists; refusing to invite a duplicate"
        )
    print(expected)
    return 0


def _authorization_command(args: argparse.Namespace) -> int:
    protocol, _identity, _runner, binding, token = _cli_binding(args)
    repository = _validated_repository(args.repository)
    comments = _github_pages(
        f"/repos/{repository}/issues/{protocol.issue}/comments", token=token
    )
    amendment = amendment_line(protocol, **binding)
    matches = [
        cast(dict[str, Any], comment)
        for comment in comments
        if _owner_matches(comment, body=amendment)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one standalone owner amendment comment; "
            f"found {len(matches)}"
        )
    amendment_id, _created_at, amendment_time = _comment_receipt(
        matches[0], issue=protocol.issue, label="amendment"
    )
    if amendment_time >= dt.datetime.now(dt.UTC):
        raise RuntimeError("owner amendment must be durable before authorization generation")
    expected = authorization_line(
        protocol,
        amendment_comment_id=amendment_id,
        amendment_sha256=_canonical_sha256(amendment),
        **binding,
    )
    existing = [comment for comment in comments if _owner_matches(comment, body=expected)]
    if existing:
        raise RuntimeError(
            "the exact owner authorization already exists; refusing to invite a duplicate"
        )
    print(expected)
    return 0


def _launch_command(args: argparse.Namespace) -> int:
    token = _github_token()
    repository = _validated_repository(args.repository)
    root = Path(args.root).resolve(strict=True)
    namespace = claim_local_launch(
        protocol_key=cast(str, args.protocol),
        root=root,
        repository=repository,
        ref_name=cast(str, args.ref_name),
        cpuset=cast(str, args.cpuset),
        data_home=Path(args.data_home),
        token=token,
    )
    print(
        _canonical_json(
            {
                "schema": "asi.ipmnist_local_prereg.launch_cli.v1",
                "namespace": namespace.relative_to(root).as_posix(),
            }
        )
    )
    return 0


def _record_cache_command(args: argparse.Namespace) -> int:
    payload = record_cache_receipt(
        protocol_key=cast(str, args.protocol),
        root=Path(args.root),
        data_home=Path(args.data_home),
    )
    print(_canonical_json(payload))
    return 0


def _validate_command(args: argparse.Namespace) -> int:
    payload = validate_and_publish_result(
        protocol_key=cast(str, args.protocol),
        root=Path(args.root),
    )
    print(_canonical_json(payload))
    return 0


def _add_protocol_root_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protocol", choices=sorted(LOCAL_PROTOCOLS), required=True)
    parser.add_argument("--root", type=Path, required=True)


def _add_binding_arguments(parser: argparse.ArgumentParser) -> None:
    _add_protocol_root_arguments(parser)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--ref-name", required=True)
    parser.add_argument("--cpuset", required=True)
    parser.add_argument("--data-home", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    amendment = subparsers.add_parser(
        "amendment", help="print the one exact owner amendment after read-only preflight"
    )
    _add_binding_arguments(amendment)
    amendment.set_defaults(handler=_amendment_command)

    authorization = subparsers.add_parser(
        "authorization",
        help="verify the owner amendment and print the one exact authorization",
    )
    _add_binding_arguments(authorization)
    authorization.set_defaults(handler=_authorization_command)

    launch = subparsers.add_parser(
        "launch", help="verify GitHub records and atomically claim the fresh namespace"
    )
    _add_binding_arguments(launch)
    launch.set_defaults(handler=_launch_command)

    cache = subparsers.add_parser(
        "record-cache", help="hash the authorized canonical cache into an exclusive receipt"
    )
    _add_protocol_root_arguments(cache)
    cache.add_argument("--data-home", type=Path, required=True)
    cache.set_defaults(handler=_record_cache_command)

    validate = subparsers.add_parser(
        "validate",
        help="consume the sole validation attempt and publish a strict terminal result",
    )
    _add_protocol_root_arguments(validate)
    validate.set_defaults(handler=_validate_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = cast(Any, args.handler)
    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
