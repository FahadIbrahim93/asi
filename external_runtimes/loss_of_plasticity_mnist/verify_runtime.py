"""Verify the prospective official MNIST runtime without executing a workload."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import platform
import stat
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NoReturn, cast

SOURCE_COMMIT = "a6b79580d85f3025bdb601566d3627c5f489f13b"
SOURCE_TREE = "3e2c26af8f756c87c891f42bc9699ebc3d9ebe0d"
SOURCE_ROOT = Path(f"/opt/loss-of-plasticity-{SOURCE_COMMIT}")
QUALIFICATION_ROOT = Path("/opt/qualification")
MAX_PLAN_BYTES = 64 * 1024
JsonValue = Any

REQUIRED_SOURCE_SHA256 = {
    "LICENSE": "eed4ca91042d3b727df8667cc8335e4c6b4d3e2c41cd48dd7fc8f1880c3fa313",
    "README.md": "73630621c37d267daa833009cdccbecfe46ab8ae1471c4e754afeb4de592271e",
    "requirements.txt": "b036597f2815b989c88b31059cca1cb90244530861e73e62905e78075455880c",
    "setup.py": "195a1a4ae14d5667222ffde9be3cd25e24f215527154d3aa2b3e19aede09230f",
    "lop/permuted_mnist/README.md": (
        "bc4b12a360d33609f569fc31497a27ef5fdc58a5186d7de21c3d5b8f2e224082"
    ),
    "lop/permuted_mnist/load_mnist.py": (
        "daedaa08b95430b2ef0908864321a64e8187853808ce1c09c1b8f6ba6e526ff3"
    ),
    "lop/permuted_mnist/online_expr.py": (
        "c55f04567e4f94356f9f90e1bbd2ff90e8f61f48b7fcece4fcba8e21de9414fa"
    ),
    "lop/permuted_mnist/cfg/bp/std_net.json": (
        "c8c5064a6a1114ad1e1c8d2879a69a6212904e5b9d8da731a29d5c2ce311e247"
    ),
    "lop/permuted_mnist/cfg/cbp.json": (
        "4715c269de83d86278038f5faba4945805f02cd6b11bcfe3000473de3e53b1cf"
    ),
}

PACKAGE_VERSIONS = {
    "certifi": "2026.7.22",
    "charset-normalizer": "3.5.1",
    "filelock": "3.16.1",
    "fsspec": "2025.3.0",
    "idna": "3.15",
    "jinja2": "3.1.6",
    "markupsafe": "2.1.5",
    "mpmath": "1.3.0",
    "networkx": "3.1",
    "numpy": "1.24.1",
    "pillow": "10.4.0",
    "requests": "2.32.4",
    "scipy": "1.10.1",
    "sympy": "1.13.3",
    "torch": "2.1.0+cpu",
    "torchvision": "0.16.0+cpu",
    "tqdm": "4.66.1",
    "typing-extensions": "4.13.2",
    "urllib3": "2.2.3",
}


def _reject_constant(token: str) -> NoReturn:
    raise ValueError(f"non-finite JSON token {token}")


def _pairs(items: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in items:
        if type(key) is not str or key in result:
            raise ValueError("plan contains duplicate or non-string keys")
        result[key] = value
    return result


def _preflight(value: object) -> None:
    stack: list[tuple[object, int]] = [(value, 0)]
    nodes = 0
    text_bytes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > 10_000 or depth > 32:
            raise ValueError("plan exceeds its node or depth limit")
        actual = type(item)
        if item is None or actual is bool:
            continue
        if actual is int:
            if not -(1 << 63) <= cast(int, item) <= (1 << 63) - 1:
                raise ValueError("plan integer exceeds signed 64-bit bounds")
            continue
        if actual is float:
            if not math.isfinite(cast(float, item)):
                raise ValueError("plan contains a non-finite float")
            continue
        if actual is str:
            text_bytes += len(cast(str, item).encode("utf-8"))
        elif actual is list:
            sequence = cast("list[object]", item)
            if len(sequence) > 2048:
                raise ValueError("plan list exceeds its item limit")
            stack.extend((child, depth + 1) for child in sequence)
        elif actual is dict:
            mapping = cast("dict[object, object]", item)
            keys = tuple(mapping.keys())
            if len(keys) > 2048 or any(type(key) is not str for key in keys):
                raise ValueError("plan object has invalid keys or too many fields")
            for key in cast("tuple[str, ...]", keys):
                text_bytes += len(key.encode("utf-8"))
                stack.append((mapping[key], depth + 1))
        else:
            raise ValueError("plan must use exact JSON containers and scalars")
        if text_bytes > MAX_PLAN_BYTES:
            raise ValueError("plan exceeds its cumulative text limit")


def _exact_keys(
    value: object, expected: Sequence[str], *, name: str
) -> dict[str, JsonValue]:
    if type(value) is not dict:
        raise ValueError(f"{name} fields differ")
    mapping = cast("dict[object, JsonValue]", value)
    keys = tuple(mapping.keys())
    expected_keys = tuple(expected)
    if any(type(key) is not str for key in keys):
        raise ValueError(f"{name} keys must be exact strings")
    if len(keys) != len(expected_keys) or frozenset(keys) != frozenset(expected_keys):
        raise ValueError(f"{name} fields differ")
    return cast("dict[str, JsonValue]", value)


def _load_plan() -> dict[str, JsonValue]:
    raw = (QUALIFICATION_ROOT / "qualification-plan.json").read_bytes()
    if len(raw) > MAX_PLAN_BYTES:
        raise ValueError("qualification plan exceeds its byte limit")
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_reject_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("qualification plan is not bounded valid UTF-8 JSON") from error
    _preflight(value)
    return _exact_keys(
        value,
        ("schema", "qualification_issue", "authority", "qualification_inputs", "runtime",
         "prospective_diagnostic", "claims", "blockers"),
        name="qualification plan",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_object(kind: bytes, payload: bytes) -> bytes:
    return hashlib.sha1(kind + b" " + str(len(payload)).encode("ascii") + b"\0" + payload).digest()


def _source_tree(directory: Path) -> bytes:
    entries: list[tuple[bytes, bytes]] = []
    for path in directory.iterdir():
        name = path.name.encode("utf-8")
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            mode = b"40000"
            identity = _source_tree(path)
            sort_key = name + b"/"
        elif stat.S_ISREG(metadata.st_mode):
            mode = b"100755" if metadata.st_mode & stat.S_IXUSR else b"100644"
            identity = _git_object(b"blob", path.read_bytes())
            sort_key = name
        else:
            raise ValueError("source contains a non-file, non-directory entry")
        entries.append((sort_key, mode + b" " + name + b"\0" + identity))
    return _git_object(b"tree", b"".join(value for _, value in sorted(entries)))


def _validate_plan(plan: dict[str, JsonValue]) -> None:
    if plan["schema"] != "asi.loss_of_plasticity_mnist.prospective_runtime.v1":
        raise ValueError("plan schema differs")
    if plan["qualification_issue"] != 1583:
        raise ValueError("plan issue differs")
    authority = _exact_keys(
        plan["authority"],
        ("paper_revision", "repository", "commit", "git_tree", "source_archive_sha256",
         "license", "license_sha256", "source_revision_postdates_paper",
         "required_file_sha256"),
        name="authority",
    )
    expected_authority = {
        "paper_revision": "arXiv:2306.13812v3",
        "repository": "https://github.com/shibhansh/loss-of-plasticity.git",
        "commit": SOURCE_COMMIT,
        "git_tree": SOURCE_TREE,
        "source_archive_sha256": (
            "fe8e7973eda2201865b4d2f76e4aa6a1d68959da1dbd4e50fc29f4d3dad8e5b7"
        ),
        "license": "MIT",
        "license_sha256": REQUIRED_SOURCE_SHA256["LICENSE"],
        "source_revision_postdates_paper": True,
        "required_file_sha256": REQUIRED_SOURCE_SHA256,
    }
    if authority != expected_authority:
        raise ValueError("source authority differs from the audited official revision")
    inputs = _exact_keys(
        plan["qualification_inputs"],
        ("base_image_digest", "dockerfile_sha256", "requirements_in_sha256",
         "requirements_lock_sha256", "fetch_source_sha256", "verify_runtime_sha256"),
        name="qualification inputs",
    )
    if inputs["base_image_digest"] != (
        "python:3.8.18-slim-bookworm@sha256:"
        "e796941013b10bb53a0924d8705485a1afe654bbbc6fe71d32509101e44b6414"
    ):
        raise ValueError("base image identity differs")
    files = {
        "dockerfile_sha256": "Dockerfile.source",
        "requirements_in_sha256": "requirements.in",
        "requirements_lock_sha256": "requirements.lock",
        "fetch_source_sha256": "fetch_source.py",
        "verify_runtime_sha256": "verify_runtime.py",
    }
    for field, relative in files.items():
        expected = inputs[field]
        if type(expected) is not str or _sha256(QUALIFICATION_ROOT / relative) != expected:
            raise ValueError(f"{relative} differs from the qualification plan")
    runtime = _exact_keys(
        plan["runtime"],
        ("platform", "python", "python_implementation", "pip", "setuptools", "accelerator",
         "torch_cuda", "package_versions", "compatibility_deviations",
         "future_invocation_requirements"),
        name="runtime",
    )
    if runtime["package_versions"] != PACKAGE_VERSIONS:
        raise ValueError("runtime package plan differs from the exact lock")
    if runtime["compatibility_deviations"] != [
        "upstream invokes Python 3.8 but pins unsupported SciPy 1.11.2; qualification "
        "runtime uses final Python-3.8-compatible SciPy 1.10.1",
        "MNIST-only runtime omits upstream ImageNet, plotting, and RL-only dependencies",
    ]:
        raise ValueError("runtime compatibility deviations differ")
    diagnostic = _exact_keys(
        plan["prospective_diagnostic"],
        ("family", "input_permutation_is_cumulative", "labels_permuted", "tasks_per_run",
         "examples_per_task", "data_steps_per_run", "runs", "batch_size", "hidden_layers",
         "hidden_width", "learner_task_information", "learner_boundary_information",
         "mechanism_off_control", "dataset_in_image", "workload_executed"),
        name="prospective diagnostic",
    )
    if diagnostic != {
        "family": "online_permuted_mnist",
        "input_permutation_is_cumulative": True,
        "labels_permuted": False,
        "tasks_per_run": 800,
        "examples_per_task": 60_000,
        "data_steps_per_run": 48_000_000,
        "runs": 10,
        "batch_size": 1,
        "hidden_layers": 3,
        "hidden_width": 2_000,
        "learner_task_information": [],
        "learner_boundary_information": [],
        "mechanism_off_control": (
            "CBP with replacement rate exactly zero; reduction remains to be verified"
        ),
        "dataset_in_image": False,
        "workload_executed": False,
    }:
        raise ValueError("prospective diagnostic differs from the audited paper/code plan")
    claims = _exact_keys(
        plan["claims"],
        ("runtime_build_only", "external_workload_executed", "execution_attested",
         "negative_outcome_retained", "paper_parity_claimed", "performance_metrics_computed",
         "scientific_promotion_allowed", "external_execution_authorized"),
        name="claims",
    )
    if claims != {
        "runtime_build_only": True,
        "external_workload_executed": False,
        "execution_attested": False,
        "negative_outcome_retained": False,
        "paper_parity_claimed": False,
        "performance_metrics_computed": False,
        "scientific_promotion_allowed": False,
        "external_execution_authorized": False,
    }:
        raise ValueError("plan claims exceed a prospective nonparity runtime")
    blockers = plan["blockers"]
    if type(blockers) is not list or len(blockers) != 9 or any(
        type(item) is not str for item in blockers
    ):
        raise ValueError("plan must retain all nine exact blockers")


def _validate_source() -> None:
    if not SOURCE_ROOT.is_dir() or _source_tree(SOURCE_ROOT).hex() != SOURCE_TREE:
        raise ValueError("installed source differs from the exact official Git tree")
    for relative, expected in REQUIRED_SOURCE_SHA256.items():
        if _sha256(SOURCE_ROOT / relative) != expected:
            raise ValueError(f"official source file differs: {relative}")
    if (SOURCE_ROOT / "lop/permuted_mnist/data").exists():
        raise ValueError("prospective runtime must not contain MNIST data")


def _validate_runtime(plan: dict[str, JsonValue]) -> None:
    runtime = cast("dict[str, JsonValue]", plan["runtime"])
    expected_scalars = {
        "platform": "linux-x86_64",
        "python": "3.8.18",
        "python_implementation": "CPython",
        "pip": "23.0.1",
        "setuptools": "57.5.0",
        "accelerator": "cpu",
        "torch_cuda": None,
    }
    if any(runtime[name] != value for name, value in expected_scalars.items()):
        raise ValueError("runtime scalar identity differs")
    if (
        platform.system() != "Linux"
        or platform.machine() != "x86_64"
        or platform.python_version() != "3.8.18"
        or platform.python_implementation() != "CPython"
    ):
        raise ValueError("host runtime differs from the prospective image")
    if importlib.metadata.version("pip") != "23.0.1" or importlib.metadata.version(
        "setuptools"
    ) != "57.5.0":
        raise ValueError("base packaging runtime differs")
    for distribution, expected in PACKAGE_VERSIONS.items():
        if importlib.metadata.version(distribution) != expected:
            raise ValueError(f"runtime package differs: {distribution}")
    for module_name in ("numpy", "scipy", "torch", "torchvision", "lop.permuted_mnist.online_expr"):
        importlib.import_module(module_name)
    torch = importlib.import_module("torch")
    if cast(object, torch.version.cuda) is not None or bool(torch.cuda.is_available()):
        raise ValueError("prospective MNIST runtime must remain CPU-only")


def main() -> int:
    plan = _load_plan()
    _validate_plan(plan)
    _validate_source()
    _validate_runtime(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
