"""Bounded real-engine COOM CO8 qualification smoke.

This script runs only inside the isolated image described by the sibling
Dockerfile. It is deliberately fixed-action and mechanism-off: its output is a
runtime qualification receipt, not a performance result.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import stat
import sys
import time
from pathlib import Path

import numpy as np

SCHEMA = "asi.coom_external_runtime_qualification.development.v1"
SOURCE_COMMIT = "7929801176c6e2e036c7c1c7dd6ce9b84a9d1f3e"
SOURCE_TREE = "6e935b4ad6f3e52280de871e56937071aa5cd13f"
SOURCE_ARCHIVE_SHA256 = "a4736e9916468482d75831d53a12a8601c4da91cd40b9b24d313522034a15661"
SOURCE_LICENSE_SHA256 = "47c8691ec5399bc8c58bcfaf0ba43b4ff48e6917c894c03748e3e0d14345d649"
SOURCE_ASSET_MANIFEST_SHA256 = "deaa00979139cf80055f9d04d65800abc78c4feb11e061274e1a4486f9fa6cab"
PATCH_SHA256 = "25bc846908e573ff1c7d02909a9bb895570e0beafe1a928dbd0d5fd4b63835a7"
PATCHED_REWARD_WRAPPER_SHA256 = (
    "0ab457a6bc95dc2551b2c81608d1619549e56ced47e2c85949c39b87b8b5a8cf"
)
EXPECTED_TRACE_SHA256 = "c74968494ccebaaeac4bc1e0c0f1db7546ac5091b831c05a4c0c727266da696f"
TASK_NAMES = (
    "pitfall-default",
    "arms_dealer-default",
    "hide_and_seek-default",
    "floor_is_lava-default",
    "chainsaw-default",
    "raise_the_roof-default",
    "run_and_gun-default",
    "health_gathering-default",
)
SEED = 1_582_000
STEPS_PER_TASK = 2
_QUALIFICATION_ROOT = Path("/opt/qualification")
_MAX_MANIFEST_BYTES = 8192


def _array_sha256(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _asset_manifest() -> tuple[int, int, str]:
    root = Path("/opt/coom")
    paths = sorted((*root.rglob("*.cfg"), *root.rglob("*.wad")))
    digest = hashlib.sha256()
    total = 0
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        raw = path.read_bytes()
        total += len(raw)
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return len(paths), total, digest.hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_object(kind: bytes, payload: bytes) -> bytes:
    header = kind + b" " + str(len(payload)).encode("ascii") + b"\0"
    return hashlib.sha1(header + payload, usedforsecurity=False).digest()


def _source_tree_sha1(root: Path) -> str:
    """Recompute the upstream Git tree, reversing only the reviewed import patch."""

    patched_path = Path("COOM/wrappers/reward.py")

    def tree(directory: Path) -> bytes:
        entries: list[tuple[bytes, bytes]] = []
        for path in directory.iterdir():
            relative = path.relative_to(root)
            name = path.name.encode("utf-8")
            metadata = path.lstat()
            if stat.S_ISDIR(metadata.st_mode):
                mode = b"40000"
                identity = tree(path)
                sort_key = name + b"/"
            elif stat.S_ISLNK(metadata.st_mode):
                mode = b"120000"
                identity = _git_object(b"blob", os.readlink(path).encode("utf-8"))
                sort_key = name
            elif stat.S_ISREG(metadata.st_mode):
                mode = b"100755" if metadata.st_mode & stat.S_IXUSR else b"100644"
                raw = path.read_bytes()
                if relative == patched_path:
                    current = b"from gymnasium import RewardWrapper"
                    original = b"from gym import RewardWrapper"
                    if raw.count(current) != 1 or original in raw:
                        raise ValueError("reviewed COOM import patch cannot be reversed exactly")
                    raw = raw.replace(current, original)
                identity = _git_object(b"blob", raw)
                sort_key = name
            else:
                raise ValueError("COOM source contains an unsupported filesystem entry")
            entries.append((sort_key, mode + b" " + name + b"\0" + identity))
        payload = b"".join(value for _, value in sorted(entries))
        return _git_object(b"tree", payload)

    return tree(root).hex()


def _exact_keys(value: object, expected: set[str], *, name: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{name} fields differ from the qualification schema")
    return value


def _sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be an exact lowercase SHA-256")
    return value


def _load_qualification_manifest() -> dict[str, object]:
    raw = (_QUALIFICATION_ROOT / "qualification-manifest.json").read_bytes()
    if len(raw) > _MAX_MANIFEST_BYTES:
        raise ValueError("qualification manifest exceeds its byte limit")

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if type(key) is not str or key in result:
                raise ValueError("qualification manifest contains duplicate or non-string keys")
            result[key] = value
        return result

    value = json.loads(
        raw,
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token {token}")
        ),
    )
    manifest = _exact_keys(
        value,
        {
            "schema",
            "base_image_digest",
            "dockerfile_sha256",
            "requirements_lock_sha256",
            "smoke_sha256",
            "patch_sha256",
        },
        name="qualification manifest",
    )
    if manifest["schema"] != "asi.coom_external_runtime.inputs.v1":
        raise ValueError("qualification manifest schema differs")
    if manifest["base_image_digest"] != (
        "python:3.12.12-slim-bookworm@sha256:"
        "593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c"
    ):
        raise ValueError("base image digest differs from the qualification contract")
    files = {
        "dockerfile_sha256": "Dockerfile.source",
        "requirements_lock_sha256": "requirements.lock",
        "smoke_sha256": "smoke.py",
        "patch_sha256": "coom-gymnasium.patch",
    }
    for field, relative in files.items():
        expected = _sha256(manifest[field], name=field)
        if _file_sha256(_QUALIFICATION_ROOT / relative) != expected:
            raise ValueError(f"{relative} differs from the qualification manifest")
    return manifest


def _trace() -> tuple[list[dict[str, object]], int]:
    from COOM.env.builder import build_multi_discrete_actions, make_sequence
    from COOM.utils.config import Sequence

    start = time.perf_counter_ns()
    records: list[dict[str, object]] = []
    environments = make_sequence(
        Sequence.CO8,
        doom_kwargs={
            "seed": SEED,
            "render": False,
            "test_only": False,
            "resolution": "160X120",
            "frame_skip": 4,
            "action_space_fn": build_multi_discrete_actions,
            "num_tasks": 8,
        },
        wrapper_config={
            "augment": False,
            "resize": True,
            "frame_height": 84,
            "frame_width": 84,
            "rescale": True,
            "normalize_observation": False,
            "frame_stack": False,
            "lstm": False,
            "record": False,
            "sparse_rewards": False,
        },
    )
    try:
        for task_index, environment in enumerate(environments):
            observation, reset_info = environment.reset()
            if type(reset_info) is not dict or reset_info:
                raise ValueError("COOM reset info must remain the exact empty safe subset")
            reset = np.asarray(observation)
            steps: list[dict[str, object]] = []
            for _ in range(STEPS_PER_TASK):
                observation, reward, terminated, truncated, info = environment.step(0)
                if type(info) is not dict or info:
                    raise ValueError("COOM step info must remain the exact empty safe subset")
                value = np.asarray(observation)
                steps.append(
                    {
                        "action": 0,
                        "observation_sha256": _array_sha256(value),
                        "observation_shape": list(value.shape),
                        "observation_dtype": value.dtype.str,
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "info": {},
                    }
                )
            records.append(
                {
                    "task_index": task_index,
                    "name": environment.unwrapped.name,
                    "reset_info": {},
                    "reset_observation_sha256": _array_sha256(reset),
                    "reset_observation_shape": list(reset.shape),
                    "reset_observation_dtype": reset.dtype.str,
                    "steps": steps,
                }
            )
    finally:
        for environment in environments:
            environment.close()
    return records, time.perf_counter_ns() - start


def _validate_receipt(receipt: object) -> None:
    root = _exact_keys(
        receipt,
        {
            "schema",
            "qualification_inputs",
            "source",
            "runtime",
            "trace",
            "trace_sha256",
            "resource_receipt",
            "claims",
        },
        name="receipt",
    )
    if root["schema"] != SCHEMA:
        raise ValueError("receipt schema differs")
    manifest = _load_qualification_manifest()
    if root["qualification_inputs"] != manifest:
        raise ValueError("receipt qualification inputs differ from verified local bytes")
    source = _exact_keys(
        root["source"],
        {
            "repository",
            "commit",
            "git_tree",
            "archive_sha256",
            "license",
            "license_sha256",
            "asset_count",
            "asset_bytes",
            "asset_manifest_sha256",
            "qualification_patch_sha256",
            "qualification_patch_scope",
            "patched_reward_wrapper_sha256",
        },
        name="source",
    )
    expected_source = {
        "repository": "https://github.com/TTomilin/COOM.git",
        "commit": SOURCE_COMMIT,
        "git_tree": SOURCE_TREE,
        "archive_sha256": SOURCE_ARCHIVE_SHA256,
        "license": "MIT",
        "license_sha256": SOURCE_LICENSE_SHA256,
        "asset_count": 33,
        "asset_bytes": 4_153_440,
        "asset_manifest_sha256": SOURCE_ASSET_MANIFEST_SHA256,
        "qualification_patch_sha256": PATCH_SHA256,
        "qualification_patch_scope": "gym RewardWrapper import only",
        "patched_reward_wrapper_sha256": PATCHED_REWARD_WRAPPER_SHA256,
    }
    if source != expected_source:
        raise ValueError("source identity differs from the exact qualification inputs")
    runtime = _exact_keys(
        root["runtime"],
        {
            "python",
            "python_implementation",
            "platform",
            "numpy",
            "scipy",
            "gymnasium",
            "vizdoom",
            "opencv_python_headless",
        },
        name="runtime",
    )
    expected_versions = {
        "python": "3.12.12",
        "python_implementation": "CPython",
        "numpy": "1.26.4",
        "scipy": "1.11.4",
        "gymnasium": "0.28.1",
        "vizdoom": "1.3.0",
        "opencv_python_headless": "4.11.0.86",
    }
    if any(runtime.get(name) != value for name, value in expected_versions.items()):
        raise ValueError("runtime versions differ from the hash-locked qualification")
    if type(runtime["platform"]) is not str or not 1 <= len(runtime["platform"]) <= 256:
        raise ValueError("platform telemetry must be a bounded exact string")
    trace = _exact_keys(
        root["trace"],
        {"seed", "sequence", "steps_per_task", "fixed_action", "frame_skip", "resize", "records"},
        name="trace",
    )
    if {key: trace[key] for key in trace if key != "records"} != {
        "seed": SEED,
        "sequence": "CO8",
        "steps_per_task": 2,
        "fixed_action": 0,
        "frame_skip": 4,
        "resize": [84, 84],
    }:
        raise ValueError("trace protocol differs from the frozen qualification smoke")
    records = trace["records"]
    if type(records) is not list or len(records) != 8:
        raise ValueError("trace must contain exactly eight ordered task records")
    for task_index, (record_value, task_name) in enumerate(zip(records, TASK_NAMES, strict=True)):
        record = _exact_keys(
            record_value,
            {
                "task_index",
                "name",
                "reset_info",
                "reset_observation_sha256",
                "reset_observation_shape",
                "reset_observation_dtype",
                "steps",
            },
            name=f"task {task_index}",
        )
        if (
            record["task_index"] != task_index
            or type(record["task_index"]) is not int
            or record["name"] != task_name
            or record["reset_info"] != {}
            or record["reset_observation_shape"] != [84, 84, 3]
            or record["reset_observation_dtype"] != "<f8"
        ):
            raise ValueError("task record identity or reset payload differs")
        _sha256(record["reset_observation_sha256"], name="reset observation hash")
        steps = record["steps"]
        if type(steps) is not list or len(steps) != 2:
            raise ValueError("each task must contain exactly two step records")
        for step_value in steps:
            step = _exact_keys(
                step_value,
                {
                    "action",
                    "info",
                    "observation_dtype",
                    "observation_sha256",
                    "observation_shape",
                    "reward",
                    "terminated",
                    "truncated",
                },
                name="step",
            )
            if (
                type(step["action"]) is not int
                or step["action"] != 0
                or step["info"] != {}
                or step["observation_dtype"] != "<f8"
                or step["observation_shape"] != [84, 84, 3]
                or type(step["reward"]) not in (int, float)
                or not math.isfinite(float(step["reward"]))
                or type(step["terminated"]) is not bool
                or type(step["truncated"]) is not bool
            ):
                raise ValueError("step payload differs from the exact bounded contract")
            _sha256(step["observation_sha256"], name="step observation hash")
    trace_bytes = json.dumps(trace, sort_keys=True, separators=(",", ":")).encode("utf-8")
    observed_trace_sha256 = hashlib.sha256(trace_bytes).hexdigest()
    if (
        root["trace_sha256"] != observed_trace_sha256
        or observed_trace_sha256 != EXPECTED_TRACE_SHA256
    ):
        raise ValueError("trace differs from the independently repeated deterministic golden")
    resources = _exact_keys(
        root["resource_receipt"],
        {
            "task_resets",
            "environment_steps",
            "environment_step_queries",
            "policy_queries",
            "learner_updates",
            "model_queries",
            "elapsed_ns_telemetry_only",
        },
        name="resource receipt",
    )
    expected_resources = {
        "task_resets": 8,
        "environment_steps": 16,
        "environment_step_queries": 16,
        "policy_queries": 0,
        "learner_updates": 0,
        "model_queries": 0,
    }
    if any(
        type(resources.get(name)) is not int or resources[name] != value
        for name, value in expected_resources.items()
    ):
        raise ValueError("resource receipt differs from exact fixed-action work")
    elapsed = resources["elapsed_ns_telemetry_only"]
    if type(elapsed) is not int or elapsed < 0:
        raise ValueError("timing telemetry must be a nonnegative exact integer")
    claims = _exact_keys(
        root["claims"],
        {
            "external_runtime_executed",
            "execution_attested",
            "mechanism_off",
            "performance_metrics_computed",
            "paper_parity_claimed",
            "scientific_promotion_allowed",
            "negative_outcome_retained",
        },
        name="claims",
    )
    if claims != {
        "external_runtime_executed": True,
        "execution_attested": False,
        "mechanism_off": True,
        "performance_metrics_computed": False,
        "paper_parity_claimed": False,
        "scientific_promotion_allowed": False,
        "negative_outcome_retained": False,
    }:
        raise ValueError("receipt claims exceed the bounded unattested qualification")


def main() -> None:
    qualification_inputs = _load_qualification_manifest()
    root = Path("/opt/coom")
    if _source_tree_sha1(root) != SOURCE_TREE:
        raise SystemExit("COOM source archive does not reconstruct the pinned Git tree")
    if _file_sha256(root / "LICENSE.txt") != SOURCE_LICENSE_SHA256:
        raise SystemExit("COOM license bytes differ from the audited source pin")
    if (
        _file_sha256(root / "COOM/wrappers/reward.py")
        != PATCHED_REWARD_WRAPPER_SHA256
    ):
        raise SystemExit("COOM qualification patch result differs from the reviewed bytes")
    asset_count, asset_bytes, asset_sha256 = _asset_manifest()
    if (asset_count, asset_bytes, asset_sha256) != (
        33,
        4_153_440,
        SOURCE_ASSET_MANIFEST_SHA256,
    ):
        raise SystemExit("COOM WAD/config asset manifest differs from the audited source pin")
    records, elapsed_ns = _trace()
    trace = {
        "seed": SEED,
        "sequence": "CO8",
        "steps_per_task": STEPS_PER_TASK,
        "fixed_action": 0,
        "frame_skip": 4,
        "resize": [84, 84],
        "records": records,
    }
    trace_bytes = json.dumps(trace, sort_keys=True, separators=(",", ":")).encode("utf-8")
    receipt = {
        "schema": SCHEMA,
        "qualification_inputs": qualification_inputs,
        "source": {
            "repository": "https://github.com/TTomilin/COOM.git",
            "commit": SOURCE_COMMIT,
            "git_tree": SOURCE_TREE,
            "archive_sha256": SOURCE_ARCHIVE_SHA256,
            "license": "MIT",
            "license_sha256": SOURCE_LICENSE_SHA256,
            "asset_count": asset_count,
            "asset_bytes": asset_bytes,
            "asset_manifest_sha256": asset_sha256,
            "qualification_patch_sha256": PATCH_SHA256,
            "patched_reward_wrapper_sha256": PATCHED_REWARD_WRAPPER_SHA256,
            "qualification_patch_scope": "gym RewardWrapper import only",
        },
        "runtime": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "numpy": importlib.metadata.version("numpy"),
            "scipy": importlib.metadata.version("scipy"),
            "gymnasium": importlib.metadata.version("gymnasium"),
            "vizdoom": importlib.metadata.version("vizdoom"),
            "opencv_python_headless": importlib.metadata.version("opencv-python-headless"),
        },
        "trace": trace,
        "trace_sha256": hashlib.sha256(trace_bytes).hexdigest(),
        "resource_receipt": {
            "task_resets": 8,
            "environment_steps": 8 * STEPS_PER_TASK,
            "environment_step_queries": 8 * STEPS_PER_TASK,
            "policy_queries": 0,
            "learner_updates": 0,
            "model_queries": 0,
            "elapsed_ns_telemetry_only": elapsed_ns,
        },
        "claims": {
            "external_runtime_executed": True,
            "execution_attested": False,
            "mechanism_off": True,
            "performance_metrics_computed": False,
            "paper_parity_claimed": False,
            "scientific_promotion_allowed": False,
            "negative_outcome_retained": False,
        },
    }
    _validate_receipt(receipt)
    json.dump(receipt, sys.stdout, sort_keys=True, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
