"""Bounded real-engine COOM CO8 qualification smoke.

This script runs only inside the isolated image described by the sibling
Dockerfile. It is deliberately fixed-action and mechanism-off: its output is a
runtime qualification receipt, not a performance result.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from COOM.env.builder import build_multi_discrete_actions, make_sequence
from COOM.utils.config import Sequence

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
SEED = 1_582_000
STEPS_PER_TASK = 2


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


def _trace() -> tuple[list[dict[str, object]], int]:
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
            reset = np.asarray(observation)
            steps: list[dict[str, object]] = []
            for _ in range(STEPS_PER_TASK):
                observation, reward, terminated, truncated, info = environment.step(0)
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
                        "info": info,
                    }
                )
            records.append(
                {
                    "task_index": task_index,
                    "name": environment.unwrapped.name,
                    "reset_info": reset_info,
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


def main() -> None:
    root = Path("/opt/coom")
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
    json.dump(receipt, sys.stdout, sort_keys=True, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
