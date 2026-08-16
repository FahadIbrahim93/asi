#!/usr/bin/env python3
"""Fail-closed orchestration checks for manual IPMNIST preregistration runs."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, cast

WORKFLOW_PATH: Final = ".github/workflows/ipmnist-prereg.yml"
DRIVER_PATH: Final = ".github/scripts/ipmnist_prereg.py"
OWNER_LOGIN: Final = "lalalune"
EXPECTED_CONFIG: Final = {
    "n_tasks": 60,
    "task_length": 5000,
    "input_dim": 784,
    "hidden1": 300,
    "hidden2": 150,
    "n_classes": 10,
}
EXPECTED_POLICY: Final = {
    "evidence_class": "development_screening_diagnostic",
    "development_only": True,
    "scientific_promotion_allowed": False,
}


@dataclass(frozen=True)
class Protocol:
    key: str
    issue: int
    namespace: str
    control: str
    candidate: str
    seeds: tuple[int, ...]


PROTOCOLS: Final = {
    "issue51": Protocol(
        key="issue51",
        issue=51,
        namespace="replication_r1",
        control="sigma0_shiftnorm_d099",
        candidate="rls_head_resid_l1_preset005",
        seeds=(0, 1, 2),
    ),
    "issue188": Protocol(
        key="issue188",
        issue=188,
        namespace="gate_ablation_r3",
        control="rls_head_resid_l1_preset005",
        candidate="rls_head_resid_l1_preset005_nogate",
        seeds=tuple(range(3, 13)),
    ),
}


def protocol_for(key: str) -> Protocol:
    try:
        return PROTOCOLS[key]
    except KeyError as exc:
        raise ValueError(f"unsupported preregistration protocol: {key!r}") from exc


def _lower_hex(value: str, length: int, *, name: str) -> str:
    if len(value) != length or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be exactly {length} lowercase hexadecimal characters")
    return value


def authorization_line(
    protocol: Protocol,
    *,
    source: str,
    tree: str,
    uv_lock_sha256: str,
    workflow_blob_sha1: str,
    driver_blob_sha1: str,
    ref_name: str,
) -> str:
    source = _lower_hex(source, 40, name="source")
    tree = _lower_hex(tree, 40, name="tree")
    uv_lock_sha256 = _lower_hex(uv_lock_sha256, 64, name="uv_lock_sha256")
    workflow_blob_sha1 = _lower_hex(workflow_blob_sha1, 40, name="workflow_blob_sha1")
    driver_blob_sha1 = _lower_hex(driver_blob_sha1, 40, name="driver_blob_sha1")
    if not ref_name or any(char.isspace() for char in ref_name):
        raise ValueError("ref_name must be a non-empty tag name without whitespace")
    seeds = ",".join(str(seed) for seed in protocol.seeds)
    return (
        f"ASI_PREREG_LAUNCH_V1 issue={protocol.issue} protocol={protocol.key} "
        f"source={source} tree={tree} uv_lock_sha256={uv_lock_sha256} "
        f"workflow_blob_sha1={workflow_blob_sha1} driver_blob_sha1={driver_blob_sha1} "
        f"ref={ref_name} runner=macos-14-arm64 seeds={seeds} "
        "protocol_approval=approved seed_budget=approved "
        "compute=authorized-uncompensated"
    )


def classify_outcome(
    protocol_key: str, *, mean_diff: float, stderr_diff: float, per_seed_diff: tuple[float, ...]
) -> str:
    if not math.isfinite(mean_diff) or not math.isfinite(stderr_diff) or stderr_diff < 0.0:
        raise ValueError("paired summary statistics must be finite and stderr non-negative")
    if not per_seed_diff or not all(math.isfinite(value) for value in per_seed_diff):
        raise ValueError("paired per-seed differences must be non-empty and finite")
    if protocol_key == "issue51":
        if any(value <= 0.0 for value in per_seed_diff):
            return "not_replicated"
        if 0.004882 <= mean_diff <= 0.005950:
            return "replicated"
        return "directionally_replicated"
    if protocol_key == "issue188":
        margin = 0.0015
        if mean_diff - 2.0 * stderr_diff > -margin:
            return "not_load_bearing"
        if mean_diff + 2.0 * stderr_diff < -margin:
            return "load_bearing"
        return "inconclusive"
    raise ValueError(f"unsupported preregistration protocol: {protocol_key!r}")


def _parse_utc(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def _github_json(path: str, *, token: str) -> Any:
    url = f"https://api.github.com{path}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "asi-ipmnist-prereg-v1",
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


def _workflow_runs(repository: str, *, token: str) -> list[dict[str, Any]]:
    encoded = urllib.parse.quote(Path(WORKFLOW_PATH).name, safe="")
    path = f"/repos/{repository}/actions/workflows/{encoded}/runs?event=workflow_dispatch"
    runs: list[dict[str, Any]] = []
    for page in range(1, 101):
        payload = _github_json(f"{path}&per_page=100&page={page}", token=token)
        if isinstance(payload, dict) and isinstance(payload.get("workflow_runs"), list):
            page_runs = [cast(dict[str, Any], value) for value in payload["workflow_runs"]]
            runs.extend(page_runs)
            if len(page_runs) < 100:
                return runs
            continue
        time.sleep(2.0)
    raise RuntimeError("GitHub Actions pagination exceeded 10,000 workflow runs")


def verify_launch_authorization(
    *,
    protocol_key: str,
    repository: str,
    source: str,
    tree: str,
    uv_lock_sha256: str,
    workflow_blob_sha1: str,
    driver_blob_sha1: str,
    ref_name: str,
    run_id: int,
    run_attempt: int,
    token: str,
) -> dict[str, Any]:
    protocol = protocol_for(protocol_key)
    expected_line = authorization_line(
        protocol,
        source=source,
        tree=tree,
        uv_lock_sha256=uv_lock_sha256,
        workflow_blob_sha1=workflow_blob_sha1,
        driver_blob_sha1=driver_blob_sha1,
        ref_name=ref_name,
    )
    if run_attempt != 1:
        raise RuntimeError("rerun attempts are forbidden; dispatch a new reviewed source instead")

    current = _github_json(f"/repos/{repository}/actions/runs/{run_id}", token=token)
    if not isinstance(current, dict):
        raise RuntimeError("current workflow run metadata is unavailable")
    expected_title = f"ipmnist-{protocol.key}-{source}"
    required_current = {
        "id": run_id,
        "event": "workflow_dispatch",
        "head_sha": source,
        "display_title": expected_title,
        "run_attempt": 1,
        "path": WORKFLOW_PATH,
    }
    mismatched = {
        key: (current.get(key), expected)
        for key, expected in required_current.items()
        if current.get(key) != expected
    }
    if mismatched:
        raise RuntimeError(f"current workflow run binding mismatch: {mismatched}")

    matching_runs = [
        run
        for run in _workflow_runs(repository, token=token)
        if run.get("event") == "workflow_dispatch"
        and run.get("head_sha") == source
        and run.get("display_title") == expected_title
    ]
    matching_ids = sorted(int(run["id"]) for run in matching_runs)
    if matching_ids != [run_id]:
        raise RuntimeError(
            "this protocol/source must have exactly one dispatch; "
            f"observed matching run IDs {matching_ids}"
        )

    comments = _github_pages(f"/repos/{repository}/issues/{protocol.issue}/comments", token=token)
    matches = [
        cast(dict[str, Any], comment)
        for comment in comments
        if isinstance(comment, dict)
        and comment.get("body") == expected_line
        and isinstance(comment.get("user"), dict)
        and comment["user"].get("login") == OWNER_LOGIN
        and comment.get("author_association") == "OWNER"
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one standalone owner authorization comment; found {len(matches)}"
        )
    comment = matches[0]
    created_at = cast(str, current.get("created_at"))
    if _parse_utc(cast(str, comment["created_at"])) >= _parse_utc(created_at):
        raise RuntimeError("owner authorization must be durable before workflow dispatch")
    return {
        "schema": "asi.ipmnist_prereg.launch_preflight.v1",
        "protocol": asdict(protocol),
        "source": source,
        "tree": tree,
        "uv_lock_sha256": uv_lock_sha256,
        "workflow_blob_sha1": workflow_blob_sha1,
        "driver_blob_sha1": driver_blob_sha1,
        "ref_name": ref_name,
        "runner": "macos-14-arm64",
        "run_id": run_id,
        "run_attempt": run_attempt,
        "run_url": current.get("html_url"),
        "authorization_comment_id": comment.get("id"),
        "authorization_comment_url": comment.get("html_url"),
        "authorization_created_at": comment.get("created_at"),
        "authorization_line": expected_line,
        "authorization_sha256": hashlib.sha256(expected_line.encode()).hexdigest(),
    }


def _strict_json(path: Path) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{path}: duplicate JSON key {key!r}")
            value[key] = item
        return value

    def reject_constant(value: str) -> Any:
        raise ValueError(f"{path}: non-finite JSON constant {value!r}")

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=reject_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected one JSON object")
    return payload


def _require_exact_keys(value: dict[str, Any], expected: set[str], *, context: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{context}: key mismatch; missing={sorted(expected - set(value))}, "
            f"unexpected={sorted(set(value) - expected)}"
        )


def _validate_runtime(environment: dict[str, Any]) -> None:
    python = cast(dict[str, Any], environment["python"])
    host = cast(dict[str, Any], environment["platform"])
    packages = cast(dict[str, Any], environment["packages"])
    jax = cast(dict[str, Any], environment["jax"])
    process = cast(dict[str, Any], environment["process_environment"])
    if python != {"implementation": "CPython", "version": "3.12.12"}:
        raise ValueError(f"unexpected Python receipt: {python}")
    if host.get("system") != "Darwin" or host.get("machine") != "arm64":
        raise ValueError(f"runner must be Darwin arm64, got {host}")
    expected_packages = {
        "jax": "0.11.0",
        "jaxlib": "0.11.0",
        "numpy": "2.5.1",
        "scikit-learn": "1.9.0",
    }
    if any(packages.get(name) != version for name, version in expected_packages.items()):
        raise ValueError(f"unexpected locked package receipt: {packages}")
    devices = jax.get("devices")
    if jax.get("backend") != "cpu" or not isinstance(devices, list) or len(devices) != 1:
        raise ValueError(f"expected exactly one JAX CPU device, got {jax}")
    device = cast(dict[str, Any], devices[0])
    if device.get("platform") != "cpu":
        raise ValueError(f"expected a CPU JAX device, got {device}")
    expected_process = {
        "CUDA_VISIBLE_DEVICES": "",
        "JAX_DEFAULT_MATMUL_PRECISION": None,
        "JAX_ENABLE_X64": "false",
        "JAX_PLATFORM_NAME": "cpu",
        "JAX_PLATFORMS": "cpu",
        "OMP_NUM_THREADS": "1",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=1",
    }
    if process != expected_process:
        raise ValueError(f"unexpected process-environment receipt: {process}")


def validate_result_bundle(
    *, protocol_key: str, root: Path, source: str, tree: str, uv_lock_sha256: str
) -> dict[str, Any]:
    from alberta_framework.benchmarks.ipmnist_screening import (
        SHARD_SCHEMA,
        SUMMARY_SCHEMA,
        load_shard,
    )

    protocol = protocol_for(protocol_key)
    source = _lower_hex(source, 40, name="source")
    tree = _lower_hex(tree, 40, name="tree")
    uv_lock_sha256 = _lower_hex(uv_lock_sha256, 64, name="uv_lock_sha256")
    namespace = root / "outputs" / "ipmnist_screening" / protocol.namespace
    shards_dir = namespace / "shards"
    expected_pairs = {
        (arm, seed) for arm in (protocol.control, protocol.candidate) for seed in protocol.seeds
    }
    expected_paths = {shards_dir / f"{arm}_seed{seed}.json" for arm, seed in expected_pairs}
    observed_paths = set(shards_dir.glob("*.json"))
    if observed_paths != expected_paths:
        raise ValueError(
            "shard filename coverage mismatch; "
            f"missing={sorted(str(path) for path in expected_paths - observed_paths)}, "
            f"unexpected={sorted(str(path) for path in observed_paths - expected_paths)}"
        )
    shards = [load_shard(path) for path in sorted(expected_paths)]
    observed_pairs = {(shard["config_name"], shard["seed"]) for shard in shards}
    if observed_pairs != expected_pairs or len(shards) != len(expected_pairs):
        raise ValueError("shard payload arm/seed coverage is not exact")
    first = shards[0]
    for shard in shards:
        if shard["schema"] != SHARD_SCHEMA:
            raise ValueError("all shards must use the strict v2 schema")
        if shard["config"] != EXPECTED_CONFIG:
            raise ValueError(f"unexpected protocol config in shard: {shard['config']}")
        if shard["noise_mode"] != "step" or shard["noise_pool_steps"] is not None:
            raise ValueError("all shards must use exact step noise")
        provenance = cast(dict[str, Any], shard["source_provenance"])
        if (
            provenance.get("git_commit") != source
            or provenance.get("git_tree") != tree
            or provenance.get("uv_lock_sha256") != uv_lock_sha256
            or provenance.get("worktree_clean") is not True
        ):
            raise ValueError(f"shard source provenance mismatch: {provenance}")
        if shard["source_provenance"] != first["source_provenance"]:
            raise ValueError("shards do not share exact source provenance")
        if shard["dataset_provenance"] != first["dataset_provenance"]:
            raise ValueError("shards do not share exact dataset provenance")
        if shard["environment"] != first["environment"]:
            raise ValueError("shards do not share exact runtime provenance")
    _validate_runtime(cast(dict[str, Any], first["environment"]))

    summary_path = namespace / (
        "summary.json" if protocol.key == "issue51" else "summary_resid_gate_ablation_r3.json"
    )
    summary = _strict_json(summary_path)
    expected_summary_keys = {
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
    }
    _require_exact_keys(summary, expected_summary_keys, context=str(summary_path))
    if summary["schema"] != SUMMARY_SCHEMA or summary["evidence_policy"] != EXPECTED_POLICY:
        raise ValueError("summary must be a strict-v2 permanently nonpromoting artifact")
    if (
        summary["protocol_config"] != EXPECTED_CONFIG
        or summary["noise_mode"] != "step"
        or summary["noise_pool_steps"] is not None
        or summary["control_name"] != protocol.control
        or summary["n_shards"] != len(expected_pairs)
        or summary["source_provenance"] != first["source_provenance"]
        or summary["dataset_provenance"] != first["dataset_provenance"]
        or summary["environment"] != first["environment"]
    ):
        raise ValueError("summary protocol or provenance binding mismatch")
    results = summary["results"]
    if not isinstance(results, list) or len(results) != 2:
        raise ValueError("summary must contain exactly the control and candidate")
    by_name = {
        result.get("config_name"): result
        for result in results
        if isinstance(result, dict) and isinstance(result.get("config_name"), str)
    }
    if set(by_name) != {protocol.control, protocol.candidate}:
        raise ValueError(f"summary arm set mismatch: {sorted(by_name)}")
    for name, result in by_name.items():
        if result.get("seeds") != list(protocol.seeds) or result.get("n_seeds") != len(
            protocol.seeds
        ):
            raise ValueError(f"summary seed coverage mismatch for {name}")
    candidate = by_name[protocol.candidate]
    paired = candidate.get("paired_vs_control")
    if not isinstance(paired, dict):
        raise ValueError("candidate summary is missing its paired comparison")
    if paired.get("control") != protocol.control or paired.get("seeds") != list(protocol.seeds):
        raise ValueError("paired comparison does not bind the frozen control/seeds")
    raw_diffs = paired.get("per_seed_diff")
    if not isinstance(raw_diffs, list) or len(raw_diffs) != len(protocol.seeds):
        raise ValueError("paired per-seed difference coverage mismatch")
    diffs = tuple(float(value) for value in raw_diffs)
    mean_diff = float(paired["mean_diff"])
    stderr_diff = float(paired["stderr_diff"])
    outcome = classify_outcome(
        protocol.key,
        mean_diff=mean_diff,
        stderr_diff=stderr_diff,
        per_seed_diff=diffs,
    )
    manifest = summary["shard_manifest"]
    if not isinstance(manifest, list) or len(manifest) != len(expected_pairs):
        raise ValueError("summary shard manifest coverage mismatch")
    manifest_pairs = {(entry.get("config_name"), entry.get("seed")) for entry in manifest}
    if manifest_pairs != expected_pairs:
        raise ValueError("summary shard manifest arm/seed identities mismatch")
    return {
        "schema": "asi.ipmnist_prereg.result_validation.v1",
        "protocol": asdict(protocol),
        "source": source,
        "tree": tree,
        "uv_lock_sha256": uv_lock_sha256,
        "summary": summary_path.relative_to(root).as_posix(),
        "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "mean_diff": mean_diff,
        "stderr_diff": stderr_diff,
        "per_seed_diff": list(diffs),
        "outcome": outcome,
        "runtime": first["environment"],
        "dataset_provenance": first["dataset_provenance"],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _preflight_command(args: argparse.Namespace) -> int:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is required")
    payload = verify_launch_authorization(
        protocol_key=args.protocol,
        repository=args.repository,
        source=args.source,
        tree=args.tree,
        uv_lock_sha256=args.uv_lock_sha256,
        workflow_blob_sha1=args.workflow_blob_sha1,
        driver_blob_sha1=args.driver_blob_sha1,
        ref_name=args.ref_name,
        run_id=args.run_id,
        run_attempt=args.run_attempt,
        token=token,
    )
    _write_json(args.output, payload)
    print(json.dumps(payload, allow_nan=False, sort_keys=True))
    return 0


def _validate_command(args: argparse.Namespace) -> int:
    payload = validate_result_bundle(
        protocol_key=args.protocol,
        root=args.root.resolve(strict=True),
        source=args.source,
        tree=args.tree,
        uv_lock_sha256=args.uv_lock_sha256,
    )
    _write_json(args.output, payload)
    print(json.dumps(payload, allow_nan=False, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--protocol", choices=sorted(PROTOCOLS), required=True)
    preflight.add_argument("--repository", required=True)
    preflight.add_argument("--source", required=True)
    preflight.add_argument("--tree", required=True)
    preflight.add_argument("--uv-lock-sha256", required=True)
    preflight.add_argument("--workflow-blob-sha1", required=True)
    preflight.add_argument("--driver-blob-sha1", required=True)
    preflight.add_argument("--ref-name", required=True)
    preflight.add_argument("--run-id", type=int, required=True)
    preflight.add_argument("--run-attempt", type=int, required=True)
    preflight.add_argument("--output", type=Path, required=True)
    preflight.set_defaults(handler=_preflight_command)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--protocol", choices=sorted(PROTOCOLS), required=True)
    validate.add_argument("--root", type=Path, required=True)
    validate.add_argument("--source", required=True)
    validate.add_argument("--tree", required=True)
    validate.add_argument("--uv-lock-sha256", required=True)
    validate.add_argument("--output", type=Path, required=True)
    validate.set_defaults(handler=_validate_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = cast(Any, args.handler)
    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
