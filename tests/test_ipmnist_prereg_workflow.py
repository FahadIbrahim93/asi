from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any, cast

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_DRIVER = runpy.run_path(_ROOT / ".github" / "scripts" / "ipmnist_prereg.py")
_PROTOCOLS = cast(dict[str, Any], _DRIVER["PROTOCOLS"])
_authorization_line = cast(Any, _DRIVER["authorization_line"])
_classify_outcome = cast(Any, _DRIVER["classify_outcome"])


def test_prereg_protocols_pin_exact_arms_and_seeds() -> None:
    issue51 = _PROTOCOLS["issue51"]
    assert issue51.issue == 51
    assert issue51.namespace == "replication_r1"
    assert issue51.control == "sigma0_shiftnorm_d099"
    assert issue51.candidate == "rls_head_resid_l1_preset005"
    assert issue51.seeds == (0, 1, 2)

    issue188 = _PROTOCOLS["issue188"]
    assert issue188.issue == 188
    assert issue188.namespace == "gate_ablation_r3"
    assert issue188.control == "rls_head_resid_l1_preset005"
    assert issue188.candidate == "rls_head_resid_l1_preset005_nogate"
    assert issue188.seeds == tuple(range(3, 13))


def test_authorization_line_binds_every_launch_identity() -> None:
    line = _authorization_line(
        _PROTOCOLS["issue51"],
        source="1" * 40,
        tree="2" * 40,
        uv_lock_sha256="3" * 64,
        workflow_blob_sha1="4" * 40,
        driver_blob_sha1="5" * 40,
        ref_name="ipmnist-prereg-example",
    )
    assert line == (
        "ASI_PREREG_LAUNCH_V1 issue=51 protocol=issue51 "
        f"source={'1' * 40} tree={'2' * 40} uv_lock_sha256={'3' * 64} "
        f"workflow_blob_sha1={'4' * 40} driver_blob_sha1={'5' * 40} "
        "ref=ipmnist-prereg-example runner=macos-14-arm64 seeds=0,1,2 "
        "protocol_approval=approved seed_budget=approved compute=authorized-uncompensated"
    )


@pytest.mark.parametrize(
    ("mean_diff", "diffs", "expected"),
    [
        (0.004882, (0.004, 0.005, 0.006), "replicated"),
        (0.005950, (0.004, 0.005, 0.006), "replicated"),
        (0.006, (0.004, 0.005, 0.006), "directionally_replicated"),
        (0.005, (0.004, 0.0, 0.006), "not_replicated"),
    ],
)
def test_issue51_outcomes_are_frozen(
    mean_diff: float, diffs: tuple[float, ...], expected: str
) -> None:
    assert (
        _classify_outcome("issue51", mean_diff=mean_diff, stderr_diff=0.0001, per_seed_diff=diffs)
        == expected
    )


@pytest.mark.parametrize(
    ("mean_diff", "stderr_diff", "expected"),
    [
        (-0.001, 0.0001, "not_load_bearing"),
        (-0.002, 0.0001, "load_bearing"),
        (-0.0015, 0.0001, "inconclusive"),
        (-0.0013, 0.0001, "inconclusive"),
    ],
)
def test_issue188_outcomes_are_frozen(mean_diff: float, stderr_diff: float, expected: str) -> None:
    assert (
        _classify_outcome(
            "issue188",
            mean_diff=mean_diff,
            stderr_diff=stderr_diff,
            per_seed_diff=(mean_diff,) * 10,
        )
        == expected
    )


def test_outcome_validation_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        _classify_outcome(
            "issue188",
            mean_diff=float("nan"),
            stderr_diff=0.0,
            per_seed_diff=(0.0,) * 10,
        )
