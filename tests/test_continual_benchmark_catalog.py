from __future__ import annotations

import json

import pytest

from alberta_framework.benchmarks.continual_benchmark_catalog import (
    BENCHMARKS,
    CATALOG_SCHEMA,
    benchmark_readiness,
    benchmark_specs,
    catalog_payload,
    main,
)
from alberta_framework.benchmarks.external_qualification import EXTERNAL_QUALIFICATION_PLANS


def test_catalog_ids_and_source_pins_are_well_formed() -> None:
    specs = benchmark_specs()
    assert len(specs) >= 12
    assert len(BENCHMARKS) == len(specs)
    assert tuple(BENCHMARKS) == tuple(spec.benchmark_id for spec in specs)
    for spec in specs:
        assert spec.benchmark_id == spec.benchmark_id.lower()
        assert spec.source_url.startswith("https://")
        if spec.source_commit is not None:
            assert len(spec.source_commit) == 40
            int(spec.source_commit, 16)


def test_catalog_payload_is_explicitly_nonpromoting() -> None:
    payload = catalog_payload()
    assert payload["schema"] == CATALOG_SCHEMA
    assert payload["nonpromoting"] is True
    assert len(payload["benchmarks"]) == len(benchmark_specs())  # type: ignore[arg-type]


def test_integrated_native_reference_life_is_ready() -> None:
    readiness = benchmark_readiness(BENCHMARKS["reference-life"])
    assert readiness.ready is True
    assert readiness.missing_commands == ()
    assert readiness.missing_modules == ()


def test_scaffolded_external_suite_is_not_reported_runnable() -> None:
    readiness = benchmark_readiness(BENCHMARKS["continual-world-cw20"])
    assert readiness.ready is False
    assert readiness.integration == "isolated"


@pytest.mark.parametrize("benchmark_id", ("split-mnist", "rotated-mnist", "split-cifar100"))
def test_supplied_array_adapter_does_not_claim_canonical_suite_readiness(
    benchmark_id: str,
) -> None:
    readiness = benchmark_readiness(BENCHMARKS[benchmark_id])
    assert readiness.ready is False
    assert readiness.status == "scaffolded"
    assert readiness.missing_modules == ()


def test_overlapping_external_pins_match_qualification_authority() -> None:
    plans = {plan.lane_id: plan for plan in EXTERNAL_QUALIFICATION_PLANS}
    mapping = {
        "split-mnist": "native-supervised-suite",
        "rotated-mnist": "native-supervised-suite",
        "split-cifar100": "native-supervised-suite",
        "clear10": "clear",
        "continual-world-cw20": "continual-world-cw20",
        "cora": "cora",
        "coom": "coom-vizdoom",
        "loss-of-plasticity": "loss-of-plasticity",
        "dreamerv3": "dreamer-family",
    }
    for benchmark_id, slug in mapping.items():
        spec = BENCHMARKS[benchmark_id]
        commits = {revision.commit for revision in plans[slug].code_revisions}
        assert spec.source_commit in commits


def test_catalog_cli_lists_selected_benchmark(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list", "reference-life"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [item["benchmark_id"] for item in payload["benchmarks"]] == ["reference-life"]


def test_doctor_exit_is_fail_closed(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["doctor", "continual-world-cw20"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["readiness"][0]["ready"] is False


def test_unknown_benchmark_is_rejected() -> None:
    with pytest.raises(SystemExit, match="2"):
        main(["list", "not-a-benchmark"])
