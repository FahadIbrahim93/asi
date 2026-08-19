from __future__ import annotations

import copy
import json
import subprocess
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.benchmarks import telapa_qualification as telapa
from alberta_framework.benchmarks.telapa_qualification import (
    SCHEMA,
    SwitchingPolicyLifeAdapter,
    TeLAPACatalogEntry,
    TeLAPASmokeConfig,
    _bounded_json_bytes,
    _preflight_json_tree,
    rollout_latent_descriptor,
    run_smoke,
    validate_result,
)

pytestmark = pytest.mark.unit


def _result() -> dict[str, Any]:
    return run_smoke(TeLAPASmokeConfig(steps=8, phase_length=2))


def test_catalog_fails_closed_without_immutable_anonymous_revision() -> None:
    catalog = TeLAPACatalogEntry()
    catalog.validate()
    assert catalog.repository_revision is None
    assert catalog.repository_tree_digest is None
    assert catalog.immutable_external_source_established is False
    assert catalog.paper_parity_allowed is False
    with pytest.raises(ValueError, match="fail closed"):
        TeLAPACatalogEntry(immutable_external_source_established=True).validate()


def test_latent_descriptor_is_jittable_and_deterministic() -> None:
    observations = jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    actions = jnp.asarray([1, 0], dtype=jnp.int32)
    rewards = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    eager = rollout_latent_descriptor(observations, actions, rewards)
    compiled = jax.jit(rollout_latent_descriptor)(observations, actions, rewards)
    np.testing.assert_array_equal(eager, compiled)
    np.testing.assert_array_equal(eager, jnp.asarray([0.5, 0.5, 0.5, 0.5]))


def test_current_life_adapter_is_deterministic_for_the_same_key() -> None:
    first = SwitchingPolicyLifeAdapter(phase_length=2, learning_rate=0.125)
    second = SwitchingPolicyLifeAdapter(phase_length=2, learning_rate=0.125)
    first_state, first_policy = first.init(jax.random.key(7))
    second_state, second_policy = second.init(jax.random.key(7))
    first_step = first.step(first_state, first_policy, key=jax.random.key(8))
    second_step = second.step(second_state, second_policy, key=jax.random.key(8))
    np.testing.assert_array_equal(first_step[1], second_step[1])
    np.testing.assert_array_equal(first_step[2], second_step[2])
    assert first_step[3:] == second_step[3:]
    with pytest.raises(ValueError, match="learning_rate"):
        SwitchingPolicyLifeAdapter(phase_length=2, learning_rate=True)
    with pytest.raises(ValueError, match="live policy"):
        first.step(first_state, np.ones((2, 2), dtype=np.float64), key=jax.random.key(8))


def test_end_to_end_matrix_has_exact_mechanism_off_parity() -> None:
    result = _result()
    validate_result(json.loads(json.dumps(result)))
    assert result["schema"] == SCHEMA
    records = result["records"]
    assert isinstance(records, list)
    assert {record["arm"] for record in records} == {
        "diverse_archive",
        "one_model",
        "fixed_snapshot",
        "mechanism_off",
    }
    fixed = next(record for record in records if record["arm"] == "fixed_snapshot")
    off = next(record for record in records if record["arm"] == "mechanism_off")
    for field in (
        "observation_sha256",
        "action_sha256",
        "reward_sha256",
        "initial_policy_sha256",
        "final_policy_sha256",
    ):
        assert fixed[field] == off[field]
    assert off["archive_entry_count"] == 0
    assert off["resource_receipt"]["archive_persistent_bytes"] == 0


def test_matched_axes_and_exact_resource_receipts() -> None:
    result = _result()
    for record in result["records"]:
        receipt = record["resource_receipt"]
        assert receipt["environment_steps"] == 8
        assert receipt["observations_consumed"] == 8
        assert receipt["policy_updates"] == 8
        assert receipt["policy_queries"] == 8
        assert receipt["descriptor_model_queries"] == 4
        assert receipt["task_boundary_disclosures"] == 4
        assert receipt["observation_bytes"] == 64
        assert receipt["action_bytes"] == 32
        assert receipt["reward_bytes"] == 32
        assert receipt["active_policy_persistent_bytes"] == 16
        assert receipt["environment_state_persistent_bytes"] == 8
        assert receipt["timing"] is None
        assert receipt["timing_policy"] == "telemetry_only"


@pytest.mark.parametrize(
    ("path", "replacement", "match"),
    (
        (("scientific_promotion_allowed",), True, "must remain false"),
        (("catalog", "repository_revision"), "main", "immutable identity"),
        (("allowed_information", "future_task_information_visible"), True, "allowed information"),
        (("negative_retention", "required"), False, "negative retention"),
        (("records", 0, "seed"), True, "seed/arm"),
        (("records", 0, "resource_receipt", "environment_steps"), 7, "resource receipt"),
        (("records", 0, "resource_receipt", "environment_steps"), True, "resource receipt"),
        (("records", 0, "resource_receipt", "timing"), 0.1, "timing"),
        (("records", 0, "action_sha256"), "x" * 64, "SHA-256"),
    ),
)
def test_validator_rejects_hostile_or_promoting_payloads(
    path: tuple[object, ...], replacement: object, match: str
) -> None:
    payload = copy.deepcopy(_result())
    target: object = payload
    for component in path[:-1]:
        target = target[component]  # type: ignore[index]
    target[path[-1]] = replacement  # type: ignore[index]
    with pytest.raises(ValueError, match=match):
        validate_result(payload)


def test_validator_rejects_nan_extra_fields_and_unbounded_config() -> None:
    payload = _result()
    payload["extra"] = True
    with pytest.raises(ValueError, match="fields"):
        validate_result(payload)
    payload = _result()
    payload["records"][0]["resource_receipt"]["archive_entry_queries"] = float("nan")
    with pytest.raises(ValueError, match="finite JSON"):
        validate_result(payload)
    with pytest.raises(ValueError, match=r"\[1, 64\]"):
        TeLAPASmokeConfig(steps=65)
    with pytest.raises(ValueError, match="frozen"):
        TeLAPASmokeConfig(seeds=(True,))
    with pytest.raises(ValueError, match="worst-case"):
        TeLAPASmokeConfig(steps=64, phase_length=1, archive_byte_budget=128)


def test_validator_rejects_forged_archive_totals_and_current_identity() -> None:
    forged = copy.deepcopy(_result())
    receipt = forged["records"][0]["resource_receipt"]
    receipt["archive_entry_queries"] = 2**62
    receipt["archive_persistent_bytes"] = 2**62
    forged["records"][0]["archive_entry_count"] = 0
    with pytest.raises(ValueError, match="budget|replay"):
        validate_result(forged)

    forged = copy.deepcopy(_result())
    forged["identity"]["lane_source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="current tree/runtime"):
        validate_result(forged)


def test_json_preflight_rejects_deep_cyclic_and_oversized_trees_before_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deep: list[object] = []
    cursor = deep
    for _ in range(65):
        child: list[object] = []
        cursor.append(child)
        cursor = child
    with pytest.raises(ValueError, match="depth"):
        _preflight_json_tree(deep)

    cyclic: list[object] = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="cycles"):
        _preflight_json_tree(cyclic)

    shared: list[object] = []
    with pytest.raises(ValueError, match="aliases"):
        _preflight_json_tree([shared, shared])

    oversized = [None] * 10_001
    monkeypatch.setattr(telapa.json, "dumps", lambda *_args, **_kwargs: pytest.fail("serialized"))
    with pytest.raises(ValueError, match="item limit"):
        _bounded_json_bytes(oversized)


def test_json_preflight_rejects_nonexact_types_without_conversion_hooks() -> None:
    class HostileList(list[object]):
        def __iter__(self) -> object:
            raise AssertionError("conversion hook ran")

    with pytest.raises(ValueError, match="exact primitive"):
        _preflight_json_tree(HostileList())
    with pytest.raises(ValueError, match="exact primitive"):
        _preflight_json_tree(("tuple",))


def test_serializer_recursion_is_normalized_after_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_recursively(*_args: object, **_kwargs: object) -> str:
        raise RecursionError("hostile serializer recursion")

    monkeypatch.setattr(telapa.json, "dumps", fail_recursively)
    with pytest.raises(ValueError, match="bounded exact tree"):
        _bounded_json_bytes({"legal": [1, 2, 3]})


def test_cli_executes_and_round_trips() -> None:
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "alberta_framework.benchmarks.telapa_qualification",
            "--steps",
            "4",
            "--phase-length",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(process.stdout)
    validate_result(payload)
    assert len(payload["records"]) == 12
