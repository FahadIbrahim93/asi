from __future__ import annotations

import copy

import numpy as np
import pytest

import alberta_framework.evaluation.bimu_matched_nonpromoting as bimu_plan
from alberta_framework.evaluation.bimu_matched_nonpromoting import (
    FROZEN_BIMU_MATCHED_PLAN,
    INVALID_PRIOR_ATTEMPT,
    _plan_payload,
    build_bimu_execution_manifest,
    validate_bimu_execution_manifest,
)


def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.arange(8 * 4, dtype=np.float32).reshape(8, 4) / 32.0
    y = np.arange(8, dtype=np.int32) % 2
    return x[:4], y[:4], x[4:], y[4:]


def test_frozen_bimu_plan_is_matched_and_prospective() -> None:
    plan = FROZEN_BIMU_MATCHED_PLAN
    assert plan.seeds == (157001, 157002, 157003)
    assert plan.arm_names == ("memory_off", "bimu")
    assert plan.control_config.memory_window is None
    assert plan.candidate_config.memory_window == 128
    control = plan.control_config.to_protocol_payload()
    candidate = plan.candidate_config.to_protocol_payload()
    assert {key for key in control if control[key] != candidate[key]} == {"memory_window"}
    assert plan.dataset_sha256 == "85c681c2f5fc5c274870b30c9accb3d2a6e9eb90a4575a2bf1ccca64f58b6227"
    assert INVALID_PRIOR_ATTEMPT["pull_request"] == 1686
    assert INVALID_PRIOR_ATTEMPT["seed"] == 23
    payload = _plan_payload(plan)
    assert payload["expected_counters_per_arm"]["observations"] == 1280
    assert payload["expected_counters_per_arm"]["model_forward_queries"] == 10240
    assert payload["expected_resources_per_arm"] == {
        "trainable_scalar_count": 25408,
        "parameter_numeric_bytes": 101632,
        "optimizer_state_numeric_bytes": 8,
        "initial_persistent_numeric_bytes": 101640,
        "final_persistent_numeric_bytes": 101640,
        "dataset_numeric_bytes": 1607680,
        "timing_qualified": False,
        "aggregate_working_set_bytes_claimed": False,
        "numeric_resource_ceiling_bytes": 256 * 1024 * 1024,
    }
    assert payload["comparison_scope"]["paper_comparable"] is False
    with pytest.raises(TypeError):
        INVALID_PRIOR_ATTEMPT["seed"] = 157001  # type: ignore[index]


def test_manifest_binds_exact_data_source_runtime_and_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    import alberta_framework.evaluation.bimu_matched_nonpromoting as module

    tiny_plan = module._test_plan(input_dim=4, n_classes=2, examples=4)
    monkeypatch.setattr(module, "FROZEN_BIMU_MATCHED_PLAN", tiny_plan)
    manifest = build_bimu_execution_manifest(*_data())
    validate_bimu_execution_manifest(manifest, *_data())
    assert manifest["policy"]["scientific_promotion_allowed"] is False
    assert manifest["identity"]["consistency_not_attestation"] is True


def test_manifest_rejects_forged_nested_identity() -> None:
    import alberta_framework.evaluation.bimu_matched_nonpromoting as module

    plan = module._test_plan(input_dim=4, n_classes=2, examples=4)
    original = module.FROZEN_BIMU_MATCHED_PLAN
    module.FROZEN_BIMU_MATCHED_PLAN = plan
    try:
        manifest = build_bimu_execution_manifest(*_data())
        forged = copy.deepcopy(manifest)
        forged["plan"]["seeds"][0] = 23
        with pytest.raises(ValueError, match="plan"):
            validate_bimu_execution_manifest(forged, *_data())
    finally:
        module.FROZEN_BIMU_MATCHED_PLAN = original


def test_manifest_rejects_hostile_key_without_hooks() -> None:
    class Hostile(str):
        calls = 0

        def __hash__(self) -> int:
            self.calls += 1
            return super().__hash__()

        def __eq__(self, other: object) -> bool:
            self.calls += 1
            raise AssertionError("must not compare")

    key = Hostile("schema")
    payload = {key: "x"}
    key.calls = 0
    with pytest.raises(ValueError, match="exact JSON"):
        validate_bimu_execution_manifest(payload, *_data())
    assert key.calls == 0


def test_manifest_rejects_hostile_metaclass_without_hooks() -> None:
    class HostileMeta(type):
        calls = 0

        def __eq__(cls, other: object) -> bool:
            cls.calls += 1
            raise AssertionError("must not compare runtime types")

    class Hostile(metaclass=HostileMeta):
        pass

    HostileMeta.calls = 0
    with pytest.raises(ValueError, match="exact JSON"):
        bimu_plan._json_preflight({"value": Hostile()})
    assert HostileMeta.calls == 0
