"""Hostile integer validation for reference life scorecard."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.unit


class _HostileInt(int):
    calls = 0

    def __float__(self) -> float:
        type(self).calls += 1
        raise AssertionError("hostile float")

    def __int__(self) -> int:
        type(self).calls += 1
        raise AssertionError("hostile int")

    def __lt__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile lt")

    def __eq__(self, other: object) -> bool:
        type(self).calls += 1
        raise AssertionError("hostile eq")


class _HostileMeta(type):
    calls = 0

    def __eq__(cls, other: object) -> bool:
        del other
        cls.calls += 1
        raise AssertionError("hostile metaclass eq")


class _MetaclassHostileInt(int, metaclass=_HostileMeta):
    pass


def test_streaming_observe_rejects_hostile_before_float() -> None:
    from alberta_framework.benchmarks.reference_life_scorecard import StreamingRunSummary

    summary = StreamingRunSummary.for_switching(horizon=10, phase_length=3, post_switch_window=2)
    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="must be a finite number"):
        summary.observe(
            reward=hostile,
            oracle_reward=1.0,
            regime_id=0,
            parameters_changed=False,
            next_state_index=0,
        )
    assert _HostileInt.calls == 0
    _HostileInt.calls = 0
    hostile2 = _HostileInt(2)
    with pytest.raises(ValueError, match="must be a finite number"):
        summary.observe(
            reward=1.0,
            oracle_reward=hostile2,
            regime_id=0,
            parameters_changed=False,
            next_state_index=0,
        )
    assert _HostileInt.calls == 0
    # bool rejected
    with pytest.raises(ValueError, match="must be a finite number"):
        summary.observe(
            reward=True,
            oracle_reward=1.0,
            regime_id=0,
            parameters_changed=False,
            next_state_index=0,
        )
    # valid still works
    summary.observe(
        reward=1,
        oracle_reward=2.0,
        regime_id=0,
        parameters_changed=False,
        next_state_index=0,
    )
    assert summary.accepted_events == 1


def test_streaming_observe_uses_type_identity_without_metaclass_equality() -> None:
    from alberta_framework.benchmarks.reference_life_scorecard import StreamingRunSummary

    summary = StreamingRunSummary.for_switching(horizon=10, phase_length=3, post_switch_window=2)
    _HostileMeta.calls = 0
    with pytest.raises(ValueError, match="reward must be a finite number"):
        summary.observe(
            reward=_MetaclassHostileInt(1),
            oracle_reward=1.0,
            regime_id=0,
            parameters_changed=False,
            next_state_index=0,
        )
    assert _HostileMeta.calls == 0


def test_run_shard_rejects_hostile_regime_without_dispatching_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alberta_framework.benchmarks import reference_life_scorecard as scorecard
    from alberta_framework.reference_life import LifePhase

    plan = scorecard.build_development_plan()
    spec = scorecard.iter_run_specs(plan)[0]
    state = SimpleNamespace(
        phase=LifePhase.QUIESCENT,
        agent_state=None,
        accepted_events=0,
        environment_rng_cursor=0,
    )
    event = SimpleNamespace(
        transaction=SimpleNamespace(
            reward=1.0,
            next_decision_observation=SimpleNamespace(
                to_numpy=lambda: np.asarray([1.0, 0.0])
            ),
        ),
        oracle_reward=1.0,
        regime_id=_HostileInt(0),
        step_result=SimpleNamespace(parameters_changed=False),
    )
    runner = SimpleNamespace(
        agent_adapter=SimpleNamespace(),
        init=lambda: state,
        step=lambda _state: SimpleNamespace(
            accepted=True,
            event=event,
            state=state,
            rejection_reason=None,
        ),
    )
    monkeypatch.setattr(scorecard, "build_scorecard_runner", lambda *_args: runner)
    monkeypatch.setattr(scorecard, "_block_agent_state", lambda _state: None)
    monkeypatch.setattr(scorecard, "_agent_resource_payload", lambda *_args: {})
    monkeypatch.setattr(
        scorecard,
        "_resolved_components",
        lambda *_args: {
            "arm_definition": plan.arm_definition(spec.arm),
            "agent_manifest": None,
            "environment_manifest": None,
            "life_config": None,
            "life_config_sha256": None,
        },
    )

    _HostileInt.calls = 0
    record = scorecard.run_scorecard_shard(plan, spec)

    assert record["status"] == "failed"
    assert record["failure"]["stage"] == "step"
    assert record["failure"]["message"] == "regime_id must be an integer"
    assert _HostileInt.calls == 0


def test_artifact_builder_rejects_hostile_schedule_before_integer_coercion() -> None:
    from alberta_framework.benchmarks import reference_life_scorecard as scorecard

    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="integer schedule_index"):
        scorecard.build_scorecard_artifact(
            scorecard.build_development_plan(),
            [{"schedule_index": _HostileInt(0)}],
        )
    assert _HostileInt.calls == 0


def test_artifact_builder_admits_exact_record_containers_before_hooks() -> None:
    from collections.abc import Iterator, Mapping

    from alberta_framework.benchmarks import reference_life_scorecard as scorecard

    class HostileRecords(list[object]):
        calls = 0

        def __iter__(self) -> Iterator[object]:
            self.calls += 1
            raise AssertionError("must not iterate")

    class HostileRecord(Mapping[str, object]):
        calls = 0

        def __getitem__(self, key: str) -> object:
            self.calls += 1
            raise AssertionError("must not index")

        def __iter__(self) -> Iterator[str]:
            self.calls += 1
            raise AssertionError("must not iterate")

        def __len__(self) -> int:
            self.calls += 1
            raise AssertionError("must not size")

    plan = scorecard.build_development_plan()
    hostile_records = HostileRecords()
    with pytest.raises(ValueError, match="exact list or tuple"):
        scorecard.build_scorecard_artifact(plan, hostile_records)
    assert hostile_records.calls == 0

    hostile_record = HostileRecord()
    with pytest.raises(ValueError, match="exact string-keyed object"):
        scorecard.build_scorecard_artifact(plan, [hostile_record])
    assert hostile_record.calls == 0


def test_reward_sum_rejects_hostile_before_float() -> None:
    from alberta_framework.benchmarks.reference_life_scorecard import _reward_sum

    hostile = _HostileInt(5)
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="must be finite"):
        _reward_sum({"outcome": {"reward_sum": hostile}})
    assert _HostileInt.calls == 0
    with pytest.raises(ValueError, match="must be finite"):
        _reward_sum({"outcome": {"reward_sum": True}})
    assert _HostileInt.calls == 0
    assert _reward_sum({"outcome": {"reward_sum": 5}}) == 5.0
    assert _reward_sum({"outcome": {"reward_sum": 5.0}}) == 5.0


def test_finite_nonnegative_rejects_hostile_before_float() -> None:
    from alberta_framework.benchmarks.reference_life_scorecard import _require_finite_nonnegative

    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="must be a finite nonnegative"):
        _require_finite_nonnegative(hostile, path="p")
    assert _HostileInt.calls == 0
    with pytest.raises(ValueError, match="must be a finite nonnegative"):
        _require_finite_nonnegative(True, path="p")
    assert _HostileInt.calls == 0
    assert _require_finite_nonnegative(1.0, path="p") == 1.0
    assert _require_finite_nonnegative(0, path="p") == 0.0


def test_finite_number_rejects_hostile_before_float() -> None:
    from alberta_framework.benchmarks.reference_life_scorecard import _require_finite_number

    hostile = _HostileInt(1)
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="must be a finite number"):
        _require_finite_number(hostile, path="p")
    assert _HostileInt.calls == 0
    with pytest.raises(ValueError, match="must be a finite number"):
        _require_finite_number(True, path="p")
    assert _HostileInt.calls == 0
    assert _require_finite_number(1, path="p") == 1.0
