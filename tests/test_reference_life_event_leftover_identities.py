"""Leftover-identity gates for reference-life event and regime records."""

from __future__ import annotations

import dataclasses
import json

import pytest

from alberta_framework.reference_agent import (
    DispatchCommand,
    DispatchReceipt,
    StepResult,
    Transaction,
)
from alberta_framework.reference_life import (
    HaltStage,
    LifeHalt,
    PendingOutcome,
    RecoveryMode,
    ReferenceEnvironmentExecution,
    ReferenceEnvironmentStart,
    ReferenceLifeEvent,
    ReferenceLifeRun,
    ReferenceLifeRunner,
    ReferenceLifeState,
)

_DIGEST = "0" * 64


class _ExplodingTypeHookMeta(type):
    equality_calls = 0

    def __hash__(cls) -> int:
        raise AssertionError("hostile runtime-class hash executed")

    def __eq__(cls, other: object) -> bool:
        del other
        _ExplodingTypeHookMeta.equality_calls += 1
        raise AssertionError("hostile runtime-class equality executed")


class _HostileScalar(metaclass=_ExplodingTypeHookMeta):
    pass


class _HostileString(str):
    calls = 0

    def strip(self, chars: str | None = None) -> str:
        del chars
        type(self).calls += 1
        raise AssertionError("hostile string method executed")


def _legal_event(**overrides: object) -> ReferenceLifeEvent:
    command = object.__new__(DispatchCommand)
    receipt = object.__new__(DispatchReceipt)
    transaction = object.__new__(Transaction)
    step_result = object.__new__(StepResult)
    object.__setattr__(receipt, "command", command)
    object.__setattr__(transaction, "receipt", receipt)
    object.__setattr__(step_result, "transaction", transaction)
    payload: dict[str, object] = {
        "command": command,
        "receipt": receipt,
        "transaction": transaction,
        "step_result": step_result,
        "regime_id": 0,
        "oracle_reward": 0.5,
        "transcript_sha256": _DIGEST,
        "recovered": False,
    }
    payload.update(overrides)
    return ReferenceLifeEvent(**payload)  # type: ignore[arg-type]


def test_reference_life_event_rejects_leftover_identities() -> None:
    """Public life events must not keep leftover recovered/regime/reward identities."""

    with pytest.raises(ValueError, match="recovered"):
        _legal_event(recovered=1)
    with pytest.raises(ValueError, match="recovered"):
        _legal_event(recovered=0)
    with pytest.raises(ValueError, match="recovered"):
        _legal_event(recovered="FIXED")
    with pytest.raises(ValueError, match="regime_id"):
        _legal_event(regime_id=True)
    with pytest.raises(ValueError, match="regime_id"):
        _legal_event(regime_id=False)
    with pytest.raises(ValueError, match="oracle_reward"):
        _legal_event(oracle_reward=True)
    with pytest.raises(ValueError, match="oracle_reward"):
        _legal_event(oracle_reward=float("nan"))
    with pytest.raises(ValueError, match="transcript_sha256"):
        _legal_event(transcript_sha256="FIXED")
    with pytest.raises(ValueError, match="transcript_sha256"):
        _legal_event(transcript_sha256=True)

    legal = _legal_event(recovered=True, oracle_reward=0.5)
    dumped = json.dumps(
        {
            "recovered": legal.recovered,
            "regime_id": legal.regime_id,
            "oracle_reward": legal.oracle_reward,
        },
        allow_nan=False,
    )
    assert dumped == '{"recovered": true, "regime_id": 0, "oracle_reward": 0.5}'
    assert '"recovered": 1' not in dumped
    assert legal.recovered is True
    assert legal.regime_id == 0

    with pytest.raises(ValueError, match="command"):
        _legal_event(command=object())
    with pytest.raises(ValueError, match="receipt"):
        _legal_event(receipt=object())
    with pytest.raises(ValueError, match="transaction"):
        _legal_event(transaction=object())
    with pytest.raises(ValueError, match="step_result"):
        _legal_event(step_result=object())


def test_reference_life_hosts_reject_leftover_regime_and_reward_identities() -> None:
    """Sibling start/execution/pending hosts must reject leftover True==1 identities."""

    with pytest.raises(ValueError, match="regime_id"):
        ReferenceEnvironmentStart(state=object(), observation=object(), regime_id=True)
    with pytest.raises(ValueError, match="regime_id"):
        PendingOutcome(transaction=object(), regime_id=True, oracle_reward=0.0)
    with pytest.raises(ValueError, match="oracle reward"):
        PendingOutcome(transaction=object(), regime_id=0, oracle_reward=True)
    with pytest.raises(ValueError, match="reward"):
        ReferenceEnvironmentExecution(
            command=object(),
            state=object(),
            applied_action=object(),
            next_observation=object(),
            reward=True,
            discount=1.0,
            terminated=False,
            truncated=False,
            autoreset=False,
            regime_id=0,
            oracle_reward=0.0,
        )
    with pytest.raises(ValueError, match="terminated"):
        ReferenceEnvironmentExecution(
            command=object(),
            state=object(),
            applied_action=object(),
            next_observation=object(),
            reward=0.0,
            discount=1.0,
            terminated=1,
            truncated=False,
            autoreset=False,
            regime_id=0,
            oracle_reward=0.0,
        )


def test_real_type_gates_do_not_invoke_hostile_runtime_class_hooks() -> None:
    _ExplodingTypeHookMeta.equality_calls = 0
    hostile = _HostileScalar()

    with pytest.raises(ValueError, match="recovered"):
        _legal_event(recovered=hostile)
    with pytest.raises(ValueError, match="regime_id"):
        _legal_event(regime_id=hostile)
    with pytest.raises(ValueError, match="oracle_reward"):
        _legal_event(oracle_reward=hostile)
    assert _ExplodingTypeHookMeta.equality_calls == 0


def test_reference_life_event_requires_one_component_identity_chain() -> None:
    event = _legal_event()
    with pytest.raises(ValueError, match="receipt does not belong"):
        dataclasses.replace(event, command=object.__new__(DispatchCommand))

    replacement_receipt = object.__new__(DispatchReceipt)
    object.__setattr__(replacement_receipt, "command", event.command)
    with pytest.raises(ValueError, match="transaction does not belong"):
        dataclasses.replace(event, receipt=replacement_receipt)

    replacement_transaction = object.__new__(Transaction)
    object.__setattr__(replacement_transaction, "receipt", event.receipt)
    with pytest.raises(ValueError, match="step result does not belong"):
        dataclasses.replace(event, transaction=replacement_transaction)


def test_reference_life_run_rejects_leftover_container_and_transcript_identities() -> None:
    state = object.__new__(ReferenceLifeState)
    object.__setattr__(state, "transcript_sha256", _DIGEST)
    event = _legal_event()

    with pytest.raises(ValueError, match="state"):
        ReferenceLifeRun(state=object(), events=())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exact tuple"):
        ReferenceLifeRun(state=state, events=[event])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="only ReferenceLifeEvent"):
        ReferenceLifeRun(state=state, events=(object(),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="transcript"):
        ReferenceLifeRun(state=state, events=(_legal_event(transcript_sha256="1" * 64),))

    run = ReferenceLifeRun(state=state, events=(event,))
    assert run.events == (event,)


def test_life_veto_reasons_reject_hostile_string_identities_before_dispatch() -> None:
    hostile = _HostileString("operator stop")

    with pytest.raises(ValueError, match="halt reason"):
        LifeHalt(
            stage=HaltStage.PRE_DISPATCH,
            recovery_mode=RecoveryMode.ABORT_ONLY,
            reason=hostile,
        )
    with pytest.raises(ValueError, match="abort reason"):
        ReferenceLifeRunner.abort(  # type: ignore[arg-type]
            object.__new__(ReferenceLifeRunner),
            object(),
            reason=hostile,
        )

    assert _HostileString.calls == 0
