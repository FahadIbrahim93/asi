"""Regression checks for exact-type gates audited after PRs 1383--1397."""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.forager_matched_campaign import (
    ForagerMatchedCampaignError,
)
from alberta_framework.benchmarks.forager_matched_campaign import (
    _plain as campaign_plain,
)
from alberta_framework.benchmarks.forager_matched_qualification import (
    ForagerMatchedQualificationError,
    _plain_json,
)
from alberta_framework.benchmarks.forager_matched_seal import (
    ForagerMatchedSealError,
)
from alberta_framework.benchmarks.forager_matched_seal import (
    _plain as seal_plain,
)
from alberta_framework.benchmarks.forager_rng_parity import (
    ForagerRngParityError,
    _validate_json_value,
)
from alberta_framework.benchmarks.ipmnist_screening import (
    _validated_wall_clock_seconds,
)
from alberta_framework.benchmarks.micro_continual import _require_positive_int32
from alberta_framework.benchmarks.upgd_ipmnist import (
    _require_finite_real as require_ipmnist_real,
)
from alberta_framework.benchmarks.upgd_label_emnist import _validated_hyperparameter
from alberta_framework.core.baseline_optimizers import (
    _require_int32 as require_baseline_int,
)
from alberta_framework.core.intelligence_amplification import (
    _require_int32 as require_ia_int,
)
from alberta_framework.core.learners import _require_int32 as require_learner_int
from alberta_framework.core.off_policy_td import _validated_config_float
from alberta_framework.core.optimizers import _require_int32 as require_optimizer_int
from alberta_framework.core.reward_model import _require_int32 as require_reward_int
from alberta_framework.evaluation.continual_ia import _require_integer
from alberta_framework.evaluation.ftl_decision_fidelity import (
    _require_int32 as require_fidelity_int,
)


class _HostileType(type):
    calls = 0

    def __hash__(cls) -> int:  # pragma: no cover - must not execute
        type(cls).calls += 1
        raise AssertionError("metaclass hash hook must not run")

    def __eq__(cls, other: object) -> bool:  # pragma: no cover - must not execute
        type(cls).calls += 1
        raise AssertionError("metaclass equality hook must not run")


class _HostileValue(metaclass=_HostileType):
    pass


@pytest.fixture(autouse=True)
def _reset_calls() -> None:
    _HostileType.calls = 0


def test_numeric_gates_reject_before_hostile_metaclass_dispatch() -> None:
    hostile = _HostileValue()
    checks = (
        lambda: require_ia_int("value", hostile, minimum=1),
        lambda: require_learner_int("value", hostile, minimum=1),
        lambda: require_baseline_int("value", hostile, minimum=1),
        lambda: require_optimizer_int("value", hostile, minimum=1),
        lambda: require_reward_int("value", hostile, minimum=1),
        lambda: _validated_config_float("value", hostile),
        lambda: _require_positive_int32(hostile, name="value"),
        lambda: require_ipmnist_real("value", hostile),
        lambda: _validated_hyperparameter("step_size", hostile),
        lambda: _validated_wall_clock_seconds(hostile, "record"),
        lambda: _require_integer("value", hostile, minimum=0),
        lambda: require_fidelity_int("value", hostile, minimum=0),
    )
    for check in checks:
        with pytest.raises((TypeError, ValueError)):
            check()
    assert _HostileType.calls == 0


def test_json_gates_reject_before_hostile_metaclass_dispatch() -> None:
    hostile = _HostileValue()
    checks = (
        (lambda: campaign_plain(hostile), ForagerMatchedCampaignError),
        (lambda: _plain_json(hostile), ForagerMatchedQualificationError),
        (lambda: seal_plain(hostile), ForagerMatchedSealError),
        (lambda: _validate_json_value(hostile), ForagerRngParityError),
    )
    for check, error in checks:
        with pytest.raises(error):
            check()
    assert _HostileType.calls == 0
