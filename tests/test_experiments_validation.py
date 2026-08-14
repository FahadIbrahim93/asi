"""Contract tests for validating public multi-seed experiment inputs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Never

import pytest

from alberta_framework.core.learners import LinearLearner
from alberta_framework.streams.base import ScanStream
from alberta_framework.streams.synthetic import RandomWalkStream
from alberta_framework.utils.experiments import ExperimentConfig, run_multi_seed_experiment

pytestmark = pytest.mark.unit


def _fail_if_called() -> Never:
    raise AssertionError("experiment factory must not be called")


def _stream_factory() -> ScanStream[Any]:
    return RandomWalkStream(feature_dim=2)


def _config(
    name: str,
    *,
    learner_factory: Callable[[], LinearLearner] = LinearLearner,
    stream_factory: Callable[[], ScanStream[Any]] = _stream_factory,
    num_steps: int = 2,
) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        learner_factory=learner_factory,
        stream_factory=stream_factory,
        num_steps=num_steps,
    )


def test_duplicate_names_reject_before_distinct_factories_execute() -> None:
    configs = [
        _config(
            "baseline",
            learner_factory=_fail_if_called,
            stream_factory=_fail_if_called,
            num_steps=1,
        ),
        _config(
            "baseline",
            learner_factory=_fail_if_called,
            stream_factory=_fail_if_called,
            num_steps=2,
        ),
    ]

    with pytest.raises(
        ValueError,
        match=r"^Experiment configuration names must be unique; duplicates: 'baseline'$",
    ):
        run_multi_seed_experiment(configs, seeds=[0, 1], parallel=False, show_progress=False)


def test_repeated_config_object_rejects_before_factory_executes() -> None:
    config = _config(
        "baseline",
        learner_factory=_fail_if_called,
        stream_factory=_fail_if_called,
    )

    with pytest.raises(
        ValueError,
        match=r"^Experiment configuration names must be unique; duplicates: 'baseline'$",
    ):
        run_multi_seed_experiment([config, config], seeds=[0], parallel=False, show_progress=False)


def test_multiple_duplicate_names_are_reported_deterministically() -> None:
    configs = [
        _config(
            name,
            learner_factory=_fail_if_called,
            stream_factory=_fail_if_called,
        )
        for name in ("zeta", "alpha", "zeta", "beta", "alpha", "beta", "zeta")
    ]

    with pytest.raises(ValueError) as exc_info:
        run_multi_seed_experiment(configs, seeds=[0], parallel=False, show_progress=False)

    assert str(exc_info.value) == (
        "Experiment configuration names must be unique; duplicates: 'alpha', 'beta', 'zeta'"
    )


def test_duplicate_seeds_reject_before_factories_execute() -> None:
    configs = [
        _config(
            "baseline",
            learner_factory=_fail_if_called,
            stream_factory=_fail_if_called,
        )
    ]

    with pytest.raises(
        ValueError,
        match=r"^Experiment seeds must be unique; duplicates: 0$",
    ):
        run_multi_seed_experiment(configs, seeds=[0, 0, 1], parallel=False, show_progress=False)


def test_multiple_duplicate_seeds_are_reported_deterministically() -> None:
    configs = [
        _config(
            "baseline",
            learner_factory=_fail_if_called,
            stream_factory=_fail_if_called,
        )
    ]

    with pytest.raises(ValueError) as exc_info:
        run_multi_seed_experiment(
            configs, seeds=[7, 3, 7, 0, 3, 7], parallel=False, show_progress=False
        )

    assert str(exc_info.value) == "Experiment seeds must be unique; duplicates: 3, 7"


def test_unique_names_preserve_config_and_seed_order() -> None:
    results = run_multi_seed_experiment(
        [_config("second"), _config("first")],
        seeds=[7, 3],
        parallel=False,
        show_progress=False,
    )

    assert list(results) == ["second", "first"]
    assert results["second"].seeds == [7, 3]
    assert results["first"].seeds == [7, 3]
    assert all(
        summary.n_seeds == 2
        for result in results.values()
        for summary in result.summary.values()
    )


def test_empty_config_sequence_still_returns_empty_results() -> None:
    assert (
        run_multi_seed_experiment([], seeds=[0], parallel=False, show_progress=False) == {}
    )
