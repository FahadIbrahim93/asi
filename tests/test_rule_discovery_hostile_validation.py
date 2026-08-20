"""Hostile validation for rule discovery search-resource gates.

Protocol-bound per-name caps plus combined work-unit products must reject
before JAX allocation, stream construction, or range loops. The exact-type
hostile gate stays; 2**31-1 is not a legal search int.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from alberta_framework.benchmarks.micro_continual import MICRO_SUITE, MicroTaskConfig
from alberta_framework.benchmarks.rule_discovery import (
    _SEARCH_INT_MAX_BY_NAME,
    _SEARCH_LOGICAL_STEPS_MAX,
    _SEARCH_STREAM_NAMED_BYTES_MAX,
    _SEARCH_STREAM_STEPS_MAX,
    GENOME_SIZE,
    _require_logical_steps,
    _require_search_int,
    _require_search_work_unit,
    _resolved_suite,
    evaluate_population,
    run_search,
)


class _HostileInt(int):
    calls = 0

    def __repr__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("repr hook")

    def __str__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("str hook")


class _EvilStr(str):
    calls = 0

    def __repr__(self) -> str:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("str hook")


class _HostileArray:
    calls = 0

    @property
    def __class__(self) -> type[object]:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("class hook")

    @property
    def shape(self) -> tuple[int, int]:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("shape hook")

    def __array__(self) -> np.ndarray:  # pragma: no cover
        type(self).calls += 1
        raise AssertionError("array hook")


def test_require_search_int_rejects_hostile_without_hooks() -> None:
    evil = _HostileInt(1)
    _HostileInt.calls = 0
    with pytest.raises(ValueError, match="must be an integer") as exc:
        _require_search_int("my_param", evil, minimum=1)  # type: ignore[arg-type]
    assert _HostileInt.calls == 0
    assert "!r" not in str(exc.value)
    assert "HostileInt" not in str(exc.value)


def test_require_search_int_rejects_string_subclass_name() -> None:
    evil = _EvilStr("my_param")
    _EvilStr.calls = 0
    with pytest.raises(ValueError, match="exact string"):
        _require_search_int(evil, 1, minimum=0)  # type: ignore[arg-type]
    assert _EvilStr.calls == 0


def test_require_search_int_rejects_below_minimum_sanitized() -> None:
    with pytest.raises(ValueError, match="must be an integer") as exc:
        _require_search_int("my_param", 0, minimum=1)
    assert "!r" not in str(exc.value)
    assert "my_param" in str(exc.value)


def test_source_has_no_repr_leak() -> None:
    text = pathlib.Path("alberta_framework/benchmarks/rule_discovery.py").read_text()
    assert "!r" not in text


def test_valid_int_passes() -> None:
    assert _require_search_int("my_param", 5, minimum=1) == 5


def test_require_search_int_rejects_unbounded_and_int32_max() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        _require_search_int("task_length", 10**12, minimum=1)
    with pytest.raises(ValueError, match="must be an integer"):
        _require_search_int("n_random", 2**31 - 1, minimum=0)
    with pytest.raises(ValueError, match="must be an integer"):
        _require_search_int("generations", 2**31 - 1, minimum=0)


def test_require_search_int_last_fit_and_first_overflow() -> None:
    for name, maximum in _SEARCH_INT_MAX_BY_NAME.items():
        assert _require_search_int(name, maximum, minimum=0) == maximum
        with pytest.raises(ValueError, match="must be an integer"):
            _require_search_int(name, maximum + 1, minimum=0)


def test_require_search_work_unit_combined_product() -> None:
    last_fit_rows = _SEARCH_INT_MAX_BY_NAME["n_random"]
    _require_search_work_unit(n_random=last_fit_rows)
    with pytest.raises(ValueError, match="candidate arrays"):
        _require_search_work_unit(n_random=last_fit_rows + 1)
    _require_search_work_unit(
        n_random=0, n_tasks=2, task_length=_SEARCH_STREAM_STEPS_MAX // 2
    )
    with pytest.raises(ValueError, match="stream steps"):
        _require_search_work_unit(n_random=0, n_tasks=2, task_length=5_000_001)


def test_require_search_work_unit_adjacent_population_generations() -> None:
    _require_search_work_unit(n_random=0, population=1_024, generations=15)
    with pytest.raises(ValueError, match="candidate evaluations"):
        _require_search_work_unit(n_random=0, population=1_024, generations=16)


def test_require_search_work_unit_adjacent_children_generations() -> None:
    _require_search_work_unit(n_random=4_096, children=256, generations=48)
    with pytest.raises(ValueError, match="candidate evaluations"):
        _require_search_work_unit(n_random=4_097, children=256, generations=48)


def test_require_logical_steps_adjacent_candidate_stream_product() -> None:
    _require_logical_steps(
        candidate_evals=1,
        stream_steps=_SEARCH_LOGICAL_STEPS_MAX,
        seed_count=1,
    )
    with pytest.raises(ValueError, match="candidate-example steps"):
        _require_logical_steps(
            candidate_evals=1,
            stream_steps=_SEARCH_LOGICAL_STEPS_MAX + 1,
            seed_count=1,
        )


def test_resolved_suite_rejects_single_override_stream_product() -> None:
    last_fit_length = _SEARCH_STREAM_STEPS_MAX // 12
    _resolved_suite(None, last_fit_length)
    with pytest.raises(ValueError, match="stream steps"):
        _resolved_suite(None, last_fit_length + 1)


def test_evaluate_population_rejects_oversize_block_before_materialize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_materialization(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("oversize genome block reached stream materialization")

    monkeypatch.setattr(
        "alberta_framework.benchmarks.rule_discovery._materialize_eval",
        unexpected_materialization,
    )

    with pytest.raises(ValueError, match="must be an integer"):
        evaluate_population(
            np.empty((_SEARCH_INT_MAX_BY_NAME["n_random"] + 1, GENOME_SIZE), np.float32),
            MICRO_SUITE["M1"],
            seeds=(0,),
            batch_size=1,
        )


def test_evaluate_population_rejects_stream_named_bytes_before_materialize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_materialization(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("oversize stream reached materialization")

    monkeypatch.setattr(
        "alberta_framework.benchmarks.rule_discovery._materialize_eval",
        unexpected_materialization,
    )
    steps = _SEARCH_STREAM_NAMED_BYTES_MAX // (4 * 64 + 8) + 1
    config = MicroTaskConfig(
        name="hostile",
        kind="input_permutation",
        role="search",
        input_dim=64,
        n_classes=10,
        n_tasks=1,
        task_length=steps,
        hidden1=32,
        hidden2=16,
        crop=False,
    )
    with pytest.raises(ValueError, match="named arrays"):
        evaluate_population(
            np.zeros((1, GENOME_SIZE), dtype=np.float32),
            config,
            seeds=(0,),
            batch_size=1,
        )


def test_run_search_rejects_invalid_population_relationships() -> None:
    with pytest.raises(ValueError, match="initial candidate pool"):
        run_search(n_random=19, population=20, generations=0, elite=1)
    with pytest.raises(ValueError, match="at least one child"):
        run_search(n_random=19, population=19, generations=1, elite=19)


def test_evaluate_population_rejects_hostile_array_without_dispatch() -> None:
    value = _HostileArray()
    _HostileArray.calls = 0
    with pytest.raises(ValueError, match="exact trusted"):
        evaluate_population(
            value,  # type: ignore[arg-type]
            MICRO_SUITE["M1"],
            seeds=(0,),
            batch_size=1,
        )
    assert _HostileArray.calls == 0


def test_evaluate_population_rejects_cartesian_work_before_materialize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_materialization(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("oversize Cartesian work reached stream materialization")

    monkeypatch.setattr(
        "alberta_framework.benchmarks.rule_discovery._materialize_eval",
        unexpected_materialization,
    )
    genomes = np.zeros((1_008, GENOME_SIZE), dtype=np.float32)
    with pytest.raises(ValueError, match="candidate-example steps"):
        evaluate_population(
            genomes,
            MICRO_SUITE["M1"],
            seeds=tuple(range(1_024)),
            batch_size=1,
        )
