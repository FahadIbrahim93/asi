"""Cardinality bounds for StackedLinearHorde demon counts and sequences.

Peer modules bound demon and demon-list cardinalities (e.g.
``_MAX_HORDE_DEMONS = 4096`` in ``types.py``), but ``StackedHordeConfig``
allowed ``n_demons`` up to ``_INT32_MAX`` and ``_decode_sequence`` accepted
lists/tuples of any size and list subclasses. Issue #2225.
"""

from __future__ import annotations

import pytest

from alberta_framework.core import stacked_horde
from alberta_framework.core.stacked_horde import (
    StackedHordeConfig,
    nexting_spec,
)

pytestmark = pytest.mark.unit


def _config(n_demons: int) -> StackedHordeConfig:
    return StackedHordeConfig(
        n_demons=n_demons,
        feature_dim=2,
        gammas=(0.9,) * n_demons,
        lamdas=(0.5,) * n_demons,
        cumulant_indices=tuple(range(n_demons)),
        step_size=0.1,
    )


def test_max_stacked_horde_demons_is_4096() -> None:
    assert stacked_horde._MAX_STACKED_HORDE_DEMONS == 4096


def test_config_rejects_oversized_n_demons() -> None:
    with pytest.raises(ValueError, match="n_demons"):
        _config(4097)


def test_config_accepts_boundary_n_demons() -> None:
    cfg = _config(4096)
    assert cfg.n_demons == 4096


def test_from_config_rejects_oversized_sequences() -> None:
    n = 4097
    payload = {
        "type": "StackedHordeConfig",
        "n_demons": n,
        "feature_dim": 2,
        "gammas": [0.9] * n,
        "lamdas": [0.5] * n,
        "cumulant_indices": list(range(n)),
        "step_size": 0.1,
    }
    with pytest.raises(ValueError):
        StackedHordeConfig.from_config(payload)


class _HostileList(list[object]):
    calls = 0

    def __len__(self) -> int:  # type: ignore[override]
        type(self).calls += 1
        raise AssertionError("hostile list length")

    def __iter__(self):  # type: ignore[no-untyped-def]
        type(self).calls += 1
        raise AssertionError("hostile list iteration")


def test_from_config_rejects_hostile_list_subclass_before_hooks() -> None:
    _HostileList.calls = 0
    hostile = _HostileList([0.9, 0.9])
    payload = {
        "type": "StackedHordeConfig",
        "n_demons": 2,
        "feature_dim": 2,
        "gammas": hostile,
        "lamdas": [0.5, 0.5],
        "cumulant_indices": [0, 1],
        "step_size": 0.1,
    }
    with pytest.raises(ValueError, match="gammas"):
        StackedHordeConfig.from_config(payload)
    assert _HostileList.calls == 0


def test_nexting_spec_rejects_oversized_product() -> None:
    # 4097 cumulants x 1 gamma = 4097 demons > 4096
    with pytest.raises(ValueError):
        nexting_spec(
            feature_dim=2,
            cumulant_indices=tuple(range(4097)),
            gammas=(0.9,),
        )


def test_nexting_spec_accepts_boundary_product() -> None:
    cfg = nexting_spec(
        feature_dim=2,
        cumulant_indices=tuple(range(1024)),
        gammas=(0.0, 0.5, 0.9, 0.99),
    )
    assert cfg.n_demons == 4096
