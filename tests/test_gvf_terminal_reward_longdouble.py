"""Regression: GVFSpec.terminal_reward must admit np.longdouble like its sibling
fields gamma/lamda.

np.longdouble is numpy's own scalar type with dtype character 'g', distinct from
'd' even where the two have the same width, so the rejection reproduces on every
platform. Issue #2283.
"""

from __future__ import annotations

import numpy as np
import pytest

from alberta_framework import DemonType, GVFSpec

pytestmark = pytest.mark.unit


def _base() -> dict:
    return dict(
        name="d",
        demon_type=DemonType.PREDICTION,
        gamma=0.0,
        lamda=0.0,
        cumulant_index=0,
    )


def test_terminal_reward_accepts_every_numpy_float_family() -> None:
    for family in (np.float16, np.float32, np.float64, np.longdouble):
        spec = GVFSpec(**_base(), terminal_reward=family(1.5))
        assert spec.terminal_reward == pytest.approx(1.5)


def test_terminal_reward_accepts_longdouble_zero_and_negative() -> None:
    spec = GVFSpec(**_base(), terminal_reward=np.longdouble(0.0))
    assert spec.terminal_reward == 0.0
    spec = GVFSpec(**_base(), terminal_reward=np.longdouble(-2.5))
    assert spec.terminal_reward == pytest.approx(-2.5)


def test_gamma_lamda_still_accept_longdouble() -> None:
    spec = GVFSpec(
        name="d",
        demon_type=DemonType.PREDICTION,
        gamma=np.longdouble(0.5),
        lamda=np.longdouble(0.5),
        cumulant_index=0,
    )
    assert spec.gamma == pytest.approx(0.5)
    assert spec.lamda == pytest.approx(0.5)


def test_terminal_reward_still_rejects_non_finite_and_subclasses() -> None:
    with pytest.raises(ValueError):
        GVFSpec(**_base(), terminal_reward=np.longdouble("inf"))
    with pytest.raises(ValueError):
        GVFSpec(**_base(), terminal_reward=np.longdouble("nan"))

    class _HostileFloat(float):
        pass

    with pytest.raises(ValueError):
        GVFSpec(**_base(), terminal_reward=_HostileFloat(1.5))
