"""Regression: C-int scalar spellings (np.intc/np.uintc) must pass the exact-type
integer allowlists in ipmnist_gradual and causal_map_forager.

numpy exposes five signed C integer scalar types (b h i l q) but only four
fixed-width names, so allowlists spelled with fixed-width names only miss exactly
one family per width sign on every platform. Issue #2295.
"""

from __future__ import annotations

import numpy as np
import pytest

from alberta_framework.benchmarks.causal_map_forager import CausalMapForagerConfig
from alberta_framework.benchmarks.ipmnist_gradual import (
    GradualTransitionConfig,
    output_interpolation,
    task_sampling_mask,
    transition_alpha,
)

pytestmark = pytest.mark.unit

C_INT_SIGNED = [np.int8, np.int16, np.int32, np.int64, np.intc]
C_INT_UNSIGNED = [np.uint8, np.uint16, np.uint32, np.uint64, np.uintc]
ALL_C_INT = C_INT_SIGNED + C_INT_UNSIGNED


def test_gradual_transition_config_accepts_every_c_int_family() -> None:
    for family in ALL_C_INT:
        config = GradualTransitionConfig(
            mode="output_interpolation", transition_steps=family(4)
        )
        assert config.transition_steps >= 1


def test_transition_alpha_accepts_every_c_int_family() -> None:
    config = GradualTransitionConfig(mode="output_interpolation", transition_steps=8)
    for family in ALL_C_INT:
        alpha = transition_alpha(family(4), config)
        assert 0.0 <= alpha <= 1.0


def test_output_interpolation_accepts_every_c_int_family() -> None:
    old_label = np.intc(0)
    new_label = np.intc(1)
    for family in ALL_C_INT:
        result = output_interpolation(
            old_label, new_label, 0.5, n_classes=family(4)
        )
        assert result.shape == (4,)


def test_task_sampling_mask_accepts_every_c_int_family() -> None:
    # seed goes through require_jax_seed, which deliberately demands a built-in
    # int (JAX key domain) — that is separate from the #2295 allowlist defect.
    mask = task_sampling_mask(seed=7, transition_id=np.intc(2), count=16, alpha=0.5)
    assert mask.shape == (16,)
    for family in C_INT_UNSIGNED:
        mask = task_sampling_mask(
            seed=7, transition_id=family(2), count=family(16), alpha=0.5
        )
        assert mask.shape == (16,)


def test_causal_config_accepts_every_c_int_family() -> None:
    for family in ALL_C_INT:
        config = CausalMapForagerConfig(initial_retry_delay=family(1))
        assert config.initial_retry_delay >= 1
