"""Option-search n_options ceiling before jnp.arange hang.

Public last-fit is 4096 (same as backup_budget). Tests construct 65 options.
Origin accepted INT32-legal n_options whenever backup_budget * n_options
stayed under 262_144 diagnostic slots, then scanned jnp.arange(n_options).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from alberta_framework.core.option_search_control import (
    _MAX_BACKUP_BUDGET,
    OptionSearchControl,
    OptionSearchControlConfig,
)


class _Agent:
    def __init__(self, n_options: object) -> None:
        self.config = SimpleNamespace(n_options=n_options)


def test_documented_protocol_ceiling() -> None:
    assert _MAX_BACKUP_BUDGET == 4096


def test_last_fit_option_count_is_accepted() -> None:
    OptionSearchControl(_Agent(_MAX_BACKUP_BUDGET), OptionSearchControlConfig())  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [4097, 10_001, 2**31 - 1])
def test_rejects_oversized_option_counts_before_slot_product(value: int) -> None:
    with pytest.raises(ValueError, match="n_options must be an integer in"):
        OptionSearchControl(_Agent(value), OptionSearchControlConfig())  # type: ignore[arg-type]
