"""Protocol ceilings for causal-map JAX chunk scans.

Public Forager last-fit is jax_chunk_size=10_000. Origin accepted INT32-legal
chunks and scanned jnp.arange(chunk) — hang, not leftover INT32 math.
"""

from __future__ import annotations

import pytest

from alberta_framework.benchmarks.forager import ForagerBenchmarkConfig


def test_rejects_oversized_causal_map_chunks() -> None:
    with pytest.raises(ValueError, match="jax_chunk_size"):
        ForagerBenchmarkConfig(steps=10_001, jax_chunk_size=10_001)


def test_causal_map_accepts_the_shared_last_fit_chunk() -> None:
    cfg = ForagerBenchmarkConfig(steps=10_000, jax_chunk_size=10_000)
    assert cfg.jax_chunk_size == 10_000
