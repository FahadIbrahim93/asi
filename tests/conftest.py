"""Pytest configuration and fixtures for Alberta Framework tests."""

import functools
import inspect
from typing import Any

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest


def _install_numpy_assert_allclose_strict_compatibility() -> None:
    """Bridge Chex's NumPy-2 assertion call on the supported NumPy 1.26 floor."""

    original = np.testing.assert_allclose
    if "strict" in inspect.signature(original).parameters:
        return

    @functools.wraps(original)
    def compatible_assert_allclose(
        actual: Any,
        desired: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        strict = bool(kwargs.pop("strict", False))
        if strict:
            actual_array = np.asarray(actual)
            desired_array = np.asarray(desired)
            if actual_array.shape != desired_array.shape:
                raise AssertionError(
                    "strict allclose requires matching shapes: "
                    f"{actual_array.shape} != {desired_array.shape}"
                )
            if actual_array.dtype != desired_array.dtype:
                raise AssertionError(
                    "strict allclose requires matching dtypes: "
                    f"{actual_array.dtype} != {desired_array.dtype}"
                )
        original(actual, desired, *args, **kwargs)

    np.testing.assert_allclose = compatible_assert_allclose  # type: ignore[assignment]


_install_numpy_assert_allclose_strict_compatibility()


@pytest.fixture
def rng_key():
    """Provide a deterministic JAX random key."""
    return jr.key(42)


@pytest.fixture
def feature_dim():
    """Default feature dimension for tests."""
    return 10


@pytest.fixture
def sample_observation(feature_dim, rng_key):
    """Generate a sample observation vector."""
    return jr.normal(rng_key, (feature_dim,), dtype=jnp.float32)


@pytest.fixture
def sample_target():
    """Generate a sample target value."""
    return jnp.array([1.5], dtype=jnp.float32)
