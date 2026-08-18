import copy
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.evaluation.optimizer_geometry import (
    FROZEN_GEOMETRY_CONFIG,
    GEOMETRY_PROTOCOL,
    GEOMETRY_RESULT_SCHEMA,
    flad_noise_component,
    flad_noise_component_transaction,
    muon_ogd_dual_update,
    orthogonal_correction,
    run_streaming_matrix_evaluation,
    spectral_matrix_sign,
    spectral_matrix_sign_transaction,
    validate_streaming_matrix_result,
)


def _paper_ns5(matrix: jax.Array, *, steps: int) -> jax.Array:
    value = matrix / jnp.maximum(jnp.linalg.norm(matrix), jnp.asarray(1e-12, matrix.dtype))
    transposed = value.shape[0] > value.shape[1]
    if transposed:
        value = value.T
    for _ in range(steps):
        value = 1.5 * value - 0.5 * (value @ value.T) @ value
    return value.T if transposed else value


def test_orthogonal_correction_removes_protected_direction() -> None:
    update = jnp.array([2.0, 3.0])
    basis = jnp.array([[1.0, 0.0]])
    np.testing.assert_allclose(orthogonal_correction(update, basis), [0.0, 3.0])
    np.testing.assert_array_equal(orthogonal_correction(update, jnp.zeros((0, 2))), update)


@pytest.mark.parametrize("shape", [(2, 2), (2, 3), (3, 2)])
def test_spectral_matrix_sign_matches_pinned_muon_ogd_ns5(shape: tuple[int, int]) -> None:
    matrix = jnp.arange(1, shape[0] * shape[1] + 1, dtype=jnp.float32).reshape(shape)
    np.testing.assert_allclose(
        spectral_matrix_sign(matrix, steps=5), _paper_ns5(matrix, steps=5), rtol=1e-6
    )


def test_spectral_matrix_sign_zero_and_jit_paths() -> None:
    zero = jnp.zeros((3, 2), dtype=jnp.float32)
    np.testing.assert_array_equal(spectral_matrix_sign(zero), zero)
    jitted = jax.jit(spectral_matrix_sign)
    np.testing.assert_allclose(jitted(jnp.eye(2, dtype=jnp.float32)), jnp.eye(2), rtol=1e-5)


def test_muon_ogd_empty_constraints_reduces_exactly_to_ns5() -> None:
    matrix = jnp.array([[2.0, 1.0], [0.5, -1.0]], dtype=jnp.float32)
    update, dual = muon_ogd_dual_update(
        matrix,
        jnp.zeros((0, 2, 2), dtype=jnp.float32),
        jnp.zeros((0,), dtype=jnp.float32),
        dual_learning_rate=0.25,
        dual_steps=2,
    )
    np.testing.assert_array_equal(dual, jnp.zeros((0,), dtype=jnp.float32))
    np.testing.assert_allclose(update, spectral_matrix_sign(matrix), rtol=1e-6)


def test_flad_zero_gradient_is_primal_and_derivative_safe() -> None:
    perturbation = jnp.array([2.0, 4.0])
    zero = jnp.zeros(2)
    np.testing.assert_array_equal(flad_noise_component(perturbation, zero), perturbation)
    delta_jacobian, gradient_jacobian = jax.jacrev(flad_noise_component, argnums=(0, 1))(
        perturbation, zero
    )
    assert bool(jnp.all(jnp.isfinite(delta_jacobian)))
    assert bool(jnp.all(jnp.isfinite(gradient_jacobian)))
    np.testing.assert_array_equal(delta_jacobian, jnp.eye(2))


def test_flad_jit_removes_gradient_aligned_component() -> None:
    result = jax.jit(flad_noise_component)(jnp.array([2.0, 4.0]), jnp.array([1.0, 0.0]))
    np.testing.assert_allclose(result, [0.0, 4.0])


def test_streaming_matrix_evaluation_is_frozen_matched_and_nonpromoting() -> None:
    result = run_streaming_matrix_evaluation()
    assert result["schema"] == GEOMETRY_RESULT_SCHEMA
    assert result["config"] == dict(FROZEN_GEOMETRY_CONFIG)
    assert GEOMETRY_PROTOCOL["stage"] == "frozen_small_streaming_matrix_pre_ipmnist"
    assert GEOMETRY_PROTOCOL["scientific_promotion_allowed"] is False
    arms = result["arms"]
    assert isinstance(arms, list)
    assert len(arms) == 6
    for arm in arms:
        resources = arm["resources"]
        assert resources["observations"] == FROZEN_GEOMETRY_CONFIG["updates"]
        assert resources["updates"] == FROZEN_GEOMETRY_CONFIG["updates"]
        assert resources["data_steps"] == FROZEN_GEOMETRY_CONFIG["updates"]
        assert resources["persistent_bytes"] > 0
        assert resources["timing_qualified"] is False
    assert result["policy"] == {
        "status": "development-only-nonpromoting",
        "development_only": True,
        "scientific_promotion_allowed": False,
        "negative_outcomes_retained": True,
    }
    validate_streaming_matrix_result(json.loads(json.dumps(result)))


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("config", "seed"), 1),
        (("config", "updates"), 8.0),
        (("policy", "scientific_promotion_allowed"), True),
        (("arms", 0, "resources", "updates"), 7),
        (("arms", 0, "metrics", "final_target_mse"), 0.0),
        (("comparisons", 0, "outcome"), "unexpected"),
    ],
)
def test_streaming_matrix_validator_rejects_tampering(
    path: tuple[object, ...], replacement: object
) -> None:
    result = copy.deepcopy(run_streaming_matrix_evaluation())
    target: object = result
    for component in path[:-1]:
        target = target[component]  # type: ignore[index]
    target[path[-1]] = replacement  # type: ignore[index]
    with pytest.raises(ValueError):
        validate_streaming_matrix_result(result)


def test_streaming_matrix_validator_admits_exact_builtin_containers_before_hooks() -> None:
    class HostileDict(dict[object, object]):
        calls = 0

        def __iter__(self):  # type: ignore[no-untyped-def]
            self.calls += 1
            raise AssertionError("must not iterate")

        def __getitem__(self, key: object) -> object:
            self.calls += 1
            raise AssertionError("must not index")

    hostile = HostileDict()
    with pytest.raises(ValueError, match="string-keyed mapping"):
        validate_streaming_matrix_result(hostile)
    assert hostile.calls == 0


@pytest.mark.parametrize("field", ["protocol", "arms"])
def test_streaming_matrix_validator_rejects_nested_container_subclasses_without_hooks(
    field: str,
) -> None:
    class HostileDict(dict[object, object]):
        calls = 0

        def __iter__(self):  # type: ignore[no-untyped-def]
            self.calls += 1
            raise AssertionError("must not iterate")

    class HostileList(list[object]):
        calls = 0

        def __iter__(self):  # type: ignore[no-untyped-def]
            self.calls += 1
            raise AssertionError("must not iterate")

    result = run_streaming_matrix_evaluation()
    hostile: HostileDict | HostileList
    if field == "protocol":
        hostile = HostileDict()
    else:
        hostile = HostileList()
    result[field] = hostile
    with pytest.raises(ValueError):
        validate_streaming_matrix_result(result)
    assert hostile.calls == 0


def test_streaming_matrix_validator_rejects_scalar_subclasses_without_equality_hooks() -> None:
    class HostileStr(str):
        calls = 0

        def __eq__(self, other: object) -> bool:
            self.calls += 1
            raise AssertionError("must not compare")

    class HostileFloat(float):
        calls = 0

        def __eq__(self, other: object) -> bool:
            self.calls += 1
            raise AssertionError("must not compare")

    result = run_streaming_matrix_evaluation()
    hostile_string = HostileStr("unexpected")
    result["schema"] = hostile_string
    with pytest.raises(ValueError):
        validate_streaming_matrix_result(result)
    assert hostile_string.calls == 0

    result = run_streaming_matrix_evaluation()
    hostile_number = HostileFloat(0.0)
    result["arms"][0]["metrics"]["final_target_mse"] = hostile_number  # type: ignore[index]
    with pytest.raises(ValueError):
        validate_streaming_matrix_result(result)
    assert hostile_number.calls == 0


def test_geometry_primitives_are_outer_jit_safe() -> None:
    corrected = jax.jit(orthogonal_correction)(jnp.array([1.0, 2.0]), jnp.array([[1.0, 0.0]]))
    np.testing.assert_allclose(corrected, [0.0, 2.0])


def test_geometry_rejects_empty_flad_and_hostile_array() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        flad_noise_component(jnp.zeros(0), jnp.zeros(0))

    class Hostile:
        calls = 0

        def __array__(self) -> np.ndarray:
            self.calls += 1
            raise AssertionError("must not run")

    hostile = Hostile()
    with pytest.raises(ValueError, match="exact NumPy or JAX"):
        spectral_matrix_sign(hostile)  # type: ignore[arg-type]
    assert hostile.calls == 0


def test_geometry_float32_overflow_is_invalid_not_laundered() -> None:
    maximum = jnp.finfo(jnp.float32).max
    matrix, matrix_valid = jax.jit(spectral_matrix_sign_transaction)(jnp.full((2, 2), maximum))
    assert bool(jnp.all(jnp.isfinite(matrix)))
    assert not bool(matrix_valid)
    component, component_valid = jax.jit(flad_noise_component_transaction)(
        jnp.full((2,), maximum), jnp.full((2,), maximum)
    )
    assert bool(jnp.all(jnp.isfinite(component)))
    assert not bool(component_valid)
