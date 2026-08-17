"""Surprise-driven cumulant discovery for Horde demons (Step 3 Phase F).

Background
==========
Horde (Sutton et al. 2011) requires a hand-curated set of GVF demons --
each demon's cumulant, gamma, lambda, and target policy are specified up
front. For Step 3 to scale beyond hand-engineered question sets, we need
**discovery**: a mechanism that proposes candidate cumulants, evaluates
their utility, keeps the best, and discards the rest.

This module implements the simplest practical discovery method --
**surprise-driven retention**. Candidates are random projections of the
input observation; each candidate has a "demon" attached as a
single-output ``LinearLearner``-style predictor; candidate utility is the
EMA of squared TD error (the "surprise"). Periodically we replace the
lowest-utility candidates with new random projections.

Why squared TD error? A TD error of zero means the demon has accurately
predicted its cumulant; the cumulant is therefore *not informative*
about future dynamics. Demons with persistent positive squared TD error
are predicting a signal that is hard to know. (Cf. White, Modayil &
Sutton 2014, "Surprise as an intrinsic motivation for hierarchical RL.")
Caveat: the converse selection also happens -- a cumulant that is
irreducible noise sustains high squared TD error forever, so this
utility cannot separate hard-but-learnable structure from unpredictable
noise. Distinguishing the two needs a learning-progress signal (error
*decrease* over time) rather than error magnitude.

Limitations
===========
This is intentionally minimal:
- Uses ``OffPolicyTDLinearLearner`` per candidate (linear, on-policy when
  rho=1) so the candidate predictor has no learned features beyond the
  raw observation.
- Random projections are the cheapest possible cumulant generator.
- Retains the K highest-utility candidates; no shadow / promotion logic
  like Step 2's interaction features. (A future iteration could borrow
  from ``FixedBudgetInteractionLearner``.)
- For Veeriah-style meta-gradient discovery (NeurIPS 2019), the cumulant
  parameters would need to be learned by gradient descent on a downstream
  task loss. That is the natural follow-up.

The output of discovery is a tuple ``(active_cumulants, utilities)`` that
can be plugged into a downstream Horde -- the surviving demons are the
GVF cumulants worth predicting.

Reference:
    White, A., Modayil, J., & Sutton, R.S. (2014). "Surprise as an
    intrinsic motivation for hierarchical reinforcement learning."
    Veeriah, V., et al. (2019). "Discovery of Useful Questions as
    Auxiliary Tasks." NeurIPS.
"""

from __future__ import annotations

import functools
import operator
from typing import Any, SupportsIndex, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Float, Int, PRNGKeyArray

from alberta_framework.core._float32_scalars import validated_float32_scalar

_INT32_MAX = 2**31 - 1
_MAX_PERSISTENT_STATE_BYTES = 256 * 1024 * 1024
_ACTUAL_INT_TYPES = frozenset(
    {int, *(np.dtype(code).type for code in ("b", "B", "h", "H", "i", "I", "l", "L", "q", "Q"))}
)


def _require_int32(name: str, value: object, *, minimum: int) -> int:
    if type(value) not in _ACTUAL_INT_TYPES:
        raise ValueError(f"{name} must be an integer in [{minimum}, {_INT32_MAX}]")
    canonical = operator.index(cast(SupportsIndex, value))
    if not minimum <= canonical <= _INT32_MAX:
        raise ValueError(f"{name} must be an integer in [{minimum}, {_INT32_MAX}]")
    return canonical


def _require_float32(
    name: str,
    value: object,
    *,
    lower: float | None = None,
    upper: float | None = None,
    upper_inclusive: bool = True,
    positive: bool = False,
) -> float:
    try:
        return validated_float32_scalar(
            name,
            value,
            lower=lower,
            upper=upper,
            upper_inclusive=upper_inclusive,
            positive=positive,
        )
    except Exception as error:
        raise ValueError(f"{name} is outside its finite float32 domain") from error


def _persistent_resources(raw_dim: int, n_candidates: int) -> dict[str, int]:
    """Exact retained-array budget, excluding compiler and transient buffers."""
    persistent_scalars = 2 * n_candidates * raw_dim + 3 * n_candidates + 2
    persistent_bytes = 4 * persistent_scalars
    if persistent_scalars > _INT32_MAX:
        raise ValueError("derived cumulant-discovery persistent scalars must fit signed int32")
    if persistent_bytes > _MAX_PERSISTENT_STATE_BYTES:
        raise ValueError("derived cumulant-discovery persistent state exceeds 256 MiB")
    return {
        "persistent_scalars": persistent_scalars,
        "persistent_bytes": persistent_bytes,
    }

# =============================================================================
# State
# =============================================================================


@chex.dataclass(frozen=True)
class CumulantDiscoveryState:
    """State for surprise-driven cumulant discovery.

    Attributes:
        projections: ``(n_candidates, raw_dim)`` random projection matrix
            -- each row defines one candidate cumulant ``c_i = w_i . obs``.
        weights: ``(n_candidates, raw_dim)`` per-candidate predictor weights.
        biases: ``(n_candidates,)`` per-candidate predictor biases.
        utility: ``(n_candidates,)`` EMA of squared TD error (surprise).
        ages: ``(n_candidates,)`` number of update steps since last reset.
        key: JAX random key for sampling fresh projections on replacement.
    """

    projections: Float[Array, "n_candidates raw_dim"]
    weights: Float[Array, "n_candidates raw_dim"]
    biases: Float[Array, " n_candidates"]
    utility: Float[Array, " n_candidates"]
    ages: Int[Array, " n_candidates"]
    key: PRNGKeyArray


# =============================================================================
# Discovery learner
# =============================================================================


class CumulantDiscovery:
    """Surprise-driven cumulant discovery.

    Maintains ``n_candidates`` parallel candidate cumulants (random
    linear projections of the observation) plus per-candidate linear
    value predictors. After each step, the predictor's TD error is the
    "surprise"; an EMA of squared surprise is the utility. Periodically,
    candidates with the lowest utility (subject to a maturity threshold)
    are replaced with fresh random projections.

    Args:
        raw_dim: Dimension of the raw observation
        n_candidates: Number of cumulant candidates to maintain
        decay_rate: Utility EMA decay (``utility := decay*utility +
            (1-decay)*surprise``). Default 0.99.
        replacement_rate: Per-step probability of replacing the
            lowest-utility candidate (eligible by maturity). Default
            ``5e-3`` (replace ~one candidate every 200 steps).
        maturity_threshold: Minimum age before a candidate may be
            replaced. Default 200 steps.
        predictor_step_size: Per-candidate linear predictor step size.
            Default 0.05.
        gamma: Pseudo-termination discount used for the demon's TD error
            (default 0.0 = supervised next-step prediction). Use
            ``gamma > 0`` to retain candidates whose temporal returns
            are surprising.
        enabled: If False, replacement never occurs (useful for
            ablations).
    """

    def __init__(
        self,
        raw_dim: int,
        n_candidates: int = 16,
        decay_rate: float = 0.99,
        replacement_rate: float = 5e-3,
        maturity_threshold: int = 200,
        predictor_step_size: float = 0.05,
        gamma: float = 0.0,
        enabled: bool = True,
    ):
        raw_dim = _require_int32("raw_dim", raw_dim, minimum=1)
        n_candidates = _require_int32("n_candidates", n_candidates, minimum=1)
        maturity_threshold = _require_int32(
            "maturity_threshold", maturity_threshold, minimum=0
        )
        _persistent_resources(raw_dim, n_candidates)
        decay_rate = _require_float32(
            "decay_rate",
            decay_rate,
            positive=True,
            upper=1.0,
            upper_inclusive=False,
        )
        replacement_rate = _require_float32(
            "replacement_rate", replacement_rate, lower=0.0, upper=1.0
        )
        predictor_step_size = _require_float32(
            "predictor_step_size", predictor_step_size, positive=True
        )
        gamma = _require_float32("gamma", gamma, lower=0.0, upper=1.0)
        if type(enabled) not in (bool, np.bool_):
            raise ValueError("enabled must be a boolean")
        enabled = bool(enabled)

        self._raw_dim = raw_dim
        self._n_candidates = n_candidates
        self._decay_rate = decay_rate
        self._replacement_rate = replacement_rate
        self._maturity_threshold = maturity_threshold
        self._predictor_step_size = predictor_step_size
        self._gamma = gamma
        self._enabled = enabled

    @property
    def n_candidates(self) -> int:
        return self._n_candidates

    @property
    def raw_dim(self) -> int:
        return self._raw_dim

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def persistent_resource_budget(self) -> dict[str, int]:
        """Exact persistent JAX-array envelope."""
        return _persistent_resources(self._raw_dim, self._n_candidates)

    def init(self, key: Array) -> CumulantDiscoveryState:
        """Initialize state with random projections, zero predictors,
        zero utility, zero ages."""
        k_proj, k_state = jr.split(key)
        # Unit-norm random projections
        raw_proj = jr.normal(
            k_proj, (self._n_candidates, self._raw_dim), dtype=jnp.float32
        )
        norms = jnp.linalg.norm(raw_proj, axis=1, keepdims=True) + 1e-8
        projections = raw_proj / norms
        return CumulantDiscoveryState(  # type: ignore[call-arg]
            projections=projections,
            weights=jnp.zeros(
                (self._n_candidates, self._raw_dim), dtype=jnp.float32
            ),
            biases=jnp.zeros(self._n_candidates, dtype=jnp.float32),
            utility=jnp.zeros(self._n_candidates, dtype=jnp.float32),
            ages=jnp.zeros(self._n_candidates, dtype=jnp.int32),
            key=k_state,
        )

    def cumulants(
        self,
        state: CumulantDiscoveryState,
        observation: Float[Array, " raw_dim"],
    ) -> Float[Array, " n_candidates"]:
        """Compute the candidate cumulant values for an observation.

        In a GVF update, cumulants are transition signals: the update from
        ``s_t`` to ``s_{t+1}`` should use ``c_{t+1}``, so callers feeding a
        Horde should normally pass the *next* observation here.
        """
        return state.projections @ observation

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: CumulantDiscoveryState,
        observation: Float[Array, " raw_dim"],
        next_observation: Float[Array, " raw_dim"],
    ) -> CumulantDiscoveryState:
        """Apply one update step.

        For each candidate i:
            cumulant_i = projections_i . next_obs
            V_i        = weights_i . obs + bias_i
            V_i_next   = weights_i . next_obs + bias_i
            delta_i    = cumulant_i + gamma * V_i_next - V_i
            weights_i += alpha * delta_i * obs
            bias_i    += alpha * delta_i
            utility_i := decay * utility_i + (1 - decay) * delta_i^2
            ages_i    += 1

        Args:
            state: Current discovery state
            observation: Current raw observation, ``s_t``.
            next_observation: Next raw observation, ``s_{t+1}``; candidate
                cumulants are evaluated on this observation to match the
                nexting/GVF convention ``G_t = c_{t+1} + gamma G_{t+1}``.

        Returns:
            Updated discovery state
        """
        alpha = jnp.asarray(self._predictor_step_size, dtype=jnp.float32)
        gamma = jnp.asarray(self._gamma, dtype=jnp.float32)
        decay = jnp.asarray(self._decay_rate, dtype=jnp.float32)

        # Per-candidate transition cumulant c_{t+1} and predictions.
        cumulants = state.projections @ next_observation  # (n,)
        v = state.weights @ observation + state.biases  # (n,)
        v_next = state.weights @ next_observation + state.biases  # (n,)

        td = cumulants + gamma * v_next - v  # (n,)

        # Predictor update (per-candidate semi-gradient TD step)
        # weights_i += alpha * td_i * obs
        proposed_weights = state.weights + alpha * td[:, None] * observation[None, :]
        proposed_biases = state.biases + alpha * td
        proposed_utility = decay * state.utility + (1.0 - decay) * (td**2)
        proposed_state = CumulantDiscoveryState(  # type: ignore[call-arg]
            projections=state.projections,
            weights=proposed_weights,
            biases=proposed_biases,
            utility=proposed_utility,
            ages=jnp.minimum(state.ages, jnp.asarray(_INT32_MAX - 1, dtype=jnp.int32)) + 1,
            key=state.key,
        )
        # Inf next obs makes 0 @ inf = NaN in V(s') at zero init, then
        # alpha * nan * obs poisons every candidate. Hold the finite state.
        inputs_valid = jnp.all(jnp.isfinite(observation)) & jnp.all(
            jnp.isfinite(next_observation)
        )
        proposed_finite = (
            jnp.all(jnp.isfinite(proposed_weights))
            & jnp.all(jnp.isfinite(proposed_biases))
            & jnp.all(jnp.isfinite(proposed_utility))
        )
        return cast(
            CumulantDiscoveryState,
            jax.lax.cond(
                inputs_valid & proposed_finite,
                lambda: proposed_state,
                lambda: state,
            ),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def maybe_replace(
        self,
        state: CumulantDiscoveryState,
    ) -> CumulantDiscoveryState:
        """Possibly replace the lowest-utility eligible candidate with a
        fresh random projection.

        If ``enabled=False``, returns the input state unchanged.

        On each call, with probability ``replacement_rate``, the
        lowest-utility candidate that has reached maturity is replaced.
        Replacement: re-sample the projection row, zero the predictor
        weights/bias, reset utility to 0 and age to 0.
        """
        if not self._enabled:
            return state

        k_trigger, k_proj, k_next = jr.split(state.key, 3)

        # Sample whether to attempt replacement
        trigger = jr.uniform(k_trigger, ()) < self._replacement_rate

        # Among mature candidates, find the one with lowest utility
        mature = state.ages >= self._maturity_threshold
        # Disqualify immature candidates by setting their utility to +inf
        masked_utility = jnp.where(
            mature, state.utility, jnp.full_like(state.utility, jnp.inf)
        )
        # If no candidates are mature, this index is arbitrary
        worst_idx = jnp.argmin(masked_utility)
        any_mature = jnp.any(mature)

        # Fresh projection
        raw_new = jr.normal(k_proj, (self._raw_dim,), dtype=jnp.float32)
        new_proj_row = raw_new / (jnp.linalg.norm(raw_new) + 1e-8)

        do_replace = trigger & any_mature

        new_projections = jnp.where(
            do_replace,
            state.projections.at[worst_idx].set(new_proj_row),
            state.projections,
        )
        new_weights = jnp.where(
            do_replace,
            state.weights.at[worst_idx].set(jnp.zeros(self._raw_dim, dtype=jnp.float32)),
            state.weights,
        )
        new_biases = jnp.where(
            do_replace,
            state.biases.at[worst_idx].set(jnp.float32(0.0)),
            state.biases,
        )
        new_utility = jnp.where(
            do_replace,
            state.utility.at[worst_idx].set(jnp.float32(0.0)),
            state.utility,
        )
        new_ages = jnp.where(
            do_replace,
            state.ages.at[worst_idx].set(jnp.int32(0)),
            state.ages,
        )

        return CumulantDiscoveryState(  # type: ignore[call-arg]
            projections=new_projections,
            weights=new_weights,
            biases=new_biases,
            utility=new_utility,
            ages=new_ages,
            key=k_next,
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "type": "CumulantDiscovery",
            "raw_dim": self._raw_dim,
            "n_candidates": self._n_candidates,
            "decay_rate": self._decay_rate,
            "replacement_rate": self._replacement_rate,
            "maturity_threshold": self._maturity_threshold,
            "predictor_step_size": self._predictor_step_size,
            "gamma": self._gamma,
            "enabled": self._enabled,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> CumulantDiscovery:
        """Reconstruct from dict."""
        if type(config) is not dict:
            raise ValueError("cumulant-discovery config must be an exact dictionary")
        config = dict(config)
        expected = {
            "type",
            "raw_dim",
            "n_candidates",
            "decay_rate",
            "replacement_rate",
            "maturity_threshold",
            "predictor_step_size",
            "gamma",
            "enabled",
        }
        if set(config) != expected:
            raise ValueError("cumulant-discovery config keys are not the exact schema")
        if config.pop("type") != "CumulantDiscovery":
            raise ValueError("cumulant-discovery config type is unsupported")
        for name in ("raw_dim", "n_candidates", "maturity_threshold"):
            if type(config[name]) is not int:
                raise ValueError(f"serialized {name} must be an exact JSON integer")
        for name in (
            "decay_rate",
            "replacement_rate",
            "predictor_step_size",
            "gamma",
        ):
            if type(config[name]) is not float:
                raise ValueError(f"serialized {name} must be an exact JSON float")
        if type(config["enabled"]) is not bool:
            raise ValueError("serialized enabled must be an exact JSON boolean")
        return cls(**config)
