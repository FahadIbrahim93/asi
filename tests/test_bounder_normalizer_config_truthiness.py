"""Unit tests verifying bounder/normalizer config adoption never coerces truthiness."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from alberta_framework.core.actor_critic import (
    ActorCriticAgent,
    ActorCriticConfig,
    ContinuousActorCriticAgent,
    ContinuousActorCriticConfig,
)
from alberta_framework.core.horde import HordeLearner
from alberta_framework.core.horde_actor_critic import (
    HordeActorCriticAgent,
    HordeActorCriticConfig,
    NonlinearHordeActorCriticAgent,
    NonlinearHordeActorCriticConfig,
    NonlinearQHordeActorCriticAgent,
    NonlinearQHordeActorCriticConfig,
    QHordeActorCriticAgent,
    QHordeActorCriticConfig,
)
from alberta_framework.core.normalizers import EMANormalizer
from alberta_framework.core.off_policy_horde import OffPolicyHordeLearner
from alberta_framework.core.optimizers import LMS, ObGDBounding
from alberta_framework.core.types import DemonType, GVFSpec, HordeSpec


class _HostileBounderConfig(dict[str, object]):
    calls = 0

    def __bool__(self) -> bool:
        type(self).calls += 1
        raise AssertionError("bounder config truth hook executed")


class _HostileNormalizerConfig(dict[str, object]):
    calls = 0

    def __bool__(self) -> bool:
        type(self).calls += 1
        raise AssertionError("normalizer config truth hook executed")


def _sample_horde_spec() -> HordeSpec:
    demon = GVFSpec(
        name="d0",
        demon_type=DemonType.PREDICTION,
        gamma=0.9,
        lamda=0.8,
        cumulant_index=0,
    )
    return HordeSpec(
        demons=(demon,),
        gammas=jnp.array([0.9]),
        lamdas=jnp.array([0.8]),
    )


def _sample_control_horde_spec() -> HordeSpec:
    demon = GVFSpec(
        name="control_d0",
        demon_type=DemonType.CONTROL,
        gamma=0.0,
        lamda=0.8,
        cumulant_index=0,
    )
    return HordeSpec(
        demons=(demon,),
        gammas=jnp.array([0.0]),
        lamdas=jnp.array([0.8]),
    )


class TestActorCriticBounderConfigTruthiness:
    def test_actor_critic_config_does_not_invoke_truthiness(self) -> None:
        _HostileBounderConfig.calls = 0
        agent = ActorCriticAgent(
            ActorCriticConfig(n_actions=2),
            bounder=ObGDBounding(kappa=1.5),
        )
        payload = agent.to_config()
        payload["bounder"] = _HostileBounderConfig(payload["bounder"])

        with pytest.raises(ValueError, match="bounder config must be an exact dict"):
            ActorCriticAgent.from_config(payload)

        assert _HostileBounderConfig.calls == 0

    def test_continuous_actor_critic_config_does_not_invoke_truthiness(self) -> None:
        _HostileBounderConfig.calls = 0
        agent = ContinuousActorCriticAgent(
            ContinuousActorCriticConfig(action_dim=1),
            bounder=ObGDBounding(kappa=1.5),
        )
        payload = agent.to_config()
        payload["bounder"] = _HostileBounderConfig(payload["bounder"])

        with pytest.raises(ValueError, match="bounder config must be an exact dict"):
            ContinuousActorCriticAgent.from_config(payload)

        assert _HostileBounderConfig.calls == 0


class TestHordeActorCriticBounderConfigTruthiness:
    def test_horde_actor_critic_config_does_not_invoke_truthiness(self) -> None:
        _HostileBounderConfig.calls = 0
        agent = HordeActorCriticAgent(
            HordeActorCriticConfig(
                n_actions=2,
                actor_step_size=0.05,
                actor_lamda=0.7,
            ),
            critic=HordeLearner(horde_spec=_sample_horde_spec()),
            actor_bounder=ObGDBounding(kappa=1.5),
        )
        payload = agent.to_config()
        payload["actor_bounder"] = _HostileBounderConfig(payload["actor_bounder"])

        with pytest.raises(ValueError, match="bounder config must be an exact dict"):
            HordeActorCriticAgent.from_config(payload)

        assert _HostileBounderConfig.calls == 0

    def test_q_horde_actor_critic_config_does_not_invoke_truthiness(self) -> None:
        _HostileBounderConfig.calls = 0
        agent = QHordeActorCriticAgent(
            QHordeActorCriticConfig(
                n_actions=1,
                gamma=0.9,
                actor_step_size=0.05,
                actor_lamda=0.7,
            ),
            critic=HordeLearner(horde_spec=_sample_control_horde_spec()),
            actor_bounder=ObGDBounding(kappa=1.5),
        )
        payload = agent.to_config()
        payload["actor_bounder"] = _HostileBounderConfig(payload["actor_bounder"])

        with pytest.raises(ValueError, match="bounder config must be an exact dict"):
            QHordeActorCriticAgent.from_config(payload)

        assert _HostileBounderConfig.calls == 0

    def test_nonlinear_horde_actor_critic_config_does_not_invoke_truthiness(self) -> None:
        _HostileBounderConfig.calls = 0
        agent = NonlinearHordeActorCriticAgent(
            config=NonlinearHordeActorCriticConfig(n_actions=2, value_head_index=0),
            critic=HordeLearner(horde_spec=_sample_horde_spec()),
            actor_bounder=ObGDBounding(kappa=1.5),
        )
        payload = agent.to_config()
        payload["actor_bounder"] = _HostileBounderConfig(payload["actor_bounder"])

        with pytest.raises(ValueError, match="bounder config must be an exact dict"):
            NonlinearHordeActorCriticAgent.from_config(payload)

        assert _HostileBounderConfig.calls == 0

    def test_nonlinear_q_horde_actor_critic_config_does_not_invoke_truthiness(self) -> None:
        _HostileBounderConfig.calls = 0
        agent = NonlinearQHordeActorCriticAgent(
            config=NonlinearQHordeActorCriticConfig(n_actions=1),
            critic=HordeLearner(horde_spec=_sample_control_horde_spec()),
            actor_bounder=ObGDBounding(kappa=1.5),
        )
        payload = agent.to_config()
        payload["actor_bounder"] = _HostileBounderConfig(payload["actor_bounder"])

        with pytest.raises(ValueError, match="bounder config must be an exact dict"):
            NonlinearQHordeActorCriticAgent.from_config(payload)

        assert _HostileBounderConfig.calls == 0


class TestOffPolicyHordeBounderNormalizerConfigTruthiness:
    def test_off_policy_horde_bounder_config_does_not_invoke_truthiness(self) -> None:
        _HostileBounderConfig.calls = 0
        horde = OffPolicyHordeLearner(
            horde_spec=_sample_horde_spec(),
            optimizer=LMS(step_size=0.01),
            bounder=ObGDBounding(kappa=1.5),
        )
        payload = horde.to_config()
        payload["bounder"] = _HostileBounderConfig(payload["bounder"])

        with pytest.raises(ValueError, match="bounder config must be an exact dict"):
            OffPolicyHordeLearner.from_config(payload)

        assert _HostileBounderConfig.calls == 0

    def test_off_policy_horde_normalizer_config_does_not_invoke_truthiness(self) -> None:
        _HostileNormalizerConfig.calls = 0
        horde = OffPolicyHordeLearner(
            horde_spec=_sample_horde_spec(),
            optimizer=LMS(step_size=0.01),
            normalizer=EMANormalizer(),
        )
        payload = horde.to_config()
        payload["normalizer"] = _HostileNormalizerConfig(payload["normalizer"])

        with pytest.raises(ValueError, match="normalizer config must be an exact dict"):
            OffPolicyHordeLearner.from_config(payload)

        assert _HostileNormalizerConfig.calls == 0

    def test_off_policy_horde_normalizer_config_round_trips_exact_dict(self) -> None:
        horde = OffPolicyHordeLearner(
            horde_spec=_sample_horde_spec(),
            optimizer=LMS(step_size=0.01),
            normalizer=EMANormalizer(),
        )

        restored = OffPolicyHordeLearner.from_config(horde.to_config())

        assert restored._normalizer is not None
