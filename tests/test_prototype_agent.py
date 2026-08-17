"""Mechanism tests for the experimental PrototypeAgent composition surface.

These tests establish routing, shape, update, and isolation invariants.  They
do not establish an integrated Alberta Plan completion result.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.checkpoints import (
    load_checkpoint_metadata,
    save_checkpoint,
)
from alberta_framework.core.dreaming import DreamingConfig
from alberta_framework.core.intelligence_amplification import IAConfig
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import (
    PROTOTYPE_CHECKPOINT_SCHEMA,
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeAgentState,
    PrototypeArrayResult,
    PrototypeUpdateResult,
    feature_to_subtask_specs,
    load_prototype_checkpoint,
    save_prototype_checkpoint,
)
from alberta_framework.core.types import DemonType, GVFSpec, create_horde_spec
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SPEC0 = SubtaskSpec(
    feature_index=0,
    threshold=0.5,
    pseudo_reward_scale=1.0,
    max_option_steps=8,
)
_SPEC1 = SubtaskSpec(
    feature_index=1,
    threshold=0.3,
    pseudo_reward_scale=2.0,
    max_option_steps=4,
)

OBS_DIM = 4
N_PRIM = 2


def test_authoritative_transition_is_compiled_as_one_reusable_jax_boundary() -> None:
    """Validation stays eager while normalized transitions reuse one compiled core."""

    assert not hasattr(PrototypeAgent.update_transition, "lower")
    assert hasattr(PrototypeAgent._update_transition_impl, "lower")


def _oak_cfg(
    specs: tuple[SubtaskSpec, ...] = (_SPEC0,),
    obs_dim: int = OBS_DIM,
    n_prim: int = N_PRIM,
) -> OaKConfig:
    stomp = STOMPConfig(
        subtask_specs=specs,
        observation_dim=obs_dim,
        n_primitive_actions=n_prim,
    )
    return OaKConfig(stomp=stomp)


def _wm_cfg(
    obs_dim: int = OBS_DIM,
    n_actions: int = N_PRIM,
) -> ActionConditionedWorldModelConfig:
    return ActionConditionedWorldModelConfig(
        observation_dim=obs_dim,
        n_actions=n_actions,
        hidden_sizes=(),  # linear for speed
        step_size=0.1,
        error_decay=0.99,
    )


def _minimal_config() -> PrototypeAgentConfig:
    """OaK-only, no world model, no horde, no IA."""
    return PrototypeAgentConfig(oak=_oak_cfg())


def _full_config(n_dreams: int = 2) -> PrototypeAgentConfig:
    """All components enabled."""
    horde_spec = create_horde_spec(
        [
            GVFSpec(
                name="v0.9",
                demon_type=DemonType.PREDICTION,
                cumulant_index=0,
                gamma=0.9,
                lamda=0.0,
            ),
            GVFSpec(
                name="r",
                demon_type=DemonType.PREDICTION,
                cumulant_index=0,
                gamma=0.0,
                lamda=0.0,
            ),
        ]
    )
    from alberta_framework.core.intelligence_amplification import ExoCerebellumConfig

    ia_cortex = OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=(_SPEC0,),
            observation_dim=OBS_DIM,
            n_primitive_actions=N_PRIM,
        )
    )
    ia_cfg = IAConfig(
        cerebellum=ExoCerebellumConfig(n_demons=2, obs_dim=OBS_DIM, step_size=0.05),
        cortex=ia_cortex,
    )
    return PrototypeAgentConfig(
        oak=_oak_cfg(),
        world_model=_wm_cfg(),
        dreaming=DreamingConfig(warmup_steps=1, max_model_error_ema=1e6),
        buffer_capacity=20,
        n_dreams_per_step=n_dreams,
        horde_spec=horde_spec,
        horde_hidden_sizes=(),
        horde_step_size=0.1,
        ia=ia_cfg,
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestPrototypeAgentConfigValidation:
    def test_buffer_capacity_positive(self) -> None:
        with pytest.raises(ValueError, match="buffer_capacity"):
            PrototypeAgentConfig(oak=_oak_cfg(), buffer_capacity=0)

    def test_n_dreams_non_negative(self) -> None:
        with pytest.raises(ValueError, match="n_dreams_per_step"):
            PrototypeAgentConfig(oak=_oak_cfg(), n_dreams_per_step=-1)

    def test_dreams_require_world_model(self) -> None:
        with pytest.raises(ValueError, match="world_model"):
            PrototypeAgentConfig(oak=_oak_cfg(), n_dreams_per_step=2, world_model=None)

    def test_unknown_dream_next_observation_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="dream_next_observation_mode"):
            PrototypeAgentConfig(
                oak=_oak_cfg(),
                dream_next_observation_mode="unknown",  # type: ignore[arg-type]
            )

    def test_sample_one_hot_dreams_reject_gru_observations(self) -> None:
        from alberta_framework.core.prototype_agent import GRUPerceptionConfig

        with pytest.raises(ValueError, match="GRU-augmented"):
            PrototypeAgentConfig(
                oak=_oak_cfg(obs_dim=2),
                gru_perception=GRUPerceptionConfig(
                    observation_dim=1,
                    hidden_dim=1,
                ),
                dream_next_observation_mode="sample_one_hot",
            )

    def test_world_model_observation_dim_must_match_oak(self) -> None:
        with pytest.raises(ValueError, match="world_model.observation_dim"):
            PrototypeAgentConfig(
                oak=_oak_cfg(obs_dim=2),
                world_model=_wm_cfg(obs_dim=3),
                dream_next_observation_mode="sample_one_hot",
            )

    def test_world_model_action_count_must_match_oak(self) -> None:
        with pytest.raises(ValueError, match="world_model.n_actions"):
            PrototypeAgentConfig(
                oak=_oak_cfg(n_prim=2),
                world_model=_wm_cfg(n_actions=3),
                dream_next_observation_mode="sample_one_hot",
            )

    def test_ia_obs_dim_must_match_oak(self) -> None:
        from alberta_framework.core.intelligence_amplification import ExoCerebellumConfig

        bad_cortex = OaKConfig(
            stomp=STOMPConfig(
                subtask_specs=(_SPEC0,),
                observation_dim=OBS_DIM + 1,  # mismatched
                n_primitive_actions=N_PRIM,
            )
        )
        ia_bad = IAConfig(
            cerebellum=ExoCerebellumConfig(n_demons=2, obs_dim=OBS_DIM + 1, step_size=0.05),
            cortex=bad_cortex,
        )
        with pytest.raises(ValueError, match="observation_dim"):
            PrototypeAgentConfig(oak=_oak_cfg(obs_dim=OBS_DIM), ia=ia_bad)

    def test_horde_step_size_positive(self) -> None:
        with pytest.raises(ValueError, match="horde_step_size"):
            PrototypeAgentConfig(oak=_oak_cfg(), horde_step_size=0.0)


# ---------------------------------------------------------------------------
# Config roundtrip
# ---------------------------------------------------------------------------


class TestPrototypeAgentConfigRoundtrip:
    def test_minimal_roundtrip(self) -> None:
        cfg = _minimal_config()
        restored = PrototypeAgentConfig.from_config(cfg.to_config())
        assert restored.oak.observation_dim == cfg.oak.observation_dim
        assert restored.world_model is None
        assert restored.horde_spec is None
        assert restored.ia is None

    def test_full_roundtrip(self) -> None:
        cfg = _full_config()
        restored = PrototypeAgentConfig.from_config(cfg.to_config())
        assert restored.oak.observation_dim == cfg.oak.observation_dim
        assert restored.world_model is not None
        assert restored.horde_spec is not None
        assert restored.ia is not None
        assert restored.n_dreams_per_step == cfg.n_dreams_per_step
        assert restored.buffer_capacity == cfg.buffer_capacity

    def test_legacy_dream_mode_preserves_serialized_config(self) -> None:
        cfg = _full_config()
        payload = cfg.to_config()
        assert cfg.dream_next_observation_mode == "model_prediction"
        assert "dream_next_observation_mode" not in payload
        assert (
            PrototypeAgentConfig.from_config(payload).dream_next_observation_mode
            == "model_prediction"
        )

    def test_sample_one_hot_dream_mode_roundtrip(self) -> None:
        cfg = PrototypeAgentConfig(
            oak=_oak_cfg(),
            world_model=_wm_cfg(),
            dreaming=DreamingConfig(warmup_steps=0),
            n_dreams_per_step=1,
            dream_next_observation_mode="sample_one_hot",
        )
        payload = cfg.to_config()
        assert payload["dream_next_observation_mode"] == "sample_one_hot"
        restored = PrototypeAgentConfig.from_config(payload)
        assert restored.dream_next_observation_mode == "sample_one_hot"

    def test_from_config_rejects_wrong_type_and_unknown_fields(self) -> None:
        payload = _minimal_config().to_config()
        wrong_type = dict(payload, type="NotPrototypeAgentConfig")
        with pytest.raises(ValueError, match="payload type"):
            PrototypeAgentConfig.from_config(wrong_type)
        unknown = dict(payload, unexpected=1)
        with pytest.raises(ValueError, match="unknown fields: unexpected"):
            PrototypeAgentConfig.from_config(unknown)


# ---------------------------------------------------------------------------
# Init and start
# ---------------------------------------------------------------------------


class TestPrototypeAgentInit:
    def test_minimal_init_and_start_contract(self) -> None:
        agent = PrototypeAgent(_minimal_config())
        state = agent.init(jr.key(0))
        assert isinstance(state, PrototypeAgentState)
        assert state.world_model_state is None
        assert state.buffer_state is None
        assert state.horde_state is None
        assert state.ia_state is None
        assert state.step_count == 0
        n_total = N_PRIM + 1  # 1 option
        bls = state.oak_state.stomp_state.base_learner_state
        assert len(bls.head_params.weights) == n_total
        obs = jnp.ones(OBS_DIM)
        primed = agent.start(state, obs)
        chex.assert_trees_all_close(primed.oak_state.stomp_state.base_last_obs, obs, atol=1e-6)


# ---------------------------------------------------------------------------
# Act
# ---------------------------------------------------------------------------


class TestPrototypeAgentAct:
    def test_act_returns_valid_action(self) -> None:
        agent = PrototypeAgent(_minimal_config())
        state = agent.start(agent.init(jr.key(0)), jnp.zeros(OBS_DIM))
        obs = jnp.zeros(OBS_DIM)
        action = agent.act(state, obs)
        chex.assert_shape(action, ())
        assert int(action) < N_PRIM


# ---------------------------------------------------------------------------
# Update: minimal (OaK only)
# ---------------------------------------------------------------------------


class TestPrototypeAgentUpdateMinimal:
    def test_update_contract(self) -> None:
        agent = PrototypeAgent(_minimal_config())
        state = agent.start(agent.init(jr.key(0)), jnp.zeros(OBS_DIM))
        result = agent.update(state, jnp.array(1.0), jnp.ones(OBS_DIM))

        assert isinstance(result, PrototypeUpdateResult)
        assert int(result.state.step_count) == 1
        chex.assert_shape(result.action, ())
        assert jnp.isfinite(result.oak_td_error)
        assert result.world_model_error is None
        assert result.dream_td_errors is None
        assert result.horde_td_errors is None
        assert result.ia_augmented_obs is None
        assert result.ia_recommendation is None


# ---------------------------------------------------------------------------
# Update: full agent (world model + dreaming + horde + IA)
# ---------------------------------------------------------------------------


class TestPrototypeAgentUpdateFull:
    def test_full_update_contract(self) -> None:
        agent = PrototypeAgent(_full_config(n_dreams=2))
        observation = jnp.zeros(OBS_DIM)
        state = agent.start(agent.init(jr.key(0)), observation)
        assert state.world_model_state is not None
        assert state.buffer_state is not None
        assert state.horde_state is not None
        assert state.ia_state is not None
        chex.assert_trees_all_close(
            state.ia_state.cortex_state.stomp_state.base_last_obs,
            observation,
            atol=1e-6,
        )
        cumulants = jnp.array([0.5, 0.3], dtype=jnp.float32)
        result = agent.update(state, jnp.array(1.0), jnp.ones(OBS_DIM), cumulants)

        assert result.world_model_error is not None
        assert jnp.isfinite(result.world_model_error)
        assert result.dream_td_errors is not None
        chex.assert_shape(result.dream_td_errors, (2,))
        assert result.horde_td_errors is not None
        chex.assert_shape(result.horde_td_errors, (2,))  # 2 demons
        assert result.ia_augmented_obs is not None
        chex.assert_shape(result.ia_augmented_obs, (OBS_DIM + 2,))  # obs + 2 cerebellum demons
        assert result.ia_recommendation is not None
        chex.assert_shape(result.ia_recommendation, ())
        assert int(result.state.buffer_state.size) == 1
        assert int(result.state.world_model_state.step_count) == 1


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------


class TestPrototypeAgentScan:
    def test_scan_minimal_contract(self) -> None:
        agent = PrototypeAgent(_minimal_config())
        state = agent.start(agent.init(jr.key(0)), jnp.zeros(OBS_DIM))
        n_steps = 3
        rewards = jr.normal(jr.key(42), (n_steps,))
        next_obs = jr.normal(jr.key(43), (n_steps, OBS_DIM))
        result = agent.scan(state, rewards, next_obs)

        assert isinstance(result, PrototypeArrayResult)
        chex.assert_shape(result.actions, (n_steps,))
        chex.assert_shape(result.oak_td_errors, (n_steps,))
        chex.assert_shape(result.oak_average_rewards, (n_steps,))
        assert int(result.state.step_count) == n_steps
        chex.assert_tree_all_finite(result.oak_td_errors)

    def test_scan_world_model_config_update(self) -> None:
        """Scan with world model enabled runs without error."""
        cfg = PrototypeAgentConfig(
            oak=_oak_cfg(),
            world_model=_wm_cfg(),
            dreaming=DreamingConfig(warmup_steps=100),  # warmup prevents dreaming
            n_dreams_per_step=0,
        )
        agent = PrototypeAgent(cfg)
        state = agent.start(agent.init(jr.key(0)), jnp.zeros(OBS_DIM))
        n_steps = 2
        result = agent.scan(state, jnp.zeros(n_steps), jnp.zeros((n_steps, OBS_DIM)))
        assert int(result.state.world_model_state.step_count) == n_steps


# ---------------------------------------------------------------------------
# Curation
# ---------------------------------------------------------------------------


class TestPrototypeAgentCurate:
    def test_curate_preserves_configuration_and_can_continue(self) -> None:
        cfg = PrototypeAgentConfig(
            oak=_oak_cfg(),
            world_model=_wm_cfg(),
            dreaming=DreamingConfig(warmup_steps=1000),
            n_dreams_per_step=0,
            dream_next_observation_mode="sample_one_hot",
        )
        agent = PrototypeAgent(cfg)
        state = agent.start(agent.init(jr.key(0)), jax.nn.one_hot(0, OBS_DIM))
        state = agent.update(
            state,
            jnp.array(0.0),
            jax.nn.one_hot(1, OBS_DIM),
        ).state
        new_agent, new_state = agent.curate(state, jr.key(2))

        assert isinstance(new_agent, PrototypeAgent)
        assert isinstance(new_state, PrototypeAgentState)
        assert new_agent.config.dream_next_observation_mode == "sample_one_hot"
        assert int(new_state.world_model_state.step_count) == int(
            state.world_model_state.step_count
        )
        result = new_agent.update(
            new_state,
            jnp.array(0.5),
            jax.nn.one_hot(2, OBS_DIM),
        )
        assert jnp.isfinite(result.oak_td_error)


# ---------------------------------------------------------------------------
# Auto subtask specs
# ---------------------------------------------------------------------------


class TestAutoSubtaskSpecs:
    def test_auto_subtask_specs_are_bounded_and_unique(self) -> None:
        agent = PrototypeAgent(_minimal_config())
        state = agent.start(agent.init(jr.key(0)), jnp.zeros(OBS_DIM))
        specs = agent.auto_subtask_specs(state, n_subtasks=3)

        assert len(specs) == 3
        indices = [spec.feature_index for spec in specs]
        assert all(0 <= index < OBS_DIM for index in indices)
        assert len(indices) == len(set(indices))


# ---------------------------------------------------------------------------
# feature_to_subtask_specs standalone
# ---------------------------------------------------------------------------


class TestFeatureToSubtaskSpecs:
    def test_count_cap_and_indices(self) -> None:
        agent = PrototypeAgent(_minimal_config())
        state = agent.init(jr.key(0))
        specs = feature_to_subtask_specs(state.oak_state, n_subtasks=2)
        assert len(specs) == 2
        specs = feature_to_subtask_specs(state.oak_state, n_subtasks=100)
        assert len(specs) <= OBS_DIM
        assert all(0 <= spec.feature_index < OBS_DIM for spec in specs)

    def test_applies_subtask_parameters(self) -> None:
        agent = PrototypeAgent(_minimal_config())
        state = agent.init(jr.key(0))
        specs = feature_to_subtask_specs(
            state.oak_state,
            n_subtasks=2,
            threshold=0.7,
            max_option_steps=15,
        )
        assert all(spec.threshold == pytest.approx(0.7) for spec in specs)
        assert all(spec.max_option_steps == 15 for spec in specs)


# ---------------------------------------------------------------------------
# Config serialization agent roundtrip
# ---------------------------------------------------------------------------


class TestPrototypeAgentSerializationRoundtrip:
    def test_agent_config_roundtrips(self) -> None:
        minimal = PrototypeAgent(_minimal_config())
        restored_minimal = PrototypeAgent.from_config(minimal.to_config())
        assert restored_minimal.config.oak.observation_dim == OBS_DIM
        assert restored_minimal.config.world_model is None

        full = PrototypeAgent(_full_config())
        restored_full = PrototypeAgent.from_config(full.to_config())
        assert restored_full.config.n_dreams_per_step == full.config.n_dreams_per_step
        assert restored_full.config.horde_spec is not None
        assert restored_full.config.ia is not None

    def test_primitive_only_checkpoint_roundtrips_empty_option_state_exactly(
        self,
        tmp_path,
    ) -> None:
        config = PrototypeAgentConfig(
            oak=_oak_cfg(specs=(), obs_dim=2, n_prim=2),
        )
        agent = PrototypeAgent(config)
        lifecycle_id = jnp.asarray((1, 2), dtype=jnp.uint32)
        state = agent.start(
            agent.init(jr.key(17), lifecycle_id=lifecycle_id),
            jnp.asarray((1.0, 0.0), dtype=jnp.float32),
        )
        assert state.oak_state.cumulative_pseudo_rewards.size == 0

        checkpoint = tmp_path / "primitive-only"
        save_prototype_checkpoint(agent, state, checkpoint)
        metadata = load_checkpoint_metadata(checkpoint)
        restored_agent, restored_state = load_prototype_checkpoint(checkpoint)

        assert (
            metadata["empty_array_codec"]
            == "alberta.prototype_agent.empty_array_projection.v1"
        )
        assert restored_agent.to_config() == agent.to_config()
        chex.assert_trees_all_equal(restored_state, state)

    def test_rbg_explicit_lifecycle_checkpoint_roundtrips_exactly(
        self,
        tmp_path,
    ) -> None:
        agent = PrototypeAgent(
            PrototypeAgentConfig(
                oak=_oak_cfg(specs=(), obs_dim=2, n_prim=2),
            )
        )
        state = agent.start(
            agent.init(
                jr.key(23, impl="rbg"),
                lifecycle_id=jnp.asarray((3, 4), dtype=jnp.uint32),
            ),
            jnp.asarray((0.25, -0.5), dtype=jnp.float32),
        )
        checkpoint = tmp_path / "primitive-only-rbg"
        save_prototype_checkpoint(agent, state, checkpoint)

        hinted_agent, hinted_state = load_prototype_checkpoint(
            checkpoint,
            template_key=jr.key(0, impl="rbg"),
        )
        restored_agent, restored_state = load_prototype_checkpoint(checkpoint)

        assert load_checkpoint_metadata(checkpoint)["prng_impl"] == "rbg"
        assert (
            hinted_agent.to_config()
            == restored_agent.to_config()
            == agent.to_config()
        )
        chex.assert_trees_all_equal(hinted_state, state)
        chex.assert_trees_all_equal(restored_state, state)
        with pytest.raises(ValueError, match="PRNG implementation"):
            load_prototype_checkpoint(checkpoint, template_key=jr.key(0))

    @pytest.mark.parametrize(
        ("shape", "dtype"),
        [
            pytest.param((0, 1), jnp.float32, id="shape"),
            pytest.param((0,), jnp.int32, id="dtype"),
        ],
    )
    def test_checkpoint_save_rejects_malformed_empty_leaf_contract(
        self,
        tmp_path,
        shape,
        dtype,
    ) -> None:
        agent = PrototypeAgent(
            PrototypeAgentConfig(
                oak=_oak_cfg(specs=(), obs_dim=2, n_prim=2),
            )
        )
        state = agent.start(
            agent.init(jr.key(18)),
            jnp.asarray((1.0, 0.0), dtype=jnp.float32),
        )
        malformed_state = state.replace(
            oak_state=state.oak_state.replace(
                cumulative_pseudo_rewards=jnp.zeros(shape, dtype=dtype),
            )
        )
        assert bool(agent._checkpoint_state_valid(malformed_state))

        with pytest.raises(ValueError, match="array contract"):
            save_prototype_checkpoint(
                agent,
                malformed_state,
                tmp_path / "malformed-empty-leaf",
            )

    @pytest.mark.parametrize(
        ("field", "dtype"),
        [
            pytest.param("utility_ema", jnp.int32, id="utility-dtype"),
            pytest.param("execution_counts", jnp.float32, id="count-dtype"),
        ],
    )
    def test_checkpoint_save_rejects_malformed_nonempty_array_dtype(
        self,
        tmp_path,
        field,
        dtype,
    ) -> None:
        agent = PrototypeAgent(_minimal_config())
        state = agent.start(
            agent.init(jr.key(20)),
            jnp.zeros((OBS_DIM,), dtype=jnp.float32),
        )
        original = getattr(state.oak_state, field)
        assert original.size > 0
        malformed_state = state.replace(
            oak_state=state.oak_state.replace(
                **{field: jnp.asarray(original, dtype=dtype)},
            )
        )
        assert bool(agent._checkpoint_state_valid(malformed_state))

        with pytest.raises(ValueError, match="array contract"):
            save_prototype_checkpoint(
                agent,
                malformed_state,
                tmp_path / "malformed-nonempty-leaf",
            )

    def test_checkpoint_save_rejects_scalar_replacing_array_leaf(
        self,
        tmp_path,
    ) -> None:
        agent = PrototypeAgent(_minimal_config())
        state = agent.start(
            agent.init(jr.key(21)),
            jnp.zeros((OBS_DIM,), dtype=jnp.float32),
        )
        malformed_state = state.replace(step_count=0)
        assert bool(agent._checkpoint_state_valid(malformed_state))

        with pytest.raises(ValueError, match="array contract"):
            save_prototype_checkpoint(
                agent,
                malformed_state,
                tmp_path / "scalar-state-leaf",
            )

    def test_checkpoint_save_rejects_numpy_replacing_jax_array_leaf(
        self,
        tmp_path,
    ) -> None:
        agent = PrototypeAgent(_minimal_config())
        state = agent.start(
            agent.init(jr.key(22)),
            jnp.zeros((OBS_DIM,), dtype=jnp.float32),
        )
        malformed_state = state.replace(
            step_count=np.asarray(0, dtype=np.int32),
        )
        assert bool(agent._checkpoint_state_valid(malformed_state))

        with pytest.raises(ValueError, match="array contract"):
            save_prototype_checkpoint(
                agent,
                malformed_state,
                tmp_path / "numpy-state-leaf",
            )

    def test_v3_checkpoint_without_empty_array_codec_remains_loadable(
        self,
        tmp_path,
    ) -> None:
        import alberta_framework.core.prototype_agent as prototype_module

        agent = PrototypeAgent(_minimal_config())
        state = agent.start(
            agent.init(jr.key(19)),
            jnp.zeros((OBS_DIM,), dtype=jnp.float32),
        )
        config = agent.to_config()
        checkpoint = tmp_path / "legacy-v3-storage"
        save_checkpoint(
            state,
            checkpoint,
            metadata={
                "schema": PROTOTYPE_CHECKPOINT_SCHEMA,
                "agent_config": config,
                "config_sha256": prototype_module._prototype_config_digest(config),
            },
        )

        restored_agent, restored_state = load_prototype_checkpoint(checkpoint)

        assert restored_agent.to_config() == config
        chex.assert_trees_all_equal(restored_state, state)

    def test_checkpoint_loader_rejects_wrong_schema_or_config_digest(
        self,
        monkeypatch,
        tmp_path,
    ) -> None:
        import alberta_framework.core.prototype_agent as prototype_module

        config = PrototypeAgent(_minimal_config()).to_config()
        monkeypatch.setattr(
            prototype_module,
            "load_checkpoint_metadata",
            lambda _path: {
                "schema": "alberta.prototype_agent.v0",
                "agent_config": config,
                "config_sha256": "unused",
            },
        )
        with pytest.raises(ValueError, match="PrototypeAgent v1"):
            load_prototype_checkpoint(tmp_path / "not-read")

        monkeypatch.setattr(
            prototype_module,
            "load_checkpoint_metadata",
            lambda _path: {
                "schema": PROTOTYPE_CHECKPOINT_SCHEMA,
                "agent_config": config,
                "config_sha256": "tampered",
            },
        )
        with pytest.raises(ValueError, match="digest"):
            load_prototype_checkpoint(tmp_path / "not-read")

        noncanonical = dict(config)
        noncanonical["dream_next_observation_mode"] = "model_prediction"
        monkeypatch.setattr(
            prototype_module,
            "load_checkpoint_metadata",
            lambda _path: {
                "schema": PROTOTYPE_CHECKPOINT_SCHEMA,
                "agent_config": noncanonical,
                "config_sha256": prototype_module._prototype_config_digest(
                    noncanonical
                ),
            },
        )
        with pytest.raises(ValueError, match="not canonical"):
            load_prototype_checkpoint(tmp_path / "not-read")

        monkeypatch.setattr(
            prototype_module,
            "load_checkpoint_metadata",
            lambda _path: {
                "schema": PROTOTYPE_CHECKPOINT_SCHEMA,
                "agent_config": config,
                "config_sha256": prototype_module._prototype_config_digest(config),
                "empty_array_codec": "unknown.empty-array-codec.v9",
            },
        )
        with pytest.raises(ValueError, match="empty-array codec"):
            load_prototype_checkpoint(tmp_path / "not-read")

        monkeypatch.setattr(
            prototype_module,
            "load_checkpoint_metadata",
            lambda _path: {
                "schema": PROTOTYPE_CHECKPOINT_SCHEMA,
                "agent_config": config,
                "config_sha256": prototype_module._prototype_config_digest(config),
                "empty_array_codec": (
                    "alberta.prototype_agent.empty_array_projection.v1"
                ),
                "prng_impl": "unknown-prng-v9",
            },
        )
        with pytest.raises(ValueError, match="PRNG implementation"):
            load_prototype_checkpoint(tmp_path / "not-read")


# ---------------------------------------------------------------------------
# Dreaming mechanics
# ---------------------------------------------------------------------------


class TestPrototypeAgentDreaming:
    def test_dreams_zero_before_warmup(self) -> None:
        """During warmup, dream TD errors should all be zero (gated)."""
        cfg = PrototypeAgentConfig(
            oak=_oak_cfg(),
            world_model=_wm_cfg(),
            dreaming=DreamingConfig(warmup_steps=10000, max_model_error_ema=1e6),
            buffer_capacity=50,
            n_dreams_per_step=3,
        )
        agent = PrototypeAgent(cfg)
        state = agent.start(agent.init(jr.key(0)), jnp.zeros(OBS_DIM))
        result = agent.update(state, jnp.array(1.0), jnp.ones(OBS_DIM))
        assert result.dream_td_errors is not None
        # All gated dreams produce 0.0
        chex.assert_trees_all_close(result.dream_td_errors, jnp.zeros(3), atol=1e-6)


# ---------------------------------------------------------------------------
# GRU Perception (Step 8 sub-component a)
# ---------------------------------------------------------------------------

GRU_OBS_DIM = 4
GRU_HIDDEN = 8
GRU_AUG_DIM = GRU_OBS_DIM + GRU_HIDDEN


def _gru_config() -> PrototypeAgentConfig:
    from alberta_framework.core.prototype_agent import GRUPerceptionConfig

    return PrototypeAgentConfig(
        oak=_oak_cfg(obs_dim=GRU_AUG_DIM),
        gru_perception=GRUPerceptionConfig(
            observation_dim=GRU_OBS_DIM,
            hidden_dim=GRU_HIDDEN,
        ),
    )


class TestGRUPerceptionConfig:
    def test_augmented_dim(self) -> None:
        from alberta_framework.core.prototype_agent import GRUPerceptionConfig

        cfg = GRUPerceptionConfig(observation_dim=4, hidden_dim=16)
        assert cfg.augmented_dim() == 20

    def test_config_roundtrip(self) -> None:
        from alberta_framework.core.prototype_agent import GRUPerceptionConfig

        cfg = GRUPerceptionConfig(observation_dim=6, hidden_dim=32)
        restored = GRUPerceptionConfig.from_config(cfg.to_config())
        assert restored.observation_dim == 6
        assert restored.hidden_dim == 32

    def test_oak_dim_mismatch_raises(self) -> None:
        from alberta_framework.core.prototype_agent import GRUPerceptionConfig

        with pytest.raises(ValueError, match="oak.observation_dim"):
            PrototypeAgentConfig(
                oak=_oak_cfg(obs_dim=4),  # wrong — should be 4+8=12
                gru_perception=GRUPerceptionConfig(observation_dim=4, hidden_dim=8),
            )

    def test_world_model_dim_mismatch_raises(self) -> None:
        from alberta_framework.core.prototype_agent import GRUPerceptionConfig

        with pytest.raises(ValueError, match="world_model.observation_dim"):
            PrototypeAgentConfig(
                oak=_oak_cfg(obs_dim=12),  # correct: 4+8
                world_model=ActionConditionedWorldModelConfig(
                    observation_dim=4,  # wrong — should be 12
                    n_actions=2,
                ),
                gru_perception=GRUPerceptionConfig(observation_dim=4, hidden_dim=8),
            )

    def test_prototype_config_roundtrip_with_gru(self) -> None:
        cfg = _gru_config()
        restored = PrototypeAgentConfig.from_config(cfg.to_config())
        assert restored.gru_perception is not None
        assert restored.gru_perception.observation_dim == GRU_OBS_DIM
        assert restored.gru_perception.hidden_dim == GRU_HIDDEN


class TestGRUPerceptionStateInit:
    def test_state_initialization_contract(self) -> None:
        agent = PrototypeAgent(_gru_config())
        state = agent.init(jr.key(0))
        assert state.gru_state is not None
        chex.assert_shape(state.gru_state.hidden, (GRU_HIDDEN,))
        assert float(jnp.max(jnp.abs(state.gru_state.hidden))) == pytest.approx(0.0)
        gru = state.gru_state
        chex.assert_shape(gru.W_z, (GRU_HIDDEN, GRU_OBS_DIM))
        chex.assert_shape(gru.U_z, (GRU_HIDDEN, GRU_HIDDEN))
        chex.assert_shape(gru.b_z, (GRU_HIDDEN,))
        assert PrototypeAgent(_minimal_config()).init(jr.key(1)).gru_state is None


class TestGRUPerceptionUpdate:
    def test_start_and_update_contract(self) -> None:
        agent = PrototypeAgent(_gru_config())
        state0 = agent.init(jr.key(0))
        obs = jr.normal(jr.key(1), (GRU_OBS_DIM,))
        state = agent.start(state0, obs)
        assert float(jnp.max(jnp.abs(state.gru_state.hidden))) > 0.0
        stored = state.oak_state.stomp_state.base_last_obs
        chex.assert_shape(stored, (GRU_AUG_DIM,))
        h0 = state.gru_state.hidden
        result = agent.update(
            state,
            jnp.array(1.0),
            jr.normal(jr.key(2), (GRU_OBS_DIM,)),
        )
        h1 = result.state.gru_state.hidden
        assert not jnp.allclose(h0, h1)
        assert jnp.isfinite(result.oak_td_error)
        assert jnp.all(jnp.isfinite(result.state.gru_state.hidden))


class TestAutoCurate:
    """Tests for auto_curate_every config field and maybe_curate() method."""

    def _agent(self, auto_curate_every: int = 0) -> tuple[PrototypeAgent, PrototypeAgentState]:
        cfg = PrototypeAgentConfig(
            oak=_oak_cfg(specs=(_SPEC0, _SPEC1)),
            auto_curate_every=auto_curate_every,
        )
        agent = PrototypeAgent(cfg)
        state = agent.start(agent.init(jr.key(0)), jnp.zeros(OBS_DIM))
        return agent, state

    def test_config_validation_and_roundtrip(self) -> None:
        cfg = PrototypeAgentConfig(oak=_oak_cfg(), auto_curate_every=50)
        cfg2 = PrototypeAgentConfig.from_config(cfg.to_config())
        assert cfg2.auto_curate_every == 50
        with pytest.raises(ValueError, match="auto_curate_every"):
            PrototypeAgentConfig(oak=_oak_cfg(), auto_curate_every=-1)

    def test_maybe_curate_noop_branches(self) -> None:
        agent, state = self._agent(auto_curate_every=0)
        new_agent, new_state = agent.maybe_curate(state, jr.key(1))
        assert new_agent is agent
        assert new_state is state

        agent, state = self._agent(auto_curate_every=10)
        new_agent, new_state = agent.maybe_curate(state, jr.key(2))
        assert new_agent is agent
        assert new_state is state
        obs = jr.normal(jr.key(7), (OBS_DIM,))
        result = agent.update(state, jnp.array(0.0), obs)
        state1 = result.state
        assert int(state1.step_count) == 1
        new_agent, new_state = agent.maybe_curate(state1, jr.key(3))
        assert new_agent is agent
        assert new_state is state1
