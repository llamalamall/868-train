"""Tests for hybrid bundle versioning and weights-only warmstart semantics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.hybrid.checkpoint import HybridCheckpointManager
from src.hybrid.meta_controller import MetaControllerDQN, MetaDQNConfig
from src.hybrid.threat_controller import ThreatControllerDRQN, ThreatDRQNConfig
from src.hybrid.types import ObjectivePhase, ThreatOverride


def _meta_controller(*, seed: int = 7) -> MetaControllerDQN:
    controller = MetaControllerDQN(
        config=MetaDQNConfig(
            feature_count=4,
            hidden_size=8,
            replay_capacity=16,
            min_replay_size=32,
            batch_size=2,
            target_sync_interval=4,
            epsilon_start=0.60,
            epsilon_end=0.05,
            epsilon_decay_steps=100,
        ),
        seed=seed,
    )
    controller.start_episode()
    controller.observe(
        features=(1.0, 0.0, 0.0, 0.0),
        chosen_phase=ObjectivePhase.COLLECT_SIPHONS,
        reward=0.5,
        next_features=(0.0, 1.0, 0.0, 0.0),
        done=False,
        next_allowed_phases=(ObjectivePhase.COLLECT_SIPHONS, ObjectivePhase.EXIT_SECTOR),
    )
    return controller


def _threat_controller(*, seed: int = 11) -> ThreatControllerDRQN:
    controller = ThreatControllerDRQN(
        config=ThreatDRQNConfig(
            feature_count=5,
            hidden_size=8,
            replay_capacity=16,
            min_replay_size=32,
            batch_size=2,
            target_sync_interval=4,
            epsilon_start=0.80,
            epsilon_end=0.05,
            epsilon_decay_steps=100,
            sequence_length=4,
        ),
        seed=seed,
    )
    controller.start_episode()
    controller.observe(
        features=(1.0, 0.0, 0.0, 0.0, 0.0),
        chosen_override=ThreatOverride.EVADE,
        reward=-0.25,
        next_features=(0.0, 1.0, 0.0, 0.0, 0.0),
        done=False,
        next_allowed_overrides=(ThreatOverride.ROUTE_DEFAULT, ThreatOverride.EVADE),
    )
    return controller


def test_hybrid_bundle_resume_load_preserves_saved_training_state(tmp_path: Path) -> None:
    meta = _meta_controller()
    threat = _threat_controller()
    bundle = HybridCheckpointManager.save_bundle(
        run_directory=tmp_path / "bundle",
        meta_controller=meta,
        threat_controller=threat,
        hybrid_config={
            "meta_config": {"feature_count": meta.feature_count, "epsilon_start": meta.config.epsilon_start},
            "threat_config": {
                "feature_count": threat.feature_count,
                "epsilon_start": threat.config.epsilon_start,
            },
        },
        training_state={"episode_results": [{"steps": 3}]},
    )

    loaded_meta, loaded_threat, hybrid_config, training_state = HybridCheckpointManager.load_bundle(
        run_directory=bundle.run_directory,
        for_training=True,
    )

    assert loaded_meta.training_snapshot()["total_env_steps"] == 1
    assert loaded_meta.training_snapshot()["replay_size"] == 1
    assert loaded_threat.training_snapshot()["total_env_steps"] == 1
    assert loaded_threat.training_snapshot()["replay_transition_count"] == 1
    assert hybrid_config["meta_config"]["feature_count"] == 4
    assert hybrid_config["threat_config"]["feature_count"] == 5
    assert training_state["episode_results"] == [{"steps": 3}]


def test_hybrid_bundle_legacy_version_is_evaluation_only_for_training_loads(tmp_path: Path) -> None:
    meta = _meta_controller()
    threat = _threat_controller()
    bundle = HybridCheckpointManager.save_bundle(
        run_directory=tmp_path / "bundle",
        meta_controller=meta,
        threat_controller=threat,
        hybrid_config={"label": "legacy-check"},
        training_state={"episodes": 1},
    )

    hybrid_payload = json.loads(bundle.hybrid_config_path.read_text(encoding="utf-8"))
    hybrid_payload["version"] = 1
    bundle.hybrid_config_path.write_text(json.dumps(hybrid_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="evaluation-only"):
        HybridCheckpointManager.load_bundle(run_directory=bundle.run_directory, for_training=True)

    loaded_meta, loaded_threat, _hybrid_config, _training_state = HybridCheckpointManager.load_bundle(
        run_directory=bundle.run_directory,
        for_training=False,
    )
    assert loaded_meta.feature_count == 4
    assert loaded_threat.feature_count == 5


def test_meta_warmstart_copy_transfers_weights_without_training_state(tmp_path: Path) -> None:
    del tmp_path
    source = _meta_controller(seed=23)
    source_snapshot = source.training_snapshot()
    sample = (0.2, 0.4, 0.6, 0.8)
    source_q = source._online.q_values(sample)

    fresh = MetaControllerDQN(
        config=MetaDQNConfig(
            feature_count=4,
            hidden_size=8,
            replay_capacity=5,
            min_replay_size=32,
            batch_size=2,
            target_sync_interval=4,
            epsilon_start=0.25,
            epsilon_end=0.05,
            epsilon_decay_steps=10,
        ),
        seed=23,
    )
    before_copy = fresh.training_snapshot()

    fresh.copy_weights_from(source)
    after_copy = fresh.training_snapshot()
    fresh_q = fresh._online.q_values(sample)

    assert source_snapshot["total_env_steps"] == 1
    assert source_snapshot["replay_size"] == 1
    assert before_copy["total_env_steps"] == 0
    assert before_copy["replay_size"] == 0
    assert after_copy["total_env_steps"] == 0
    assert after_copy["replay_size"] == 0
    assert after_copy["episodes_seen"] == 0
    assert after_copy["config"]["replay_capacity"] == 5
    assert after_copy["epsilon"] == pytest.approx(0.25)
    assert fresh_q == pytest.approx(source_q)
