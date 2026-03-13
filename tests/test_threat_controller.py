"""Tests for threat controller checkpoint compatibility."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.hybrid.threat_controller import ThreatControllerDRQN


def test_threat_load_ignores_legacy_sequence_length_in_checkpoint(
    monkeypatch: Any, tmp_path: Path
) -> None:
    checkpoint_path = tmp_path / "threat_drqn.pt"
    checkpoint_path.write_bytes(b"legacy")
    payload = {
        "seed": 123,
        "config": {
            "gamma": 0.9,
            "feature_count": 24,
            "hidden_size": 80,
            "sequence_length": 6,
        },
    }

    class _FakeTorch:
        @staticmethod
        def load(path: Path, map_location: str = "cpu") -> dict[str, Any]:
            assert path == checkpoint_path
            assert map_location == "cpu"
            return payload

    captured: dict[str, Any] = {}

    def _fake_require_torch() -> tuple[Any, Any, Any]:
        return (_FakeTorch(), object(), object())

    def _fake_init(self: ThreatControllerDRQN, *, config: Any, seed: int | None = None) -> None:
        captured["config"] = config
        captured["seed"] = seed
        self.config = config
        self._seed = seed

    def _fake_load_checkpoint_payload(self: ThreatControllerDRQN, raw_payload: dict[str, Any]) -> None:
        captured["payload"] = raw_payload

    monkeypatch.setattr("src.hybrid.threat_controller._require_torch", _fake_require_torch)
    monkeypatch.setattr(ThreatControllerDRQN, "__init__", _fake_init)
    monkeypatch.setattr(ThreatControllerDRQN, "load_checkpoint_payload", _fake_load_checkpoint_payload)

    controller = ThreatControllerDRQN.load(checkpoint_path)

    assert controller.config.gamma == 0.9
    assert controller.config.feature_count == 24
    assert controller.config.hidden_size == 80
    assert not hasattr(controller.config, "sequence_length")
    assert captured["seed"] == 123
    assert captured["payload"] is payload


def test_threat_load_checkpoint_payload_accepts_version_2() -> None:
    controller = ThreatControllerDRQN.__new__(ThreatControllerDRQN)

    class _Loader:
        def __init__(self) -> None:
            self.loaded: dict[str, Any] | None = None

        def load_state_dict(self, state_dict: dict[str, Any]) -> None:
            self.loaded = state_dict

    controller._online = _Loader()
    controller._target = _Loader()
    controller._optimizer = _Loader()
    controller._hidden_state = "stale"
    controller._replay = []
    controller._replay_cursor = 0
    controller._total_env_steps = 0
    controller._optimization_steps = 0
    controller._episodes_seen = 0

    class _Config:
        replay_capacity = 10

    controller.config = _Config()
    controller._normalize_features = lambda values: tuple(float(value) for value in values)

    payload = {
        "version": 2,
        "online_state_dict": {"online": 1},
        "target_state_dict": {"target": 2},
        "optimizer_state_dict": {"optimizer": 3},
        "training_state": {
            "total_env_steps": 7,
            "optimization_steps": 4,
            "episodes_seen": 2,
            "replay_cursor": 1,
        },
        "replay": [
            {
                "features": [1, 2],
                "action_index": 0,
                "reward": 1.5,
                "next_features": [3, 4],
                "done": False,
                "next_valid_action_indices": [0, 1],
            }
        ],
    }

    controller.load_checkpoint_payload(payload)

    assert controller._online.loaded == {"online": 1}
    assert controller._target.loaded == {"target": 2}
    assert controller._optimizer.loaded == {"optimizer": 3}
    assert controller._total_env_steps == 7
    assert controller._optimization_steps == 4
    assert controller._episodes_seen == 2
    assert controller._replay_cursor == 0
    assert controller._hidden_state is None
    assert len(controller._replay) == 1
    assert controller._replay[0].features == (1.0, 2.0)


def test_threat_load_checkpoint_payload_rejects_unknown_version() -> None:
    controller = ThreatControllerDRQN.__new__(ThreatControllerDRQN)

    with pytest.raises(ValueError, match="Unsupported threat checkpoint version 3"):
        controller.load_checkpoint_payload({"version": 3})
