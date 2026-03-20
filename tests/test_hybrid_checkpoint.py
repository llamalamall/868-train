"""Tests for hybrid checkpoint bundle persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.hybrid.checkpoint import HybridCheckpointManager


class _StubController:
    def __init__(self, payload: str) -> None:
        self._payload = payload

    def save(self, path: Path) -> Path:
        path.write_text(self._payload, encoding="utf-8")
        return path


def test_save_bundle_persists_extended_hybrid_payloads(tmp_path: Path) -> None:
    run_directory = tmp_path / "bundle"
    bundle = HybridCheckpointManager.save_bundle(
        run_directory=run_directory,
        meta_controller=_StubController("meta"),
        threat_controller=_StubController("threat"),
        hybrid_config={
            "command": "train-full-hierarchical",
            "run_tag": "hybrid-full-beta-fixedmeta",
            "warmstart_checkpoint": "artifacts/hybrid/20260314-03-hybrid-beta",
            "resume_checkpoint": None,
        },
        training_state={
            "episodes_requested": 1,
            "episodes_completed": 1,
            "max_steps_per_episode": 450,
            "saved_at_utc": "2026-03-14T15:00:00Z",
            "summary": {
                "episodes": 1,
                "non_death_terminal_episodes": 1,
                "hit_step_limit_episodes": 0,
            },
            "results": [
                {
                    "episode_id": "episode-00001",
                    "hit_step_limit": False,
                    "terminal_classification": "non_death_terminal",
                    "phase_switches": 2,
                    "threat_active_steps": 0,
                }
            ],
        },
    )

    hybrid_config = json.loads(bundle.hybrid_config_path.read_text(encoding="utf-8"))
    training_state = json.loads(bundle.training_state_path.read_text(encoding="utf-8"))

    assert hybrid_config["version"] == HybridCheckpointManager.VERSION
    assert hybrid_config["run_tag"] == "hybrid-full-beta-fixedmeta"
    assert hybrid_config["warmstart_checkpoint"] == "artifacts/hybrid/20260314-03-hybrid-beta"
    assert hybrid_config["resume_checkpoint"] is None
    assert training_state["summary"]["episodes"] == 1
    assert training_state["results"][0]["terminal_classification"] == "non_death_terminal"
    assert training_state["results"][0]["phase_switches"] == 2
    assert training_state["results"][0]["threat_active_steps"] == 0


def test_load_warmstart_meta_accepts_meta_only_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_directory = tmp_path / "warmstart"
    run_directory.mkdir()
    (run_directory / "meta_controller.pt").write_text("meta", encoding="utf-8")
    (run_directory / "hybrid_config.json").write_text(
        json.dumps({"version": 1, "command": "train-meta-no-enemies"}),
        encoding="utf-8",
    )
    (run_directory / "training_state.json").write_text(
        json.dumps({"version": 1, "episodes_requested": 1, "episodes_completed": 1}),
        encoding="utf-8",
    )
    loaded_controller = object()
    monkeypatch.setattr(
        "src.hybrid.checkpoint.MetaControllerDQN.load",
        lambda path: loaded_controller,
    )

    meta_controller, hybrid_config, training_state = HybridCheckpointManager.load_warmstart_meta(
        run_directory=run_directory
    )

    assert meta_controller is loaded_controller
    assert hybrid_config["command"] == "train-meta-no-enemies"
    assert training_state["episodes_completed"] == 1
