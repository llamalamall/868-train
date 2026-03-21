"""Tests for Hybrid runner GUI helper behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.gui.hybrid_runner_gui import (
    _AUTO_LATEST_BETA_META_CHECKPOINT,
    _HYBRID_CHECKPOINT_DIR,
    _initial_browse_dir,
    _is_live_monitor_runner_module,
    _latest_completed_meta_checkpoint,
    _monitor_action_card_values,
    _run_hybrid_preset_overrides,
)


def _write_hybrid_run(
    root: Path,
    *,
    name: str,
    command: str,
    saved_at_utc: str,
    episodes_requested: int = 10,
    episodes_completed: int = 10,
) -> Path:
    run_dir = root / name
    run_dir.mkdir()
    (run_dir / "hybrid_config.json").write_text(
        json.dumps({"version": 1, "command": command}),
        encoding="utf-8",
    )
    (run_dir / "training_state.json").write_text(
        json.dumps(
            {
                "version": 1,
                "episodes_requested": episodes_requested,
                "episodes_completed": episodes_completed,
                "saved_at_utc": saved_at_utc,
                "results": [],
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_initial_browse_dir_uses_hybrid_checkpoint_directory_for_hybrid_paths() -> None:
    assert _initial_browse_dir(dest="checkpoint", current_value="") == _HYBRID_CHECKPOINT_DIR
    assert _initial_browse_dir(dest="checkpoint_root", current_value="") == _HYBRID_CHECKPOINT_DIR
    assert _initial_browse_dir(dest="resume_checkpoint", current_value="") == _HYBRID_CHECKPOINT_DIR
    assert _initial_browse_dir(dest="warmstart_checkpoint", current_value="") == _HYBRID_CHECKPOINT_DIR


def test_run_hybrid_presets_include_expected_profiles() -> None:
    movement = _run_hybrid_preset_overrides(command_name="movement-test")
    meta = _run_hybrid_preset_overrides(command_name="train-meta-no-enemies")
    full = _run_hybrid_preset_overrides(command_name="train-full-hierarchical")
    evaluate = _run_hybrid_preset_overrides(command_name="eval-hybrid")

    assert "defaults" in movement
    assert "gate a smoke" in movement
    assert "defaults" in meta
    assert "gate b baseline" in meta
    assert "defaults" in full
    assert "gate c baseline" in full
    assert "defaults" in evaluate
    assert "beta verification" in evaluate


def test_latest_completed_meta_checkpoint_prefers_newest_completed_run(tmp_path: Path, monkeypatch) -> None:
    older = _write_hybrid_run(
        tmp_path,
        name="20260314-03-hybrid-beta",
        command="train-meta-no-enemies",
        saved_at_utc="2026-03-14T03:00:00Z",
    )
    newer = _write_hybrid_run(
        tmp_path,
        name="20260314-07-hybrid-beta",
        command="train-meta-no-enemies",
        saved_at_utc="2026-03-14T07:00:00Z",
    )
    _write_hybrid_run(
        tmp_path,
        name="20260314-08-hybrid-incomplete",
        command="train-meta-no-enemies",
        saved_at_utc="2026-03-14T08:00:00Z",
        episodes_requested=20,
        episodes_completed=10,
    )

    monkeypatch.setattr("src.gui.hybrid_runner_gui._HYBRID_CHECKPOINT_DIR", tmp_path)

    assert _latest_completed_meta_checkpoint() == newer
    assert older != newer


def test_latest_completed_meta_checkpoint_raises_when_none_found(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("src.gui.hybrid_runner_gui._HYBRID_CHECKPOINT_DIR", tmp_path)

    with pytest.raises(ValueError, match="No completed train-meta-no-enemies hybrid checkpoints found"):
        _latest_completed_meta_checkpoint()


def test_hybrid_full_presets_use_auto_latest_beta_meta_checkpoint() -> None:
    presets = _run_hybrid_preset_overrides(command_name="train-full-hierarchical")

    assert presets["beta full fixed meta"]["warmstart_checkpoint"] == _AUTO_LATEST_BETA_META_CHECKPOINT
    assert presets["beta full long warmup"]["warmstart_checkpoint"] == _AUTO_LATEST_BETA_META_CHECKPOINT


def test_monitor_action_card_values_cover_hybrid_payload() -> None:
    assert _monitor_action_card_values(
        "action=move_up phase=collect_siphons next_target=(3,4) reason=scripted_phase_only"
    ) == {
        "action": "move_up",
        "reason": "scripted_phase_only",
        "phase": "collect_siphons",
        "target": "(3,4)",
        "loss": "-",
    }


def test_is_live_monitor_runner_module_accepts_hybrid_runner() -> None:
    assert _is_live_monitor_runner_module("src.hybrid.runner", ("movement-test",)) is True
    assert _is_live_monitor_runner_module("src.memory.state_monitor_tui", ()) is False
