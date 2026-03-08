"""Tests for DQN runner GUI helper behavior."""

from __future__ import annotations

from src.env import dqn_policy_runner
from src.gui.dqn_runner_gui import (
    _CHECKPOINT_DIR,
    _initial_browse_dir,
    _iter_parser_actions,
    _run_dqn_preset_overrides,
    _sort_form_actions,
)


def test_sort_form_actions_prioritizes_exe_and_checkpoint() -> None:
    parser = dqn_policy_runner._build_parser()
    sorted_actions = _sort_form_actions(_iter_parser_actions(parser))
    first_two_dests = [action.dest for action in sorted_actions[:2]]
    assert first_two_dests == ["exe", "checkpoint"]


def test_initial_browse_dir_uses_checkpoint_directory_for_checkpoint_fields() -> None:
    assert _initial_browse_dir(dest="checkpoint", current_value="") == _CHECKPOINT_DIR
    assert _initial_browse_dir(dest="checkpoint_a", current_value="") == _CHECKPOINT_DIR
    assert _initial_browse_dir(dest="checkpoint_b", current_value="") == _CHECKPOINT_DIR


def test_run_dqn_presets_include_expected_profiles() -> None:
    presets = _run_dqn_preset_overrides()
    assert "defaults" in presets
    assert "reward survival" in presets
    assert "reward exploration" in presets
