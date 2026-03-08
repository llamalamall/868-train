"""Tests for DQN runner GUI helper behavior."""

from __future__ import annotations

import pytest

from src.env import dqn_policy_runner
from src.gui.dqn_runner_gui import (
    _CHECKPOINT_DIR,
    _estimate_epsilon_eta_seconds,
    _format_duration_seconds,
    _initial_browse_dir,
    _iter_parser_actions,
    _parse_episode_progress,
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


def test_parse_episode_progress_handles_dqn_episode_id_with_fallback_total() -> None:
    current, total = _parse_episode_progress("episode-00007", fallback_total=20)
    assert current == 7
    assert total == 20


def test_parse_episode_progress_handles_compare_fraction() -> None:
    current, total = _parse_episode_progress("3/10", fallback_total=99)
    assert current == 3
    assert total == 10


def test_format_duration_seconds_uses_compact_clock_text() -> None:
    assert _format_duration_seconds(None) == "-"
    assert _format_duration_seconds(4.1) == "4s"
    assert _format_duration_seconds(125.0) == "2m 05s"
    assert _format_duration_seconds(3720.0) == "1h 02m"


def test_estimate_epsilon_eta_seconds_computes_linear_decay_remaining_time() -> None:
    eta = _estimate_epsilon_eta_seconds(
        current_epsilon=0.5,
        epsilon_start=0.8,
        epsilon_end=0.2,
        epsilon_decay_steps=600,
        seconds_per_step=0.5,
    )
    assert eta == pytest.approx(150.0)
