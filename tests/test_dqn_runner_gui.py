"""Tests for DQN runner GUI helper behavior."""

from __future__ import annotations

import pytest

from src.env import dqn_policy_runner
from src.hybrid import runner as hybrid_runner
from src.gui.dqn_runner_gui import (
    _HYBRID_CHECKPOINT_DIR,
    _SMOKE_TEST_REWARD_DESTS,
    _CHECKPOINT_DIR,
    _estimate_epsilon_eta_seconds,
    _format_duration_seconds,
    _initial_browse_dir,
    _is_boolean_flag,
    _get_subparser,
    _iter_parser_actions,
    _parse_episode_progress,
    _resolve_reward_metric_value,
    _run_dqn_preset_overrides,
    _run_hybrid_preset_overrides,
    _strip_textual_markup,
    _sort_form_actions,
)
from src.training.rewards import RewardWeights


def test_sort_form_actions_prioritizes_exe_and_checkpoint() -> None:
    parser = dqn_policy_runner._build_parser()
    sorted_actions = _sort_form_actions(_iter_parser_actions(parser))
    first_three_dests = [action.dest for action in sorted_actions[:3]]
    assert first_three_dests == ["exe", "checkpoint", "restore_save_file"]


def test_initial_browse_dir_uses_checkpoint_directory_for_checkpoint_fields() -> None:
    assert _initial_browse_dir(dest="checkpoint", current_value="") == _CHECKPOINT_DIR
    assert _initial_browse_dir(dest="checkpoint_a", current_value="") == _CHECKPOINT_DIR
    assert _initial_browse_dir(dest="checkpoint_b", current_value="") == _CHECKPOINT_DIR


def test_initial_browse_dir_uses_hybrid_checkpoint_directory_for_hybrid_paths() -> None:
    assert _initial_browse_dir(dest="checkpoint_root", current_value="") == _HYBRID_CHECKPOINT_DIR
    assert _initial_browse_dir(dest="resume_checkpoint", current_value="") == _HYBRID_CHECKPOINT_DIR
    assert _initial_browse_dir(dest="warmstart_checkpoint", current_value="") == _HYBRID_CHECKPOINT_DIR


def test_no_enemies_action_is_treated_as_boolean_flag() -> None:
    parser = dqn_policy_runner._build_parser()
    action_by_dest = {action.dest: action for action in _iter_parser_actions(parser)}

    assert _is_boolean_flag(action_by_dest["no_enemies"]) is True
    assert _is_boolean_flag(action_by_dest["episodes"]) is False


def test_run_dqn_presets_include_expected_profiles() -> None:
    presets = _run_dqn_preset_overrides()
    assert "defaults" in presets
    assert "reward survival" in presets
    assert "reward exploration" in presets
    assert "phase progression (no enemies)" in presets
    assert "smoke test - siphon objective" in presets
    assert "smoke test - enemy objective" in presets
    assert "smoke test - exit objective" in presets


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
    assert "eval quick" in evaluate


def test_phase_progression_profile_ignores_enemy_rewards() -> None:
    presets = _run_dqn_preset_overrides()
    profile = presets["phase progression (no enemies)"]

    assert profile["mode"] == "train"
    assert profile["no_enemies"] is True
    assert profile["reward_enemy_damaged"] == pytest.approx(0.0)
    assert profile["reward_enemy_cleared"] == pytest.approx(0.0)
    assert profile["reward_phase_progress"] > 0.0


def test_smoke_test_presets_zero_all_rewards_except_target_objective() -> None:
    presets = _run_dqn_preset_overrides()
    default_weights = RewardWeights()
    smoke_profiles = {
        "smoke test - siphon objective": ("reward_siphon_collected", default_weights.siphon_collected),
        "smoke test - enemy objective": ("reward_enemy_cleared", default_weights.enemy_cleared),
        "smoke test - exit objective": ("reward_map_clear_bonus", default_weights.map_clear_bonus),
    }

    for profile_name, (active_reward_dest, active_reward_value) in smoke_profiles.items():
        profile = presets[profile_name]
        assert profile["episodes"] == 5
        for reward_dest in _SMOKE_TEST_REWARD_DESTS:
            expected_value = active_reward_value if reward_dest == active_reward_dest else 0.0
            assert profile[reward_dest] == pytest.approx(expected_value)


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


def test_resolve_reward_metric_value_uses_reward_line_total_when_training_waits() -> None:
    reward_value = _resolve_reward_metric_value(
        training_line="episode=1 step=2 total=1.200 waiting=step",
        reward_line="reward total=+0.420 survival=+0.050",
        previous_value="-",
    )
    assert reward_value == "+0.420"


def test_strip_textual_markup_removes_color_tokens_from_board_text() -> None:
    raw = "map [yellow]#[/] [bright_white]P[/] [magenta]E[/]"
    assert _strip_textual_markup(raw) == "map # P E"


def test_hybrid_gui_action_discovery_includes_meta_reward_weight_flags() -> None:
    parser = hybrid_runner._build_parser()
    movement_parser = _get_subparser(parser, command_name="movement-test")
    discovered = {action.dest for action in _iter_parser_actions(movement_parser)}

    assert "restore_save_file" in discovered
    assert "restore_save_delay" in discovered
    assert "meta_reward_objective_complete" in discovered
    assert "meta_reward_phase_progress" in discovered
    assert "meta_reward_step_cost" in discovered
    assert "meta_reward_premature_exit_penalty" in discovered
    assert "meta_reward_sector_advance" in discovered
