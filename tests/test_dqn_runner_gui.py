"""Tests for DQN runner GUI helper behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.env import dqn_policy_runner
from src.hybrid import runner as hybrid_runner
from src.gui.dqn_runner_gui import (
    _AUTO_LATEST_BETA_META_CHECKPOINT,
    _HYBRID_CHECKPOINT_DIR,
    _CHECKPOINT_DIR,
    _LiveMonitorSessionBinding,
    _RunningPythonProcess,
    _REWARD_HISTORY_LIMIT,
    _SMOKE_TEST_REWARD_DESTS,
    _discover_live_monitor_session_binding,
    _estimate_epsilon_eta_seconds,
    _extract_cli_option_value,
    _format_epsilon_progress_text,
    _format_phase_breakdown_tooltip,
    _format_reward_breakdown_tooltip,
    _format_duration_seconds,
    _initial_browse_dir,
    _is_boolean_flag,
    _get_subparser,
    _iter_parser_actions,
    _monitor_action_card_values,
    _monitor_key_label_for_action,
    _parse_next_available_actions,
    _parse_episode_progress,
    _resolve_reward_metric_value,
    _resolve_preset_overrides,
    _run_dqn_preset_overrides,
    _run_hybrid_preset_overrides,
    _select_live_monitor_binding,
    _split_windows_command_line,
    _strip_textual_markup,
    _sort_form_actions,
)
from src.training.rewards import RewardWeights


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
        json.dumps(
            {
                "version": 1,
                "command": command,
            }
        ),
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
    assert "beta efficient warmstart" in meta
    assert "gamma logic rerun" in meta
    assert "efficient anti-churn" in meta
    assert "meta ack sweep balanced" in meta
    assert "meta ack sweep conservative" in meta
    assert "meta ack sweep fast poll" in meta
    assert "beta efficient conservative ack" in meta
    assert "defaults" in full
    assert "gate c baseline" in full
    assert "beta full fixed meta" in full
    assert "beta full long warmup" in full
    assert "defaults" in evaluate
    assert "eval quick" in evaluate
    assert "beta verification" in evaluate


def test_beta_meta_preset_uses_expected_efficiency_settings() -> None:
    presets = _run_hybrid_preset_overrides(command_name="train-meta-no-enemies")
    profile = presets["beta efficient warmstart"]

    assert profile == {
        "episodes": 120,
        "max_steps": 320,
        "no_enemies": True,
        "meta_learning_rate": 0.0005,
        "meta_epsilon_decay_steps": 3000,
        "phase_lock_min_steps": 6,
        "target_stall_release_steps": 4,
        "run_tag": "hybrid-meta-beta-efficient",
    }


def test_gamma_logic_rerun_profile_matches_latest_gamma_hyperparameters() -> None:
    presets = _run_hybrid_preset_overrides(command_name="train-meta-no-enemies")
    profile = presets["gamma logic rerun"]

    assert profile == {
        "episodes": 120,
        "max_steps": 350,
        "no_enemies": True,
        "meta_gamma": 0.99,
        "meta_learning_rate": 0.001,
        "meta_epsilon_start": 0.6,
        "meta_epsilon_end": 0.05,
        "meta_epsilon_decay_steps": 5000,
        "phase_lock_min_steps": 6,
        "target_stall_release_steps": 4,
        "run_tag": "hybrid-meta-gamma-fixedlogic",
    }


def test_efficient_anti_churn_profile_sets_phase_lock_knobs() -> None:
    presets = _run_hybrid_preset_overrides(command_name="train-meta-no-enemies")
    profile = presets["efficient anti-churn"]

    assert profile == {
        "episodes": 120,
        "max_steps": 320,
        "no_enemies": True,
        "meta_learning_rate": 0.0005,
        "meta_epsilon_decay_steps": 3000,
        "phase_lock_min_steps": 6,
        "target_stall_release_steps": 4,
        "run_tag": "hybrid-meta-efficient-fixedlogic",
    }


def test_meta_ack_sweep_balanced_profile_sets_timing_knobs() -> None:
    presets = _run_hybrid_preset_overrides(command_name="train-meta-no-enemies")
    profile = presets["meta ack sweep balanced"]

    assert profile == {
        "episodes": 30,
        "max_steps": 320,
        "no_enemies": True,
        "meta_learning_rate": 0.0005,
        "meta_epsilon_decay_steps": 3000,
        "phase_lock_min_steps": 6,
        "target_stall_release_steps": 4,
        "post_action_delay": 0.03,
        "action_ack_timeout": 0.50,
        "action_ack_poll_interval": 0.02,
        "action_ack_backoff_max_level": 0,
        "run_tag": "hybrid-meta-ack-balanced",
    }


def test_meta_ack_sweep_conservative_profile_sets_timing_knobs() -> None:
    presets = _run_hybrid_preset_overrides(command_name="train-meta-no-enemies")
    profile = presets["meta ack sweep conservative"]

    assert profile == {
        "episodes": 30,
        "max_steps": 320,
        "no_enemies": True,
        "meta_learning_rate": 0.0005,
        "meta_epsilon_decay_steps": 3000,
        "phase_lock_min_steps": 6,
        "target_stall_release_steps": 4,
        "post_action_delay": 0.05,
        "action_ack_timeout": 0.70,
        "action_ack_poll_interval": 0.02,
        "action_ack_backoff_max_level": 0,
        "run_tag": "hybrid-meta-ack-conservative",
    }


def test_meta_ack_sweep_fast_poll_profile_sets_timing_knobs() -> None:
    presets = _run_hybrid_preset_overrides(command_name="train-meta-no-enemies")
    profile = presets["meta ack sweep fast poll"]

    assert profile == {
        "episodes": 30,
        "max_steps": 320,
        "no_enemies": True,
        "meta_learning_rate": 0.0005,
        "meta_epsilon_decay_steps": 3000,
        "phase_lock_min_steps": 6,
        "target_stall_release_steps": 4,
        "post_action_delay": 0.02,
        "action_ack_timeout": 0.50,
        "action_ack_poll_interval": 0.01,
        "action_ack_backoff_max_level": 0,
        "run_tag": "hybrid-meta-ack-fastpoll",
    }


def test_beta_efficient_conservative_ack_profile_sets_promoted_timing_knobs() -> None:
    presets = _run_hybrid_preset_overrides(command_name="train-meta-no-enemies")
    profile = presets["beta efficient conservative ack"]

    assert profile == {
        "episodes": 120,
        "max_steps": 320,
        "no_enemies": True,
        "meta_learning_rate": 0.0005,
        "meta_epsilon_decay_steps": 3000,
        "phase_lock_min_steps": 6,
        "target_stall_release_steps": 4,
        "post_action_delay": 0.05,
        "action_ack_timeout": 0.70,
        "action_ack_poll_interval": 0.02,
        "action_ack_backoff_max_level": 0,
        "run_tag": "hybrid-meta-beta-efficient-conservative-ack",
    }


def test_beta_full_presets_resolve_latest_beta_meta_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    older_beta = _write_hybrid_run(
        tmp_path,
        name="20260314-03-hybrid-beta",
        command="train-meta-no-enemies",
        saved_at_utc="2026-03-14T14:14:50Z",
    )
    latest_beta = _write_hybrid_run(
        tmp_path,
        name="20260314-07-hybrid-beta",
        command="train-meta-no-enemies",
        saved_at_utc="2026-03-14T17:20:00Z",
    )
    _write_hybrid_run(
        tmp_path,
        name="20260314-04-hybrid-full-beta",
        command="train-full-hierarchical",
        saved_at_utc="2026-03-14T14:46:10Z",
    )
    monkeypatch.setattr("src.gui.dqn_runner_gui._HYBRID_CHECKPOINT_DIR", tmp_path)

    presets = _run_hybrid_preset_overrides(command_name="train-full-hierarchical")
    fixed_meta = _resolve_preset_overrides(overrides=presets["beta full fixed meta"])
    long_warmup = _resolve_preset_overrides(overrides=presets["beta full long warmup"])

    assert fixed_meta["warmstart_checkpoint"] == str(latest_beta)
    assert fixed_meta["episodes"] == 320
    assert fixed_meta["joint_finetune"] is False
    assert fixed_meta["meta_freeze_episodes"] == 0
    assert fixed_meta["threat_learning_rate"] == pytest.approx(0.0007)
    assert fixed_meta["threat_epsilon_start"] == pytest.approx(0.80)
    assert fixed_meta["threat_epsilon_end"] == pytest.approx(0.05)
    assert fixed_meta["threat_epsilon_decay_steps"] == 15000
    assert fixed_meta["phase_lock_min_steps"] == 6
    assert fixed_meta["target_stall_release_steps"] == 4
    assert fixed_meta["run_tag"] == "hybrid-full-beta-fixedmeta"
    assert long_warmup["warmstart_checkpoint"] == str(latest_beta)
    assert long_warmup["joint_finetune"] is True
    assert long_warmup["meta_freeze_episodes"] == 80
    assert long_warmup["phase_lock_min_steps"] == 6
    assert long_warmup["target_stall_release_steps"] == 4
    assert long_warmup["run_tag"] == "hybrid-full-beta-longwarmup"
    assert older_beta != latest_beta


def test_resolve_preset_overrides_falls_back_to_latest_non_beta_meta_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_hybrid_run(
        tmp_path,
        name="20260314-03-hybrid-beta",
        command="train-meta-no-enemies",
        saved_at_utc="2026-03-14T14:14:50Z",
        episodes_requested=120,
        episodes_completed=60,
    )
    latest_non_beta = _write_hybrid_run(
        tmp_path,
        name="20260314-02-hybrid-quicktune",
        command="train-meta-no-enemies",
        saved_at_utc="2026-03-14T15:00:00Z",
    )
    monkeypatch.setattr("src.gui.dqn_runner_gui._HYBRID_CHECKPOINT_DIR", tmp_path)

    resolved = _resolve_preset_overrides(
        overrides={"warmstart_checkpoint": _AUTO_LATEST_BETA_META_CHECKPOINT}
    )

    assert resolved["warmstart_checkpoint"] == str(latest_non_beta)


def test_resolve_preset_overrides_raises_when_no_completed_meta_checkpoint_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("src.gui.dqn_runner_gui._HYBRID_CHECKPOINT_DIR", tmp_path)

    with pytest.raises(ValueError, match="No completed train-meta-no-enemies hybrid checkpoints found"):
        _resolve_preset_overrides(overrides={"warmstart_checkpoint": _AUTO_LATEST_BETA_META_CHECKPOINT})


def test_beta_eval_preset_uses_verification_settings() -> None:
    presets = _run_hybrid_preset_overrides(command_name="eval-hybrid")
    profile = presets["beta verification"]

    assert profile == {
        "episodes": 30,
        "max_steps": 450,
        "no_enemies": False,
        "phase_lock_min_steps": 6,
        "target_stall_release_steps": 4,
    }


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


def test_format_epsilon_progress_text_includes_end_target() -> None:
    assert _format_epsilon_progress_text(current_epsilon=0.6, epsilon_end=0.05) == "60.0% -> 5.0%"


def test_format_epsilon_progress_text_falls_back_to_end_target_when_current_missing() -> None:
    assert _format_epsilon_progress_text(current_epsilon=None, epsilon_end=0.05) == "end 5.0%"


def test_resolve_reward_metric_value_uses_reward_line_total_when_training_waits() -> None:
    reward_value = _resolve_reward_metric_value(
        training_line="episode=1 step=2 total=1.200 waiting=step",
        reward_line="reward total=+0.420 survival=+0.050",
        previous_value="-",
    )
    assert reward_value == "+0.420"


def test_monitor_action_card_values_cover_dqn_and_hybrid_payloads() -> None:
    assert _monitor_action_card_values("action=move_up reason=dqn_select_action loss=0.012345") == {
        "action": "move_up",
        "reason": "dqn_select_action",
        "phase": "-",
        "target": "-",
        "loss": "0.012345",
    }
    assert _monitor_action_card_values(
        "action=move_right phase=collect_siphons next_target=(3,4) reason=scripted_phase_only"
    ) == {
        "action": "move_right",
        "reason": "scripted_phase_only",
        "phase": "collect_siphons",
        "target": "(3,4)",
        "loss": "-",
    }


def test_parse_next_available_actions_preserves_full_input_set() -> None:
    parsed = _parse_next_available_actions(
        "next_available_actions=move_up,move_left,move_right,space,prog_slot_1,prog_slot_9,prog_slot_10"
    )

    assert parsed == (
        "move_up",
        "move_left",
        "move_right",
        "space",
        "prog_slot_1",
        "prog_slot_9",
        "prog_slot_10",
    )


def test_monitor_key_label_for_action_maps_to_visual_keycaps() -> None:
    assert _monitor_key_label_for_action("move_down") == "DOWN"
    assert _monitor_key_label_for_action("prog_slot_10") == "0"
    assert _monitor_key_label_for_action("confirm") is None


def test_split_windows_command_line_preserves_quoted_paths() -> None:
    command_line = (
        'python -m src.hybrid.runner movement-test --external-status-file '
        '"C:\\Users\\John White\\AppData\\Local\\Temp\\status file.json"'
    )

    parsed = _split_windows_command_line(command_line)

    assert parsed == (
        "python",
        "-m",
        "src.hybrid.runner",
        "movement-test",
        "--external-status-file",
        "C:\\Users\\John White\\AppData\\Local\\Temp\\status file.json",
    )


def test_extract_cli_option_value_supports_equals_syntax() -> None:
    arguments = (
        "compare",
        "--external-status-file=C:\\tmp\\compare-status.json",
        "--checkpoint-a",
        "a.json",
    )

    assert _extract_cli_option_value(arguments, "--external-status-file") == "C:\\tmp\\compare-status.json"


def test_discover_live_monitor_session_binding_uses_runner_args_when_present() -> None:
    processes = (
        _RunningPythonProcess(
            pid=4200,
            parent_pid=None,
            executable_name="python.exe",
            command_line=(
                'python -m src.hybrid.runner movement-test --exe 868-HACK.exe '
                '--external-status-file "C:\\tmp\\status.json" '
                '--external-control-file "C:\\tmp\\control.json"'
            ),
        ),
    )

    binding = _discover_live_monitor_session_binding(
        processes,
        preferred_executable_name="868-HACK.exe",
    )

    assert binding == _LiveMonitorSessionBinding(
        runner_pid=4200,
        runner_module="src.hybrid.runner",
        status_file=Path("C:/tmp/status.json"),
        control_file=Path("C:/tmp/control.json"),
        executable_name="868-HACK.exe",
        source_pid=4200,
        source_module="src.hybrid.runner",
    )


def test_discover_live_monitor_session_binding_falls_back_to_child_tui_process() -> None:
    processes = (
        _RunningPythonProcess(
            pid=4300,
            parent_pid=None,
            executable_name="python.exe",
            command_line='python -m src.training.evaluate compare --exe 868-HACK.exe',
        ),
        _RunningPythonProcess(
            pid=4305,
            parent_pid=4300,
            executable_name="python.exe",
            command_line=(
                'python -m src.memory.state_monitor_tui --exe 868-HACK.exe '
                '--external-status-file "C:\\tmp\\compare-status.json"'
            ),
        ),
    )

    binding = _discover_live_monitor_session_binding(
        processes,
        preferred_runner_pid=4300,
        preferred_executable_name="868-HACK.exe",
    )

    assert binding == _LiveMonitorSessionBinding(
        runner_pid=4300,
        runner_module="src.training.evaluate",
        status_file=Path("C:/tmp/compare-status.json"),
        control_file=None,
        executable_name="868-HACK.exe",
        source_pid=4305,
        source_module="src.memory.state_monitor_tui",
    )


def test_select_live_monitor_binding_prefers_matching_executable_name() -> None:
    bindings = (
        _LiveMonitorSessionBinding(
            runner_pid=5100,
            runner_module="src.hybrid.runner",
            status_file=Path("C:/tmp/other-status.json"),
            control_file=Path("C:/tmp/other-control.json"),
            executable_name="other.exe",
            source_pid=5100,
            source_module="src.hybrid.runner",
        ),
        _LiveMonitorSessionBinding(
            runner_pid=5000,
            runner_module="src.hybrid.runner",
            status_file=Path("C:/tmp/status.json"),
            control_file=Path("C:/tmp/control.json"),
            executable_name="868-HACK.exe",
            source_pid=5000,
            source_module="src.hybrid.runner",
        ),
    )

    selected = _select_live_monitor_binding(
        bindings,
        preferred_executable_name="868-HACK.exe",
    )

    assert selected == bindings[1]


def test_format_reward_breakdown_tooltip_formats_reward_components() -> None:
    tooltip = _format_reward_breakdown_tooltip(
        "reward total=+0.420 survival=+0.050 objective=collect_siphons"
    )

    assert "reward breakdown" in tooltip
    assert "total" in tooltip
    assert "+0.420" in tooltip
    assert "survival" in tooltip
    assert "objective" in tooltip
    assert "collect_siphons" in tooltip


def test_format_phase_breakdown_tooltip_formats_phase_details() -> None:
    tooltip = _format_phase_breakdown_tooltip(
        "action=move_right phase=collect_siphons next_target=(3,4) reason=scripted_phase_only"
    )

    assert "phase detail" in tooltip
    assert "Reason" in tooltip
    assert "scripted_phase_only" in tooltip
    assert "Action" in tooltip
    assert "move_right" in tooltip
    assert "Phase" in tooltip
    assert "collect_siphons" in tooltip
    assert "Target" in tooltip
    assert "(3,4)" in tooltip


def test_reward_history_limit_tracks_500_steps() -> None:
    assert _REWARD_HISTORY_LIMIT == 500


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
    assert "meta_reward_final_sector_win" in discovered
    assert "meta_reward_currency_gain" in discovered
    assert "meta_reward_energy_gain" in discovered
    assert "meta_reward_score_gain" in discovered
    assert "meta_reward_prog_gain" in discovered
    assert "meta_reward_step_limit_penalty" in discovered
    assert "meta_reward_stagnation_penalty" in discovered
    assert "meta_reward_stagnation_grace_steps" in discovered
