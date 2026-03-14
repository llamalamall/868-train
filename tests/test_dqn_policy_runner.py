"""Tests for DQN policy runner CLI wiring helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.env.dqn_policy_runner import (
    _build_dqn_config,
    _build_parser,
    _default_checkpoint_path,
    _event_indicates_fail_terminal,
    _format_monitor_actions,
    _reason_indicates_fail_terminal,
    _restore_selected_save_file,
    _resolve_checkpoint_path,
    _validate_args,
)


def test_dqn_runner_parser_defaults() -> None:
    parser = _build_parser()
    args = parser.parse_args([])

    assert args.mode == "train"
    assert args.exe == "868-HACK.exe"
    assert args.episodes == 100
    assert args.max_steps == 1000
    assert args.checkpoint is None
    assert args.checkpoint_every == 0
    assert args.movement_keys == "arrows"
    assert args.prog_actions is True
    assert args.launch_exe is True
    assert args.window_input is True
    assert args.step_through is False
    assert args.require_non_terminal_reset is True
    assert args.no_enemies is False
    assert args.tui is False
    assert args.game_tick_ms == 1
    assert args.disable_idle_frame_delay is True
    assert args.disable_background_motion is True
    assert args.disable_wall_animations is True
    assert args.post_action_delay == 0.01
    assert args.restore_save_delay == 0.35
    assert args.wait_for_action_processing is True
    assert args.action_ack_timeout == 0.35
    assert args.action_ack_poll_interval == 0.05
    assert args.gamma == 0.99
    assert args.learning_rate == 0.005
    assert args.min_replay_size == 256
    assert args.batch_size == 64
    assert args.epsilon_start == 0.8
    assert args.epsilon_end == 0.05
    assert args.epsilon_decay_steps == 5000
    assert args.reward_energy_delta == 0.02
    assert args.reward_score_delta == 0.01
    assert args.reward_safe_tile_bonus == 0.02
    assert args.reward_danger_tile_penalty == 0.08
    assert args.reward_sector_advance == 1.0


def test_dqn_runner_parser_accepts_overrides() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--mode",
            "eval",
            "--episodes",
            "3",
            "--max-steps",
            "90",
            "--checkpoint",
            "artifacts/checkpoints/final.json",
            "--movement-keys",
            "numpad",
            "--no-prog-actions",
            "--no-launch-exe",
            "--no-window-input",
            "--step-through",
            "--no-enemies",
            "--no-require-non-terminal-reset",
            "--no-tui",
            "--game-tick-ms",
            "8",
            "--disable-idle-frame-delay",
            "--disable-background-motion",
            "--disable-wall-animations",
            "--post-action-delay",
            "0.4",
            "--restore-save-delay",
            "1.25",
            "--no-wait-for-action-processing",
            "--action-ack-timeout",
            "1.5",
            "--action-ack-poll-interval",
            "0.2",
            "--gamma",
            "0.9",
            "--learning-rate",
            "0.02",
            "--target-sync-interval",
            "50",
        ]
    )

    assert args.mode == "eval"
    assert args.episodes == 3
    assert args.max_steps == 90
    assert args.checkpoint == "artifacts/checkpoints/final.json"
    assert args.restore_save_file is None
    assert args.movement_keys == "numpad"
    assert args.prog_actions is False
    assert args.launch_exe is False
    assert args.window_input is False
    assert args.step_through is True
    assert args.no_enemies is True
    assert args.require_non_terminal_reset is False
    assert args.tui is False
    assert args.game_tick_ms == 8
    assert args.disable_idle_frame_delay is True
    assert args.disable_background_motion is True
    assert args.disable_wall_animations is True
    assert args.post_action_delay == 0.4
    assert args.restore_save_delay == 1.25
    assert args.wait_for_action_processing is False
    assert args.action_ack_timeout == 1.5
    assert args.action_ack_poll_interval == 0.2
    assert args.gamma == 0.9
    assert args.learning_rate == 0.02
    assert args.target_sync_interval == 50


def test_dqn_runner_parser_rejects_out_of_range_game_tick_ms() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--game-tick-ms", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--game-tick-ms", "17"])


def test_default_checkpoint_path_uses_artifacts_checkpoints_dir() -> None:
    path = _default_checkpoint_path(
        now_utc=datetime(2026, 3, 6, 12, 34, 56, tzinfo=timezone.utc)
    )

    assert path == Path("artifacts") / "checkpoints" / "dqn-20260306-123456.json"


def test_resolve_checkpoint_path_prefers_explicit_value() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--checkpoint", "artifacts/checkpoints/custom.json"])

    assert _resolve_checkpoint_path(args) == Path("artifacts/checkpoints/custom.json")


def test_build_dqn_config_maps_cli_values() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--gamma",
            "0.93",
            "--learning-rate",
            "0.015",
            "--replay-capacity",
            "777",
            "--min-replay-size",
            "22",
            "--batch-size",
            "11",
            "--target-sync-interval",
            "17",
            "--epsilon-start",
            "0.8",
            "--epsilon-end",
            "0.2",
            "--epsilon-decay-steps",
            "333",
        ]
    )

    config = _build_dqn_config(args)

    assert config.gamma == 0.93
    assert config.learning_rate == 0.015
    assert config.replay_capacity == 777
    assert config.min_replay_size == 22
    assert config.batch_size == 11
    assert config.target_sync_interval == 17
    assert config.epsilon_start == 0.8
    assert config.epsilon_end == 0.2
    assert config.epsilon_decay_steps == 333


def test_validate_args_requires_checkpoint_in_eval_mode() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--mode", "eval"])

    with pytest.raises(SystemExit):
        _validate_args(parser, args)


def test_validate_args_rejects_negative_checkpoint_every() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--checkpoint-every", "-1"])

    with pytest.raises(SystemExit):
        _validate_args(parser, args)


def test_validate_args_rejects_missing_restore_save_file() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--restore-save-file", "does-not-exist.bin"])

    with pytest.raises(SystemExit):
        _validate_args(parser, args)


def test_validate_args_rejects_negative_restore_save_delay() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--restore-save-delay", "-0.01"])

    with pytest.raises(SystemExit):
        _validate_args(parser, args)


def test_reason_and_event_fail_detection_helpers() -> None:
    assert _reason_indicates_fail_terminal("state:fail_state")
    assert _reason_indicates_fail_terminal("memory:player_dead")
    assert _reason_indicates_fail_terminal("state:start_screen")
    assert not _reason_indicates_fail_terminal("timeout:max_steps")
    assert _event_indicates_fail_terminal({"done": True, "terminal_reason": "state:fail_state"})
    assert not _event_indicates_fail_terminal({"done": False, "terminal_reason": "state:fail_state"})
    assert not _event_indicates_fail_terminal({"done": True, "terminal_reason": "timeout:max_steps"})


def test_restore_selected_save_file_copies_contents(tmp_path: Path) -> None:
    source = tmp_path / "source-save"
    source.write_text("save-data", encoding="utf-8")
    target = tmp_path / "target" / "savegame_868"

    _restore_selected_save_file(source_path=source, target_path=target)

    assert target.read_text(encoding="utf-8") == "save-data"


def test_format_monitor_actions_preserves_full_action_list() -> None:
    formatted = _format_monitor_actions(
        ("move_up", "move_down", "move_left", "move_right", "confirm"),
        limit=3,
    )

    assert formatted == "move_up,move_down,move_left,move_right,confirm"
