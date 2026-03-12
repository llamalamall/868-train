"""Tests for heuristic policy runner argument wiring."""

from __future__ import annotations

import pytest

from src.env.heuristic_policy_runner import _build_parser, _validate_args


def test_heuristic_runner_parser_defaults() -> None:
    parser = _build_parser()
    args = parser.parse_args([])

    assert args.exe == "868-HACK.exe"
    assert args.episodes == 3
    assert args.max_steps == 200
    assert args.low_health_threshold == 3
    assert args.enemy_prediction_horizon_steps == 2
    assert args.movement_keys == "arrows"
    assert args.prog_actions is True
    assert args.launch_exe is True
    assert args.window_input is True
    assert args.step_through is False
    assert args.require_non_terminal_reset is True
    assert args.tui is True
    assert args.post_action_delay == 0.2
    assert args.wait_for_action_processing is True
    assert args.action_ack_timeout == 0.35
    assert args.action_ack_poll_interval == 0.05
    assert args.restore_save_file is None
    assert args.restore_save_delay == pytest.approx(0.35)
    assert args.reset_sequence == "confirm"
    assert args.verbose_actions is False


def test_heuristic_runner_parser_accepts_config_overrides() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--episodes",
            "9",
            "--max-steps",
            "321",
            "--movement-keys",
            "numpad",
            "--no-prog-actions",
            "--no-launch-exe",
            "--no-window-input",
            "--step-through",
            "--no-require-non-terminal-reset",
            "--no-tui",
            "--post-action-delay",
            "0.45",
            "--no-wait-for-action-processing",
            "--action-ack-timeout",
            "1.4",
            "--action-ack-poll-interval",
            "0.3",
            "--restore-save-delay",
            "1.2",
            "--low-health-threshold",
            "2",
            "--enemy-prediction-horizon-steps",
            "5",
            "--verbose-actions",
            "--reward-score-delta",
            "0.2",
            "--reward-fail-penalty",
            "18.5",
        ]
    )

    assert args.episodes == 9
    assert args.max_steps == 321
    assert args.movement_keys == "numpad"
    assert args.prog_actions is False
    assert args.launch_exe is False
    assert args.window_input is False
    assert args.step_through is True
    assert args.require_non_terminal_reset is False
    assert args.tui is False
    assert args.post_action_delay == 0.45
    assert args.wait_for_action_processing is False
    assert args.action_ack_timeout == 1.4
    assert args.action_ack_poll_interval == 0.3
    assert args.restore_save_delay == pytest.approx(1.2)
    assert args.low_health_threshold == 2
    assert args.enemy_prediction_horizon_steps == 5
    assert args.verbose_actions is True
    assert args.reward_score_delta == 0.2
    assert args.reward_fail_penalty == 18.5


def test_heuristic_runner_validate_args_rejects_missing_restore_save_file() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--restore-save-file", "does-not-exist.bin"])

    with pytest.raises(SystemExit):
        _validate_args(parser, args)


def test_heuristic_runner_validate_args_rejects_negative_restore_save_delay() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--restore-save-delay", "-0.01"])

    with pytest.raises(SystemExit):
        _validate_args(parser, args)
