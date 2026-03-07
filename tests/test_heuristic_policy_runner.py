"""Tests for heuristic policy runner argument wiring."""

from __future__ import annotations

from src.env.heuristic_policy_runner import _build_parser


def test_heuristic_runner_parser_defaults() -> None:
    parser = _build_parser()
    args = parser.parse_args([])

    assert args.exe == "868-HACK.exe"
    assert args.episodes == 3
    assert args.max_steps == 200
    assert args.low_health_threshold == 3
    assert args.movement_keys == "arrows"
    assert args.prog_actions is True
    assert args.require_non_terminal_reset is True
    assert args.tui is True
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
            "--no-require-non-terminal-reset",
            "--no-tui",
            "--low-health-threshold",
            "2",
            "--verbose-actions",
            "--reward-fail-penalty",
            "18.5",
        ]
    )

    assert args.episodes == 9
    assert args.max_steps == 321
    assert args.movement_keys == "numpad"
    assert args.prog_actions is False
    assert args.require_non_terminal_reset is False
    assert args.tui is False
    assert args.low_health_threshold == 2
    assert args.verbose_actions is True
    assert args.reward_fail_penalty == 18.5
