"""Tests for re-heuristic policy runner argument wiring."""

from __future__ import annotations

from src.env.re_heuristic_policy_runner import _build_parser


def test_re_heuristic_runner_parser_defaults() -> None:
    parser = _build_parser()
    args = parser.parse_args([])

    assert args.exe == "868-HACK.exe"
    assert args.episodes == 3
    assert args.max_steps == 200
    assert args.low_health_threshold == 3
    assert args.enemy_prediction_horizon_steps == 2
    assert args.movement_keys == "arrows"
    assert args.prog_actions is True
    assert args.step_through is False
    assert args.require_non_terminal_reset is True
    assert args.tui is True
    assert args.reset_sequence == "confirm"
    assert args.verbose_actions is False
    assert args.mined_rules is True
    assert args.rule_pack_path is None


def test_re_heuristic_runner_parser_accepts_config_overrides() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--episodes",
            "11",
            "--max-steps",
            "222",
            "--movement-keys",
            "numpad",
            "--no-prog-actions",
            "--step-through",
            "--no-require-non-terminal-reset",
            "--no-tui",
            "--low-health-threshold",
            "1",
            "--enemy-prediction-horizon-steps",
            "6",
            "--verbose-actions",
            "--reward-fail-penalty",
            "19.5",
            "--no-mined-rules",
            "--rule-pack-path",
            "artifacts/re_heuristic/accepted_rule_pack.json",
        ]
    )

    assert args.episodes == 11
    assert args.max_steps == 222
    assert args.movement_keys == "numpad"
    assert args.prog_actions is False
    assert args.step_through is True
    assert args.require_non_terminal_reset is False
    assert args.tui is False
    assert args.low_health_threshold == 1
    assert args.enemy_prediction_horizon_steps == 6
    assert args.verbose_actions is True
    assert args.reward_fail_penalty == 19.5
    assert args.mined_rules is False
    assert args.rule_pack_path == "artifacts/re_heuristic/accepted_rule_pack.json"

