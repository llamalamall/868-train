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
    assert args.avoid_enemy_distance == 1
    assert args.movement_keys == "arrows"
    assert args.reset_sequence == "confirm"


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
            "--low-health-threshold",
            "2",
            "--avoid-enemy-distance",
            "3",
            "--reward-fail-penalty",
            "18.5",
        ]
    )

    assert args.episodes == 9
    assert args.max_steps == 321
    assert args.movement_keys == "numpad"
    assert args.low_health_threshold == 2
    assert args.avoid_enemy_distance == 3
    assert args.reward_fail_penalty == 18.5
