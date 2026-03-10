"""Tests for random policy runner helper wiring."""

from __future__ import annotations

import argparse

import pytest

from src.env.random_policy_runner import (
    _build_action_config,
    _build_parser,
    _build_reward_config,
    _build_reward_fn,
)
from src.state.schema import FieldState, GameStateSnapshot


def _field(value: object) -> FieldState:
    return FieldState(value=value, status="ok")


def _snapshot(*, health: int, currency: int, failed: bool = False) -> GameStateSnapshot:
    return GameStateSnapshot(
        timestamp_utc="2026-03-06T00:00:00+00:00",
        health=_field(health),
        energy=_field(0),
        currency=_field(currency),
        fail_state=_field(failed),
    )


def test_build_action_config_wasd_includes_letter_key_codes() -> None:
    config = _build_action_config("wasd")

    assert config.action_key_bindings["move_up"] == "W"
    assert config.action_key_bindings["move_left"] == "A"
    assert config.action_key_bindings["move_down"] == "S"
    assert config.action_key_bindings["move_right"] == "D"
    assert config.key_codes["W"] == 0x57
    assert config.key_codes["A"] == 0x41
    assert config.key_codes["S"] == 0x53
    assert config.key_codes["D"] == 0x44


def test_build_action_config_numpad_includes_numpad_key_codes() -> None:
    config = _build_action_config("numpad")

    assert config.action_key_bindings["move_up"] == "NUMPAD8"
    assert config.action_key_bindings["move_left"] == "NUMPAD4"
    assert config.action_key_bindings["move_down"] == "NUMPAD2"
    assert config.action_key_bindings["move_right"] == "NUMPAD6"
    assert config.key_codes["NUMPAD8"] == 0x68
    assert config.key_codes["NUMPAD4"] == 0x64
    assert config.key_codes["NUMPAD2"] == 0x62
    assert config.key_codes["NUMPAD6"] == 0x66


def test_build_action_config_includes_prog_slot_actions_by_default() -> None:
    config = _build_action_config("arrows")

    assert config.action_key_bindings["prog_slot_1"] == "1"
    assert config.action_key_bindings["prog_slot_5"] == "5"
    assert config.action_key_bindings["prog_slot_10"] == "0"
    assert config.key_codes["1"] == 0x31
    assert config.key_codes["0"] == 0x30


def test_build_action_config_can_disable_prog_slot_actions() -> None:
    config = _build_action_config("arrows", include_prog_actions=False)

    assert all(not action.startswith("prog_slot_") for action in config.action_key_bindings)


def test_build_action_config_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="movement_keys must be one of"):
        _build_action_config("vim")


def test_random_runner_parser_prog_actions_defaults_to_enabled() -> None:
    parser = _build_parser()
    args = parser.parse_args([])

    assert args.prog_actions is True
    assert args.launch_exe is True
    assert args.window_input is True
    assert args.step_through is False
    assert args.require_non_terminal_reset is True
    assert args.tui is True
    assert args.game_tick_ms == 16
    assert args.post_action_delay == 0.2
    assert args.wait_for_action_processing is True
    assert args.action_ack_timeout == 0.35
    assert args.action_ack_poll_interval == 0.05
    assert args.reward_sector_advance == 1.0


def test_random_runner_parser_accepts_prog_actions_toggle() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--no-prog-actions",
            "--no-launch-exe",
            "--no-window-input",
            "--step-through",
            "--no-require-non-terminal-reset",
            "--no-tui",
            "--game-tick-ms",
            "8",
            "--post-action-delay",
            "0.3",
            "--no-wait-for-action-processing",
            "--action-ack-timeout",
            "1.2",
            "--action-ack-poll-interval",
            "0.25",
        ]
    )

    assert args.prog_actions is False
    assert args.launch_exe is False
    assert args.window_input is False
    assert args.step_through is True
    assert args.require_non_terminal_reset is False
    assert args.tui is False
    assert args.game_tick_ms == 8
    assert args.post_action_delay == 0.3
    assert args.wait_for_action_processing is False
    assert args.action_ack_timeout == 1.2
    assert args.action_ack_poll_interval == 0.25


def test_random_runner_parser_rejects_out_of_range_game_tick_ms() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--game-tick-ms", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--game-tick-ms", "17"])


def test_build_reward_fn_applies_configured_components_and_writes_breakdown() -> None:
    args = argparse.Namespace(
        reward_survival=0.2,
        reward_step_penalty=0.0,
        reward_health_delta=1.0,
        reward_currency_delta=0.5,
        reward_energy_delta=0.0,
        reward_score_delta=0.0,
        reward_siphon_collected=0.0,
        reward_enemy_damaged=0.0,
        reward_enemy_cleared=0.0,
        reward_phase_progress=0.0,
        reward_backtrack_penalty=0.0,
        reward_map_clear_bonus=0.0,
        reward_premature_exit_penalty=0.0,
        reward_invalid_action_penalty=0.0,
        reward_fail_penalty=9.0,
        reward_safe_tile_bonus=0.0,
        reward_danger_tile_penalty=0.0,
        reward_resource_proximity=0.0,
        reward_prog_collected_base=0.0,
        reward_points_collected=0.0,
        reward_damage_taken_penalty=0.0,
        reward_sector_advance=0.0,
        reward_clip_abs=5.0,
    )
    reward_config = _build_reward_config(args)
    reward_fn = _build_reward_fn(reward_config=reward_config, print_breakdown=False)

    previous = _snapshot(health=10, currency=1, failed=False)
    current = _snapshot(health=8, currency=4, failed=False)
    info: dict[str, object] = {"step_index": 3, "action": "move_right"}

    reward = reward_fn(previous, current, False, info)

    assert reward == pytest.approx(-0.3)
    breakdown = info["reward_breakdown"]
    assert isinstance(breakdown, dict)
    assert breakdown["survival"] == 0.2
    assert breakdown["step_penalty"] == 0.0
    assert breakdown["health_change"] == -2.0
    assert breakdown["currency_change"] == 1.5
    assert breakdown["energy_change"] == 0.0
    assert breakdown["score_change"] == 0.0
    assert breakdown["siphon_collected"] == 0.0
    assert breakdown["enemy_damaged"] == 0.0
    assert breakdown["enemy_cleared"] == 0.0
    assert breakdown["prog_collected"] == 0.0
    assert breakdown["resource_proximity"] == 0.0
    assert breakdown["fail_penalty"] == 0.0
    assert breakdown["sector_advance"] == 0.0
    assert breakdown["total"] == pytest.approx(-0.3)
