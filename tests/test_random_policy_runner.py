"""Tests for random policy runner helper wiring."""

from __future__ import annotations

import argparse

import pytest

from src.env.random_policy_runner import _build_action_config, _build_reward_config, _build_reward_fn
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


def test_build_action_config_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="movement_keys must be one of"):
        _build_action_config("vim")


def test_build_reward_fn_applies_configured_components_and_writes_breakdown() -> None:
    args = argparse.Namespace(
        reward_survival=0.2,
        reward_health_delta=1.0,
        reward_currency_delta=0.5,
        reward_fail_penalty=9.0,
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
    assert breakdown["health_change"] == -2.0
    assert breakdown["currency_change"] == 1.5
    assert breakdown["fail_penalty"] == 0.0
    assert breakdown["total"] == pytest.approx(-0.3)
