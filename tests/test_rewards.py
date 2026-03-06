"""Tests for reward shaping v1."""

from __future__ import annotations

import pytest

from src.state.schema import FieldState, GameStateSnapshot
from src.training.rewards import RewardConfig, RewardWeights, compute_reward


def _ok_field(value: object) -> FieldState:
    return FieldState(value=value, status="ok")


def _missing_field() -> FieldState:
    return FieldState(value=None, status="missing", error_code="not_available")


def _snapshot(
    *,
    health: FieldState,
    currency: FieldState,
    fail_state: FieldState,
    timestamp: str = "2026-03-06T00:00:00+00:00",
) -> GameStateSnapshot:
    return GameStateSnapshot(
        timestamp_utc=timestamp,
        health=health,
        energy=_missing_field(),
        currency=currency,
        fail_state=fail_state,
    )


def test_compute_reward_is_deterministic_for_same_inputs() -> None:
    config = RewardConfig(
        weights=RewardWeights(
            survival=0.2,
            health_delta=1.5,
            currency_delta=0.4,
            fail_penalty=12.0,
        )
    )
    previous_state = _snapshot(
        health=_ok_field(10),
        currency=_ok_field(2),
        fail_state=_ok_field(False),
    )
    current_state = _snapshot(
        health=_ok_field(8),
        currency=_ok_field(6),
        fail_state=_ok_field(False),
    )

    first = compute_reward(
        previous_state=previous_state,
        current_state=current_state,
        done=False,
        config=config,
    )
    second = compute_reward(
        previous_state=previous_state,
        current_state=current_state,
        done=False,
        config=config,
    )

    assert first == second
    assert first.breakdown.survival == 0.2
    assert first.breakdown.health_change == -3.0
    assert first.breakdown.currency_change == 1.6
    assert first.breakdown.fail_penalty == 0.0
    assert first.total == pytest.approx(-1.2)


def test_reward_weights_can_be_loaded_from_partial_config_mapping() -> None:
    weights = RewardWeights.from_mapping({"currency_delta": 0.9, "fail_penalty": 15.0})

    assert weights.survival == 0.1
    assert weights.health_delta == 0.5
    assert weights.currency_delta == 0.9
    assert weights.fail_penalty == 15.0


def test_compute_reward_applies_terminal_fail_penalty() -> None:
    config = RewardConfig(
        weights=RewardWeights(
            survival=0.1,
            health_delta=1.0,
            currency_delta=0.25,
            fail_penalty=20.0,
        )
    )
    previous_state = _snapshot(
        health=_ok_field(5),
        currency=_ok_field(4),
        fail_state=_ok_field(False),
    )
    current_state = _snapshot(
        health=_ok_field(5),
        currency=_ok_field(5),
        fail_state=_ok_field(True),
    )

    result = compute_reward(
        previous_state=previous_state,
        current_state=current_state,
        done=True,
        config=config,
    )

    assert result.breakdown.survival == 0.0
    assert result.breakdown.health_change == 0.0
    assert result.breakdown.currency_change == 0.25
    assert result.breakdown.fail_penalty == -20.0
    assert result.total == pytest.approx(-19.75)


def test_compute_reward_uses_zero_deltas_when_core_fields_missing() -> None:
    config = RewardConfig(weights=RewardWeights(survival=0.3, fail_penalty=9.0))
    previous_state = _snapshot(
        health=_missing_field(),
        currency=_ok_field(4),
        fail_state=_ok_field(False),
    )
    current_state = _snapshot(
        health=_ok_field(8),
        currency=_missing_field(),
        fail_state=_ok_field(False),
    )

    result = compute_reward(
        previous_state=previous_state,
        current_state=current_state,
        done=False,
        config=config,
    )

    assert result.breakdown.survival == 0.3
    assert result.breakdown.health_change == 0.0
    assert result.breakdown.currency_change == 0.0
    assert result.breakdown.fail_penalty == 0.0
    assert result.total == 0.3


def test_compute_reward_logs_component_breakdown(caplog: pytest.LogCaptureFixture) -> None:
    previous_state = _snapshot(
        health=_ok_field(7),
        currency=_ok_field(1),
        fail_state=_ok_field(False),
    )
    current_state = _snapshot(
        health=_ok_field(6),
        currency=_ok_field(3),
        fail_state=_ok_field(False),
    )

    caplog.set_level("DEBUG")
    _ = compute_reward(
        previous_state=previous_state,
        current_state=current_state,
        done=False,
    )

    assert "Reward breakdown" in caplog.text
    assert "health_change" in caplog.text
    assert "currency_change" in caplog.text
    assert "fail_penalty" in caplog.text
