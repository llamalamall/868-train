"""Tests for objective-driven reward shaping."""

from __future__ import annotations

import pytest

from src.state.schema import EnemyState, FieldState, GameStateSnapshot, GridPosition, MapState
from src.training.rewards import RewardConfig, RewardWeights, compute_reward


def _ok_field(value: object) -> FieldState:
    return FieldState(value=value, status="ok")


def _missing_field() -> FieldState:
    return FieldState(value=None, status="missing", error_code="not_available")


def _snapshot(
    *,
    health: FieldState,
    energy: FieldState | None = None,
    currency: FieldState,
    fail_state: FieldState,
    map_state: MapState | None = None,
    timestamp: str = "2026-03-06T00:00:00+00:00",
) -> GameStateSnapshot:
    return GameStateSnapshot(
        timestamp_utc=timestamp,
        health=health,
        energy=energy or _missing_field(),
        currency=currency,
        fail_state=fail_state,
        map=map_state or MapState(status="missing"),
    )


def test_compute_reward_is_deterministic_for_same_inputs() -> None:
    config = RewardConfig(
        weights=RewardWeights(
            survival=0.2,
            step_penalty=0.05,
            health_delta=1.5,
            currency_delta=0.4,
            siphon_collected=0.0,
            enemy_cleared=0.0,
            phase_progress=0.0,
            map_clear_bonus=0.0,
            premature_exit_penalty=0.0,
            invalid_action_penalty=0.0,
            fail_penalty=12.0,
        ),
        reward_clip_abs=100.0,
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
    assert first.breakdown.step_penalty == -0.05
    assert first.breakdown.health_change == -3.0
    assert first.breakdown.currency_change == 1.6
    assert first.breakdown.fail_penalty == 0.0
    assert first.total == pytest.approx(-1.25)


def test_compute_reward_applies_objective_components_for_siphon_enemy_and_map_clear() -> None:
    config = RewardConfig(
        weights=RewardWeights(
            survival=0.0,
            step_penalty=0.0,
            health_delta=0.0,
            currency_delta=0.0,
            siphon_collected=2.5,
            enemy_cleared=1.5,
            phase_progress=0.0,
            map_clear_bonus=8.0,
            premature_exit_penalty=0.0,
            invalid_action_penalty=0.0,
            fail_penalty=0.0,
        ),
        reward_clip_abs=100.0,
    )
    previous_state = _snapshot(
        health=_ok_field(10),
        currency=_ok_field(1),
        fail_state=_ok_field(False),
        map_state=MapState(
            status="ok",
            width=3,
            height=3,
            player_position=GridPosition(0, 0),
            exit_position=GridPosition(2, 0),
            siphons=(GridPosition(1, 0),),
            enemies=(
                EnemyState(
                    slot=1,
                    type_id=2,
                    position=GridPosition(2, 2),
                    hp=1,
                    state=1,
                    in_bounds=True,
                ),
            ),
        ),
    )
    current_state = _snapshot(
        health=_ok_field(10),
        currency=_ok_field(1),
        fail_state=_ok_field(False),
        map_state=MapState(
            status="ok",
            width=3,
            height=3,
            player_position=GridPosition(2, 0),
            exit_position=GridPosition(2, 0),
            siphons=(),
            enemies=(),
        ),
    )

    result = compute_reward(
        previous_state=previous_state,
        current_state=current_state,
        done=False,
        config=config,
    )

    assert result.breakdown.siphon_collected == 2.5
    assert result.breakdown.enemy_cleared == 1.5
    assert result.breakdown.map_clear_bonus == 8.0
    assert result.total == pytest.approx(12.0)


def test_compute_reward_uses_phase_progress_when_distance_improves() -> None:
    config = RewardConfig(
        weights=RewardWeights(
            survival=0.0,
            step_penalty=0.0,
            health_delta=0.0,
            currency_delta=0.0,
            siphon_collected=0.0,
            enemy_cleared=0.0,
            phase_progress=0.25,
            map_clear_bonus=0.0,
            premature_exit_penalty=0.0,
            invalid_action_penalty=0.0,
            fail_penalty=0.0,
        ),
        reward_clip_abs=100.0,
    )
    previous_state = _snapshot(
        health=_ok_field(10),
        currency=_ok_field(1),
        fail_state=_ok_field(False),
        map_state=MapState(
            status="ok",
            width=3,
            height=1,
            player_position=GridPosition(0, 0),
            exit_position=GridPosition(2, 0),
            siphons=(GridPosition(2, 0),),
        ),
    )
    current_state = _snapshot(
        health=_ok_field(10),
        currency=_ok_field(1),
        fail_state=_ok_field(False),
        map_state=MapState(
            status="ok",
            width=3,
            height=1,
            player_position=GridPosition(1, 0),
            exit_position=GridPosition(2, 0),
            siphons=(GridPosition(2, 0),),
        ),
    )

    result = compute_reward(
        previous_state=previous_state,
        current_state=current_state,
        done=False,
        config=config,
    )

    assert result.breakdown.phase_progress == pytest.approx(0.25)
    assert result.total == pytest.approx(0.25)


def test_compute_reward_applies_premature_exit_and_invalid_action_penalties() -> None:
    config = RewardConfig(
        weights=RewardWeights(
            survival=0.0,
            step_penalty=0.0,
            health_delta=0.0,
            currency_delta=0.0,
            siphon_collected=0.0,
            enemy_cleared=0.0,
            phase_progress=0.0,
            map_clear_bonus=0.0,
            premature_exit_penalty=2.5,
            invalid_action_penalty=0.75,
            fail_penalty=0.0,
        ),
        reward_clip_abs=100.0,
    )
    previous_state = _snapshot(
        health=_ok_field(10),
        currency=_ok_field(0),
        fail_state=_ok_field(False),
    )
    current_state = _snapshot(
        health=_ok_field(10),
        currency=_ok_field(0),
        fail_state=_ok_field(False),
    )

    result = compute_reward(
        previous_state=previous_state,
        current_state=current_state,
        done=False,
        config=config,
        info={"premature_exit_attempt": True, "action_effective": False},
    )

    assert result.breakdown.premature_exit_penalty == -2.5
    assert result.breakdown.invalid_action_penalty == -0.75
    assert result.total == pytest.approx(-3.25)


def test_compute_reward_clips_total_reward() -> None:
    config = RewardConfig(
        weights=RewardWeights(
            survival=0.0,
            step_penalty=0.0,
            health_delta=0.0,
            currency_delta=0.0,
            siphon_collected=0.0,
            enemy_cleared=0.0,
            phase_progress=0.0,
            map_clear_bonus=0.0,
            premature_exit_penalty=0.0,
            invalid_action_penalty=0.0,
            fail_penalty=20.0,
        ),
        reward_clip_abs=5.0,
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

    assert result.breakdown.fail_penalty == -20.0
    assert result.total == -5.0
