"""Tests for objective-driven reward shaping."""

from __future__ import annotations

import pytest

from src.state.schema import (
    EnemyState,
    FieldState,
    GameStateSnapshot,
    GridPosition,
    InventoryState,
    MapCellState,
    MapState,
    ResourceCellState,
)
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
    inventory: InventoryState | None = None,
    map_state: MapState | None = None,
    extra_fields: dict[str, FieldState] | None = None,
    timestamp: str = "2026-03-06T00:00:00+00:00",
) -> GameStateSnapshot:
    return GameStateSnapshot(
        timestamp_utc=timestamp,
        health=health,
        energy=energy or _missing_field(),
        currency=currency,
        fail_state=fail_state,
        inventory=inventory or InventoryState(status="missing"),
        map=map_state or MapState(status="missing"),
        extra_fields=extra_fields or {},
    )


def test_compute_reward_is_deterministic_for_same_inputs() -> None:
    config = RewardConfig(
        weights=RewardWeights(
            survival=0.2,
            step_penalty=0.05,
            health_delta=1.5,
            currency_delta=0.4,
            energy_delta=0.0,
            score_delta=0.0,
            siphon_collected=0.0,
            enemy_cleared=0.0,
            phase_progress=0.0,
            map_clear_bonus=0.0,
            premature_exit_penalty=0.0,
            invalid_action_penalty=0.0,
            fail_penalty=12.0,
            safe_tile_bonus=0.0,
            danger_tile_penalty=0.0,
            resource_proximity=0.0,
            prog_collected_base=0.0,
            points_collected=0.0,
            damage_taken_penalty=0.0,
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
            energy_delta=0.0,
            score_delta=0.0,
            siphon_collected=2.5,
            enemy_cleared=1.5,
            phase_progress=0.0,
            map_clear_bonus=8.0,
            premature_exit_penalty=0.0,
            invalid_action_penalty=0.0,
            fail_penalty=0.0,
            safe_tile_bonus=0.0,
            danger_tile_penalty=0.0,
            resource_proximity=0.0,
            prog_collected_base=0.0,
            points_collected=0.0,
            damage_taken_penalty=0.0,
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


def test_compute_reward_enemy_cleared_uses_enemy_identity_diff_not_count_delta() -> None:
    config = RewardConfig(
        weights=RewardWeights(
            survival=0.0,
            step_penalty=0.0,
            health_delta=0.0,
            currency_delta=0.0,
            energy_delta=0.0,
            score_delta=0.0,
            siphon_collected=0.0,
            enemy_cleared=1.5,
            phase_progress=0.0,
            map_clear_bonus=0.0,
            premature_exit_penalty=0.0,
            invalid_action_penalty=0.0,
            fail_penalty=0.0,
            safe_tile_bonus=0.0,
            danger_tile_penalty=0.0,
            resource_proximity=0.0,
            prog_collected_base=0.0,
            points_collected=0.0,
            damage_taken_penalty=0.0,
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
            enemies=(
                EnemyState(
                    slot=1,
                    type_id=3,
                    position=GridPosition(2, 0),
                    hp=2,
                    state=0,
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
            height=1,
            player_position=GridPosition(0, 0),
            enemies=(
                EnemyState(
                    slot=2,
                    type_id=0,
                    position=GridPosition(1, 0),
                    hp=4,
                    state=0,
                    in_bounds=True,
                ),
            ),
        ),
    )

    result = compute_reward(
        previous_state=previous_state,
        current_state=current_state,
        done=False,
        config=config,
    )

    # One enemy (slot=1) was removed while another spawned, so clear count is still 1.
    assert result.breakdown.enemy_cleared == pytest.approx(1.5)


def test_compute_reward_map_clear_does_not_ignore_type_zero_enemies() -> None:
    config = RewardConfig(
        weights=RewardWeights(
            survival=0.0,
            step_penalty=0.0,
            health_delta=0.0,
            currency_delta=0.0,
            energy_delta=0.0,
            score_delta=0.0,
            siphon_collected=0.0,
            enemy_cleared=0.0,
            phase_progress=0.0,
            map_clear_bonus=8.0,
            premature_exit_penalty=0.0,
            invalid_action_penalty=0.0,
            fail_penalty=0.0,
            safe_tile_bonus=0.0,
            danger_tile_penalty=0.0,
            resource_proximity=0.0,
            prog_collected_base=0.0,
            points_collected=0.0,
            damage_taken_penalty=0.0,
        ),
        reward_clip_abs=100.0,
    )
    previous_state = _snapshot(
        health=_ok_field(10),
        currency=_ok_field(1),
        fail_state=_ok_field(False),
        map_state=MapState(
            status="ok",
            width=2,
            height=1,
            player_position=GridPosition(0, 0),
            exit_position=GridPosition(1, 0),
            siphons=(),
            enemies=(
                EnemyState(
                    slot=1,
                    type_id=0,
                    position=GridPosition(0, 0),
                    hp=3,
                    state=0,
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
            width=2,
            height=1,
            player_position=GridPosition(1, 0),
            exit_position=GridPosition(1, 0),
            siphons=(),
            enemies=(
                EnemyState(
                    slot=1,
                    type_id=0,
                    position=GridPosition(1, 0),
                    hp=3,
                    state=0,
                    in_bounds=True,
                ),
            ),
        ),
    )

    result = compute_reward(
        previous_state=previous_state,
        current_state=current_state,
        done=False,
        config=config,
    )

    assert result.breakdown.map_clear_bonus == 0.0


def test_compute_reward_rewards_nonlethal_enemy_damage() -> None:
    config = RewardConfig(
        weights=RewardWeights(
            survival=0.0,
            step_penalty=0.0,
            health_delta=0.0,
            currency_delta=0.0,
            energy_delta=0.0,
            score_delta=0.0,
            siphon_collected=0.0,
            enemy_damaged=0.5,
            enemy_cleared=0.0,
            phase_progress=0.0,
            map_clear_bonus=0.0,
            premature_exit_penalty=0.0,
            invalid_action_penalty=0.0,
            fail_penalty=0.0,
            safe_tile_bonus=0.0,
            danger_tile_penalty=0.0,
            resource_proximity=0.0,
            prog_collected_base=0.0,
            points_collected=0.0,
            damage_taken_penalty=0.0,
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
            enemies=(
                EnemyState(
                    slot=1,
                    type_id=3,
                    position=GridPosition(1, 0),
                    hp=5,
                    state=0,
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
            height=1,
            player_position=GridPosition(0, 0),
            enemies=(
                EnemyState(
                    slot=1,
                    type_id=3,
                    position=GridPosition(1, 0),
                    hp=3,
                    state=0,
                    in_bounds=True,
                ),
            ),
        ),
    )

    result = compute_reward(
        previous_state=previous_state,
        current_state=current_state,
        done=False,
        config=config,
    )

    assert result.breakdown.enemy_damaged == pytest.approx(1.0)
    assert result.breakdown.enemy_cleared == 0.0
    assert result.total == pytest.approx(1.0)


def test_compute_reward_uses_phase_progress_when_distance_improves() -> None:
    config = RewardConfig(
        weights=RewardWeights(
            survival=0.0,
            step_penalty=0.0,
            health_delta=0.0,
            currency_delta=0.0,
            energy_delta=0.0,
            score_delta=0.0,
            siphon_collected=0.0,
            enemy_cleared=0.0,
            phase_progress=0.25,
            map_clear_bonus=0.0,
            premature_exit_penalty=0.0,
            invalid_action_penalty=0.0,
            fail_penalty=0.0,
            safe_tile_bonus=0.0,
            danger_tile_penalty=0.0,
            resource_proximity=0.0,
            prog_collected_base=0.0,
            points_collected=0.0,
            damage_taken_penalty=0.0,
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


def test_compute_reward_applies_backtrack_penalty_when_distance_worsens() -> None:
    config = RewardConfig(
        weights=RewardWeights(
            survival=0.0,
            step_penalty=0.0,
            health_delta=0.0,
            currency_delta=0.0,
            energy_delta=0.0,
            score_delta=0.0,
            siphon_collected=0.0,
            enemy_cleared=0.0,
            phase_progress=0.25,
            backtrack_penalty=0.5,
            map_clear_bonus=0.0,
            premature_exit_penalty=0.0,
            invalid_action_penalty=0.0,
            fail_penalty=0.0,
            safe_tile_bonus=0.0,
            danger_tile_penalty=0.0,
            resource_proximity=0.0,
            prog_collected_base=0.0,
            points_collected=0.0,
            damage_taken_penalty=0.0,
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
            player_position=GridPosition(1, 0),
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
            player_position=GridPosition(0, 0),
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

    assert result.breakdown.phase_progress == 0.0
    assert result.breakdown.backtrack_penalty == pytest.approx(-0.5)
    assert result.total == pytest.approx(-0.5)


def test_compute_reward_applies_premature_exit_and_invalid_action_penalties() -> None:
    config = RewardConfig(
        weights=RewardWeights(
            survival=0.0,
            step_penalty=0.0,
            health_delta=0.0,
            currency_delta=0.0,
            energy_delta=0.0,
            score_delta=0.0,
            siphon_collected=0.0,
            enemy_cleared=0.0,
            phase_progress=0.0,
            map_clear_bonus=0.0,
            premature_exit_penalty=2.5,
            invalid_action_penalty=0.75,
            fail_penalty=0.0,
            safe_tile_bonus=0.0,
            danger_tile_penalty=0.0,
            resource_proximity=0.0,
            prog_collected_base=0.0,
            points_collected=0.0,
            damage_taken_penalty=0.0,
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
            energy_delta=0.0,
            score_delta=0.0,
            siphon_collected=0.0,
            enemy_cleared=0.0,
            phase_progress=0.0,
            map_clear_bonus=0.0,
            premature_exit_penalty=0.0,
            invalid_action_penalty=0.0,
            fail_penalty=20.0,
            safe_tile_bonus=0.0,
            danger_tile_penalty=0.0,
            resource_proximity=0.0,
            prog_collected_base=0.0,
            points_collected=0.0,
            damage_taken_penalty=0.0,
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


def test_compute_reward_applies_map_state_harvest_and_prog_components() -> None:
    config = RewardConfig(
        weights=RewardWeights(
            survival=0.0,
            step_penalty=0.0,
            health_delta=0.0,
            currency_delta=0.0,
            energy_delta=0.5,
            score_delta=0.1,
            siphon_collected=0.0,
            enemy_cleared=0.0,
            phase_progress=0.0,
            map_clear_bonus=0.0,
            premature_exit_penalty=0.0,
            invalid_action_penalty=0.0,
            fail_penalty=0.0,
            safe_tile_bonus=0.0,
            danger_tile_penalty=0.0,
            resource_proximity=0.4,
            prog_collected_base=1.0,
            points_collected=0.2,
            damage_taken_penalty=0.0,
        ),
        reward_clip_abs=100.0,
    )
    previous_state = _snapshot(
        health=_ok_field(10),
        energy=_ok_field(2),
        currency=_ok_field(0),
        fail_state=_ok_field(False),
        inventory=InventoryState(status="ok", raw_prog_ids=(7,)),
        map_state=MapState(
            status="ok",
            width=3,
            height=1,
            player_position=GridPosition(0, 0),
            exit_position=GridPosition(2, 0),
            cells=(
                MapCellState(
                    position=GridPosition(2, 0),
                    cell_type=0,
                    tile_variant=0,
                    wall_state=0,
                    points=5,
                ),
            ),
            resource_cells=(ResourceCellState(position=GridPosition(2, 0), credits=1),),
        ),
        extra_fields={"score": _ok_field(10)},
    )
    current_state = _snapshot(
        health=_ok_field(10),
        energy=_ok_field(4),
        currency=_ok_field(0),
        fail_state=_ok_field(False),
        inventory=InventoryState(status="ok", raw_prog_ids=(7, 10)),
        map_state=MapState(
            status="ok",
            width=3,
            height=1,
            player_position=GridPosition(1, 0),
            exit_position=GridPosition(2, 0),
            cells=(
                MapCellState(
                    position=GridPosition(2, 0),
                    cell_type=0,
                    tile_variant=0,
                    wall_state=0,
                    points=3,
                ),
            ),
            resource_cells=(ResourceCellState(position=GridPosition(2, 0), credits=1),),
        ),
        extra_fields={"score": _ok_field(16)},
    )

    result = compute_reward(
        previous_state=previous_state,
        current_state=current_state,
        done=False,
        config=config,
    )

    assert result.breakdown.energy_change == pytest.approx(1.0)
    assert result.breakdown.score_change == pytest.approx(0.6)
    assert result.breakdown.prog_collected == pytest.approx(2.5)
    assert result.breakdown.points_collected == pytest.approx(0.4)
    assert result.breakdown.resource_proximity == pytest.approx(0.4)
    assert result.total == pytest.approx(4.9)


def test_compute_reward_applies_safety_and_damage_penalties() -> None:
    config = RewardConfig(
        weights=RewardWeights(
            survival=0.0,
            step_penalty=0.0,
            health_delta=0.0,
            currency_delta=0.0,
            energy_delta=0.0,
            score_delta=0.0,
            siphon_collected=0.0,
            enemy_cleared=0.0,
            phase_progress=0.0,
            map_clear_bonus=0.0,
            premature_exit_penalty=0.0,
            invalid_action_penalty=0.0,
            fail_penalty=0.0,
            safe_tile_bonus=0.2,
            danger_tile_penalty=0.3,
            resource_proximity=0.0,
            prog_collected_base=0.0,
            points_collected=0.0,
            damage_taken_penalty=0.6,
        ),
        reward_clip_abs=100.0,
    )
    previous_state = _snapshot(
        health=_ok_field(10),
        currency=_ok_field(0),
        fail_state=_ok_field(False),
        map_state=MapState(
            status="ok",
            width=3,
            height=1,
            player_position=GridPosition(0, 0),
            enemies=(),
        ),
    )
    current_state = _snapshot(
        health=_ok_field(7),
        currency=_ok_field(0),
        fail_state=_ok_field(False),
        map_state=MapState(
            status="ok",
            width=3,
            height=1,
            player_position=GridPosition(1, 0),
            enemies=(
                EnemyState(
                    slot=1,
                    type_id=3,
                    position=GridPosition(1, 1),
                    hp=1,
                    state=1,
                    in_bounds=True,
                ),
            ),
        ),
    )

    result = compute_reward(
        previous_state=previous_state,
        current_state=current_state,
        done=False,
        config=config,
    )

    assert result.breakdown.safe_tile_bonus == 0.0
    assert result.breakdown.danger_tile_penalty == pytest.approx(-0.3)
    assert result.breakdown.damage_taken_penalty == pytest.approx(-1.8)
    assert result.total == pytest.approx(-2.1)


def test_compute_reward_phase_progress_uses_wall_aware_shortest_path() -> None:
    config = RewardConfig(
        weights=RewardWeights(
            survival=0.0,
            step_penalty=0.0,
            health_delta=0.0,
            currency_delta=0.0,
            energy_delta=0.0,
            score_delta=0.0,
            siphon_collected=0.0,
            enemy_cleared=0.0,
            phase_progress=1.0,
            map_clear_bonus=0.0,
            premature_exit_penalty=0.0,
            invalid_action_penalty=0.0,
            fail_penalty=0.0,
            safe_tile_bonus=0.0,
            danger_tile_penalty=0.0,
            resource_proximity=0.0,
            prog_collected_base=0.0,
            points_collected=0.0,
            damage_taken_penalty=0.0,
        ),
        reward_clip_abs=100.0,
    )
    wall_cell = MapCellState(
        position=GridPosition(1, 0),
        cell_type=1,
        tile_variant=0,
        wall_state=0,
        is_wall=True,
    )
    previous_state = _snapshot(
        health=_ok_field(10),
        currency=_ok_field(0),
        fail_state=_ok_field(False),
        map_state=MapState(
            status="ok",
            width=3,
            height=2,
            player_position=GridPosition(0, 0),
            siphons=(GridPosition(2, 0),),
            cells=(wall_cell,),
        ),
    )
    current_state = _snapshot(
        health=_ok_field(10),
        currency=_ok_field(0),
        fail_state=_ok_field(False),
        map_state=MapState(
            status="ok",
            width=3,
            height=2,
            player_position=GridPosition(0, 1),
            siphons=(GridPosition(2, 0),),
            cells=(wall_cell,),
        ),
    )

    result = compute_reward(
        previous_state=previous_state,
        current_state=current_state,
        done=False,
        config=config,
    )

    # Manhattan distance gets worse (2 -> 3), but shortest path improves (4 -> 3).
    assert result.breakdown.phase_progress == pytest.approx(1.0)


def test_compute_reward_resource_proximity_uses_adjacent_targets_for_prog_walls() -> None:
    config = RewardConfig(
        weights=RewardWeights(
            survival=0.0,
            step_penalty=0.0,
            health_delta=0.0,
            currency_delta=0.0,
            energy_delta=0.0,
            score_delta=0.0,
            siphon_collected=0.0,
            enemy_cleared=0.0,
            phase_progress=0.0,
            map_clear_bonus=0.0,
            premature_exit_penalty=0.0,
            invalid_action_penalty=0.0,
            fail_penalty=0.0,
            safe_tile_bonus=0.0,
            danger_tile_penalty=0.0,
            resource_proximity=1.0,
            prog_collected_base=0.0,
            points_collected=0.0,
            damage_taken_penalty=0.0,
        ),
        reward_clip_abs=100.0,
    )
    prog_wall = MapCellState(
        position=GridPosition(2, 0),
        cell_type=1,
        tile_variant=0,
        wall_state=0,
        is_wall=True,
        prog_id=10,
    )
    previous_state = _snapshot(
        health=_ok_field(10),
        currency=_ok_field(0),
        fail_state=_ok_field(False),
        map_state=MapState(
            status="ok",
            width=3,
            height=2,
            player_position=GridPosition(0, 1),
            cells=(prog_wall,),
        ),
    )
    current_state = _snapshot(
        health=_ok_field(10),
        currency=_ok_field(0),
        fail_state=_ok_field(False),
        map_state=MapState(
            status="ok",
            width=3,
            height=2,
            player_position=GridPosition(1, 1),
            cells=(prog_wall,),
        ),
    )

    result = compute_reward(
        previous_state=previous_state,
        current_state=current_state,
        done=False,
        config=config,
    )

    # Progress is measured against reachable adjacent tiles to the wall harvest target.
    assert result.breakdown.resource_proximity == pytest.approx(1.0)
