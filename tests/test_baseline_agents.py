"""Tests for baseline agent policies."""

from __future__ import annotations

import logging
import random

from src.agent.baseline_heuristic import HeuristicBaselineAgent, HeuristicBaselineConfig
from src.agent.baseline_random import RandomBaselineAgent
from src.state.schema import (
    EnemyState,
    FieldState,
    GameStateSnapshot,
    GridPosition,
    InventoryState,
    MapCellState,
    MapState,
)


def _field(value: object) -> FieldState:
    return FieldState(value=value, status="ok")


def _snapshot(
    *,
    health: int = 10,
    energy: int = 0,
    player: GridPosition | None = None,
    exit_pos: GridPosition | None = None,
    enemies: tuple[EnemyState, ...] = (),
    siphons: tuple[GridPosition, ...] = (),
    walls: tuple[GridPosition, ...] = (),
    cells: tuple[MapCellState, ...] = (),
    inventory: InventoryState | None = None,
) -> GameStateSnapshot:
    wall_cells = tuple(
        MapCellState(
            position=position,
            cell_type=1,
            tile_variant=0,
            wall_state=0,
            is_wall=True,
        )
        for position in walls
    )
    merged_cells = (*wall_cells, *cells)
    map_state = MapState(
        status="ok" if player is not None else "missing",
        player_position=player,
        exit_position=exit_pos,
        enemies=enemies,
        siphons=siphons,
        cells=merged_cells,
    )
    return GameStateSnapshot(
        timestamp_utc="2026-03-06T00:00:00+00:00",
        health=_field(health),
        energy=_field(energy),
        currency=_field(0),
        fail_state=_field(False),
        inventory=inventory or InventoryState(status="missing"),
        map=map_state,
    )


def _inventory(*prog_ids: int) -> InventoryState:
    return InventoryState(status="ok", raw_prog_ids=tuple(prog_ids))


def test_random_baseline_agent_is_reproducible_with_same_seed() -> None:
    agent = RandomBaselineAgent()
    state = _snapshot()

    rng_a = random.Random(7)
    rng_b = random.Random(7)
    seq_a = [agent.select_action(state=state, action_space=("a", "b", "c"), rng=rng_a) for _ in range(8)]
    seq_b = [agent.select_action(state=state, action_space=("a", "b", "c"), rng=rng_b) for _ in range(8)]

    assert seq_a == seq_b


def test_heuristic_baseline_waits_when_health_is_low() -> None:
    agent = HeuristicBaselineAgent()
    state = _snapshot(health=2, player=GridPosition(1, 1), exit_pos=GridPosition(2, 1))

    action = agent.select_action(
        state=state,
        action_space=("move_left", "move_right", "wait"),
        rng=random.Random(1),
    )

    assert action == "wait"


def test_heuristic_baseline_moves_toward_exit_when_safe() -> None:
    agent = HeuristicBaselineAgent()
    state = _snapshot(health=9, player=GridPosition(1, 1), exit_pos=GridPosition(2, 1))

    action = agent.select_action(
        state=state,
        action_space=("move_left", "move_right", "wait"),
        rng=random.Random(1),
    )

    assert action == "move_right"


def test_heuristic_baseline_treats_move_up_as_positive_y() -> None:
    agent = HeuristicBaselineAgent()
    state = _snapshot(health=9, player=GridPosition(1, 1), exit_pos=GridPosition(1, 2))

    action = agent.select_action(
        state=state,
        action_space=("move_down", "move_up", "wait"),
        rng=random.Random(1),
    )

    assert action == "move_up"


def test_heuristic_baseline_moves_toward_enemy_in_direct_sight() -> None:
    enemy = EnemyState(slot=0, type_id=1, position=GridPosition(4, 1), hp=1, state=0, in_bounds=True)
    agent = HeuristicBaselineAgent(
        config=HeuristicBaselineConfig(enemy_prediction_horizon_steps=0)
    )
    state = _snapshot(
        health=8,
        player=GridPosition(1, 1),
        exit_pos=GridPosition(0, 1),
        enemies=(enemy,),
    )

    action = agent.select_action(
        state=state,
        action_space=("move_left", "move_right", "wait"),
        rng=random.Random(2),
    )

    assert action == "move_right"


def test_heuristic_baseline_moves_toward_glitch_on_wall_in_sight() -> None:
    glitch = EnemyState(slot=0, type_id=4, position=GridPosition(4, 1), hp=1, state=0, in_bounds=True)
    agent = HeuristicBaselineAgent(
        config=HeuristicBaselineConfig(enemy_prediction_horizon_steps=0)
    )
    state = _snapshot(
        health=8,
        player=GridPosition(1, 1),
        exit_pos=GridPosition(0, 1),
        enemies=(glitch,),
        walls=(GridPosition(4, 1),),
    )

    action = agent.select_action(
        state=state,
        action_space=("move_left", "move_right", "wait"),
        rng=random.Random(24),
    )

    assert action == "move_right"


def test_heuristic_baseline_avoids_move_that_enters_enemy_adjacency() -> None:
    enemy = EnemyState(slot=0, type_id=1, position=GridPosition(2, 2), hp=1, state=0, in_bounds=True)
    agent = HeuristicBaselineAgent()
    state = _snapshot(
        health=8,
        player=GridPosition(1, 1),
        exit_pos=GridPosition(2, 1),
        enemies=(enemy,),
    )

    action = agent.select_action(
        state=state,
        action_space=("move_right", "move_left"),
        rng=random.Random(25),
    )

    assert action == "move_left"


def test_heuristic_baseline_avoids_move_within_one_space_of_virus() -> None:
    virus = EnemyState(slot=0, type_id=2, position=GridPosition(3, 2), hp=1, state=0, in_bounds=True)
    agent = HeuristicBaselineAgent(
        config=HeuristicBaselineConfig(enemy_prediction_horizon_steps=0)
    )
    state = _snapshot(
        health=8,
        player=GridPosition(1, 1),
        exit_pos=GridPosition(2, 1),
        enemies=(virus,),
    )

    action = agent.select_action(
        state=state,
        action_space=("move_right", "move_left"),
        rng=random.Random(26),
    )

    assert action == "move_left"


def test_heuristic_baseline_uses_enemy_lookahead_horizon_for_move_safety() -> None:
    virus = EnemyState(slot=0, type_id=2, position=GridPosition(4, 2), hp=1, state=0, in_bounds=True)
    state = _snapshot(
        health=8,
        player=GridPosition(1, 1),
        exit_pos=GridPosition(2, 1),
        enemies=(virus,),
    )
    immediate_agent = HeuristicBaselineAgent(
        config=HeuristicBaselineConfig(enemy_prediction_horizon_steps=0)
    )
    lookahead_agent = HeuristicBaselineAgent(
        config=HeuristicBaselineConfig(enemy_prediction_horizon_steps=2)
    )

    immediate_action = immediate_agent.select_action(
        state=state,
        action_space=("move_right", "move_left"),
        rng=random.Random(27),
    )
    lookahead_action = lookahead_agent.select_action(
        state=state,
        action_space=("move_right", "move_left"),
        rng=random.Random(27),
    )

    assert immediate_action == "move_right"
    assert lookahead_action == "move_left"


def test_heuristic_baseline_prioritizes_enemy_attack_over_prog_and_objectives() -> None:
    enemy = EnemyState(slot=0, type_id=1, position=GridPosition(4, 1), hp=1, state=0, in_bounds=True)
    agent = HeuristicBaselineAgent(
        config=HeuristicBaselineConfig(
            low_health_threshold=3,
            enable_prog_usage=True,
            prog_energy_floor=1,
        )
    )
    state = _snapshot(
        health=2,
        energy=10,
        player=GridPosition(1, 1),
        exit_pos=GridPosition(0, 1),
        enemies=(enemy,),
        siphons=(GridPosition(1, 2),),
        inventory=_inventory(7),
    )

    action = agent.select_action(
        state=state,
        action_space=("move_right", "move_up", "prog_slot_1", "space", "wait"),
        rng=random.Random(28),
    )

    assert action == "move_right"


def test_heuristic_baseline_uses_prog_when_all_moves_are_dangerous_without_los_attack() -> None:
    enemies = (
        EnemyState(slot=0, type_id=2, position=GridPosition(0, 0), hp=1, state=0, in_bounds=True),
        EnemyState(slot=1, type_id=2, position=GridPosition(2, 0), hp=1, state=0, in_bounds=True),
        EnemyState(slot=2, type_id=2, position=GridPosition(0, 2), hp=1, state=0, in_bounds=True),
        EnemyState(slot=3, type_id=2, position=GridPosition(2, 2), hp=1, state=0, in_bounds=True),
    )
    agent = HeuristicBaselineAgent(
        config=HeuristicBaselineConfig(
            enable_prog_usage=True,
            prog_energy_floor=1,
        )
    )
    state = _snapshot(
        health=8,
        energy=10,
        player=GridPosition(1, 1),
        exit_pos=GridPosition(5, 5),
        enemies=enemies,
        inventory=_inventory(7),
    )

    action = agent.select_action(
        state=state,
        action_space=("move_left", "move_right", "move_up", "move_down", "prog_slot_1"),
        rng=random.Random(28),
    )

    assert action == "prog_slot_1"


def test_heuristic_baseline_prioritizes_siphon_before_exit() -> None:
    agent = HeuristicBaselineAgent()
    state = _snapshot(
        health=8,
        player=GridPosition(0, 0),
        exit_pos=GridPosition(2, 0),
        siphons=(GridPosition(0, 1), GridPosition(1, 1)),
    )

    action = agent.select_action(
        state=state,
        action_space=("move_right", "move_up"),
        rng=random.Random(3),
    )

    assert action == "move_up"


def test_heuristic_baseline_does_not_pursue_enemy_when_los_blocked() -> None:
    enemy = EnemyState(slot=0, type_id=1, position=GridPosition(2, 1), hp=1, state=0, in_bounds=True)
    agent = HeuristicBaselineAgent()
    state = _snapshot(
        health=8,
        player=GridPosition(0, 1),
        exit_pos=GridPosition(2, 2),
        enemies=(enemy,),
        siphons=(GridPosition(0, 2),),
        walls=(GridPosition(1, 1),),
    )

    action = agent.select_action(
        state=state,
        action_space=("move_right", "move_up"),
        rng=random.Random(4),
    )

    assert action == "move_up"


def test_heuristic_baseline_pathfinds_around_wall_to_exit() -> None:
    agent = HeuristicBaselineAgent()
    state = _snapshot(
        health=9,
        player=GridPosition(0, 0),
        exit_pos=GridPosition(2, 0),
        walls=(GridPosition(1, 0),),
    )

    action = agent.select_action(
        state=state,
        action_space=("move_right", "move_up", "move_left"),
        rng=random.Random(5),
    )

    assert action == "move_up"


def test_heuristic_baseline_targets_reachable_siphon_via_pathfinding() -> None:
    agent = HeuristicBaselineAgent()
    state = _snapshot(
        health=9,
        player=GridPosition(0, 0),
        exit_pos=GridPosition(2, 2),
        siphons=(GridPosition(2, 0), GridPosition(0, 2)),
        walls=(GridPosition(1, 0), GridPosition(1, 1), GridPosition(1, 2)),
    )

    action = agent.select_action(
        state=state,
        action_space=("move_up", "move_right"),
        rng=random.Random(6),
    )

    assert action == "move_up"


def test_heuristic_baseline_avoids_exit_step_while_siphons_remain() -> None:
    agent = HeuristicBaselineAgent()
    state = _snapshot(
        health=9,
        player=GridPosition(0, 0),
        exit_pos=GridPosition(1, 0),
        siphons=(GridPosition(0, 1),),
    )

    action = agent.select_action(
        state=state,
        action_space=("move_right", "move_up"),
        rng=random.Random(7),
    )

    assert action == "move_up"


def test_heuristic_baseline_never_fallbacks_to_exit_when_siphons_remain() -> None:
    agent = HeuristicBaselineAgent()
    state = _snapshot(
        health=9,
        player=GridPosition(0, 0),
        exit_pos=GridPosition(1, 0),
        siphons=(GridPosition(0, 1),),
    )

    action = agent.select_action(
        state=state,
        action_space=("move_right", "space"),
        rng=random.Random(8),
    )

    assert action == "space"


def test_heuristic_baseline_verbose_logging_emits_chosen_action(
    caplog,
) -> None:
    caplog.set_level(logging.INFO)
    agent = HeuristicBaselineAgent(
        config=HeuristicBaselineConfig(verbose_action_logging=True)
    )
    state = _snapshot(health=9, player=GridPosition(1, 1), exit_pos=GridPosition(2, 1))

    action = agent.select_action(
        state=state,
        action_space=("move_left", "move_right", "wait"),
        rng=random.Random(1),
    )

    assert action == "move_right"
    assert "heuristic_action" in caplog.text
    assert "choice=move_right" in caplog.text


def test_heuristic_uses_show_prog_when_recon_value_is_high() -> None:
    agent = HeuristicBaselineAgent(
        config=HeuristicBaselineConfig(
            enable_prog_usage=True,
            prog_energy_floor=1,
        )
    )
    unknown_walls = (
        MapCellState(position=GridPosition(1, 0), cell_type=1, tile_variant=0, wall_state=0, is_wall=True),
        MapCellState(position=GridPosition(2, 0), cell_type=1, tile_variant=0, wall_state=0, is_wall=True),
    )
    state = _snapshot(
        energy=10,
        player=GridPosition(0, 0),
        exit_pos=GridPosition(5, 5),
        siphons=(GridPosition(0, 1),),
        cells=unknown_walls,
        inventory=_inventory(2),
    )

    action = agent.select_action(
        state=state,
        action_space=("prog_slot_1", "move_up"),
        rng=random.Random(31),
    )

    assert action == "prog_slot_1"


def test_heuristic_uses_delay_prog_under_immediate_pressure() -> None:
    enemy = EnemyState(slot=1, type_id=1, position=GridPosition(1, 0), hp=1, state=1, in_bounds=True)
    agent = HeuristicBaselineAgent(
        config=HeuristicBaselineConfig(
            low_health_threshold=3,
            enable_prog_usage=True,
            prog_energy_floor=1,
        )
    )
    state = _snapshot(
        health=2,
        energy=10,
        player=GridPosition(0, 0),
        exit_pos=GridPosition(5, 5),
        enemies=(enemy,),
        siphons=(GridPosition(0, 1),),
        inventory=_inventory(7),
    )

    action = agent.select_action(
        state=state,
        action_space=("prog_slot_1", "wait", "move_up"),
        rng=random.Random(32),
    )

    assert action == "prog_slot_1"


def test_heuristic_skips_prog_when_energy_is_below_floor() -> None:
    agent = HeuristicBaselineAgent(
        config=HeuristicBaselineConfig(
            enable_prog_usage=True,
            prog_energy_floor=4,
        )
    )
    unknown_walls = (
        MapCellState(position=GridPosition(1, 0), cell_type=1, tile_variant=0, wall_state=0, is_wall=True),
        MapCellState(position=GridPosition(2, 0), cell_type=1, tile_variant=0, wall_state=0, is_wall=True),
    )
    state = _snapshot(
        energy=2,
        player=GridPosition(0, 0),
        exit_pos=GridPosition(5, 5),
        siphons=(GridPosition(0, 1),),
        cells=unknown_walls,
        inventory=_inventory(2),
    )

    action = agent.select_action(
        state=state,
        action_space=("prog_slot_1", "move_up"),
        rng=random.Random(33),
    )

    assert action == "move_up"


def test_heuristic_backs_off_prog_slot_after_ineffective_attempt() -> None:
    agent = HeuristicBaselineAgent(
        config=HeuristicBaselineConfig(
            enable_prog_usage=True,
            prog_energy_floor=1,
            prog_retry_backoff_steps=4,
            show_recast_gap_steps=0,
        )
    )
    unknown_walls = (
        MapCellState(position=GridPosition(1, 0), cell_type=1, tile_variant=0, wall_state=0, is_wall=True),
        MapCellState(position=GridPosition(2, 0), cell_type=1, tile_variant=0, wall_state=0, is_wall=True),
    )
    before = _snapshot(
        energy=10,
        player=GridPosition(0, 0),
        exit_pos=GridPosition(5, 5),
        siphons=(GridPosition(0, 1),),
        cells=unknown_walls,
        inventory=_inventory(2),
    )
    after = _snapshot(
        energy=10,
        player=GridPosition(0, 0),
        exit_pos=GridPosition(5, 5),
        siphons=(GridPosition(0, 1),),
        cells=unknown_walls,
        inventory=_inventory(2),
    )

    first_action = agent.select_action(
        state=before,
        action_space=("prog_slot_1", "move_up"),
        rng=random.Random(34),
    )
    second_action = agent.select_action(
        state=after,
        action_space=("prog_slot_1", "move_up"),
        rng=random.Random(34),
    )

    assert first_action == "prog_slot_1"
    assert second_action == "move_up"


def test_heuristic_post_siphon_prefers_resources_and_harvests_with_space() -> None:
    agent = HeuristicBaselineAgent(
        config=HeuristicBaselineConfig(
            resource_goal_weight=1.0,
            prog_goal_weight=0.0,
            points_goal_weight=0.0,
        )
    )
    before = _snapshot(
        player=GridPosition(1, 1),
        exit_pos=GridPosition(5, 5),
        siphons=(GridPosition(4, 4), GridPosition(5, 4)),
    )
    rich_cells = (
        MapCellState(position=GridPosition(1, 1), cell_type=0, tile_variant=0, wall_state=0, credits=2, energy=2),
        MapCellState(position=GridPosition(1, 2), cell_type=0, tile_variant=0, wall_state=0, credits=1, energy=1),
        MapCellState(position=GridPosition(2, 1), cell_type=0, tile_variant=0, wall_state=0, credits=1, energy=0),
    )
    after = _snapshot(
        player=GridPosition(1, 1),
        exit_pos=GridPosition(5, 5),
        siphons=(GridPosition(5, 4),),
        cells=rich_cells,
    )

    _ = agent.select_action(state=before, action_space=("move_up", "move_right", "space"), rng=random.Random(21))
    action = agent.select_action(
        state=after,
        action_space=("move_up", "move_right", "space"),
        rng=random.Random(21),
    )

    assert action == "space"


def test_heuristic_post_siphon_prog_plan_targets_preferred_prog_wall() -> None:
    agent = HeuristicBaselineAgent(
        config=HeuristicBaselineConfig(
            resource_goal_weight=0.0,
            prog_goal_weight=1.0,
            points_goal_weight=0.0,
        )
    )
    before = _snapshot(
        player=GridPosition(0, 0),
        exit_pos=GridPosition(5, 5),
        siphons=(GridPosition(4, 4), GridPosition(5, 4)),
    )
    prog_wall = MapCellState(
        position=GridPosition(1, 0),
        cell_type=1,
        tile_variant=0,
        wall_state=0,
        prog_id=10,  # .debug
        is_wall=True,
    )
    after = _snapshot(
        player=GridPosition(0, 0),
        exit_pos=GridPosition(5, 5),
        siphons=(GridPosition(5, 4),),
        cells=(prog_wall,),
    )

    _ = agent.select_action(state=before, action_space=("move_right", "space"), rng=random.Random(22))
    action = agent.select_action(
        state=after,
        action_space=("move_right", "space"),
        rng=random.Random(22),
    )

    assert action == "space"


def test_heuristic_post_siphon_points_plan_moves_toward_highest_points_wall() -> None:
    agent = HeuristicBaselineAgent(
        config=HeuristicBaselineConfig(
            resource_goal_weight=0.0,
            prog_goal_weight=0.0,
            points_goal_weight=1.0,
        )
    )
    before = _snapshot(
        player=GridPosition(0, 0),
        exit_pos=GridPosition(5, 5),
        siphons=(GridPosition(4, 4), GridPosition(5, 4)),
    )
    low_points_wall = MapCellState(
        position=GridPosition(2, 0),
        cell_type=2,
        tile_variant=0,
        wall_state=0,
        points=3,
        is_wall=True,
    )
    high_points_wall = MapCellState(
        position=GridPosition(0, 2),
        cell_type=2,
        tile_variant=0,
        wall_state=0,
        points=9,
        is_wall=True,
    )
    after = _snapshot(
        player=GridPosition(0, 0),
        exit_pos=GridPosition(5, 5),
        siphons=(GridPosition(5, 4),),
        cells=(low_points_wall, high_points_wall),
    )

    _ = agent.select_action(
        state=before,
        action_space=("move_up", "move_right", "space"),
        rng=random.Random(23),
    )
    action = agent.select_action(
        state=after,
        action_space=("move_up", "move_right", "space"),
        rng=random.Random(23),
    )

    assert action == "move_up"
