"""Tests for baseline agent policies."""

from __future__ import annotations

import random

from src.agent.baseline_heuristic import HeuristicBaselineAgent
from src.agent.baseline_random import RandomBaselineAgent
from src.state.schema import EnemyState, FieldState, GameStateSnapshot, GridPosition, MapState


def _field(value: object) -> FieldState:
    return FieldState(value=value, status="ok")


def _snapshot(
    *,
    health: int = 10,
    player: GridPosition | None = None,
    exit_pos: GridPosition | None = None,
    enemies: tuple[EnemyState, ...] = (),
) -> GameStateSnapshot:
    map_state = MapState(
        status="ok" if player is not None else "missing",
        player_position=player,
        exit_position=exit_pos,
        enemies=enemies,
    )
    return GameStateSnapshot(
        timestamp_utc="2026-03-06T00:00:00+00:00",
        health=_field(health),
        energy=_field(0),
        currency=_field(0),
        fail_state=_field(False),
        map=map_state,
    )


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


def test_heuristic_baseline_moves_away_from_adjacent_enemy() -> None:
    enemy = EnemyState(slot=0, type_id=1, position=GridPosition(2, 1), hp=1, state=0, in_bounds=True)
    agent = HeuristicBaselineAgent()
    state = _snapshot(
        health=8,
        player=GridPosition(1, 1),
        exit_pos=GridPosition(2, 1),
        enemies=(enemy,),
    )

    action = agent.select_action(
        state=state,
        action_space=("move_left", "move_right", "wait"),
        rng=random.Random(2),
    )

    assert action == "move_left"
