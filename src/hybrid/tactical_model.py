"""Deterministic tactical rules shared across hybrid control code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.state.schema import EnemyState, GameStateSnapshot, GridPosition

ENEMY_TYPE_VIRUS = 2
ENEMY_TYPE_CLASSIC = 3
ENEMY_TYPE_GLITCH = 4
ENEMY_TYPE_CRYPTOG = 5

MOVE_DELTAS: dict[str, tuple[int, int]] = {
    "move_up": (0, 1),
    "move_down": (0, -1),
    "move_left": (-1, 0),
    "move_right": (1, 0),
}


@dataclass(frozen=True)
class TacticalRiskSnapshot:
    """Risk summary for standing on a position over a short enemy horizon."""

    immediate_damage: bool
    horizon_damage_steps: int
    nearest_enemy_distance_after_one_turn: int | None


def wall_positions(state: GameStateSnapshot) -> set[GridPosition]:
    if state.map.status != "ok":
        return set()
    walls = {cell.position for cell in state.map.cells if cell.is_wall}
    if walls:
        return walls
    return {wall.position for wall in state.map.walls}


def move_target_position(
    *,
    state: GameStateSnapshot,
    action: str,
) -> GridPosition | None:
    if state.map.status != "ok" or state.map.player_position is None:
        return None
    player = state.map.player_position
    delta = MOVE_DELTAS.get(action)
    if delta is None:
        return player
    candidate = GridPosition(x=player.x + delta[0], y=player.y + delta[1])
    if not is_in_bounds(candidate, width=state.map.width, height=state.map.height):
        return None
    if candidate in wall_positions(state):
        return None
    return candidate


def siphon_spawn_cost_at_position(
    *,
    state: GameStateSnapshot,
    position: GridPosition | None,
) -> int:
    if state.map.status != "ok" or position is None:
        return 0
    layers = state.map.layers
    if len(layers.siphon_penalty_map) == state.map.height:
        total = 0
        for candidate in (
            position,
            GridPosition(x=position.x + 1, y=position.y),
            GridPosition(x=position.x - 1, y=position.y),
            GridPosition(x=position.x, y=position.y + 1),
            GridPosition(x=position.x, y=position.y - 1),
        ):
            if not is_in_bounds(candidate, width=state.map.width, height=state.map.height):
                continue
            row = layers.siphon_penalty_map[candidate.y]
            if len(row) != state.map.width:
                continue
            try:
                total += max(int(row[candidate.x]), 0)
            except (TypeError, ValueError):
                continue
        if total > 0:
            return total

    total = 0
    for cell in state.map.cells:
        if not cell.is_wall:
            continue
        if manhattan(cell.position, position) > 1:
            continue
        total += max(int(getattr(cell, "threat", 0)), int(getattr(cell, "wall_state", 0)), 0)
    return total


def estimate_position_risk(
    *,
    state: GameStateSnapshot,
    position: GridPosition | None,
    horizon_turns: int,
) -> TacticalRiskSnapshot:
    if (
        state.map.status != "ok"
        or position is None
        or horizon_turns <= 0
    ):
        return TacticalRiskSnapshot(
            immediate_damage=False,
            horizon_damage_steps=0,
            nearest_enemy_distance_after_one_turn=None,
        )

    walls = wall_positions(state)
    enemies = tuple(enemy for enemy in state.map.enemies if enemy.in_bounds)
    current_enemies = enemies
    immediate_damage = False
    horizon_damage_steps = 0
    nearest_after_one_turn: int | None = None
    for turn_index in range(max(int(horizon_turns), 1)):
        predicted_enemies, took_damage = simulate_enemy_turn(
            enemies=current_enemies,
            player_position=position,
            width=state.map.width,
            height=state.map.height,
            walls=walls,
        )
        if turn_index == 0:
            immediate_damage = bool(took_damage)
            nearest_after_one_turn = nearest_enemy_distance(
                enemies=predicted_enemies,
                player_position=position,
            )
        if took_damage:
            horizon_damage_steps += 1
        current_enemies = predicted_enemies
    return TacticalRiskSnapshot(
        immediate_damage=immediate_damage,
        horizon_damage_steps=horizon_damage_steps,
        nearest_enemy_distance_after_one_turn=nearest_after_one_turn,
    )


def simulate_enemy_turn(
    *,
    enemies: tuple[EnemyState, ...],
    player_position: GridPosition,
    width: int,
    height: int,
    walls: set[GridPosition],
) -> tuple[tuple[EnemyState, ...], bool]:
    predicted: list[EnemyState] = []
    took_damage = False
    for enemy in enemies:
        if not enemy.in_bounds:
            predicted.append(enemy)
            continue

        current_position = enemy.position
        if enemy_can_attack_position(
            enemy_type=enemy.type_id,
            enemy_position=current_position,
            player_position=player_position,
        ):
            took_damage = True
            predicted.append(enemy)
            continue

        if enemy.type_id == ENEMY_TYPE_VIRUS:
            first_step = predict_enemy_substep(
                enemy_type=enemy.type_id,
                enemy_position=current_position,
                player_position=player_position,
                width=width,
                height=height,
                walls=walls,
            )
            current_position = first_step
            if enemy_can_attack_position(
                enemy_type=enemy.type_id,
                enemy_position=current_position,
                player_position=player_position,
            ):
                took_damage = True
            else:
                current_position = predict_enemy_substep(
                    enemy_type=enemy.type_id,
                    enemy_position=current_position,
                    player_position=player_position,
                    width=width,
                    height=height,
                    walls=walls,
                )
        else:
            current_position = predict_enemy_substep(
                enemy_type=enemy.type_id,
                enemy_position=current_position,
                player_position=player_position,
                width=width,
                height=height,
                walls=walls,
            )

        predicted.append(
            EnemyState(
                slot=enemy.slot,
                type_id=enemy.type_id,
                position=current_position,
                hp=enemy.hp,
                state=enemy.state,
                in_bounds=enemy.in_bounds,
                incubation_timer=enemy.incubation_timer,
            )
        )
    return (tuple(predicted), took_damage)


def nearest_enemy_distance(
    *,
    enemies: Sequence[EnemyState],
    player_position: GridPosition,
) -> int | None:
    in_bounds_positions = [enemy.position for enemy in enemies if enemy.in_bounds]
    if not in_bounds_positions:
        return None
    return min(manhattan(enemy_position, player_position) for enemy_position in in_bounds_positions)


def enemy_can_attack_position(
    *,
    enemy_type: int,
    enemy_position: GridPosition,
    player_position: GridPosition,
) -> bool:
    if enemy_type == ENEMY_TYPE_VIRUS:
        return chebyshev(enemy_position, player_position) <= 1
    return manhattan(enemy_position, player_position) <= 1


def predict_enemy_substep(
    *,
    enemy_type: int,
    enemy_position: GridPosition,
    player_position: GridPosition,
    width: int,
    height: int,
    walls: set[GridPosition],
) -> GridPosition:
    can_pass_walls = enemy_type == ENEMY_TYPE_GLITCH
    candidates: list[GridPosition] = []
    for dx, dy in MOVE_DELTAS.values():
        candidate = GridPosition(x=enemy_position.x + dx, y=enemy_position.y + dy)
        if not is_in_bounds(candidate, width=width, height=height):
            continue
        if not can_pass_walls and candidate in walls:
            continue
        candidates.append(candidate)
    candidates.append(enemy_position)
    if not candidates:
        return enemy_position

    distances = {candidate: manhattan(candidate, player_position) for candidate in candidates}
    current_distance = manhattan(enemy_position, player_position)
    best_distance = min(distances.values())
    best_candidates = [candidate for candidate in candidates if distances[candidate] == best_distance]

    if enemy_type == ENEMY_TYPE_GLITCH:
        reducing_candidates = [
            candidate for candidate in candidates if distances[candidate] < current_distance
        ]
        if reducing_candidates:
            wall_reducing = [candidate for candidate in reducing_candidates if candidate in walls]
            if wall_reducing:
                return wall_reducing[0]
            best_reducing_distance = min(distances[candidate] for candidate in reducing_candidates)
            best_reducing = [
                candidate
                for candidate in reducing_candidates
                if distances[candidate] == best_reducing_distance
            ]
            return best_reducing[0]
        return best_candidates[0]

    if enemy_type == ENEMY_TYPE_CRYPTOG:
        hidden_best = [
            candidate
            for candidate in best_candidates
            if not has_line_of_sight(player=player_position, target=candidate, walls=walls)
        ]
        if hidden_best:
            return hidden_best[0]
    return best_candidates[0]


def is_in_bounds(position: GridPosition, *, width: int, height: int) -> bool:
    return 0 <= position.x < width and 0 <= position.y < height


def manhattan(a: GridPosition, b: GridPosition) -> int:
    return abs(a.x - b.x) + abs(a.y - b.y)


def chebyshev(a: GridPosition, b: GridPosition) -> int:
    return max(abs(a.x - b.x), abs(a.y - b.y))


def has_line_of_sight(
    *,
    player: GridPosition,
    target: GridPosition,
    walls: set[GridPosition],
) -> bool:
    if player == target:
        return True
    if player.x == target.x:
        start = min(player.y, target.y) + 1
        end = max(player.y, target.y)
        for y in range(start, end):
            if GridPosition(x=player.x, y=y) in walls:
                return False
        return True
    if player.y == target.y:
        start = min(player.x, target.x) + 1
        end = max(player.x, target.x)
        for x in range(start, end):
            if GridPosition(x=x, y=player.y) in walls:
                return False
        return True
    return False
