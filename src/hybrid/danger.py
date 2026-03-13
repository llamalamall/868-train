"""Shared enemy movement and threat prediction utilities for hybrid control."""

from __future__ import annotations

from dataclasses import dataclass

from src.state.schema import EnemyState, GridPosition

ENEMY_TYPE_VIRUS = 2
ENEMY_TYPE_CLASSIC = 3
ENEMY_TYPE_GLITCH = 4
ENEMY_TYPE_CRYPTOG = 5
ENEMY_TYPE_DAEMON = 7
TRACKED_ENEMY_TYPES: tuple[int, ...] = (
    ENEMY_TYPE_VIRUS,
    ENEMY_TYPE_CLASSIC,
    ENEMY_TYPE_GLITCH,
    ENEMY_TYPE_CRYPTOG,
    ENEMY_TYPE_DAEMON,
)
_MOVE_VECTORS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 0),
    (0, -1),
    (-1, 0),
)


@dataclass(frozen=True)
class EnemyTurnPrediction:
    """Predicted enemy board state and attack information for one enemy turn."""

    enemies: tuple[EnemyState, ...]
    took_damage: bool
    attack_types: tuple[int, ...]


def manhattan(a: GridPosition, b: GridPosition) -> int:
    return abs(a.x - b.x) + abs(a.y - b.y)


def chebyshev(a: GridPosition, b: GridPosition) -> int:
    return max(abs(a.x - b.x), abs(a.y - b.y))


def is_in_bounds(position: GridPosition, *, width: int, height: int) -> bool:
    return 0 <= position.x < width and 0 <= position.y < height


def has_line_of_sight(
    *,
    player: GridPosition,
    target: GridPosition,
    walls: set[GridPosition],
) -> bool:
    if player.x == target.x:
        start = min(player.y, target.y) + 1
        stop = max(player.y, target.y)
        return all(GridPosition(x=player.x, y=y) not in walls for y in range(start, stop))
    if player.y == target.y:
        start = min(player.x, target.x) + 1
        stop = max(player.x, target.x)
        return all(GridPosition(x=x, y=player.y) not in walls for x in range(start, stop))
    return False


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
    for dx, dy in _MOVE_VECTORS:
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
            candidate
            for candidate in candidates
            if distances[candidate] < current_distance
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

    return best_candidates[0]


def simulate_enemy_turn(
    *,
    enemies: tuple[EnemyState, ...],
    player_position: GridPosition,
    width: int,
    height: int,
    walls: set[GridPosition],
) -> EnemyTurnPrediction:
    predicted: list[EnemyState] = []
    attack_types: set[int] = set()
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
            attack_types.add(int(enemy.type_id))
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
                attack_types.add(int(enemy.type_id))
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
            if enemy_can_attack_position(
                enemy_type=enemy.type_id,
                enemy_position=current_position,
                player_position=player_position,
            ):
                took_damage = True
                attack_types.add(int(enemy.type_id))

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

    return EnemyTurnPrediction(
        enemies=tuple(predicted),
        took_damage=took_damage,
        attack_types=tuple(sorted(attack_types)),
    )


def assess_position_danger(
    *,
    position: GridPosition,
    enemies: tuple[EnemyState, ...],
    width: int,
    height: int,
    walls: set[GridPosition],
) -> EnemyTurnPrediction:
    return simulate_enemy_turn(
        enemies=enemies,
        player_position=position,
        width=width,
        height=height,
        walls=walls,
    )
