"""Simple heuristic baseline policy."""

from __future__ import annotations

from collections import deque
import logging
import random
from dataclasses import dataclass
from typing import Sequence

from src.state.schema import GameStateSnapshot, GridPosition

LOGGER = logging.getLogger(__name__)

_MOVE_VECTORS: dict[str, tuple[int, int]] = {
    "move_up": (0, 1),
    "move_down": (0, -1),
    "move_left": (-1, 0),
    "move_right": (1, 0),
}


@dataclass(frozen=True)
class HeuristicBaselineConfig:
    """Tunable knobs for rule-based action selection."""

    low_health_threshold: int = 3
    verbose_action_logging: bool = False


@dataclass(frozen=True)
class HeuristicBaselineAgent:
    """Rule-based policy that prioritizes survival and visible objectives."""

    config: HeuristicBaselineConfig = HeuristicBaselineConfig()

    def select_action(
        self,
        *,
        state: GameStateSnapshot,
        action_space: Sequence[str],
        rng: random.Random,
    ) -> str:
        """Choose an action from available actions using simple tactical rules."""
        actions = tuple(action_space)
        if not actions:
            raise ValueError("action_space must include at least one action.")

        actions = self._filter_exit_steps_while_siphons_remain(state=state, action_space=actions)
        if not actions:
            raise ValueError("No safe actions available after applying siphon-before-exit policy.")

        health_value = self._coerce_int(state.health.value) if state.health.status == "ok" else None
        if (
            health_value is not None
            and health_value <= self.config.low_health_threshold
            and "wait" in actions
        ):
            return self._log_choice(
                state=state,
                action="wait",
                reason="low_health_wait",
                action_space=actions,
            )

        if state.map.status == "ok" and state.map.player_position is not None:
            enemy_action = self._select_enemy_sight_move(state=state, action_space=actions)
            if enemy_action is not None:
                return self._log_choice(
                    state=state,
                    action=enemy_action,
                    reason="pursue_enemy_in_sight",
                    action_space=actions,
                )

            siphon_action = self._select_siphon_move(state=state, action_space=actions)
            if siphon_action is not None:
                return self._log_choice(
                    state=state,
                    action=siphon_action,
                    reason="collect_siphon",
                    action_space=actions,
                )

            if not state.map.siphons:
                exit_action = self._select_exit_move(state=state, action_space=actions)
                if exit_action is not None:
                    return self._log_choice(
                        state=state,
                        action=exit_action,
                        reason="move_toward_exit",
                        action_space=actions,
                    )

        if "confirm" in actions:
            return self._log_choice(
                state=state,
                action="confirm",
                reason="fallback_confirm",
                action_space=actions,
            )
        if "wait" in actions:
            return self._log_choice(
                state=state,
                action="wait",
                reason="fallback_wait",
                action_space=actions,
            )

        return self._log_choice(
            state=state,
            action=str(rng.choice(actions)),
            reason="fallback_random",
            action_space=actions,
        )

    def _filter_exit_steps_while_siphons_remain(
        self,
        *,
        state: GameStateSnapshot,
        action_space: tuple[str, ...],
    ) -> tuple[str, ...]:
        if state.map.status != "ok" or not state.map.siphons:
            return action_space
        if state.map.player_position is None or state.map.exit_position is None:
            return action_space

        player = state.map.player_position
        exit_position = state.map.exit_position
        filtered = tuple(
            action
            for action in action_space
            if not _action_steps_to_position(action=action, start=player, target=exit_position)
        )
        return filtered

    def _select_exit_move(self, *, state: GameStateSnapshot, action_space: tuple[str, ...]) -> str | None:
        player = state.map.player_position
        exit_position = state.map.exit_position
        if player is None or exit_position is None:
            return None
        route = _shortest_path_first_action(
            start=player,
            target=exit_position,
            width=state.map.width,
            height=state.map.height,
            walls=_wall_positions(state),
            allowed_first_actions=_movement_actions(action_space),
        )
        return route.action if route is not None else None

    def _select_path_to_nearest_target(
        self,
        *,
        start: GridPosition,
        targets: tuple[GridPosition, ...],
        state: GameStateSnapshot,
        action_space: tuple[str, ...],
    ) -> str | None:
        if not targets:
            return None
        walls = _wall_positions(state)
        movement_actions = _movement_actions(action_space)
        best_action: str | None = None
        best_distance: int | None = None
        for target in targets:
            route = _shortest_path_first_action(
                start=start,
                target=target,
                width=state.map.width,
                height=state.map.height,
                walls=walls,
                allowed_first_actions=movement_actions,
            )
            if route is None:
                continue
            if best_distance is None or route.distance < best_distance:
                best_distance = route.distance
                best_action = route.action
        return best_action

    def _select_siphon_move(self, *, state: GameStateSnapshot, action_space: tuple[str, ...]) -> str | None:
        player = state.map.player_position
        if player is None or not state.map.siphons:
            return None

        return self._select_path_to_nearest_target(
            start=player,
            targets=tuple(state.map.siphons),
            state=state,
            action_space=action_space,
        )

    def _select_enemy_sight_move(
        self,
        *,
        state: GameStateSnapshot,
        action_space: tuple[str, ...],
    ) -> str | None:
        player = state.map.player_position
        if player is None:
            return None

        wall_positions = _wall_positions(state)

        target_enemy: GridPosition | None = None
        target_distance: int | None = None
        for enemy in state.map.enemies:
            enemy_position = enemy.position
            if enemy_position.x == player.x:
                if not _is_vertical_line_clear(player=player, target=enemy_position, walls=wall_positions):
                    continue
            elif enemy_position.y == player.y:
                if not _is_horizontal_line_clear(player=player, target=enemy_position, walls=wall_positions):
                    continue
            else:
                continue

            distance = _manhattan(player, enemy_position)
            if target_distance is None or distance < target_distance:
                target_distance = distance
                target_enemy = enemy_position

        if target_enemy is None:
            return None
        route = _shortest_path_first_action(
            start=player,
            target=target_enemy,
            width=state.map.width,
            height=state.map.height,
            walls=wall_positions,
            allowed_first_actions=_movement_actions(action_space),
        )
        return route.action if route is not None else None

    @staticmethod
    def _coerce_int(value: object | None) -> int | None:
        try:
            if value is None or isinstance(value, bool):
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    def _log_choice(
        self,
        *,
        state: GameStateSnapshot,
        action: str,
        reason: str,
        action_space: tuple[str, ...],
    ) -> str:
        if self.config.verbose_action_logging:
            LOGGER.info(
                "heuristic_action choice=%s reason=%s health=%s player=%s exit=%s available=%s",
                action,
                reason,
                state.health.value if state.health.status == "ok" else None,
                state.map.player_position,
                state.map.exit_position,
                ",".join(action_space),
            )
        return action


def _manhattan(a: GridPosition, b: GridPosition) -> int:
    return abs(a.x - b.x) + abs(a.y - b.y)


def _movement_actions(action_space: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(action for action in action_space if action in _MOVE_VECTORS)


def _wall_positions(state: GameStateSnapshot) -> set[GridPosition]:
    wall_positions = {cell.position for cell in state.map.cells if cell.is_wall}
    if wall_positions:
        return wall_positions
    return {wall.position for wall in state.map.walls}


@dataclass(frozen=True)
class _RouteStep:
    action: str
    distance: int


def _shortest_path_first_action(
    *,
    start: GridPosition,
    target: GridPosition,
    width: int,
    height: int,
    walls: set[GridPosition],
    allowed_first_actions: tuple[str, ...],
) -> _RouteStep | None:
    if start == target:
        return None
    if not allowed_first_actions:
        return None
    if not _is_in_bounds(target, width=width, height=height):
        return None
    if target in walls:
        return None

    queue: deque[GridPosition] = deque([start])
    visited: set[GridPosition] = {start}
    parent: dict[GridPosition, tuple[GridPosition, str]] = {}

    while queue:
        current = queue.popleft()
        if current == target:
            break
        if current == start:
            actions_to_try = allowed_first_actions
        else:
            actions_to_try = tuple(_MOVE_VECTORS.keys())
        for action in actions_to_try:
            dx, dy = _MOVE_VECTORS[action]
            candidate = GridPosition(x=current.x + dx, y=current.y + dy)
            if not _is_in_bounds(candidate, width=width, height=height):
                continue
            if candidate in walls or candidate in visited:
                continue
            visited.add(candidate)
            parent[candidate] = (current, action)
            queue.append(candidate)

    if target not in parent:
        return None

    cursor = target
    first_action: str | None = None
    distance = 0
    while cursor != start:
        previous, action = parent[cursor]
        first_action = action
        distance += 1
        cursor = previous
    if first_action is None:
        return None
    return _RouteStep(action=first_action, distance=distance)


def _is_in_bounds(position: GridPosition, *, width: int, height: int) -> bool:
    return 0 <= position.x < width and 0 <= position.y < height


def _action_steps_to_position(
    *,
    action: str,
    start: GridPosition,
    target: GridPosition,
) -> bool:
    delta = _MOVE_VECTORS.get(action)
    if delta is None:
        return False
    return GridPosition(x=start.x + delta[0], y=start.y + delta[1]) == target


def _is_horizontal_line_clear(
    *,
    player: GridPosition,
    target: GridPosition,
    walls: set[GridPosition],
) -> bool:
    if player.y != target.y:
        return False
    start = min(player.x, target.x) + 1
    end = max(player.x, target.x)
    for x in range(start, end):
        if GridPosition(x=x, y=player.y) in walls:
            return False
    return True


def _is_vertical_line_clear(
    *,
    player: GridPosition,
    target: GridPosition,
    walls: set[GridPosition],
) -> bool:
    if player.x != target.x:
        return False
    start = min(player.y, target.y) + 1
    end = max(player.y, target.y)
    for y in range(start, end):
        if GridPosition(x=player.x, y=y) in walls:
            return False
    return True
