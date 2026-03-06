"""Simple heuristic baseline policy."""

from __future__ import annotations

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

    def _select_exit_move(self, *, state: GameStateSnapshot, action_space: tuple[str, ...]) -> str | None:
        player = state.map.player_position
        exit_position = state.map.exit_position
        if player is None or exit_position is None:
            return None
        return self._select_move_toward_target(
            current=player,
            target=exit_position,
            action_space=action_space,
        )

    def _select_siphon_move(self, *, state: GameStateSnapshot, action_space: tuple[str, ...]) -> str | None:
        player = state.map.player_position
        if player is None or not state.map.siphons:
            return None

        target = min(state.map.siphons, key=lambda position: _manhattan(player, position))
        return self._select_move_toward_target(
            current=player,
            target=target,
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

        wall_positions = {cell.position for cell in state.map.cells if cell.is_wall}
        if not wall_positions and state.map.walls:
            wall_positions = {wall.position for wall in state.map.walls}

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
        return self._select_move_toward_target(
            current=player,
            target=target_enemy,
            action_space=action_space,
        )

    def _select_move_toward_target(
        self,
        *,
        current: GridPosition,
        target: GridPosition,
        action_space: tuple[str, ...],
    ) -> str | None:
        current_distance = _manhattan(current, target)
        best_action: str | None = None
        best_distance = current_distance

        for action in action_space:
            delta = _MOVE_VECTORS.get(action)
            if delta is None:
                continue
            candidate = GridPosition(x=current.x + delta[0], y=current.y + delta[1])
            distance = _manhattan(candidate, target)
            if distance < best_distance:
                best_distance = distance
                best_action = action
        return best_action

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
