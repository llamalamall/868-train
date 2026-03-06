"""Simple heuristic baseline policy."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from src.state.schema import EnemyState, GameStateSnapshot, GridPosition


_MOVE_VECTORS: dict[str, tuple[int, int]] = {
    "move_up": (0, -1),
    "move_down": (0, 1),
    "move_left": (-1, 0),
    "move_right": (1, 0),
}


@dataclass(frozen=True)
class HeuristicBaselineConfig:
    """Tunable knobs for rule-based action selection."""

    low_health_threshold: int = 3
    avoid_enemy_distance: int = 1


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
            return "wait"

        if state.map.status == "ok" and state.map.player_position is not None:
            move_action = self._select_escape_move(state=state, action_space=actions)
            if move_action is not None:
                return move_action

            goal_action = self._select_goal_move(state=state, action_space=actions)
            if goal_action is not None:
                return goal_action

        if "confirm" in actions:
            return "confirm"
        if "wait" in actions:
            return "wait"

        return str(rng.choice(actions))

    def _select_goal_move(self, *, state: GameStateSnapshot, action_space: tuple[str, ...]) -> str | None:
        player = state.map.player_position
        goal = state.map.exit_position
        if player is None or goal is None:
            return None

        current_distance = _manhattan(player, goal)
        best_action: str | None = None
        best_distance = current_distance

        for action in action_space:
            delta = _MOVE_VECTORS.get(action)
            if delta is None:
                continue
            candidate = GridPosition(x=player.x + delta[0], y=player.y + delta[1])
            distance = _manhattan(candidate, goal)
            if distance < best_distance:
                best_distance = distance
                best_action = action

        return best_action

    def _select_escape_move(
        self,
        *,
        state: GameStateSnapshot,
        action_space: tuple[str, ...],
    ) -> str | None:
        player = state.map.player_position
        if player is None:
            return None

        enemies = tuple(state.map.enemies)
        if not enemies:
            return None

        current_distance = _nearest_enemy_distance(player, enemies)
        if current_distance > self.config.avoid_enemy_distance:
            return None

        best_action: str | None = None
        best_distance = current_distance
        for action in action_space:
            delta = _MOVE_VECTORS.get(action)
            if delta is None:
                continue
            candidate = GridPosition(x=player.x + delta[0], y=player.y + delta[1])
            distance = _nearest_enemy_distance(candidate, enemies)
            if distance > best_distance:
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


def _manhattan(a: GridPosition, b: GridPosition) -> int:
    return abs(a.x - b.x) + abs(a.y - b.y)


def _nearest_enemy_distance(player: GridPosition, enemies: tuple[EnemyState, ...]) -> int:
    return min(_manhattan(player, enemy.position) for enemy in enemies)
