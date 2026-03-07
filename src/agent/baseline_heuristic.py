"""Simple heuristic baseline policy."""

from __future__ import annotations

from collections import deque
import logging
import random
from dataclasses import dataclass, field
from typing import Literal, Sequence

from src.state.schema import GameStateSnapshot, GridPosition, MapCellState

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
    resource_goal_weight: float = 0.60
    prog_goal_weight: float = 0.30
    points_goal_weight: float = 0.10


@dataclass(frozen=True)
class _HarvestPlan:
    category: Literal["resources", "progs", "points"]
    target_position: GridPosition


@dataclass(frozen=True)
class _WallCandidate:
    position: GridPosition
    prog_id: int | None
    points: int


@dataclass
class HeuristicBaselineAgent:
    """Rule-based policy that prioritizes survival and visible objectives."""

    config: HeuristicBaselineConfig = HeuristicBaselineConfig()
    _last_siphon_count: int | None = field(default=None, init=False, repr=False)
    _harvest_plan: _HarvestPlan | None = field(default=None, init=False, repr=False)

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

        self._update_harvest_plan_on_siphon_change(state=state, rng=rng)

        planned_action, planned_reason = self._execute_harvest_plan(state=state, action_space=actions)
        if planned_action is not None and planned_reason is not None:
            return self._log_choice(
                state=state,
                action=planned_action,
                reason=planned_reason,
                action_space=actions,
            )

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

        return self._log_choice(
            state=state,
            action=str(rng.choice(actions)),
            reason="fallback_random",
            action_space=actions,
        )

    def _update_harvest_plan_on_siphon_change(
        self,
        *,
        state: GameStateSnapshot,
        rng: random.Random,
    ) -> None:
        if state.map.status != "ok":
            self._last_siphon_count = None
            self._harvest_plan = None
            return

        current_siphon_count = len(state.map.siphons)
        if self._last_siphon_count is None:
            self._last_siphon_count = current_siphon_count
            return

        if current_siphon_count > self._last_siphon_count:
            # Likely reset/new episode.
            self._harvest_plan = None
        elif current_siphon_count < self._last_siphon_count:
            self._harvest_plan = None
            if current_siphon_count > 0:
                self._harvest_plan = self._choose_harvest_plan(state=state, rng=rng)

        self._last_siphon_count = current_siphon_count

    def _execute_harvest_plan(
        self,
        *,
        state: GameStateSnapshot,
        action_space: tuple[str, ...],
    ) -> tuple[str | None, str | None]:
        plan = self._harvest_plan
        if plan is None:
            return (None, None)

        if state.map.status != "ok" or state.map.player_position is None:
            self._harvest_plan = None
            return (None, None)
        if not state.map.siphons:
            self._harvest_plan = None
            return (None, None)

        player = state.map.player_position
        if player == plan.target_position:
            if "space" in action_space:
                self._harvest_plan = None
                return ("space", f"harvest_{plan.category}")
            self._harvest_plan = None
            return (None, None)

        route = _shortest_path_first_action(
            start=player,
            target=plan.target_position,
            width=state.map.width,
            height=state.map.height,
            walls=_wall_positions(state),
            allowed_first_actions=_movement_actions(action_space),
        )
        if route is None:
            self._harvest_plan = None
            return (None, None)

        return (route.action, f"move_to_{plan.category}_target")

    def _choose_harvest_plan(
        self,
        *,
        state: GameStateSnapshot,
        rng: random.Random,
    ) -> _HarvestPlan | None:
        plans: dict[str, _HarvestPlan] = {}

        resource_plan = self._build_resource_plan(state=state)
        if resource_plan is not None:
            plans["resources"] = resource_plan

        prog_plan = self._build_prog_plan(state=state)
        if prog_plan is not None:
            plans["progs"] = prog_plan

        points_plan = self._build_points_plan(state=state)
        if points_plan is not None:
            plans["points"] = points_plan

        if not plans:
            return None

        weighted_options: list[tuple[str, float]] = []
        for category, weight in (
            ("resources", self.config.resource_goal_weight),
            ("progs", self.config.prog_goal_weight),
            ("points", self.config.points_goal_weight),
        ):
            if category in plans:
                weighted_options.append((category, max(float(weight), 0.0)))

        if not weighted_options:
            return None

        total_weight = sum(weight for _, weight in weighted_options)
        if total_weight <= 0:
            for category in ("resources", "progs", "points"):
                plan = plans.get(category)
                if plan is not None:
                    return plan
            return None

        pick = rng.random() * total_weight
        cumulative = 0.0
        for category, weight in weighted_options:
            cumulative += weight
            if pick <= cumulative:
                return plans[category]
        return plans[weighted_options[-1][0]]

    def _build_resource_plan(self, *, state: GameStateSnapshot) -> _HarvestPlan | None:
        player = state.map.player_position
        if state.map.status != "ok" or player is None:
            return None

        walls = _wall_positions(state)
        movement_actions = tuple(_MOVE_VECTORS.keys())
        cells_by_position = _cells_by_position(state)
        best_target: GridPosition | None = None
        best_score = 0
        best_distance: int | None = None

        for x in range(state.map.width):
            for y in range(state.map.height):
                position = GridPosition(x=x, y=y)
                if position in walls:
                    continue
                score = _resource_cluster_score(
                    position=position,
                    width=state.map.width,
                    height=state.map.height,
                    cells_by_position=cells_by_position,
                )
                if score <= 0:
                    continue

                if position == player:
                    distance = 0
                else:
                    route = _shortest_path_first_action(
                        start=player,
                        target=position,
                        width=state.map.width,
                        height=state.map.height,
                        walls=walls,
                        allowed_first_actions=movement_actions,
                    )
                    if route is None:
                        continue
                    distance = route.distance

                if (
                    score > best_score
                    or (
                        score == best_score
                        and (best_distance is None or distance < best_distance)
                    )
                ):
                    best_score = score
                    best_distance = distance
                    best_target = position

        if best_target is None:
            return None
        return _HarvestPlan(category="resources", target_position=best_target)

    def _build_prog_plan(self, *, state: GameStateSnapshot) -> _HarvestPlan | None:
        player = state.map.player_position
        if state.map.status != "ok" or player is None:
            return None

        preferred_prog_rank = {10: 0, 5: 1, 9: 2, 11: 3, 8: 4}
        best: tuple[int, int, GridPosition] | None = None

        for wall in _iter_wall_candidates(state):
            if wall.prog_id is None:
                continue
            candidate = _best_adjacent_position(
                player=player,
                target_wall=wall.position,
                state=state,
            )
            if candidate is None:
                continue
            target_position, distance = candidate
            rank = preferred_prog_rank.get(wall.prog_id, len(preferred_prog_rank) + 1)
            score = (rank, distance)
            if best is None or score < (best[0], best[1]):
                best = (rank, distance, target_position)

        if best is None:
            return None
        return _HarvestPlan(category="progs", target_position=best[2])

    def _build_points_plan(self, *, state: GameStateSnapshot) -> _HarvestPlan | None:
        player = state.map.player_position
        if state.map.status != "ok" or player is None:
            return None

        best: tuple[int, int, GridPosition] | None = None
        for wall in _iter_wall_candidates(state):
            if wall.points <= 0:
                continue
            candidate = _best_adjacent_position(
                player=player,
                target_wall=wall.position,
                state=state,
            )
            if candidate is None:
                continue
            target_position, distance = candidate
            # Max points, then shortest path.
            score = (wall.points, -distance)
            if best is None or score > (best[0], best[1]):
                best = (wall.points, -distance, target_position)

        if best is None:
            return None
        return _HarvestPlan(category="points", target_position=best[2])

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


def _cells_by_position(state: GameStateSnapshot) -> dict[GridPosition, MapCellState]:
    return {cell.position: cell for cell in state.map.cells}


def _resource_cluster_score(
    *,
    position: GridPosition,
    width: int,
    height: int,
    cells_by_position: dict[GridPosition, MapCellState],
) -> int:
    total = 0
    neighbors = (
        position,
        GridPosition(position.x + 1, position.y),
        GridPosition(position.x - 1, position.y),
        GridPosition(position.x, position.y + 1),
        GridPosition(position.x, position.y - 1),
    )
    for neighbor in neighbors:
        if not _is_in_bounds(neighbor, width=width, height=height):
            continue
        cell = cells_by_position.get(neighbor)
        if cell is None:
            continue
        total += max(int(cell.credits), 0) + max(int(cell.energy), 0)
    return total


def _iter_wall_candidates(state: GameStateSnapshot) -> tuple[_WallCandidate, ...]:
    walls: list[_WallCandidate] = []
    seen: set[GridPosition] = set()

    for wall in state.map.walls:
        walls.append(
            _WallCandidate(
                position=wall.position,
                prog_id=wall.prog_id,
                points=int(wall.points),
            )
        )
        seen.add(wall.position)

    for cell in state.map.cells:
        if not cell.is_wall or cell.position in seen:
            continue
        walls.append(
            _WallCandidate(
                position=cell.position,
                prog_id=cell.prog_id,
                points=int(cell.points),
            )
        )
    return tuple(walls)


def _best_adjacent_position(
    *,
    player: GridPosition,
    target_wall: GridPosition,
    state: GameStateSnapshot,
) -> tuple[GridPosition, int] | None:
    walls = _wall_positions(state)
    best_target: GridPosition | None = None
    best_distance: int | None = None
    movement_actions = tuple(_MOVE_VECTORS.keys())

    for dx, dy in _MOVE_VECTORS.values():
        adjacent = GridPosition(target_wall.x + dx, target_wall.y + dy)
        if not _is_in_bounds(adjacent, width=state.map.width, height=state.map.height):
            continue
        if adjacent in walls:
            continue

        if adjacent == player:
            distance = 0
        else:
            route = _shortest_path_first_action(
                start=player,
                target=adjacent,
                width=state.map.width,
                height=state.map.height,
                walls=walls,
                allowed_first_actions=movement_actions,
            )
            if route is None:
                continue
            distance = route.distance

        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_target = adjacent

    if best_target is None or best_distance is None:
        return None
    return (best_target, best_distance)


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
