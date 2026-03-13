"""Deterministic grid movement planner based on A* search."""

from __future__ import annotations

import heapq
from dataclasses import dataclass

from src.state.schema import GridPosition

_MOVE_DELTAS: dict[str, tuple[int, int]] = {
    "move_up": (0, 1),
    "move_right": (1, 0),
    "move_down": (0, -1),
    "move_left": (-1, 0),
}
_MOVE_ORDER: tuple[str, ...] = ("move_up", "move_right", "move_down", "move_left")


@dataclass(frozen=True)
class RoutePlan:
    """A path plan from start to target."""

    actions: tuple[str, ...]
    distance: int


def _manhattan(a: GridPosition, b: GridPosition) -> int:
    return abs(a.x - b.x) + abs(a.y - b.y)


def _is_in_bounds(position: GridPosition, *, width: int, height: int) -> bool:
    return 0 <= position.x < width and 0 <= position.y < height


class AStarMovementController:
    """Deterministic shortest-path planner for movement actions."""

    def plan_route(
        self,
        *,
        start: GridPosition,
        target: GridPosition,
        width: int,
        height: int,
        walls: set[GridPosition],
        allowed_first_actions: tuple[str, ...] | None = None,
    ) -> RoutePlan | None:
        """Return deterministic shortest route plan or None if unreachable."""
        if start == target:
            return RoutePlan(actions=(), distance=0)
        if start in walls or target in walls:
            return None
        if not _is_in_bounds(start, width=width, height=height):
            return None
        if not _is_in_bounds(target, width=width, height=height):
            return None

        first_actions = (
            tuple(action for action in allowed_first_actions or () if action in _MOVE_DELTAS)
            or _MOVE_ORDER
        )
        open_heap: list[tuple[int, int, int, GridPosition]] = []
        push_counter = 0
        start_h = _manhattan(start, target)
        heapq.heappush(open_heap, (start_h, 0, push_counter, start))
        best_cost: dict[GridPosition, int] = {start: 0}
        parent: dict[GridPosition, tuple[GridPosition, str]] = {}

        while open_heap:
            _, cost_so_far, _, current = heapq.heappop(open_heap)
            known = best_cost.get(current)
            if known is None or known != cost_so_far:
                continue
            if current == target:
                break

            actions_to_try = first_actions if current == start else _MOVE_ORDER
            for action in actions_to_try:
                dx, dy = _MOVE_DELTAS[action]
                candidate = GridPosition(x=current.x + dx, y=current.y + dy)
                if not _is_in_bounds(candidate, width=width, height=height):
                    continue
                if candidate in walls:
                    continue
                tentative_cost = cost_so_far + 1
                previous_best = best_cost.get(candidate)
                if previous_best is not None and tentative_cost >= previous_best:
                    continue
                best_cost[candidate] = tentative_cost
                parent[candidate] = (current, action)
                push_counter += 1
                priority = tentative_cost + _manhattan(candidate, target)
                heapq.heappush(open_heap, (priority, tentative_cost, push_counter, candidate))

        if target not in parent:
            return None

        reversed_actions: list[str] = []
        cursor = target
        while cursor != start:
            previous, action = parent[cursor]
            reversed_actions.append(action)
            cursor = previous
        reversed_actions.reverse()
        return RoutePlan(actions=tuple(reversed_actions), distance=len(reversed_actions))

    def next_action(
        self,
        *,
        start: GridPosition,
        target: GridPosition,
        width: int,
        height: int,
        walls: set[GridPosition],
        available_actions: tuple[str, ...],
    ) -> str | None:
        """Return deterministic next movement action for the route."""
        movement_actions = tuple(action for action in available_actions if action in _MOVE_DELTAS)
        if not movement_actions:
            return None
        plan = self.plan_route(
            start=start,
            target=target,
            width=width,
            height=height,
            walls=walls,
            allowed_first_actions=movement_actions,
        )
        if plan is None or not plan.actions:
            return None
        next_move = plan.actions[0]
        if next_move not in movement_actions:
            return None
        return next_move

