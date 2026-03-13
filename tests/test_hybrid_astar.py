"""Tests for deterministic hybrid A* movement planning."""

from __future__ import annotations

from src.hybrid.astar_controller import AStarMovementController
from src.state.schema import GridPosition


def test_astar_returns_expected_shortest_route() -> None:
    controller = AStarMovementController()
    plan = controller.plan_route(
        start=GridPosition(0, 0),
        target=GridPosition(2, 0),
        width=6,
        height=6,
        walls=set(),
    )

    assert plan is not None
    assert plan.distance == 2
    assert plan.actions == ("move_right", "move_right")


def test_astar_tie_breaking_is_deterministic() -> None:
    controller = AStarMovementController()
    action = controller.next_action(
        start=GridPosition(0, 0),
        target=GridPosition(1, 1),
        width=6,
        height=6,
        walls=set(),
        available_actions=("move_up", "move_right", "move_down", "move_left"),
    )

    assert action == "move_up"


def test_astar_returns_none_when_target_unreachable() -> None:
    controller = AStarMovementController()
    walls = {GridPosition(1, 0), GridPosition(0, 1)}
    action = controller.next_action(
        start=GridPosition(0, 0),
        target=GridPosition(2, 2),
        width=6,
        height=6,
        walls=walls,
        available_actions=("move_up", "move_right"),
    )

    assert action is None


def test_astar_prefers_safe_detour_when_blocked_positions_have_alternative_route() -> None:
    controller = AStarMovementController()
    plan = controller.plan_route(
        start=GridPosition(0, 0),
        target=GridPosition(2, 0),
        width=6,
        height=6,
        walls=set(),
        blocked_positions={GridPosition(1, 0)},
    )

    assert plan is not None
    assert plan.actions == ("move_up", "move_right", "move_right", "move_down")


def test_astar_falls_back_to_soft_route_when_blocked_positions_make_target_unreachable() -> None:
    controller = AStarMovementController()
    plan = controller.plan_route(
        start=GridPosition(0, 0),
        target=GridPosition(2, 0),
        width=6,
        height=6,
        walls={GridPosition(0, 1), GridPosition(1, 1), GridPosition(2, 1)},
        blocked_positions={GridPosition(1, 0)},
    )

    assert plan is not None
    assert plan.actions == ("move_right", "move_right")
