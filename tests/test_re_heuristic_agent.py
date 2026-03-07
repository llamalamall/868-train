"""Tests for re-heuristic rule-registry behavior."""

from __future__ import annotations

import random

from src.agent.re_heuristic import ReHeuristicAgent, ReHeuristicConfig, RuleDefinition, save_rule_pack
from src.state.schema import FieldState, GameStateSnapshot, GridPosition, MapState


def _snapshot(*, health: int, player_x: int = 0) -> GameStateSnapshot:
    return GameStateSnapshot(
        timestamp_utc="2026-03-07T00:00:00+00:00",
        health=FieldState(value=health, status="ok"),
        energy=FieldState(value=8, status="ok"),
        currency=FieldState(value=0, status="ok"),
        fail_state=FieldState(value=False, status="ok"),
        map=MapState(
            status="ok",
            width=3,
            height=1,
            player_position=GridPosition(player_x, 0),
            exit_position=GridPosition(2, 0),
            siphons=(GridPosition(1, 0),),
        ),
    )


def test_re_heuristic_defaults_to_baseline_behavior() -> None:
    agent = ReHeuristicAgent(config=ReHeuristicConfig(enable_mined_rules=False))
    state = _snapshot(health=2)

    action = agent.select_action(
        state=state,
        action_space=("move_left", "move_right", "wait"),
        rng=random.Random(4),
    )

    assert action == "wait"
    assert agent.last_decision_reason == "low_health_wait"
    assert agent.last_decision_rule_id == "baseline.low_health_wait"


def test_re_heuristic_honors_higher_priority_mined_rule(tmp_path) -> None:
    pack_path = tmp_path / "rules.json"
    save_rule_pack(
        path=pack_path,
        rules=(
            RuleDefinition(
                rule_id="re_mined.map_cell_semantics.override",
                priority=5000,
                reason="re_mined_override",
                action_preferences=("move_right", "wait"),
                conditions={"require_low_health": True},
            ),
        ),
    )
    agent = ReHeuristicAgent(
        config=ReHeuristicConfig(enable_mined_rules=True, rule_pack_path=str(pack_path))
    )
    state = _snapshot(health=2)

    action = agent.select_action(
        state=state,
        action_space=("move_left", "move_right", "wait"),
        rng=random.Random(5),
    )

    assert action == "move_right"
    assert agent.last_decision_reason == "re_mined_override"
    assert agent.last_decision_rule_id == "re_mined.map_cell_semantics.override"


def test_re_heuristic_reason_stays_stable_for_same_state() -> None:
    agent = ReHeuristicAgent(config=ReHeuristicConfig(enable_mined_rules=False))
    state = _snapshot(health=9)

    action_a = agent.select_action(
        state=state,
        action_space=("move_left", "move_right", "wait"),
        rng=random.Random(7),
    )
    reason_a = agent.last_decision_reason
    rule_id_a = agent.last_decision_rule_id

    action_b = agent.select_action(
        state=state,
        action_space=("move_left", "move_right", "wait"),
        rng=random.Random(99),
    )

    assert action_a == action_b
    assert agent.last_decision_reason == reason_a
    assert agent.last_decision_rule_id == rule_id_a

