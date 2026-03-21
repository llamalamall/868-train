"""Pure feature encoders for Hybrid controllers."""

from __future__ import annotations

from src.hybrid.tactical_model import TacticalRiskSnapshot
from src.hybrid.types import ObjectivePhase


def _normalized_distance(distance: int | None) -> float:
    if distance is None:
        return 1.0
    return min(float(distance) / 10.0, 1.0)


def build_meta_feature_vector(
    *,
    health: float,
    energy: float,
    currency: float,
    score: float,
    sector: float,
    siphons: float,
    enemies: float,
    resource_count: float,
    exit_known: bool,
    objective_distance: int | None,
    nearest_enemy: int | None,
    nearest_siphon: int | None,
    prog_count: int,
    scripted_phase: ObjectivePhase,
    can_siphon_now: bool,
    map_ok: bool,
) -> tuple[float, ...]:
    phase_one_hot = (
        1.0 if scripted_phase == ObjectivePhase.COLLECT_SIPHONS else 0.0,
        1.0 if scripted_phase == ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS else 0.0,
        1.0 if scripted_phase == ObjectivePhase.EXIT_SECTOR else 0.0,
    )
    return (
        health / 10.0,
        energy / 10.0,
        currency / 25.0,
        score / 500.0,
        sector / 8.0,
        siphons / 6.0,
        enemies / 8.0,
        resource_count / 12.0,
        1.0 if exit_known else 0.0,
        _normalized_distance(objective_distance),
        _normalized_distance(nearest_enemy),
        _normalized_distance(nearest_siphon),
        float(prog_count) / 10.0,
        *phase_one_hot,
        1.0 if can_siphon_now else 0.0,
        1.0 if map_ok else 0.0,
    )


def build_threat_feature_vector(
    *,
    health: float,
    energy: float,
    enemies: float,
    nearest_enemy: int | None,
    objective_distance: int | None,
    threatened: bool,
    route_action: str | None,
    route_risk: TacticalRiskSnapshot,
    wait_action_visible: bool,
    map_ok: bool,
    prog_count: int,
    can_siphon_now: bool,
    siphon_spawn_cost: int,
    current_risk: TacticalRiskSnapshot,
    combat_readiness: float,
    move_actions: tuple[str, ...],
) -> tuple[float, ...]:
    action_one_hot = tuple(1.0 if route_action == action else 0.0 for action in move_actions)
    return (
        health / 10.0,
        energy / 10.0,
        enemies / 8.0,
        _normalized_distance(nearest_enemy),
        _normalized_distance(objective_distance),
        1.0 if threatened else 0.0,
        *action_one_hot,
        1.0 if wait_action_visible else 0.0,
        1.0 if map_ok else 0.0,
        float(prog_count) / 10.0,
        1.0 if can_siphon_now else 0.0,
        1.0 if route_risk.immediate_damage else 0.0,
        _normalized_distance(route_risk.nearest_enemy_distance_after_one_turn),
        min(float(route_risk.horizon_damage_steps) / 3.0, 1.0),
        min(float(siphon_spawn_cost) / 6.0, 1.0),
        1.0 if current_risk.immediate_damage else 0.0,
        combat_readiness,
    )
