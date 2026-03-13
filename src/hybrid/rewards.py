"""Hybrid reward components for meta and threat controllers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.hybrid.types import ObjectivePhase, ThreatOverride
from src.state.schema import GameStateSnapshot, GridPosition


@dataclass(frozen=True)
class HybridMetaRewardWeights:
    """Meta-controller reward coefficients."""

    objective_complete: float = 1.50
    phase_progress: float = 0.25
    step_cost: float = 0.01
    premature_exit_penalty: float = 1.25
    sector_advance: float = 1.00
    final_sector_win: float = 25.00


@dataclass(frozen=True)
class HybridThreatRewardWeights:
    """Threat-controller reward coefficients."""

    survival: float = 0.05
    damage_taken_penalty: float = 0.35
    fail_penalty: float = 2.50
    route_rejoin_bonus: float = 0.15
    invalid_override_penalty: float = 0.10


@dataclass(frozen=True)
class HybridMetaRewardBreakdown:
    """Meta reward decomposition for one step."""

    objective_complete: float
    phase_progress: float
    step_cost: float
    premature_exit_penalty: float
    sector_advance: float
    final_sector_win: float
    total: float


@dataclass(frozen=True)
class HybridThreatRewardBreakdown:
    """Threat reward decomposition for one step."""

    survival: float
    damage_taken_penalty: float
    fail_penalty: float
    route_rejoin_bonus: float
    invalid_override_penalty: float
    total: float


def _numeric(value: object | None) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _state_extra_numeric(state: GameStateSnapshot, *, key: str) -> float | None:
    field = state.extra_fields.get(key)
    if field is None or field.status != "ok":
        return None
    return _numeric(field.value)


def _state_extra_bool(state: GameStateSnapshot, *, key: str) -> bool | None:
    field = state.extra_fields.get(key)
    if field is None or field.status != "ok":
        return None
    value = field.value
    if isinstance(value, bool):
        return value
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return None


def _siphon_count(state: GameStateSnapshot) -> int | None:
    if state.map.status != "ok":
        return None
    return len(state.map.siphons)


def _resource_value_total(state: GameStateSnapshot) -> int | None:
    if state.map.status != "ok":
        return None
    total = 0
    for cell in state.map.resource_cells:
        total += max(int(cell.credits), 0)
        total += max(int(cell.energy), 0)
        total += max(int(cell.points), 0)
    for wall in state.map.walls:
        total += max(int(wall.points), 0)
    return total


def _player_on_exit(state: GameStateSnapshot) -> bool:
    if state.map.status != "ok":
        return False
    if state.map.player_position is None or state.map.exit_position is None:
        return False
    return state.map.player_position == state.map.exit_position


def _distance_to_target(
    state: GameStateSnapshot,
    *,
    target: GridPosition | None,
) -> int | None:
    if target is None or state.map.status != "ok":
        return None
    player = state.map.player_position
    if player is None:
        return None
    return abs(player.x - target.x) + abs(player.y - target.y)


def _phase_completion_event(
    *,
    previous_state: GameStateSnapshot,
    current_state: GameStateSnapshot,
    objective_phase: ObjectivePhase,
    target: GridPosition | None,
) -> bool:
    if objective_phase == ObjectivePhase.COLLECT_SIPHONS:
        previous = _siphon_count(previous_state)
        current = _siphon_count(current_state)
        return (
            previous is not None
            and current is not None
            and current < previous
        )

    if objective_phase == ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS:
        previous_total = _resource_value_total(previous_state)
        current_total = _resource_value_total(current_state)
        if previous_total is not None and current_total is not None and current_total < previous_total:
            return True
        previous_prog_count = len(previous_state.inventory.raw_prog_ids)
        current_prog_count = len(current_state.inventory.raw_prog_ids)
        return current_prog_count > previous_prog_count

    if objective_phase == ObjectivePhase.EXIT_SECTOR:
        if not _player_on_exit(current_state):
            return False
        if _siphon_count(current_state) not in {None, 0}:
            return False
        if _resource_value_total(current_state) not in {None, 0}:
            return False
        if target is None:
            return True
        return _distance_to_target(current_state, target=target) == 0

    return False


def _sector_advance_delta(
    *,
    previous_state: GameStateSnapshot,
    current_state: GameStateSnapshot,
) -> float:
    previous = _state_extra_numeric(previous_state, key="current_sector")
    current = _state_extra_numeric(current_state, key="current_sector")
    if previous is None or current is None or current <= previous:
        return 0.0
    return float(current - previous)


def _state_indicates_victory(state: GameStateSnapshot) -> bool:
    return _state_extra_bool(state, key="victory_active") is True


def _state_is_final_sector(state: GameStateSnapshot) -> bool:
    sector = _state_extra_numeric(state, key="current_sector")
    return sector is not None and sector >= 7.0


def _reason_indicates_fail_terminal(reason: str) -> bool:
    return any(token in reason for token in ("fail", "dead", "loss"))


def _final_sector_win_event(
    *,
    previous_state: GameStateSnapshot,
    current_state: GameStateSnapshot,
    objective_phase: ObjectivePhase,
    done: bool,
    info: dict[str, Any],
) -> bool:
    if _state_indicates_victory(previous_state) or _state_indicates_victory(current_state):
        return True
    if not done or objective_phase != ObjectivePhase.EXIT_SECTOR:
        return False
    if bool(info.get("premature_exit_attempt", False)):
        return False
    reason = str(info.get("terminal_reason") or "").strip().lower()
    if reason and _reason_indicates_fail_terminal(reason):
        return False
    if not (_state_is_final_sector(previous_state) or _state_is_final_sector(current_state)):
        return False
    return _player_on_exit(previous_state) or _player_on_exit(current_state)


def _health_delta(
    *,
    previous_state: GameStateSnapshot,
    current_state: GameStateSnapshot,
) -> float:
    if previous_state.health.status != "ok" or current_state.health.status != "ok":
        return 0.0
    previous = _numeric(previous_state.health.value)
    current = _numeric(current_state.health.value)
    if previous is None or current is None:
        return 0.0
    return current - previous


class HybridRewardSuite:
    """Reward helper for split meta/threat training signals."""

    def __init__(
        self,
        *,
        meta_weights: HybridMetaRewardWeights = HybridMetaRewardWeights(),
        threat_weights: HybridThreatRewardWeights = HybridThreatRewardWeights(),
    ) -> None:
        self.meta_weights = meta_weights
        self.threat_weights = threat_weights

    def compute_meta_reward(
        self,
        *,
        previous_state: GameStateSnapshot,
        current_state: GameStateSnapshot,
        objective_phase: ObjectivePhase,
        done: bool,
        info: dict[str, Any],
    ) -> HybridMetaRewardBreakdown:
        target = info.get("objective_target")
        if not isinstance(target, GridPosition):
            target = None
        distance_before = info.get("objective_distance_before")
        if distance_before is None:
            distance_before = _distance_to_target(previous_state, target=target)
        distance_after = _distance_to_target(current_state, target=target)
        progress_delta = 0.0
        if distance_before is not None and distance_after is not None:
            progress_delta = float(distance_before - distance_after)
        completion = _phase_completion_event(
            previous_state=previous_state,
            current_state=current_state,
            objective_phase=objective_phase,
            target=target,
        )
        premature_exit = bool(
            info.get("premature_exit_attempt", False)
            or (
                _player_on_exit(current_state)
                and _siphon_count(current_state) not in {None, 0}
            )
        )
        sector_advance = _sector_advance_delta(previous_state=previous_state, current_state=current_state)
        final_sector_win = _final_sector_win_event(
            previous_state=previous_state,
            current_state=current_state,
            objective_phase=objective_phase,
            done=done,
            info=info,
        )

        objective_component = (
            abs(self.meta_weights.objective_complete)
            if completion
            else 0.0
        )
        progress_component = progress_delta * abs(self.meta_weights.phase_progress)
        step_component = -abs(self.meta_weights.step_cost)
        premature_component = (
            -abs(self.meta_weights.premature_exit_penalty)
            if premature_exit
            else 0.0
        )
        sector_component = sector_advance * abs(self.meta_weights.sector_advance)
        final_sector_win_component = (
            abs(self.meta_weights.final_sector_win)
            if final_sector_win
            else 0.0
        )
        total = (
            objective_component
            + progress_component
            + step_component
            + premature_component
            + sector_component
            + final_sector_win_component
        )
        return HybridMetaRewardBreakdown(
            objective_complete=objective_component,
            phase_progress=progress_component,
            step_cost=step_component,
            premature_exit_penalty=premature_component,
            sector_advance=sector_component,
            final_sector_win=final_sector_win_component,
            total=total,
        )

    def compute_threat_reward(
        self,
        *,
        previous_state: GameStateSnapshot,
        current_state: GameStateSnapshot,
        done: bool,
        threat_override: ThreatOverride,
        info: dict[str, Any],
    ) -> HybridThreatRewardBreakdown:
        terminal_reason = str(info.get("terminal_reason") or "").strip().lower()
        failed = done and any(token in terminal_reason for token in ("fail", "dead", "loss"))
        health_delta = _health_delta(previous_state=previous_state, current_state=current_state)
        damage_taken = max(-health_delta, 0.0)
        rejoined_route = bool(info.get("rejoined_route", False))
        invalid_override = bool(info.get("invalid_override", False))

        survival_component = abs(self.threat_weights.survival) if not done else 0.0
        damage_component = -damage_taken * abs(self.threat_weights.damage_taken_penalty)
        fail_component = -abs(self.threat_weights.fail_penalty) if failed else 0.0
        rejoin_component = (
            abs(self.threat_weights.route_rejoin_bonus)
            if rejoined_route and threat_override == ThreatOverride.ROUTE_DEFAULT
            else 0.0
        )
        invalid_component = (
            -abs(self.threat_weights.invalid_override_penalty)
            if invalid_override
            else 0.0
        )
        total = (
            survival_component
            + damage_component
            + fail_component
            + rejoin_component
            + invalid_component
        )
        return HybridThreatRewardBreakdown(
            survival=survival_component,
            damage_taken_penalty=damage_component,
            fail_penalty=fail_component,
            route_rejoin_bonus=rejoin_component,
            invalid_override_penalty=invalid_component,
            total=total,
        )
