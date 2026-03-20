"""Hybrid reward components for meta and threat controllers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from src.hybrid.tactical_model import siphon_spawn_cost_at_position
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
    currency_gain: float = 0.10
    energy_gain: float = 0.10
    score_gain: float = 0.02
    prog_gain: float = 1.50
    step_limit_penalty: float = 5.00
    stagnation_penalty: float = 0.05
    stagnation_grace_steps: int = 3


@dataclass(frozen=True)
class HybridThreatRewardWeights:
    """Threat-controller reward coefficients."""

    survival: float = 0.05
    damage_taken_penalty: float = 0.35
    fail_penalty: float = 2.50
    route_rejoin_bonus: float = 0.15
    invalid_override_penalty: float = 0.10
    enemy_damaged: float = 0.20
    enemy_cleared: float = 0.75
    spawn_debt_penalty: float = 0.15


@dataclass(frozen=True)
class HybridMetaRewardBreakdown:
    """Meta reward decomposition for one step."""

    objective_complete: float
    phase_progress: float
    step_cost: float
    premature_exit_penalty: float
    sector_advance: float
    final_sector_win: float
    currency_gain: float
    energy_gain: float
    score_gain: float
    prog_gain: float
    step_limit_penalty: float
    stagnation_penalty: float
    total: float


@dataclass(frozen=True)
class HybridThreatRewardBreakdown:
    """Threat reward decomposition for one step."""

    survival: float
    damage_taken_penalty: float
    fail_penalty: float
    route_rejoin_bonus: float
    invalid_override_penalty: float
    enemy_damaged: float
    enemy_cleared: float
    spawn_debt_penalty: float
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


def _field_numeric_delta(
    previous_state: GameStateSnapshot,
    current_state: GameStateSnapshot,
    *,
    accessor: Callable[[GameStateSnapshot], object | None],
) -> float:
    previous = _numeric(accessor(previous_state))
    current = _numeric(accessor(current_state))
    if previous is None or current is None:
        return 0.0
    return max(current - previous, 0.0)


def _extra_numeric_delta(
    previous_state: GameStateSnapshot,
    current_state: GameStateSnapshot,
    *,
    key: str,
) -> float:
    previous = _state_extra_numeric(previous_state, key=key)
    current = _state_extra_numeric(current_state, key=key)
    if previous is None or current is None:
        return 0.0
    return max(float(current - previous), 0.0)


def _prog_inventory_delta(
    previous_state: GameStateSnapshot,
    current_state: GameStateSnapshot,
) -> float:
    previous = len(previous_state.inventory.raw_prog_ids)
    current = len(current_state.inventory.raw_prog_ids)
    return float(max(current - previous, 0))


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


def _enemy_hp_by_slot(state: GameStateSnapshot) -> dict[int, int] | None:
    if state.map.status != "ok":
        return None
    hp_by_slot: dict[int, int] = {}
    for enemy in state.map.enemies:
        if not enemy.in_bounds:
            continue
        hp_by_slot[int(enemy.slot)] = max(int(enemy.hp), 0)
    return hp_by_slot


def _enemy_damage_delta(
    *,
    previous_state: GameStateSnapshot,
    current_state: GameStateSnapshot,
) -> float:
    previous_hp = _enemy_hp_by_slot(previous_state)
    current_hp = _enemy_hp_by_slot(current_state)
    if previous_hp is None or current_hp is None:
        return 0.0
    total_damage = 0
    for enemy_id, before_hp in previous_hp.items():
        after_hp = current_hp.get(enemy_id)
        if after_hp is None:
            continue
        if before_hp > after_hp:
            total_damage += before_hp - after_hp
    return float(total_damage)


def _enemy_presence_by_slot(state: GameStateSnapshot) -> dict[int, int] | None:
    if state.map.status != "ok":
        return None
    presence: dict[int, int] = {}
    for enemy in state.map.enemies:
        if not enemy.in_bounds:
            continue
        presence[int(enemy.slot)] = presence.get(int(enemy.slot), 0) + 1
    return presence


def _enemy_cleared_delta(
    *,
    previous_state: GameStateSnapshot,
    current_state: GameStateSnapshot,
) -> float:
    previous_presence = _enemy_presence_by_slot(previous_state)
    current_presence = _enemy_presence_by_slot(current_state)
    if previous_presence is None or current_presence is None:
        return 0.0
    cleared = 0
    for enemy_id, previous_count in previous_presence.items():
        current_count = current_presence.get(enemy_id, 0)
        if previous_count > current_count:
            cleared += previous_count - current_count
    return float(cleared)


def _enemy_growth_delta(
    *,
    previous_state: GameStateSnapshot,
    current_state: GameStateSnapshot,
) -> float:
    previous_presence = _enemy_presence_by_slot(previous_state)
    current_presence = _enemy_presence_by_slot(current_state)
    if previous_presence is None or current_presence is None:
        return 0.0
    growth = 0
    for enemy_id, current_count in current_presence.items():
        previous_count = previous_presence.get(enemy_id, 0)
        if current_count > previous_count:
            growth += current_count - previous_count
    return float(growth)


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
        distance_after = info.get("objective_distance_after")
        if distance_after is None:
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
        currency_gain = _field_numeric_delta(
            previous_state,
            current_state,
            accessor=lambda state: state.currency.value if state.currency.status == "ok" else None,
        )
        energy_gain = _field_numeric_delta(
            previous_state,
            current_state,
            accessor=lambda state: state.energy.value if state.energy.status == "ok" else None,
        )
        score_gain = _extra_numeric_delta(previous_state, current_state, key="score")
        prog_gain = _prog_inventory_delta(previous_state, current_state)
        step_limit_hit = bool(info.get("hit_step_limit", False))
        stagnation_steps = max(int(info.get("objective_stagnation_steps", 0)), 0)
        stagnation_grace = max(int(self.meta_weights.stagnation_grace_steps), 0)
        stagnation_threshold = max(stagnation_grace, 1)

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
        currency_component = currency_gain * abs(self.meta_weights.currency_gain)
        energy_component = energy_gain * abs(self.meta_weights.energy_gain)
        score_component = score_gain * abs(self.meta_weights.score_gain)
        prog_component = prog_gain * abs(self.meta_weights.prog_gain)
        step_limit_component = (
            -abs(self.meta_weights.step_limit_penalty)
            if step_limit_hit
            else 0.0
        )
        stagnation_component = (
            -abs(self.meta_weights.stagnation_penalty)
            if not completion and stagnation_steps >= stagnation_threshold
            else 0.0
        )
        total = (
            objective_component
            + progress_component
            + step_component
            + premature_component
            + sector_component
            + final_sector_win_component
            + currency_component
            + energy_component
            + score_component
            + prog_component
            + step_limit_component
            + stagnation_component
        )
        return HybridMetaRewardBreakdown(
            objective_complete=objective_component,
            phase_progress=progress_component,
            step_cost=step_component,
            premature_exit_penalty=premature_component,
            sector_advance=sector_component,
            final_sector_win=final_sector_win_component,
            currency_gain=currency_component,
            energy_gain=energy_component,
            score_gain=score_component,
            prog_gain=prog_component,
            step_limit_penalty=step_limit_component,
            stagnation_penalty=stagnation_component,
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
        rejoined_route = bool(
            info.get("route_rejoin_event", info.get("rejoined_route", False))
        )
        invalid_override = bool(info.get("invalid_override", False))
        enemy_damage = _enemy_damage_delta(previous_state=previous_state, current_state=current_state)
        enemy_cleared = _enemy_cleared_delta(previous_state=previous_state, current_state=current_state)
        enemy_growth = _enemy_growth_delta(previous_state=previous_state, current_state=current_state)
        chosen_action = str(info.get("action") or "").strip().lower()
        action_effective = bool(info.get("action_effective", False))
        siphon_spawn_cost = (
            float(
                siphon_spawn_cost_at_position(
                    state=previous_state,
                    position=previous_state.map.player_position if previous_state.map.status == "ok" else None,
                )
            )
            if action_effective and chosen_action in {"space", "z"}
            else 0.0
        )
        spawn_debt_units = max(enemy_growth, siphon_spawn_cost)

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
        enemy_damage_component = enemy_damage * abs(self.threat_weights.enemy_damaged)
        enemy_cleared_component = enemy_cleared * abs(self.threat_weights.enemy_cleared)
        spawn_debt_component = -spawn_debt_units * abs(self.threat_weights.spawn_debt_penalty)
        total = (
            survival_component
            + damage_component
            + fail_component
            + rejoin_component
            + invalid_component
            + enemy_damage_component
            + enemy_cleared_component
            + spawn_debt_component
        )
        return HybridThreatRewardBreakdown(
            survival=survival_component,
            damage_taken_penalty=damage_component,
            fail_penalty=fail_component,
            route_rejoin_bonus=rejoin_component,
            invalid_override_penalty=invalid_component,
            enemy_damaged=enemy_damage_component,
            enemy_cleared=enemy_cleared_component,
            spawn_debt_penalty=spawn_debt_component,
            total=total,
        )
