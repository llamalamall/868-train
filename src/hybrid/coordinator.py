"""Decision coordinator for objective selection + threat overrides."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.hybrid.astar_controller import AStarMovementController
from src.hybrid.meta_controller import MetaControllerDQN
from src.hybrid.threat_controller import ThreatControllerDRQN
from src.hybrid.types import (
    HybridDecision,
    HybridDecisionTrace,
    MetaObjectiveChoice,
    ObjectivePhase,
    ThreatOverride,
)
from src.state.schema import GameStateSnapshot, GridPosition

_MOVE_DELTAS: dict[str, tuple[int, int]] = {
    "move_up": (0, 1),
    "move_down": (0, -1),
    "move_left": (-1, 0),
    "move_right": (1, 0),
}
_MOVE_ACTIONS: tuple[str, ...] = tuple(_MOVE_DELTAS.keys())


@dataclass(frozen=True)
class HybridCoordinatorConfig:
    """Behavior knobs for the hybrid coordinator."""

    threat_trigger_distance: int = 2
    enable_prog_override: bool = True


def _manhattan(a: GridPosition, b: GridPosition) -> int:
    return abs(a.x - b.x) + abs(a.y - b.y)


def _wall_positions(state: GameStateSnapshot) -> set[GridPosition]:
    if state.map.status != "ok":
        return set()
    walls = {cell.position for cell in state.map.cells if cell.is_wall}
    if walls:
        return walls
    return {wall.position for wall in state.map.walls}


def _count_enemies(state: GameStateSnapshot) -> int:
    if state.map.status != "ok":
        return 0
    return sum(1 for enemy in state.map.enemies if enemy.in_bounds)


def _resource_targets(state: GameStateSnapshot) -> tuple[GridPosition, ...]:
    if state.map.status != "ok":
        return ()
    targets: list[GridPosition] = []
    seen: set[GridPosition] = set()
    for cell in state.map.resource_cells:
        if int(cell.credits) <= 0 and int(cell.energy) <= 0 and int(cell.points) <= 0:
            continue
        if cell.position in seen:
            continue
        seen.add(cell.position)
        targets.append(cell.position)
    for cell in state.map.cells:
        if cell.position in seen:
            continue
        if cell.prog_id is not None and cell.prog_id > 0 and not cell.is_wall:
            seen.add(cell.position)
            targets.append(cell.position)
            continue
        if cell.points > 0 and not cell.is_wall:
            seen.add(cell.position)
            targets.append(cell.position)
    return tuple(targets)


class HybridCoordinator:
    """Composes deterministic movement with hierarchical RL controllers."""

    def __init__(
        self,
        *,
        meta_controller: MetaControllerDQN,
        threat_controller: ThreatControllerDRQN,
        movement_controller: AStarMovementController = AStarMovementController(),
        config: HybridCoordinatorConfig = HybridCoordinatorConfig(),
    ) -> None:
        self.meta_controller = meta_controller
        self.threat_controller = threat_controller
        self.movement_controller = movement_controller
        self.config = config

    def start_episode(self) -> None:
        self.meta_controller.start_episode()
        self.threat_controller.start_episode()

    def allowed_meta_phases(self, state: GameStateSnapshot) -> tuple[ObjectivePhase, ...]:
        if state.map.status != "ok":
            return (ObjectivePhase.COLLECT_SIPHONS,)
        siphons_remaining = len(state.map.siphons)
        if siphons_remaining > 0:
            return (ObjectivePhase.COLLECT_SIPHONS,)
        if _resource_targets(state):
            return (
                ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS,
                ObjectivePhase.EXIT_SECTOR,
            )
        return (ObjectivePhase.EXIT_SECTOR,)

    def resolve_target_for_phase(
        self,
        *,
        state: GameStateSnapshot,
        phase: ObjectivePhase,
    ) -> GridPosition | None:
        if state.map.status != "ok" or state.map.player_position is None:
            return None

        if phase == ObjectivePhase.COLLECT_SIPHONS:
            return self._nearest_target(
                state=state,
                candidates=tuple(state.map.siphons),
            )
        if phase == ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS:
            return self._nearest_target(
                state=state,
                candidates=_resource_targets(state),
            )
        if phase == ObjectivePhase.EXIT_SECTOR:
            return state.map.exit_position
        return None

    def meta_feature_vector(
        self,
        *,
        state: GameStateSnapshot,
        scripted_phase: ObjectivePhase,
        target: GridPosition | None,
    ) -> tuple[float, ...]:
        health = self._state_numeric(state.health.value) if state.health.status == "ok" else 0.0
        energy = self._state_numeric(state.energy.value) if state.energy.status == "ok" else 0.0
        currency = self._state_numeric(state.currency.value) if state.currency.status == "ok" else 0.0
        score = self._state_extra_numeric(state, key="score") or 0.0
        sector = self._state_extra_numeric(state, key="current_sector") or 0.0
        siphons = float(len(state.map.siphons)) if state.map.status == "ok" else 0.0
        enemies = float(_count_enemies(state))
        resource_count = float(len(_resource_targets(state)))
        exit_known = (
            1.0
            if state.map.status == "ok" and state.map.exit_position is not None
            else 0.0
        )
        objective_distance = self.objective_distance(state=state, target=target)
        nearest_enemy = self.nearest_enemy_distance(state)
        nearest_siphon = self._nearest_distance_to_targets(
            state=state,
            targets=tuple(state.map.siphons) if state.map.status == "ok" else (),
        )
        phase_one_hot = (
            1.0 if scripted_phase == ObjectivePhase.COLLECT_SIPHONS else 0.0,
            1.0 if scripted_phase == ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS else 0.0,
            1.0 if scripted_phase == ObjectivePhase.EXIT_SECTOR else 0.0,
        )
        features = (
            health / 10.0,
            energy / 10.0,
            currency / 25.0,
            score / 500.0,
            sector / 8.0,
            siphons / 6.0,
            enemies / 8.0,
            resource_count / 12.0,
            exit_known,
            self._normalized_distance(objective_distance),
            self._normalized_distance(nearest_enemy),
            self._normalized_distance(nearest_siphon),
            float(len(state.inventory.raw_prog_ids[:10])) / 10.0,
            *phase_one_hot,
            1.0 if (state.can_siphon_now is True) else 0.0,
            1.0 if state.map.status == "ok" else 0.0,
        )
        return self._fit_size(features, target_size=self.meta_controller.feature_count)

    def threat_feature_vector(
        self,
        *,
        state: GameStateSnapshot,
        route_action: str | None,
        objective_distance: int | None,
    ) -> tuple[float, ...]:
        health = self._state_numeric(state.health.value) if state.health.status == "ok" else 0.0
        energy = self._state_numeric(state.energy.value) if state.energy.status == "ok" else 0.0
        enemies = float(_count_enemies(state))
        nearest_enemy = self.nearest_enemy_distance(state)
        threatened = self.is_threat_active(state)
        action_one_hot = tuple(1.0 if route_action == action else 0.0 for action in _MOVE_ACTIONS)
        features = (
            health / 10.0,
            energy / 10.0,
            enemies / 8.0,
            self._normalized_distance(nearest_enemy),
            self._normalized_distance(objective_distance),
            1.0 if threatened else 0.0,
            *action_one_hot,
            1.0 if "wait" in (state.extra_fields.keys()) else 0.0,
            1.0 if state.map.status == "ok" else 0.0,
            float(len(state.inventory.raw_prog_ids[:10])) / 10.0,
            1.0 if state.can_siphon_now is True else 0.0,
            1.0 if state.map.status == "ok" and state.map.player_position is not None else 0.0,
            1.0 if state.map.status == "ok" and state.map.exit_position is not None else 0.0,
            float(len(state.map.siphons)) / 6.0 if state.map.status == "ok" else 0.0,
            0.0,
            0.0,
            0.0,
        )
        return self._fit_size(features, target_size=self.threat_controller.feature_count)

    def objective_distance(
        self,
        *,
        state: GameStateSnapshot,
        target: GridPosition | None,
    ) -> int | None:
        if target is None or state.map.status != "ok" or state.map.player_position is None:
            return None
        player = state.map.player_position
        walls = _wall_positions(state)
        plan = self.movement_controller.plan_route(
            start=player,
            target=target,
            width=state.map.width,
            height=state.map.height,
            walls=walls,
        )
        if plan is None:
            return None
        return int(plan.distance)

    def nearest_enemy_distance(self, state: GameStateSnapshot) -> int | None:
        if state.map.status != "ok" or state.map.player_position is None:
            return None
        enemies = tuple(enemy.position for enemy in state.map.enemies if enemy.in_bounds)
        return self._nearest_distance_to_targets(state=state, targets=enemies)

    def is_threat_active(self, state: GameStateSnapshot) -> bool:
        nearest = self.nearest_enemy_distance(state)
        if nearest is None:
            return False
        return nearest <= int(max(self.config.threat_trigger_distance, 1))

    def decide(
        self,
        *,
        state: GameStateSnapshot,
        available_actions: Sequence[str],
        use_meta_controller: bool,
        use_threat_controller: bool,
        explore_meta: bool,
        explore_threat: bool,
    ) -> HybridDecisionTrace:
        actions = tuple(str(action) for action in available_actions)
        if not actions:
            raise ValueError("available_actions cannot be empty.")

        allowed_phases = self.allowed_meta_phases(state)
        scripted_phase = allowed_phases[0]
        scripted_target = self.resolve_target_for_phase(state=state, phase=scripted_phase)
        meta_features = self.meta_feature_vector(
            state=state,
            scripted_phase=scripted_phase,
            target=scripted_target,
        )
        if use_meta_controller:
            selected_phase, meta_reason, q_value = self.meta_controller.select_objective(
                features=meta_features,
                allowed_phases=allowed_phases,
                explore=explore_meta,
            )
        else:
            selected_phase = scripted_phase
            meta_reason = "scripted_phase_only"
            q_value = None

        if selected_phase not in allowed_phases:
            selected_phase = scripted_phase
            meta_reason = f"{meta_reason}|invalid_phase_fallback"
        objective_target = self.resolve_target_for_phase(state=state, phase=selected_phase)
        objective_distance = self.objective_distance(state=state, target=objective_target)

        route_action = self._route_action_for_objective(
            state=state,
            target=objective_target,
            phase=selected_phase,
            available_actions=actions,
        )
        threat_active = self.is_threat_active(state)
        threat_features = self.threat_feature_vector(
            state=state,
            route_action=route_action,
            objective_distance=objective_distance,
        )
        if use_threat_controller:
            override, threat_reason, _override_q = self.threat_controller.select_override(
                features=threat_features,
                threat_active=threat_active,
                allowed_overrides=self._allowed_threat_overrides(actions=actions),
                explore=explore_threat,
            )
        else:
            override = ThreatOverride.ROUTE_DEFAULT
            threat_reason = "threat_disabled_route_default"

        action, used_fallback, invalid_override = self._action_from_override(
            state=state,
            actions=actions,
            route_action=route_action,
            override=override,
        )
        combined_reason = f"{meta_reason}|{threat_reason}"
        if invalid_override:
            combined_reason = f"{combined_reason}|invalid_override_fallback"
        objective_choice = MetaObjectiveChoice(
            phase=selected_phase,
            target_position=objective_target,
            reason=meta_reason,
            q_value=q_value,
        )
        decision = HybridDecision(
            objective=objective_choice,
            threat_override=override,
            action=action,
            used_fallback=used_fallback,
            reason=combined_reason,
        )
        return HybridDecisionTrace(
            decision=decision,
            meta_features=meta_features,
            threat_features=threat_features,
            objective_distance_before=objective_distance,
            threat_active=threat_active,
            available_actions=actions,
        )

    def _nearest_target(
        self,
        *,
        state: GameStateSnapshot,
        candidates: tuple[GridPosition, ...],
    ) -> GridPosition | None:
        if not candidates or state.map.status != "ok" or state.map.player_position is None:
            return None
        player = state.map.player_position
        walls = _wall_positions(state)
        best_target: GridPosition | None = None
        best_distance: int | None = None
        for candidate in candidates:
            plan = self.movement_controller.plan_route(
                start=player,
                target=candidate,
                width=state.map.width,
                height=state.map.height,
                walls=walls,
            )
            if plan is None:
                continue
            if best_distance is None or plan.distance < best_distance:
                best_distance = plan.distance
                best_target = candidate
        return best_target

    def _nearest_distance_to_targets(
        self,
        *,
        state: GameStateSnapshot,
        targets: tuple[GridPosition, ...],
    ) -> int | None:
        nearest: int | None = None
        for target in targets:
            distance = self.objective_distance(state=state, target=target)
            if distance is None:
                continue
            if nearest is None or distance < nearest:
                nearest = distance
        return nearest

    def _route_action_for_objective(
        self,
        *,
        state: GameStateSnapshot,
        target: GridPosition | None,
        phase: ObjectivePhase,
        available_actions: tuple[str, ...],
    ) -> str | None:
        if state.map.status != "ok" or state.map.player_position is None:
            return None
        player = state.map.player_position
        if target is not None and target == player and phase != ObjectivePhase.EXIT_SECTOR:
            if "space" in available_actions and state.can_siphon_now is not False:
                return "space"

        if target is None:
            return None
        return self.movement_controller.next_action(
            start=player,
            target=target,
            width=state.map.width,
            height=state.map.height,
            walls=_wall_positions(state),
            available_actions=available_actions,
        )

    def _allowed_threat_overrides(self, *, actions: tuple[str, ...]) -> tuple[ThreatOverride, ...]:
        overrides = [ThreatOverride.ROUTE_DEFAULT, ThreatOverride.EVADE, ThreatOverride.ENGAGE]
        if "wait" in actions:
            overrides.append(ThreatOverride.WAIT)
        if self.config.enable_prog_override and any(action.startswith("prog_slot_") for action in actions):
            overrides.append(ThreatOverride.USE_PROG)
        return tuple(overrides)

    def _action_from_override(
        self,
        *,
        state: GameStateSnapshot,
        actions: tuple[str, ...],
        route_action: str | None,
        override: ThreatOverride,
    ) -> tuple[str, bool, bool]:
        invalid_override = False
        if override == ThreatOverride.ROUTE_DEFAULT:
            candidate = route_action
        elif override == ThreatOverride.EVADE:
            candidate = self._select_evade_action(state=state, actions=actions)
        elif override == ThreatOverride.ENGAGE:
            candidate = self._select_engage_action(state=state, actions=actions)
        elif override == ThreatOverride.WAIT:
            candidate = "wait" if "wait" in actions else None
        elif override == ThreatOverride.USE_PROG:
            candidate = self._select_prog_action(actions=actions)
        else:
            candidate = None

        if candidate is not None and candidate in actions:
            return (candidate, False, False)
        if override != ThreatOverride.ROUTE_DEFAULT:
            invalid_override = True
        fallback = self._fallback_action(actions=actions, preferred=route_action)
        return (fallback, True, invalid_override)

    def _select_evade_action(
        self,
        *,
        state: GameStateSnapshot,
        actions: tuple[str, ...],
    ) -> str | None:
        if state.map.status != "ok" or state.map.player_position is None:
            return None
        enemies = tuple(enemy.position for enemy in state.map.enemies if enemy.in_bounds)
        if not enemies:
            return None
        player = state.map.player_position
        walls = _wall_positions(state)
        best_action: str | None = None
        best_distance: int | None = None
        for action in actions:
            delta = _MOVE_DELTAS.get(action)
            if delta is None:
                continue
            candidate = GridPosition(x=player.x + delta[0], y=player.y + delta[1])
            if candidate in walls:
                continue
            nearest = min(_manhattan(candidate, enemy) for enemy in enemies)
            if best_distance is None or nearest > best_distance:
                best_distance = nearest
                best_action = action
        return best_action

    def _select_engage_action(
        self,
        *,
        state: GameStateSnapshot,
        actions: tuple[str, ...],
    ) -> str | None:
        if state.map.status != "ok" or state.map.player_position is None:
            return None
        enemies = tuple(enemy.position for enemy in state.map.enemies if enemy.in_bounds)
        if not enemies:
            return None
        player = state.map.player_position
        walls = _wall_positions(state)
        best_action: str | None = None
        best_distance: int | None = None
        for action in actions:
            delta = _MOVE_DELTAS.get(action)
            if delta is None:
                continue
            candidate = GridPosition(x=player.x + delta[0], y=player.y + delta[1])
            if candidate in walls:
                continue
            nearest = min(_manhattan(candidate, enemy) for enemy in enemies)
            if best_distance is None or nearest < best_distance:
                best_distance = nearest
                best_action = action
        return best_action

    @staticmethod
    def _select_prog_action(*, actions: tuple[str, ...]) -> str | None:
        for action in actions:
            if action.startswith("prog_slot_"):
                return action
        return None

    @staticmethod
    def _fallback_action(*, actions: tuple[str, ...], preferred: str | None) -> str:
        if preferred is not None and preferred in actions:
            return preferred
        if "wait" in actions:
            return "wait"
        for action in actions:
            if action != "cancel":
                return action
        return actions[0]

    @staticmethod
    def _fit_size(values: Sequence[float], *, target_size: int) -> tuple[float, ...]:
        normalized = [float(value) for value in values]
        if len(normalized) < target_size:
            normalized.extend(0.0 for _ in range(target_size - len(normalized)))
        elif len(normalized) > target_size:
            normalized = normalized[:target_size]
        return tuple(normalized)

    @staticmethod
    def _normalized_distance(distance: int | None) -> float:
        if distance is None:
            return 1.0
        return min(max(float(distance) / 10.0, 0.0), 1.0)

    @staticmethod
    def _state_numeric(value: object | None) -> float:
        try:
            if value is None:
                return 0.0
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _state_extra_numeric(state: GameStateSnapshot, *, key: str) -> float | None:
        field = state.extra_fields.get(key)
        if field is None or field.status != "ok":
            return None
        try:
            if field.value is None:
                return None
            return float(field.value)
        except (TypeError, ValueError):
            return None

