"""Decision coordinator for objective selection + threat overrides."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

 
from src.hybrid.objective_planning import (
    cell_index as _cell_index,
    cell_target_already_siphoned as _cell_target_already_siphoned,
    count_enemies as _count_enemies,
    credit_energy_cluster_total as _credit_energy_cluster_total,
    manhattan as _manhattan,
    resource_value_index as _resource_value_index,
    resource_targets as _resource_targets,
    siphon_reach_positions as _siphon_reach_positions,
    wall_positions as _wall_positions,
)
from src.hybrid.astar_controller import AStarMovementController
from src.hybrid.feature_encoding import build_meta_feature_vector, build_threat_feature_vector
from src.hybrid.meta_controller import MetaControllerDQN
from src.hybrid.tactical_model import (
    TacticalRiskSnapshot,
    estimate_position_risk,
    move_target_position,
    siphon_spawn_cost_at_position,
)
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
    exit_after_siphons_when_scripted: bool = False
    phase_lock_min_steps: int = 6
    target_stall_release_steps: int = 4
    enemy_prediction_horizon_turns: int = 1
    avoid_guaranteed_damage: bool = True
    dangerous_siphon_spawn_threshold: int = 2
    siphon_penalty_weight: float = 8.0


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
        self._locked_phase: ObjectivePhase | None = None
        self._locked_target: GridPosition | None = None
        self._locked_phase_steps = 0
        self._locked_target_stall_steps = 0
        self._completed_resource_targets: set[GridPosition] = set()
        self._phase_lock_overrides = 0
        self._stall_releases = 0

    def start_episode(self) -> None:
        self.meta_controller.start_episode()
        self.threat_controller.start_episode()
        self._locked_phase = None
        self._locked_target = None
        self._locked_phase_steps = 0
        self._locked_target_stall_steps = 0
        self._completed_resource_targets.clear()
        self._phase_lock_overrides = 0
        self._stall_releases = 0

    @property
    def phase_lock_overrides(self) -> int:
        return int(self._phase_lock_overrides)

    @property
    def stall_releases(self) -> int:
        return int(self._stall_releases)

    def allowed_meta_phases(self, state: GameStateSnapshot) -> tuple[ObjectivePhase, ...]:
        if state.map.status != "ok":
            return (ObjectivePhase.COLLECT_SIPHONS,)
        siphons_remaining = len(state.map.siphons)
        if siphons_remaining > 0:
            return (ObjectivePhase.COLLECT_SIPHONS,)
        if self.config.exit_after_siphons_when_scripted:
            return (ObjectivePhase.EXIT_SECTOR,)
        if self._available_resource_targets(state=state):
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
        scripted_mode: bool = False,
    ) -> GridPosition | None:
        _, target = self.resolve_objective_for_phase(
            state=state,
            phase=phase,
            scripted_mode=scripted_mode,
        )
        return target

    def resolve_objective_for_phase(
        self,
        *,
        state: GameStateSnapshot,
        phase: ObjectivePhase,
        scripted_mode: bool = False,
    ) -> tuple[ObjectivePhase, GridPosition | None]:
        if state.map.status != "ok" or state.map.player_position is None:
            return (phase, None)

        resolved_phase = self._resolved_phase_for_state(state=state, phase=phase)

        if resolved_phase == ObjectivePhase.COLLECT_SIPHONS:
            return (
                resolved_phase,
                self._nearest_target(
                    state=state,
                    candidates=tuple(state.map.siphons),
                ),
            )
        if resolved_phase == ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS:
            candidates = self._available_resource_targets(state=state)
            return (
                resolved_phase,
                self._best_resource_target(
                    state=state,
                    candidates=candidates,
                ),
            )
        if resolved_phase == ObjectivePhase.EXIT_SECTOR:
            return (resolved_phase, state.map.exit_position)
        return (resolved_phase, None)

    def _resolved_phase_for_state(
        self,
        *,
        state: GameStateSnapshot,
        phase: ObjectivePhase,
    ) -> ObjectivePhase:
        if phase != ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS or state.map.status != "ok":
            return phase
        player_siphons = self._player_siphon_count(state)
        if player_siphons is None or player_siphons > 0:
            return phase
        if state.map.siphons:
            return ObjectivePhase.COLLECT_SIPHONS
        return ObjectivePhase.EXIT_SECTOR

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
        resource_count = float(len(self._available_resource_targets(state=state)))
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
        features = build_meta_feature_vector(
            health=health,
            energy=energy,
            currency=currency,
            score=score,
            sector=sector,
            siphons=siphons,
            enemies=enemies,
            resource_count=resource_count,
            exit_known=bool(exit_known),
            objective_distance=objective_distance,
            nearest_enemy=nearest_enemy,
            nearest_siphon=nearest_siphon,
            prog_count=len(state.inventory.raw_prog_ids[:10]),
            scripted_phase=scripted_phase,
            can_siphon_now=state.can_siphon_now is True,
            map_ok=state.map.status == "ok",
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
        route_risk = self._risk_for_action(state=state, action=route_action)
        current_risk = estimate_position_risk(
            state=state,
            position=state.map.player_position if state.map.status == "ok" else None,
            horizon_turns=self.config.enemy_prediction_horizon_turns,
        )
        siphon_spawn_cost = siphon_spawn_cost_at_position(
            state=state,
            position=state.map.player_position if state.map.status == "ok" else None,
        )
        combat_readiness = self._combat_readiness_score(state)
        features = build_threat_feature_vector(
            health=health,
            energy=energy,
            enemies=enemies,
            nearest_enemy=nearest_enemy,
            objective_distance=objective_distance,
            threatened=threatened,
            route_action=route_action,
            route_risk=route_risk,
            wait_action_visible="wait" in state.extra_fields,
            map_ok=state.map.status == "ok",
            prog_count=len(state.inventory.raw_prog_ids[:10]),
            can_siphon_now=state.can_siphon_now is True,
            siphon_spawn_cost=int(siphon_spawn_cost),
            current_risk=current_risk,
            combat_readiness=combat_readiness,
            move_actions=_MOVE_ACTIONS,
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
        raw_actions = tuple(str(action) for action in available_actions)
        if not raw_actions:
            raise ValueError("available_actions cannot be empty.")
        actions = self._filter_actions_for_siphon_depletion(
            state=state,
            actions=raw_actions,
        )
        if not actions:
            raise ValueError("No non-siphon actions available while siphons are depleted.")
        scripted_mode = not bool(use_meta_controller)

        allowed_phases = self.allowed_meta_phases(state)
        scripted_phase, scripted_target = self.resolve_objective_for_phase(
            state=state,
            phase=allowed_phases[0],
            scripted_mode=scripted_mode,
        )
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
        requested_phase = selected_phase
        selected_phase, objective_target = self._objective_for_decision_phase(
            state=state,
            phase=selected_phase,
            scripted_mode=scripted_mode,
        )
        if selected_phase != requested_phase:
            meta_reason = f"{meta_reason}|phase_resolved_to_{selected_phase.value}"
        objective_distance = self.objective_distance(state=state, target=objective_target)
        if selected_phase == ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS and objective_target is None:
            selected_phase, objective_target = self._objective_for_decision_phase(
                state=state,
                phase=ObjectivePhase.EXIT_SECTOR,
                scripted_mode=scripted_mode,
            )
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
        action, guardrail_reason = self._apply_tactical_guardrails(
            state=state,
            actions=actions,
            selected_action=action,
            route_action=route_action,
        )
        combined_reason = f"{meta_reason}|{threat_reason}"
        if invalid_override:
            combined_reason = f"{combined_reason}|invalid_override_fallback"
        if guardrail_reason:
            combined_reason = f"{combined_reason}|{guardrail_reason}"
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
            requested_phase=requested_phase,
            meta_features=meta_features,
            threat_features=threat_features,
            objective_distance_before=objective_distance,
            threat_active=threat_active,
            available_actions=actions,
        )

    def observe_step_result(
        self,
        *,
        trace: HybridDecisionTrace,
        info: dict[str, object] | None = None,
        next_state: GameStateSnapshot | None = None,
    ) -> None:
        details = info or {}
        if trace.decision.objective.phase != ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS:
            self._observe_locked_target_progress(trace=trace, next_state=next_state)
            return
        target = trace.decision.objective.target_position
        if (
            target is not None
            and trace.decision.action in {"space", "z"}
            and bool(details.get("action_effective", False))
        ):
            self._completed_resource_targets.update(_siphon_reach_positions(target))
            if self._locked_phase == ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS and self._locked_target == target:
                self._locked_target = None
                self._locked_target_stall_steps = 0
        self._observe_locked_target_progress(trace=trace, next_state=next_state)

    def _objective_for_decision_phase(
        self,
        *,
        state: GameStateSnapshot,
        phase: ObjectivePhase,
        scripted_mode: bool,
    ) -> tuple[ObjectivePhase, GridPosition | None]:
        resolved_phase = self._resolved_phase_for_state(state=state, phase=phase)
        if self._locked_phase is not None and not self._is_locked_phase_valid(state=state):
            self._clear_locked_objective()
        if self._locked_phase is not None and self._locked_phase != resolved_phase:
            min_steps = self._phase_lock_min_steps(self._locked_phase)
            if self._locked_phase_steps < min_steps:
                self._phase_lock_overrides += 1
                target = self._locked_target
                if target is None:
                    locked_phase, target = self.resolve_objective_for_phase(
                        state=state,
                        phase=self._locked_phase,
                        scripted_mode=scripted_mode,
                    )
                    if locked_phase != self._locked_phase:
                        self._clear_locked_objective()
                    else:
                        self._record_locked_objective(phase=locked_phase, target=target)
                        return (locked_phase, target)
                else:
                    self._record_locked_objective(phase=self._locked_phase, target=target)
                    return (self._locked_phase, target)
            self._clear_locked_objective()
        if self._locked_phase == resolved_phase:
            locked_target = self._locked_target
            if locked_target is not None and self._is_target_valid_for_phase(
                state=state,
                phase=resolved_phase,
                target=locked_target,
                scripted_mode=scripted_mode,
            ):
                self._record_locked_objective(phase=resolved_phase, target=locked_target)
                return (resolved_phase, locked_target)

        resolved_phase, resolved_target = self.resolve_objective_for_phase(
            state=state,
            phase=resolved_phase,
            scripted_mode=scripted_mode,
        )
        self._record_locked_objective(phase=resolved_phase, target=resolved_target)
        return (resolved_phase, resolved_target)

    def _clear_locked_objective(self) -> None:
        self._locked_phase = None
        self._locked_target = None
        self._locked_phase_steps = 0
        self._locked_target_stall_steps = 0

    def _record_locked_objective(
        self,
        *,
        phase: ObjectivePhase,
        target: GridPosition | None,
    ) -> None:
        continuing_phase = self._locked_phase == phase
        previous_target = self._locked_target if continuing_phase else None
        self._locked_phase = phase
        self._locked_target = target
        if continuing_phase:
            self._locked_phase_steps += 1
        else:
            self._locked_phase_steps = 1
        if previous_target != target:
            self._locked_target_stall_steps = 0

    def _phase_lock_min_steps(self, phase: ObjectivePhase) -> int:
        if phase not in {
            ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS,
            ObjectivePhase.EXIT_SECTOR,
        }:
            return 0
        return max(int(self.config.phase_lock_min_steps), 0)

    def _is_locked_phase_valid(self, *, state: GameStateSnapshot) -> bool:
        locked_phase = self._locked_phase
        if locked_phase is None or state.map.status != "ok" or state.map.player_position is None:
            return False
        if self._resolved_phase_for_state(state=state, phase=locked_phase) != locked_phase:
            return False
        if locked_phase == ObjectivePhase.COLLECT_SIPHONS:
            return bool(state.map.siphons)
        if locked_phase == ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS:
            return bool(self._available_resource_targets(state=state))
        if locked_phase == ObjectivePhase.EXIT_SECTOR:
            return state.map.exit_position is not None
        return False

    def _observe_locked_target_progress(
        self,
        *,
        trace: HybridDecisionTrace,
        next_state: GameStateSnapshot | None,
    ) -> None:
        if (
            next_state is None
            or self._locked_phase is None
            or self._locked_target is None
            or trace.decision.objective.phase != self._locked_phase
            or trace.decision.objective.target_position != self._locked_target
        ):
            return
        if not self._is_locked_phase_valid(state=next_state):
            self._clear_locked_objective()
            return
        previous_distance = trace.objective_distance_before
        if previous_distance is None:
            return
        next_distance = self.objective_distance(state=next_state, target=self._locked_target)
        if next_distance is None or next_distance < previous_distance:
            self._locked_target_stall_steps = 0
            return
        stall_limit = max(int(self.config.target_stall_release_steps), 0)
        if stall_limit <= 0:
            return
        self._locked_target_stall_steps += 1
        if self._locked_target_stall_steps < stall_limit:
            return
        self._stall_releases += 1
        self._clear_locked_objective()

    def _is_target_valid_for_phase(
        self,
        *,
        state: GameStateSnapshot,
        phase: ObjectivePhase,
        target: GridPosition,
        scripted_mode: bool,
    ) -> bool:
        if state.map.status != "ok":
            return False
        if phase == ObjectivePhase.COLLECT_SIPHONS:
            return target in state.map.siphons
        if phase == ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS:
            return target in self._available_resource_targets(state=state)
        if phase == ObjectivePhase.EXIT_SECTOR:
            return bool(state.map.exit_position is not None and state.map.exit_position == target)
        return False

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

    def _best_resource_target(
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
        best_score: float | None = None
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
            base_value = self._resource_value_score(state=state, position=candidate)
            siphon_cost = siphon_spawn_cost_at_position(state=state, position=candidate)
            score = (
                float(base_value)
                - float(plan.distance)
                - float(siphon_cost) * float(self.config.siphon_penalty_weight)
            )
            if best_score is None or score > best_score:
                best_score = score
                best_target = candidate
        return best_target if best_target is not None else self._nearest_target(state=state, candidates=candidates)

    def _available_resource_targets(self, *, state: GameStateSnapshot) -> tuple[GridPosition, ...]:
        candidates = _resource_targets(state)
        if not self._completed_resource_targets:
            return candidates
        return tuple(
            candidate for candidate in candidates if candidate not in self._completed_resource_targets
        )

    def _resource_value_score(
        self,
        *,
        state: GameStateSnapshot,
        position: GridPosition,
    ) -> int:
        if state.map.status != "ok":
            return 0
        cells = _cell_index(state)
        resource_values = _resource_value_index(state)
        layers = state.map.layers
        total = _credit_energy_cluster_total(
            position=position,
            cell_index_map=cells,
            resource_values=resource_values,
            width=state.map.width,
            height=state.map.height,
            credits_map=layers.credits_map,
            energy_map=layers.energy_map,
        )
        for candidate in _siphon_reach_positions(position):
            cell = cells.get(candidate)
            if cell is not None and not _cell_target_already_siphoned(cell):
                total += max(int(getattr(cell, "points", 0)), 0)
                prog_id = getattr(cell, "prog_id", None)
                if prog_id is not None and int(prog_id) > 0:
                    total += 25
            for wall in state.map.walls:
                if wall.position != candidate or _cell_target_already_siphoned(wall):
                    continue
                total += max(int(getattr(wall, "points", 0)), 0)
                prog_id = getattr(wall, "prog_id", None)
                if prog_id is not None and int(prog_id) > 0:
                    total += 25
        return total

    def _combat_readiness_score(self, state: GameStateSnapshot) -> float:
        health = self._state_numeric(state.health.value) if state.health.status == "ok" else 0.0
        energy = self._state_numeric(state.energy.value) if state.energy.status == "ok" else 0.0
        inventory_load = float(len(state.inventory.raw_prog_ids[:4])) / 4.0
        readiness = ((health / 10.0) * 0.45) + ((energy / 10.0) * 0.35) + (inventory_load * 0.20)
        return min(max(readiness, 0.0), 1.0)

    def _should_delay_siphon(self, *, state: GameStateSnapshot) -> bool:
        if state.map.status != "ok" or state.map.player_position is None:
            return False
        spawn_cost = siphon_spawn_cost_at_position(state=state, position=state.map.player_position)
        if spawn_cost < int(self.config.dangerous_siphon_spawn_threshold):
            return False
        if self.is_threat_active(state):
            return True
        return self._combat_readiness_score(state) < 0.55

    def _risk_for_action(
        self,
        *,
        state: GameStateSnapshot,
        action: str | None,
    ) -> TacticalRiskSnapshot:
        if state.map.status != "ok":
            return TacticalRiskSnapshot(
                immediate_damage=False,
                horizon_damage_steps=0,
                nearest_enemy_distance_after_one_turn=None,
            )
        if action in {"space", "z", None, "wait", "cancel"}:
            position = state.map.player_position
        else:
            position = move_target_position(state=state, action=action)
        return estimate_position_risk(
            state=state,
            position=position,
            horizon_turns=max(int(self.config.enemy_prediction_horizon_turns), 1),
        )

    def _select_safest_non_siphon_action(
        self,
        *,
        state: GameStateSnapshot,
        actions: tuple[str, ...],
        preferred: str | None,
    ) -> str | None:
        candidates = tuple(action for action in actions if action not in {"space", "z", "cancel"})
        if not candidates:
            return None
        best_action: str | None = None
        best_key: tuple[float, float, float] | None = None
        for action in candidates:
            risk = self._risk_for_action(state=state, action=action)
            preferred_bonus = 0.0 if action == preferred else 1.0
            nearest_enemy = risk.nearest_enemy_distance_after_one_turn
            safety_distance = -float(nearest_enemy if nearest_enemy is not None else 99)
            key = (
                1.0 if risk.immediate_damage else 0.0,
                float(risk.horizon_damage_steps),
                preferred_bonus + safety_distance,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_action = action
        return best_action

    def _apply_tactical_guardrails(
        self,
        *,
        state: GameStateSnapshot,
        actions: tuple[str, ...],
        selected_action: str,
        route_action: str | None,
    ) -> tuple[str, str | None]:
        if state.map.status != "ok" or state.map.player_position is None:
            return (selected_action, None)
        if selected_action in {"space", "z"} and self._should_delay_siphon(state=state):
            safe_action = self._select_safest_non_siphon_action(
                state=state,
                actions=actions,
                preferred=route_action,
            )
            if safe_action is not None and safe_action != selected_action:
                return (safe_action, "guarded_dangerous_siphon")
        if self.config.avoid_guaranteed_damage and selected_action in _MOVE_ACTIONS:
            risk = self._risk_for_action(state=state, action=selected_action)
            if risk.immediate_damage:
                safe_action = self._select_safest_non_siphon_action(
                    state=state,
                    actions=actions,
                    preferred=route_action,
                )
                if safe_action is not None and safe_action != selected_action:
                    safe_risk = self._risk_for_action(state=state, action=safe_action)
                    if not safe_risk.immediate_damage or safe_risk.horizon_damage_steps < risk.horizon_damage_steps:
                        return (safe_action, "guarded_guaranteed_damage")
        return (selected_action, None)

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
            siphon_action = self._select_siphon_action(available_actions=available_actions)
            if siphon_action is not None and state.can_siphon_now is not False:
                if self._should_delay_siphon(state=state):
                    return self._select_safest_non_siphon_action(
                        state=state,
                        actions=available_actions,
                        preferred=None,
                    )
                return siphon_action

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

    @staticmethod
    def _filter_actions_for_siphon_depletion(
        *,
        state: GameStateSnapshot,
        actions: tuple[str, ...],
    ) -> tuple[str, ...]:
        player_siphons = HybridCoordinator._player_siphon_count(state)
        if player_siphons is None or player_siphons > 0:
            return actions
        return tuple(action for action in actions if action not in {"space", "z"})

    @staticmethod
    def _player_siphon_count(state: GameStateSnapshot) -> int | None:
        """Return player's current siphon count when extracted as a scalar field."""
        for key in ("siphons", "player_siphons", "siphon_count"):
            field = state.extra_fields.get(key)
            if field is None or field.status != "ok" or field.value is None:
                continue
            try:
                return int(float(field.value))
            except (TypeError, ValueError):
                continue
        return None

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
    def _select_siphon_action(*, available_actions: tuple[str, ...]) -> str | None:
        if "space" in available_actions:
            return "space"
        if "z" in available_actions:
            return "z"
        return None

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
