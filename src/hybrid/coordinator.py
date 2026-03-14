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
_CARDINAL_DELTAS: tuple[tuple[int, int], ...] = ((0, 1), (1, 0), (0, -1), (-1, 0))


@dataclass(frozen=True)
class HybridCoordinatorConfig:
    """Behavior knobs for the hybrid coordinator."""

    threat_trigger_distance: int = 2
    enable_prog_override: bool = True
    exit_after_siphons_when_scripted: bool = False


def _manhattan(a: GridPosition, b: GridPosition) -> int:
    return abs(a.x - b.x) + abs(a.y - b.y)


def _wall_positions(state: GameStateSnapshot) -> set[GridPosition]:
    if state.map.status != "ok":
        return set()
    walls = {cell.position for cell in state.map.cells if cell.is_wall}
    if walls:
        return walls
    return {wall.position for wall in state.map.walls}


def _is_in_bounds(position: GridPosition, *, width: int, height: int) -> bool:
    return 0 <= position.x < width and 0 <= position.y < height


def _adjacent_positions(position: GridPosition) -> tuple[GridPosition, ...]:
    return tuple(
        GridPosition(x=position.x + dx, y=position.y + dy)
        for dx, dy in _CARDINAL_DELTAS
    )


def _siphon_reach_positions(position: GridPosition) -> tuple[GridPosition, ...]:
    return (position, *_adjacent_positions(position))


def _cell_index(state: GameStateSnapshot) -> dict[GridPosition, object]:
    if state.map.status != "ok":
        return {}
    return {cell.position: cell for cell in state.map.cells}


def _resource_value_index(state: GameStateSnapshot) -> dict[GridPosition, tuple[int, int, int]]:
    if state.map.status != "ok":
        return {}
    cells = _cell_index(state)
    values: dict[GridPosition, tuple[int, int, int]] = {}
    for resource in state.map.resource_cells:
        raw_cell = cells.get(resource.position)
        if raw_cell is not None and _cell_target_already_siphoned(raw_cell):
            continue
        values[resource.position] = (
            max(int(resource.credits), 0),
            max(int(resource.energy), 0),
            max(int(resource.points), 0),
        )
    return values


def _read_layer_cell(
    layer: tuple[tuple[int, ...], ...],
    *,
    x: int,
    y: int,
    width: int,
    height: int,
) -> int | None:
    if x < 0 or y < 0 or x >= width or y >= height:
        return None
    if len(layer) != height:
        return None
    row = layer[y]
    if len(row) != width:
        return None
    try:
        return int(row[x])
    except (TypeError, ValueError):
        return None


def _credit_energy_cluster_total(
    *,
    position: GridPosition,
    cell_index: dict[GridPosition, object],
    resource_values: dict[GridPosition, tuple[int, int, int]],
    width: int,
    height: int,
    credits_map: tuple[tuple[int, ...], ...],
    energy_map: tuple[tuple[int, ...], ...],
) -> int:
    # Prefer precomputed outcome layers: they already include adjacent siphon reach.
    x = int(position.x)
    y = int(position.y)
    layer_credits = _read_layer_cell(
        credits_map,
        x=x,
        y=y,
        width=width,
        height=height,
    )
    layer_energy = _read_layer_cell(
        energy_map,
        x=x,
        y=y,
        width=width,
        height=height,
    )
    if layer_credits is not None and layer_energy is not None:
        return max(layer_credits, 0) + max(layer_energy, 0)

    # Fallback for synthetic states that do not populate map layers.
    total = 0
    for candidate in (position, *_adjacent_positions(position)):
        cell = cell_index.get(candidate)
        if cell is not None and _cell_target_already_siphoned(cell):
            cell_credits = 0
            cell_energy = 0
        else:
            cell_credits = max(int(getattr(cell, "credits", 0)), 0) if cell is not None else 0
            cell_energy = max(int(getattr(cell, "energy", 0)), 0) if cell is not None else 0
        resource = resource_values.get(candidate)
        resource_credits = resource[0] if resource is not None else 0
        resource_energy = resource[1] if resource is not None else 0
        total += max(cell_credits, resource_credits)
        total += max(cell_energy, resource_energy)
    return total


def _count_enemies(state: GameStateSnapshot) -> int:
    if state.map.status != "ok":
        return 0
    return sum(1 for enemy in state.map.enemies if enemy.in_bounds)


def _cell_target_already_siphoned(cell: object) -> bool:
    # Decompiled map state uses cell +0x18 as wall durability for walls and as a
    # depleted marker once a non-wall resource tile has already been siphoned.
    wall_state = int(getattr(cell, "wall_state", 0))
    is_wall = bool(getattr(cell, "is_wall", False) or hasattr(cell, "wall_type"))
    if is_wall:
        has_payload = bool(
            (getattr(cell, "prog_id", None) is not None and int(getattr(cell, "prog_id", 0)) > 0)
            or max(int(getattr(cell, "points", 0)), 0) > 0
        )
        return has_payload and wall_state <= 0
    return wall_state > 0


def _resource_targets(state: GameStateSnapshot) -> tuple[GridPosition, ...]:
    if state.map.status != "ok":
        return ()
    width = int(state.map.width)
    height = int(state.map.height)
    layers = state.map.layers
    walls = _wall_positions(state)
    cells = _cell_index(state)
    resource_values = _resource_value_index(state)
    weighted_targets: dict[GridPosition, int] = {}

    def add_target(position: GridPosition, *, weight: int) -> None:
        if not _is_in_bounds(position, width=width, height=height):
            return
        if position in walls:
            return
        cell = cells.get(position)
        if cell is not None and _cell_target_already_siphoned(cell):
            return
        bounded = max(int(weight), 0)
        current = weighted_targets.get(position)
        if current is None or bounded > current:
            weighted_targets[position] = bounded

    for cell in state.map.resource_cells:
        raw_cell = cells.get(cell.position)
        if raw_cell is not None and _cell_target_already_siphoned(raw_cell):
            continue
        cluster_total = _credit_energy_cluster_total(
            position=cell.position,
            cell_index=cells,
            resource_values=resource_values,
            width=width,
            height=height,
            credits_map=layers.credits_map,
            energy_map=layers.energy_map,
        )
        points_total = max(int(cell.points), 0)
        if cluster_total <= 0 and points_total <= 0:
            continue
        add_target(cell.position, weight=cluster_total + points_total)

    for cell in state.map.cells:
        if _cell_target_already_siphoned(cell):
            continue
        cluster_total = _credit_energy_cluster_total(
            position=cell.position,
            cell_index=cells,
            resource_values=resource_values,
            width=width,
            height=height,
            credits_map=layers.credits_map,
            energy_map=layers.energy_map,
        )
        has_prog = bool(cell.prog_id is not None and cell.prog_id > 0)
        points_total = max(int(cell.points), 0)
        if not cell.is_wall:
            if cluster_total <= 0 and not has_prog and points_total <= 0:
                continue
            add_target(
                cell.position,
                weight=cluster_total + points_total + (25 if has_prog else 0),
            )
            continue
        if not has_prog and points_total <= 0:
            continue
        wall_weight = points_total + (25 if has_prog else 0)
        for adjacent in _adjacent_positions(cell.position):
            adjacent_cluster = _credit_energy_cluster_total(
                position=adjacent,
                cell_index=cells,
                resource_values=resource_values,
                width=width,
                height=height,
                credits_map=layers.credits_map,
                energy_map=layers.energy_map,
            )
            add_target(adjacent, weight=adjacent_cluster + wall_weight)

    for wall in state.map.walls:
        if _cell_target_already_siphoned(wall):
            continue
        has_prog = bool(wall.prog_id is not None and wall.prog_id > 0)
        points_total = max(int(wall.points), 0)
        if not has_prog and points_total <= 0:
            continue
        wall_weight = points_total + (25 if has_prog else 0)
        for adjacent in _adjacent_positions(wall.position):
            adjacent_cluster = _credit_energy_cluster_total(
                position=adjacent,
                cell_index=cells,
                resource_values=resource_values,
                width=width,
                height=height,
                credits_map=layers.credits_map,
                energy_map=layers.energy_map,
            )
            add_target(adjacent, weight=adjacent_cluster + wall_weight)

    ordered = sorted(
        weighted_targets.items(),
        key=lambda item: (-item[1], item[0].y, item[0].x),
    )
    return tuple(position for position, _ in ordered)


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
        self._completed_resource_targets: set[GridPosition] = set()

    def start_episode(self) -> None:
        self.meta_controller.start_episode()
        self.threat_controller.start_episode()
        self._locked_phase = None
        self._locked_target = None
        self._completed_resource_targets.clear()

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
                self._nearest_target(
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

    def observe_step_result(
        self,
        *,
        trace: HybridDecisionTrace,
        info: dict[str, object] | None = None,
    ) -> None:
        if trace.decision.objective.phase != ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS:
            return
        target = trace.decision.objective.target_position
        if target is None:
            return
        if trace.decision.action not in {"space", "z"}:
            return
        if not bool((info or {}).get("action_effective", False)):
            return
        self._completed_resource_targets.update(_siphon_reach_positions(target))
        if self._locked_phase == ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS and self._locked_target == target:
            self._locked_target = None

    def _objective_for_decision_phase(
        self,
        *,
        state: GameStateSnapshot,
        phase: ObjectivePhase,
        scripted_mode: bool,
    ) -> tuple[ObjectivePhase, GridPosition | None]:
        resolved_phase = self._resolved_phase_for_state(state=state, phase=phase)
        if self._locked_phase is not None and self._locked_phase != resolved_phase:
            self._locked_phase = None
            self._locked_target = None
        if (
            self._locked_phase == resolved_phase
            and self._locked_target is not None
            and self._is_target_valid_for_phase(
                state=state,
                phase=resolved_phase,
                target=self._locked_target,
                scripted_mode=scripted_mode,
            )
        ):
            return (resolved_phase, self._locked_target)

        resolved_phase, resolved_target = self.resolve_objective_for_phase(
            state=state,
            phase=resolved_phase,
            scripted_mode=scripted_mode,
        )
        self._locked_phase = resolved_phase
        self._locked_target = resolved_target
        return (resolved_phase, resolved_target)

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

    def _available_resource_targets(self, *, state: GameStateSnapshot) -> tuple[GridPosition, ...]:
        candidates = _resource_targets(state)
        if not self._completed_resource_targets:
            return candidates
        return tuple(
            candidate for candidate in candidates if candidate not in self._completed_resource_targets
        )

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
