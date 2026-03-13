"""Decision coordinator for objective selection + threat overrides."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.hybrid.astar_controller import AStarMovementController
from src.hybrid.danger import (
    ENEMY_TYPE_CLASSIC,
    ENEMY_TYPE_CRYPTOG,
    ENEMY_TYPE_DAEMON,
    ENEMY_TYPE_GLITCH,
    ENEMY_TYPE_VIRUS,
    EnemyTurnPrediction,
    TRACKED_ENEMY_TYPES,
    assess_position_danger,
    enemy_can_attack_position,
    is_in_bounds,
    manhattan,
)
from src.hybrid.meta_controller import MetaControllerDQN
from src.hybrid.threat_controller import ThreatControllerDRQN
from src.hybrid.types import (
    HybridDecision,
    HybridDecisionTrace,
    MetaObjectiveChoice,
    ObjectivePhase,
    ThreatOverride,
)
from src.state.schema import EnemyState, GameStateSnapshot, GridPosition

_MOVE_DELTAS: dict[str, tuple[int, int]] = {
    "move_up": (0, 1),
    "move_down": (0, -1),
    "move_left": (-1, 0),
    "move_right": (1, 0),
}
_MOVE_ACTIONS: tuple[str, ...] = tuple(_MOVE_DELTAS.keys())
_CARDINAL_DELTAS: tuple[tuple[int, int], ...] = ((0, 1), (1, 0), (0, -1), (-1, 0))
_SAFE_PENALTY_MAX = 1
_RISKY_PENALTY_MAX = 3


@dataclass(frozen=True)
class HybridCoordinatorConfig:
    """Behavior knobs for the hybrid coordinator."""

    threat_trigger_distance: int = 2
    enable_prog_override: bool = True
    exit_after_siphons_when_scripted: bool = False
    danger_tile_cost: float = 6.0
    guaranteed_damage_cost: float = 24.0
    recent_damage_steps: int = 2


@dataclass(frozen=True)
class _HarvestTarget:
    position: GridPosition
    value: int
    penalty: int
    bucket: int


@dataclass(frozen=True)
class _ActionDangerProfile:
    action: str
    position: GridPosition
    prediction: EnemyTurnPrediction
    immediate_threat: bool
    safe_followups: int
    nearest_distance: int | None


def _wall_positions(state: GameStateSnapshot) -> set[GridPosition]:
    if state.map.status != "ok":
        return set()
    walls = {cell.position for cell in state.map.cells if cell.is_wall}
    if walls:
        return walls
    return {wall.position for wall in state.map.walls}


def _adjacent_positions(position: GridPosition) -> tuple[GridPosition, ...]:
    return tuple(
        GridPosition(x=position.x + dx, y=position.y + dy)
        for dx, dy in _CARDINAL_DELTAS
    )


def _cell_index(state: GameStateSnapshot) -> dict[GridPosition, object]:
    if state.map.status != "ok":
        return {}
    return {cell.position: cell for cell in state.map.cells}


def _resource_value_index(state: GameStateSnapshot) -> dict[GridPosition, tuple[int, int, int]]:
    if state.map.status != "ok":
        return {}
    values: dict[GridPosition, tuple[int, int, int]] = {}
    for resource in state.map.resource_cells:
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

    total = 0
    for candidate in (position, *_adjacent_positions(position)):
        cell = cell_index.get(candidate)
        cell_credits = max(int(getattr(cell, "credits", 0)), 0) if cell is not None else 0
        cell_energy = max(int(getattr(cell, "energy", 0)), 0) if cell is not None else 0
        resource = resource_values.get(candidate)
        resource_credits = resource[0] if resource is not None else 0
        resource_energy = resource[1] if resource is not None else 0
        total += max(cell_credits, resource_credits)
        total += max(cell_energy, resource_energy)
    return total


def _siphon_penalty_total(
    *,
    state: GameStateSnapshot,
    position: GridPosition,
    cell_index: dict[GridPosition, object],
    width: int,
    height: int,
) -> int:
    if state.map.status != "ok":
        return 0
    layer_penalty = _read_layer_cell(
        state.map.layers.siphon_penalty_map,
        x=int(position.x),
        y=int(position.y),
        width=width,
        height=height,
    )
    if layer_penalty is not None:
        return max(layer_penalty, 0)

    total = 0
    for candidate in (position, *_adjacent_positions(position)):
        cell = cell_index.get(candidate)
        total += max(int(getattr(cell, "threat", 0)), 0) if cell is not None else 0
    return total


def _penalty_bucket(penalty: int) -> int:
    bounded = max(int(penalty), 0)
    if bounded <= _SAFE_PENALTY_MAX:
        return 0
    if bounded <= _RISKY_PENALTY_MAX:
        return 1
    return 2


def _penalty_bucket_flags(bucket: int) -> tuple[float, float, float]:
    return (
        1.0 if bucket == 0 else 0.0,
        1.0 if bucket == 1 else 0.0,
        1.0 if bucket == 2 else 0.0,
    )


def _count_enemies(state: GameStateSnapshot) -> int:
    if state.map.status != "ok":
        return 0
    return sum(1 for enemy in state.map.enemies if enemy.in_bounds)


def _enemy_counts_by_type(state: GameStateSnapshot) -> dict[int, int]:
    counts = {enemy_type: 0 for enemy_type in TRACKED_ENEMY_TYPES}
    if state.map.status != "ok":
        return counts
    for enemy in state.map.enemies:
        if not enemy.in_bounds:
            continue
        counts[int(enemy.type_id)] = counts.get(int(enemy.type_id), 0) + 1
    return counts


def _resource_target_candidates(state: GameStateSnapshot) -> tuple[_HarvestTarget, ...]:
    if state.map.status != "ok":
        return ()
    width = int(state.map.width)
    height = int(state.map.height)
    layers = state.map.layers
    walls = _wall_positions(state)
    cells = _cell_index(state)
    resource_values = _resource_value_index(state)
    weighted_targets: dict[GridPosition, int] = {}

    def add_target(position: GridPosition, *, value: int) -> None:
        if not is_in_bounds(position, width=width, height=height):
            return
        if position in walls:
            return
        bounded_value = max(int(value), 0)
        existing = weighted_targets.get(position)
        if existing is None or bounded_value > existing:
            weighted_targets[position] = bounded_value

    for cell in state.map.resource_cells:
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
        add_target(cell.position, value=cluster_total + points_total)

    for cell in state.map.cells:
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
                value=cluster_total + points_total + (25 if has_prog else 0),
            )
            continue
        if not has_prog and points_total <= 0:
            continue
        wall_value = points_total + (25 if has_prog else 0)
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
            add_target(adjacent, value=adjacent_cluster + wall_value)

    for wall in state.map.walls:
        has_prog = bool(wall.prog_id is not None and wall.prog_id > 0)
        points_total = max(int(wall.points), 0)
        if not has_prog and points_total <= 0:
            continue
        wall_value = points_total + (25 if has_prog else 0)
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
            add_target(adjacent, value=adjacent_cluster + wall_value)

    candidates: list[_HarvestTarget] = []
    for position, value in weighted_targets.items():
        penalty = _siphon_penalty_total(
            state=state,
            position=position,
            cell_index=cells,
            width=width,
            height=height,
        )
        candidates.append(
            _HarvestTarget(
                position=position,
                value=value,
                penalty=penalty,
                bucket=_penalty_bucket(penalty),
            )
        )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (item.bucket, -item.value, item.position.y, item.position.x),
        )
    )


def _resource_targets(state: GameStateSnapshot) -> tuple[GridPosition, ...]:
    return tuple(candidate.position for candidate in _resource_target_candidates(state))


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
        self._scripted_siphoned_targets: set[GridPosition] = set()
        self._recent_damage_steps_remaining = 0

    def start_episode(self) -> None:
        self.meta_controller.start_episode()
        self.threat_controller.start_episode()
        self._locked_phase = None
        self._locked_target = None
        self._scripted_siphoned_targets.clear()
        self._recent_damage_steps_remaining = 0

    def note_transition(
        self,
        *,
        previous_state: GameStateSnapshot,
        current_state: GameStateSnapshot,
    ) -> None:
        previous = self._state_numeric(previous_state.health.value) if previous_state.health.status == "ok" else 0.0
        current = self._state_numeric(current_state.health.value) if current_state.health.status == "ok" else 0.0
        if current < previous:
            self._recent_damage_steps_remaining = max(int(self.config.recent_damage_steps), 1)
            return
        if self._recent_damage_steps_remaining > 0:
            self._recent_damage_steps_remaining -= 1

    def allowed_meta_phases(self, state: GameStateSnapshot) -> tuple[ObjectivePhase, ...]:
        if state.map.status != "ok":
            return (ObjectivePhase.COLLECT_SIPHONS,)
        if state.map.siphons:
            return (ObjectivePhase.COLLECT_SIPHONS,)
        if self.config.exit_after_siphons_when_scripted:
            return (ObjectivePhase.EXIT_SECTOR,)
        if self._resource_collection_possible(state) and _resource_targets(state):
            return (
                ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS,
                ObjectivePhase.EXIT_SECTOR,
            )
        return (ObjectivePhase.EXIT_SECTOR,)

    def _should_persist_phase_lock(
        self,
        *,
        state: GameStateSnapshot,
        phase: ObjectivePhase,
        target: GridPosition | None,
        scripted_mode: bool,
    ) -> bool:
        if (
            phase == ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS
            and not self._resource_collection_possible(state)
        ):
            return False
        if target is None:
            return False
        if (
            not scripted_mode
            and phase == ObjectivePhase.EXIT_SECTOR
            and self._resource_collection_possible(state)
            and _resource_targets(state)
        ):
            return False
        return self._is_target_valid_for_phase(
            state=state,
            phase=phase,
            target=target,
            scripted_mode=scripted_mode,
        )

    def resolve_target_for_phase(
        self,
        *,
        state: GameStateSnapshot,
        phase: ObjectivePhase,
        scripted_mode: bool = False,
    ) -> GridPosition | None:
        if state.map.status != "ok" or state.map.player_position is None:
            return None
        if phase == ObjectivePhase.COLLECT_SIPHONS:
            return self._nearest_target(
                state=state,
                candidates=tuple(state.map.siphons),
            )
        if phase == ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS:
            if not self._resource_collection_possible(state):
                return None
            return self._best_resource_target(
                state=state,
                scripted_mode=scripted_mode,
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
        exit_known = 1.0 if state.map.status == "ok" and state.map.exit_position is not None else 0.0
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
        preview_route_action = self._route_action_for_objective(
            state=state,
            target=target,
            phase=scripted_phase,
            available_actions=_MOVE_ACTIONS,
        )
        current_prediction = self._current_position_prediction(state)
        route_prediction = self._action_prediction_for_action(
            state=state,
            action=preview_route_action,
        )
        target_penalty = self.target_penalty_value(state=state, target=target)
        target_bucket = _penalty_bucket(target_penalty)
        bucket_flags = _penalty_bucket_flags(target_bucket)
        enemy_counts = _enemy_counts_by_type(state)
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
            1.0 if state.can_siphon_now is True else 0.0,
            1.0 if state.map.status == "ok" else 0.0,
            1.0 if self._position_under_immediate_attack(state=state, position=state.map.player_position) else 0.0,
            1.0 if current_prediction.took_damage else 0.0,
            1.0 if route_prediction.took_damage else 0.0,
            float(self._safe_movement_action_count(state=state)) / 4.0,
            1.0 if self._recent_damage_steps_remaining > 0 else 0.0,
            min(float(target_penalty) / 6.0, 1.0),
            *bucket_flags,
            float(enemy_counts.get(ENEMY_TYPE_VIRUS, 0)) / 4.0,
            float(enemy_counts.get(ENEMY_TYPE_CLASSIC, 0)) / 4.0,
            float(enemy_counts.get(ENEMY_TYPE_GLITCH, 0)) / 4.0,
            float(enemy_counts.get(ENEMY_TYPE_CRYPTOG, 0)) / 4.0,
            float(enemy_counts.get(ENEMY_TYPE_DAEMON, 0)) / 4.0,
        )
        return self._fit_size(features, target_size=self.meta_controller.feature_count)

    def threat_feature_vector(
        self,
        *,
        state: GameStateSnapshot,
        route_action: str | None,
        objective_distance: int | None,
        target: GridPosition | None,
    ) -> tuple[float, ...]:
        health = self._state_numeric(state.health.value) if state.health.status == "ok" else 0.0
        energy = self._state_numeric(state.energy.value) if state.energy.status == "ok" else 0.0
        enemies = float(_count_enemies(state))
        nearest_enemy = self.nearest_enemy_distance(state)
        threat_active = self.is_threat_active(state, route_action=route_action)
        current_prediction = self._current_position_prediction(state)
        route_prediction = self._action_prediction_for_action(
            state=state,
            action=route_action,
        )
        safe_action_count = self._safe_movement_action_count(state=state)
        target_penalty = self.target_penalty_value(state=state, target=target)
        target_bucket = _penalty_bucket(target_penalty)
        bucket_flags = _penalty_bucket_flags(target_bucket)
        counts = _enemy_counts_by_type(state)
        action_one_hot = tuple(1.0 if route_action == action else 0.0 for action in _MOVE_ACTIONS)
        features = (
            health / 10.0,
            energy / 10.0,
            enemies / 8.0,
            self._normalized_distance(nearest_enemy),
            self._normalized_distance(objective_distance),
            1.0 if threat_active else 0.0,
            1.0 if self._position_under_immediate_attack(state=state, position=state.map.player_position) else 0.0,
            1.0 if current_prediction.took_damage else 0.0,
            1.0 if route_prediction.took_damage else 0.0,
            1.0 if route_prediction.took_damage else 0.0,
            float(safe_action_count) / 4.0,
            1.0 if self._recent_damage_steps_remaining > 0 else 0.0,
            *action_one_hot,
            1.0 if state.map.status == "ok" else 0.0,
            float(len(state.inventory.raw_prog_ids[:10])) / 10.0,
            1.0 if state.can_siphon_now is True else 0.0,
            1.0 if state.map.status == "ok" and state.map.exit_position is not None else 0.0,
            float(len(state.map.siphons)) / 6.0 if state.map.status == "ok" else 0.0,
            min(float(target_penalty) / 6.0, 1.0),
            *bucket_flags,
            float(counts.get(ENEMY_TYPE_VIRUS, 0)) / 4.0,
            float(counts.get(ENEMY_TYPE_CLASSIC, 0)) / 4.0,
            float(counts.get(ENEMY_TYPE_GLITCH, 0)) / 4.0,
            float(counts.get(ENEMY_TYPE_CRYPTOG, 0)) / 4.0,
            float(counts.get(ENEMY_TYPE_DAEMON, 0)) / 4.0,
            self._normalized_distance(self._nearest_enemy_distance_by_type(state=state, enemy_type=ENEMY_TYPE_VIRUS)),
            self._normalized_distance(self._nearest_enemy_distance_by_type(state=state, enemy_type=ENEMY_TYPE_CLASSIC)),
            self._normalized_distance(self._nearest_enemy_distance_by_type(state=state, enemy_type=ENEMY_TYPE_GLITCH)),
            self._normalized_distance(self._nearest_enemy_distance_by_type(state=state, enemy_type=ENEMY_TYPE_CRYPTOG)),
            self._normalized_distance(self._nearest_enemy_distance_by_type(state=state, enemy_type=ENEMY_TYPE_DAEMON)),
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
        plan = self._plan_route(
            state=state,
            start=state.map.player_position,
            target=target,
            allowed_first_actions=None,
        )
        if plan is None:
            return None
        return int(plan.distance)

    def nearest_enemy_distance(self, state: GameStateSnapshot) -> int | None:
        if state.map.status != "ok" or state.map.player_position is None:
            return None
        enemies = tuple(enemy.position for enemy in state.map.enemies if enemy.in_bounds)
        return self._nearest_distance_to_targets(state=state, targets=enemies)

    def target_penalty_value(
        self,
        *,
        state: GameStateSnapshot,
        target: GridPosition | None,
    ) -> int:
        if target is None or state.map.status != "ok":
            return 0
        return _siphon_penalty_total(
            state=state,
            position=target,
            cell_index=_cell_index(state),
            width=int(state.map.width),
            height=int(state.map.height),
        )

    def is_threat_active(
        self,
        state: GameStateSnapshot,
        *,
        route_action: str | None = None,
    ) -> bool:
        current_prediction = self._current_position_prediction(state)
        if current_prediction.took_damage:
            return True
        route_prediction = self._action_prediction_for_action(state=state, action=route_action)
        return route_prediction.took_damage

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
        scripted_phase = allowed_phases[0]
        scripted_target = self.resolve_target_for_phase(
            state=state,
            phase=scripted_phase,
            scripted_mode=scripted_mode,
        )
        meta_features = self.meta_feature_vector(
            state=state,
            scripted_phase=scripted_phase,
            target=scripted_target,
        )
        locked_phase_active = bool(
            use_meta_controller
            and self._should_persist_phase_lock(
                state=state,
                phase=self._locked_phase if self._locked_phase is not None else scripted_phase,
                target=self._locked_target,
                scripted_mode=scripted_mode,
            )
        )
        if locked_phase_active and self._locked_phase is not None:
            selected_phase = self._locked_phase
            meta_reason = "meta_phase_lock"
            q_value = None
        elif use_meta_controller:
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

        objective_target = self._target_for_decision_phase(
            state=state,
            phase=selected_phase,
            scripted_mode=scripted_mode,
        )
        objective_distance = self.objective_distance(state=state, target=objective_target)
        if (
            scripted_mode
            and selected_phase == ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS
            and objective_target is None
        ):
            selected_phase = ObjectivePhase.EXIT_SECTOR
            objective_target = self.resolve_target_for_phase(
                state=state,
                phase=selected_phase,
                scripted_mode=scripted_mode,
            )
            objective_distance = self.objective_distance(state=state, target=objective_target)

        route_action = self._route_action_for_objective(
            state=state,
            target=objective_target,
            phase=selected_phase,
            available_actions=actions,
        )
        current_prediction = self._current_position_prediction(state)
        route_prediction = self._action_prediction_for_action(state=state, action=route_action)
        threat_active = current_prediction.took_damage or route_prediction.took_damage
        threat_features = self.threat_feature_vector(
            state=state,
            route_action=route_action,
            objective_distance=objective_distance,
            target=objective_target,
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
        final_prediction = self._action_prediction_for_action(state=state, action=action)
        self._record_scripted_siphon_target(
            scripted_mode=scripted_mode,
            phase=selected_phase,
            target=objective_target,
            action=action,
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
            route_action=route_action,
            objective_distance_before=objective_distance,
            threat_active=threat_active,
            current_tile_threatened=current_prediction.took_damage,
            route_tile_threatened=route_prediction.took_damage,
            predicted_damage=final_prediction.took_damage,
            safe_action_count=self._safe_movement_action_count(state=state),
            predicted_attack_types=final_prediction.attack_types,
            objective_penalty_bucket=_penalty_bucket(self.target_penalty_value(state=state, target=objective_target)),
            available_actions=actions,
        )

    def _target_for_decision_phase(
        self,
        *,
        state: GameStateSnapshot,
        phase: ObjectivePhase,
        scripted_mode: bool,
    ) -> GridPosition | None:
        if self._locked_phase is not None and self._locked_phase != phase:
            self._locked_phase = None
            self._locked_target = None
        if (
            self._locked_phase == phase
            and self._locked_target is not None
            and self._is_target_valid_for_phase(
                state=state,
                phase=phase,
                target=self._locked_target,
                scripted_mode=scripted_mode,
            )
        ):
            return self._locked_target

        resolved = self.resolve_target_for_phase(
            state=state,
            phase=phase,
            scripted_mode=scripted_mode,
        )
        if self._should_persist_phase_lock(
            state=state,
            phase=phase,
            target=resolved,
            scripted_mode=scripted_mode,
        ):
            self._locked_phase = phase
            self._locked_target = resolved
        else:
            self._locked_phase = None
            self._locked_target = None
        return resolved

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
            if not self._resource_collection_possible(state):
                return False
            if scripted_mode and target in self._scripted_siphoned_targets:
                return False
            return target in _resource_targets(state)
        if phase == ObjectivePhase.EXIT_SECTOR:
            return bool(state.map.exit_position is not None and state.map.exit_position == target)
        return False

    def _record_scripted_siphon_target(
        self,
        *,
        scripted_mode: bool,
        phase: ObjectivePhase,
        target: GridPosition | None,
        action: str,
    ) -> None:
        if not scripted_mode:
            return
        if phase != ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS:
            return
        if target is None:
            return
        if action not in {"space", "z"}:
            return
        self._scripted_siphoned_targets.add(target)

    def _best_resource_target(
        self,
        *,
        state: GameStateSnapshot,
        scripted_mode: bool,
    ) -> GridPosition | None:
        if state.map.status != "ok" or state.map.player_position is None:
            return None
        candidates = _resource_target_candidates(state)
        if scripted_mode and self._scripted_siphoned_targets:
            candidates = tuple(
                candidate
                for candidate in candidates
                if candidate.position not in self._scripted_siphoned_targets
            )
        if not candidates:
            return None

        player = state.map.player_position
        best_target: GridPosition | None = None
        best_rank: tuple[int, int, int, int, int] | None = None
        for candidate in candidates:
            plan = self._plan_route(
                state=state,
                start=player,
                target=candidate.position,
                allowed_first_actions=None,
            )
            if plan is None:
                continue
            rank = (
                int(candidate.bucket),
                -int(candidate.value),
                int(plan.distance),
                int(candidate.position.y),
                int(candidate.position.x),
            )
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_target = candidate.position
        return best_target

    def _nearest_target(
        self,
        *,
        state: GameStateSnapshot,
        candidates: tuple[GridPosition, ...],
    ) -> GridPosition | None:
        if not candidates or state.map.status != "ok" or state.map.player_position is None:
            return None
        player = state.map.player_position
        best_target: GridPosition | None = None
        best_distance: int | None = None
        for candidate in candidates:
            plan = self._plan_route(
                state=state,
                start=player,
                target=candidate,
                allowed_first_actions=None,
            )
            if plan is None:
                continue
            if best_distance is None or plan.distance < best_distance:
                best_distance = int(plan.distance)
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

    def _nearest_enemy_distance_by_type(
        self,
        *,
        state: GameStateSnapshot,
        enemy_type: int,
    ) -> int | None:
        if state.map.status != "ok":
            return None
        targets = tuple(
            enemy.position
            for enemy in state.map.enemies
            if enemy.in_bounds and int(enemy.type_id) == int(enemy_type)
        )
        return self._nearest_distance_to_targets(state=state, targets=targets)

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
            if (
                phase == ObjectivePhase.COLLECT_RESOURCES_PROGS_POINTS
                and self._should_defer_dangerous_harvest(
                    state=state,
                    target=target,
                )
            ):
                alternative = self._best_resource_target(state=state, scripted_mode=False)
                if alternative is not None and alternative != target:
                    return self._planned_next_action(
                        state=state,
                        start=player,
                        target=alternative,
                        available_actions=available_actions,
                    )
            siphon_action = self._select_siphon_action(available_actions=available_actions)
            if siphon_action is not None and state.can_siphon_now is not False:
                return siphon_action

        if target is None:
            return None
        return self._planned_next_action(
            state=state,
            start=player,
            target=target,
            available_actions=available_actions,
        )

    def _planned_next_action(
        self,
        *,
        state: GameStateSnapshot,
        start: GridPosition,
        target: GridPosition,
        available_actions: tuple[str, ...],
    ) -> str | None:
        blocked_positions, danger_costs = self._planner_danger_inputs(state)
        return self.movement_controller.next_action(
            start=start,
            target=target,
            width=state.map.width,
            height=state.map.height,
            walls=_wall_positions(state),
            available_actions=available_actions,
            blocked_positions=blocked_positions,
            danger_costs=danger_costs,
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
        for key in ("siphons", "player_siphons", "siphon_count"):
            field = state.extra_fields.get(key)
            if field is None or field.status != "ok" or field.value is None:
                continue
            try:
                return int(float(field.value))
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _resource_collection_possible(state: GameStateSnapshot) -> bool:
        if state.map.status != "ok":
            return True
        if state.map.siphons:
            return True
        player_siphons = HybridCoordinator._player_siphon_count(state)
        if player_siphons is None:
            return True
        return player_siphons > 0

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
        fallback = self._fallback_action(
            state=state,
            actions=actions,
            preferred=route_action,
        )
        return (fallback, True, invalid_override)

    def _select_evade_action(
        self,
        *,
        state: GameStateSnapshot,
        actions: tuple[str, ...],
    ) -> str | None:
        profiles = self._movement_action_profiles(state=state, actions=actions)
        if not profiles:
            return None
        safe_profiles = [profile for profile in profiles if not profile.prediction.took_damage]
        pool = safe_profiles or list(profiles)
        return min(
            pool,
            key=lambda profile: (
                profile.immediate_threat,
                -profile.safe_followups,
                -(profile.nearest_distance or 0),
                profile.action,
            ),
        ).action

    def _select_engage_action(
        self,
        *,
        state: GameStateSnapshot,
        actions: tuple[str, ...],
    ) -> str | None:
        profiles = self._movement_action_profiles(state=state, actions=actions)
        if not profiles:
            return None
        safe_profiles = [profile for profile in profiles if not profile.prediction.took_damage]
        pool = safe_profiles or list(profiles)
        return min(
            pool,
            key=lambda profile: (
                profile.prediction.took_damage,
                profile.nearest_distance if profile.nearest_distance is not None else 999,
                profile.immediate_threat,
                -profile.safe_followups,
                profile.action,
            ),
        ).action

    def _movement_action_profiles(
        self,
        *,
        state: GameStateSnapshot,
        actions: tuple[str, ...],
    ) -> tuple[_ActionDangerProfile, ...]:
        if state.map.status != "ok" or state.map.player_position is None:
            return ()
        player = state.map.player_position
        walls = _wall_positions(state)
        profiles: list[_ActionDangerProfile] = []
        for action in actions:
            delta = _MOVE_DELTAS.get(action)
            if delta is None:
                continue
            candidate = GridPosition(x=player.x + delta[0], y=player.y + delta[1])
            if candidate in walls:
                continue
            if not is_in_bounds(candidate, width=state.map.width, height=state.map.height):
                continue
            prediction = self._position_prediction(state=state, position=candidate)
            profiles.append(
                _ActionDangerProfile(
                    action=action,
                    position=candidate,
                    prediction=prediction,
                    immediate_threat=self._position_under_immediate_attack(state=state, position=candidate),
                    safe_followups=self._safe_followup_count(
                        state=state,
                        position=candidate,
                        enemy_positions=prediction.enemies,
                    ),
                    nearest_distance=self._nearest_enemy_distance_from_position(
                        position=candidate,
                        enemies=prediction.enemies,
                    ),
                )
            )
        return tuple(profiles)

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

    def _fallback_action(
        self,
        *,
        state: GameStateSnapshot,
        actions: tuple[str, ...],
        preferred: str | None,
    ) -> str:
        if preferred is not None and preferred in actions:
            return preferred
        safe_movement = self._safe_movement_actions(state=state, actions=actions)
        if safe_movement:
            return safe_movement[0]
        if "wait" in actions:
            return "wait"
        for action in actions:
            if action != "cancel":
                return action
        return actions[0]

    def _safe_movement_actions(
        self,
        *,
        state: GameStateSnapshot,
        actions: tuple[str, ...],
    ) -> tuple[str, ...]:
        safe: list[str] = []
        for profile in self._movement_action_profiles(state=state, actions=actions):
            if profile.prediction.took_damage:
                continue
            safe.append(profile.action)
        return tuple(safe)

    def _safe_movement_action_count(self, *, state: GameStateSnapshot) -> int:
        return len(self._safe_movement_actions(state=state, actions=_MOVE_ACTIONS))

    def _should_defer_dangerous_harvest(
        self,
        *,
        state: GameStateSnapshot,
        target: GridPosition,
    ) -> bool:
        if _penalty_bucket(self.target_penalty_value(state=state, target=target)) < 2:
            return False
        for candidate in _resource_target_candidates(state):
            if candidate.position == target:
                continue
            if candidate.bucket < 2:
                return True
        return False

    def _plan_route(
        self,
        *,
        state: GameStateSnapshot,
        start: GridPosition,
        target: GridPosition,
        allowed_first_actions: tuple[str, ...] | None,
    ):
        blocked_positions, danger_costs = self._planner_danger_inputs(state)
        return self.movement_controller.plan_route(
            start=start,
            target=target,
            width=state.map.width,
            height=state.map.height,
            walls=_wall_positions(state),
            allowed_first_actions=allowed_first_actions,
            blocked_positions=blocked_positions,
            danger_costs=danger_costs,
        )

    def _planner_danger_inputs(
        self,
        state: GameStateSnapshot,
    ) -> tuple[set[GridPosition], dict[GridPosition, float]]:
        if state.map.status != "ok" or not state.map.enemies:
            return (set(), {})
        blocked: set[GridPosition] = set()
        costs: dict[GridPosition, float] = {}
        walls = _wall_positions(state)
        enemies = tuple(enemy for enemy in state.map.enemies if enemy.in_bounds)
        for y in range(state.map.height):
            for x in range(state.map.width):
                position = GridPosition(x=x, y=y)
                if position in walls:
                    continue
                prediction = assess_position_danger(
                    position=position,
                    enemies=enemies,
                    width=state.map.width,
                    height=state.map.height,
                    walls=walls,
                )
                if prediction.took_damage:
                    blocked.add(position)
                    costs[position] = float(self.config.guaranteed_damage_cost)
        return (blocked, costs)

    def _current_position_prediction(self, state: GameStateSnapshot) -> EnemyTurnPrediction:
        if state.map.status != "ok" or state.map.player_position is None:
            return EnemyTurnPrediction(enemies=(), took_damage=False, attack_types=())
        return self._position_prediction(state=state, position=state.map.player_position)

    def _position_prediction(
        self,
        *,
        state: GameStateSnapshot,
        position: GridPosition,
        enemy_positions: tuple[EnemyState, ...] | None = None,
    ) -> EnemyTurnPrediction:
        if state.map.status != "ok":
            return EnemyTurnPrediction(enemies=(), took_damage=False, attack_types=())
        enemies = enemy_positions or tuple(enemy for enemy in state.map.enemies if enemy.in_bounds)
        if not enemies:
            return EnemyTurnPrediction(enemies=(), took_damage=False, attack_types=())
        return assess_position_danger(
            position=position,
            enemies=enemies,
            width=state.map.width,
            height=state.map.height,
            walls=_wall_positions(state),
        )

    def _action_prediction_for_action(
        self,
        *,
        state: GameStateSnapshot,
        action: str | None,
    ) -> EnemyTurnPrediction:
        if state.map.status != "ok" or state.map.player_position is None:
            return EnemyTurnPrediction(enemies=(), took_damage=False, attack_types=())
        position = state.map.player_position
        if action in _MOVE_DELTAS:
            delta = _MOVE_DELTAS[action]
            candidate = GridPosition(x=position.x + delta[0], y=position.y + delta[1])
            if candidate not in _wall_positions(state) and is_in_bounds(
                candidate,
                width=state.map.width,
                height=state.map.height,
            ):
                position = candidate
        return self._position_prediction(state=state, position=position)

    def _position_under_immediate_attack(
        self,
        *,
        state: GameStateSnapshot,
        position: GridPosition | None,
    ) -> bool:
        if state.map.status != "ok" or position is None:
            return False
        return any(
            enemy.in_bounds
            and enemy_can_attack_position(
                enemy_type=enemy.type_id,
                enemy_position=enemy.position,
                player_position=position,
            )
            for enemy in state.map.enemies
        )

    def _safe_followup_count(
        self,
        *,
        state: GameStateSnapshot,
        position: GridPosition,
        enemy_positions: tuple[EnemyState, ...],
    ) -> int:
        walls = _wall_positions(state)
        safe_count = 0
        candidates = (position, *_adjacent_positions(position))
        for candidate in candidates:
            if candidate in walls:
                continue
            if not is_in_bounds(candidate, width=state.map.width, height=state.map.height):
                continue
            prediction = self._position_prediction(
                state=state,
                position=candidate,
                enemy_positions=enemy_positions,
            )
            if not prediction.took_damage:
                safe_count += 1
        return safe_count

    @staticmethod
    def _nearest_enemy_distance_from_position(
        *,
        position: GridPosition,
        enemies: tuple[EnemyState, ...],
    ) -> int | None:
        nearest: int | None = None
        for enemy in enemies:
            if not enemy.in_bounds:
                continue
            distance = manhattan(position, enemy.position)
            if nearest is None or distance < nearest:
                nearest = distance
        return nearest

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
