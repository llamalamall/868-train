"""Objective-driven reward shaping with deterministic component breakdowns."""

from __future__ import annotations

import logging
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Mapping

from src.state.schema import EnemyState, FieldState, GameStateSnapshot, GridPosition

LOGGER = logging.getLogger(__name__)
_ENEMY_TYPE_VIRUS = 2
_MOVE_VECTORS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, -1),
    (-1, 0),
    (1, 0),
)


@dataclass(frozen=True)
class RewardWeights:
    """Configurable weights for each reward component."""

    survival: float = 0.02
    step_penalty: float = 0.01
    health_delta: float = 0.40
    currency_delta: float = 0.03
    energy_delta: float = 0.02
    score_delta: float = 0.01
    siphon_collected: float = 2.50
    enemy_damaged: float = 0.35
    enemy_cleared: float = 1.50
    phase_progress: float = 0.25
    map_clear_bonus: float = 8.0
    premature_exit_penalty: float = 2.5
    invalid_action_penalty: float = 0.75
    fail_penalty: float = 12.0
    safe_tile_bonus: float = 0.02
    danger_tile_penalty: float = 0.08
    resource_proximity: float = 0.05
    prog_collected_base: float = 0.5
    points_collected: float = 0.05
    damage_taken_penalty: float = 0.60

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> RewardWeights:
        """Build weights from a partial config mapping."""
        return cls(
            survival=float(values.get("survival", cls.survival)),
            step_penalty=float(values.get("step_penalty", cls.step_penalty)),
            health_delta=float(values.get("health_delta", cls.health_delta)),
            currency_delta=float(values.get("currency_delta", cls.currency_delta)),
            energy_delta=float(values.get("energy_delta", cls.energy_delta)),
            score_delta=float(values.get("score_delta", cls.score_delta)),
            siphon_collected=float(values.get("siphon_collected", cls.siphon_collected)),
            enemy_damaged=float(values.get("enemy_damaged", cls.enemy_damaged)),
            enemy_cleared=float(values.get("enemy_cleared", cls.enemy_cleared)),
            phase_progress=float(values.get("phase_progress", cls.phase_progress)),
            map_clear_bonus=float(values.get("map_clear_bonus", cls.map_clear_bonus)),
            premature_exit_penalty=float(
                values.get("premature_exit_penalty", cls.premature_exit_penalty)
            ),
            invalid_action_penalty=float(
                values.get("invalid_action_penalty", cls.invalid_action_penalty)
            ),
            fail_penalty=float(values.get("fail_penalty", cls.fail_penalty)),
            safe_tile_bonus=float(values.get("safe_tile_bonus", cls.safe_tile_bonus)),
            danger_tile_penalty=float(values.get("danger_tile_penalty", cls.danger_tile_penalty)),
            resource_proximity=float(values.get("resource_proximity", cls.resource_proximity)),
            prog_collected_base=float(values.get("prog_collected_base", cls.prog_collected_base)),
            points_collected=float(values.get("points_collected", cls.points_collected)),
            damage_taken_penalty=float(values.get("damage_taken_penalty", cls.damage_taken_penalty)),
        )


@dataclass(frozen=True)
class RewardConfig:
    """Reward configuration container for training settings."""

    weights: RewardWeights = field(default_factory=RewardWeights)
    survival_when_done: bool = False
    reward_clip_abs: float = 5.0
    prog_priority_bonus_by_id: Mapping[int, float] = field(
        default_factory=lambda: {
            10: 1.5,
            8: 1.2,
            7: 1.0,
            9: 1.0,
            18: 0.8,
        }
    )
    prog_priority_fallback_bonus: float = 0.5


@dataclass(frozen=True)
class RewardBreakdown:
    """Per-component reward contributions for debugging."""

    survival: float
    step_penalty: float
    health_change: float
    currency_change: float
    energy_change: float
    score_change: float
    siphon_collected: float
    enemy_damaged: float
    enemy_cleared: float
    phase_progress: float
    map_clear_bonus: float
    premature_exit_penalty: float
    invalid_action_penalty: float
    fail_penalty: float
    safe_tile_bonus: float
    danger_tile_penalty: float
    resource_proximity: float
    prog_collected: float
    points_collected: float
    damage_taken_penalty: float

    @property
    def total(self) -> float:
        """Return the unclipped total reward across all components."""
        return (
            self.survival
            + self.step_penalty
            + self.health_change
            + self.currency_change
            + self.energy_change
            + self.score_change
            + self.siphon_collected
            + self.enemy_damaged
            + self.enemy_cleared
            + self.phase_progress
            + self.map_clear_bonus
            + self.premature_exit_penalty
            + self.invalid_action_penalty
            + self.fail_penalty
            + self.safe_tile_bonus
            + self.danger_tile_penalty
            + self.resource_proximity
            + self.prog_collected
            + self.points_collected
            + self.damage_taken_penalty
        )


@dataclass(frozen=True)
class RewardResult:
    """Reward output for a single transition."""

    total: float
    breakdown: RewardBreakdown


def _numeric_field_value(field: FieldState) -> float | None:
    if field.status != "ok" or field.value is None:
        return None
    value = field.value
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, int | float):
        return float(value)
    return None


def _bool_field_value(field: FieldState) -> bool | None:
    if field.status != "ok" or field.value is None:
        return None
    value = field.value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    return None


def _state_extra_numeric(state: GameStateSnapshot, *, key: str) -> float | None:
    field = state.extra_fields.get(key)
    if field is None:
        return None
    return _numeric_field_value(field)


def _count_siphons(state: GameStateSnapshot) -> int | None:
    if state.map.status != "ok":
        return None
    return len(state.map.siphons)


def _count_live_enemies(state: GameStateSnapshot) -> int | None:
    if state.map.status != "ok":
        return None
    return sum(1 for enemy in state.map.enemies if enemy.in_bounds)


def _enemy_identity_counts(state: GameStateSnapshot) -> Counter[int] | None:
    if state.map.status != "ok":
        return None
    return Counter(int(enemy.slot) for enemy in state.map.enemies if enemy.in_bounds)


def _enemy_cleared_delta(
    *,
    previous_state: GameStateSnapshot,
    current_state: GameStateSnapshot,
) -> float:
    previous_counts = _enemy_identity_counts(previous_state)
    current_counts = _enemy_identity_counts(current_state)
    if previous_counts is None or current_counts is None:
        return 0.0

    cleared = 0
    for enemy_id, previous_count in previous_counts.items():
        current_count = current_counts.get(enemy_id, 0)
        if previous_count > current_count:
            cleared += previous_count - current_count
    return float(cleared)


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


def _player_on_exit(state: GameStateSnapshot) -> bool:
    if state.map.status != "ok":
        return False
    player = state.map.player_position
    exit_position = state.map.exit_position
    return player is not None and exit_position is not None and player == exit_position


def _manhattan(a: GridPosition, b: GridPosition) -> int:
    return abs(a.x - b.x) + abs(a.y - b.y)


def _chebyshev(a: GridPosition, b: GridPosition) -> int:
    return max(abs(a.x - b.x), abs(a.y - b.y))


def _is_in_bounds(position: GridPosition, *, width: int, height: int) -> bool:
    return 0 <= position.x < width and 0 <= position.y < height


def _wall_positions(state: GameStateSnapshot) -> set[GridPosition]:
    if state.map.status != "ok":
        return set()
    positions = {cell.position for cell in state.map.cells if cell.is_wall}
    if positions:
        return positions
    return {wall.position for wall in state.map.walls}


def _shortest_path_distance(state: GameStateSnapshot, *, target: GridPosition) -> int | None:
    if state.map.status != "ok" or state.map.player_position is None:
        return None

    width = int(state.map.width)
    height = int(state.map.height)
    if width <= 0 or height <= 0:
        return None
    if not _is_in_bounds(target, width=width, height=height):
        return None

    start = state.map.player_position
    if start == target:
        return 0

    walls = _wall_positions(state)
    if target in walls:
        return None

    queue: deque[tuple[GridPosition, int]] = deque([(start, 0)])
    visited: set[GridPosition] = {start}
    while queue:
        current, distance = queue.popleft()
        for dx, dy in _MOVE_VECTORS:
            candidate = GridPosition(x=current.x + dx, y=current.y + dy)
            if not _is_in_bounds(candidate, width=width, height=height):
                continue
            if candidate in walls or candidate in visited:
                continue
            if candidate == target:
                return distance + 1
            visited.add(candidate)
            queue.append((candidate, distance + 1))
    return None


def _nearest_path_distance_to_targets(
    *,
    state: GameStateSnapshot,
    targets: tuple[GridPosition, ...],
) -> int | None:
    if not targets:
        return None
    best_distance: int | None = None
    for target in targets:
        distance = _shortest_path_distance(state, target=target)
        if distance is None:
            continue
        if best_distance is None or distance < best_distance:
            best_distance = distance
    return best_distance


def _nearest_siphon_distance(state: GameStateSnapshot) -> int | None:
    if state.map.status != "ok":
        return None
    if state.map.player_position is None or not state.map.siphons:
        return None
    return _nearest_path_distance_to_targets(
        state=state,
        targets=tuple(state.map.siphons),
    )


def _nearest_enemy_distance(state: GameStateSnapshot) -> int | None:
    if state.map.status != "ok":
        return None
    if state.map.player_position is None:
        return None
    enemies = tuple(enemy.position for enemy in state.map.enemies if enemy.in_bounds)
    if not enemies:
        return None
    return _nearest_path_distance_to_targets(
        state=state,
        targets=enemies,
    )


def _exit_distance(state: GameStateSnapshot) -> int | None:
    if state.map.status != "ok":
        return None
    exit_position = state.map.exit_position
    if state.map.player_position is None or exit_position is None:
        return None
    return _shortest_path_distance(state, target=exit_position)


def _phase_progress_delta(
    *,
    previous_state: GameStateSnapshot,
    current_state: GameStateSnapshot,
) -> float:
    previous_siphons = _count_siphons(previous_state)
    previous_enemies = _count_live_enemies(previous_state)
    if previous_siphons is not None and previous_siphons > 0:
        previous_distance = _nearest_siphon_distance(previous_state)
        current_distance = _nearest_siphon_distance(current_state)
    elif previous_enemies is not None and previous_enemies > 0:
        previous_distance = _nearest_enemy_distance(previous_state)
        current_distance = _nearest_enemy_distance(current_state)
    else:
        previous_distance = _exit_distance(previous_state)
        current_distance = _exit_distance(current_state)

    if previous_distance is None or current_distance is None:
        return 0.0
    return float(previous_distance - current_distance)


def _clip(value: float, *, clip_abs: float) -> float:
    if clip_abs <= 0:
        return value
    if value > clip_abs:
        return clip_abs
    if value < -clip_abs:
        return -clip_abs
    return value


def _inventory_counts(state: GameStateSnapshot) -> Counter[int]:
    if state.inventory.status != "ok":
        return Counter()
    return Counter(int(prog_id) for prog_id in state.inventory.raw_prog_ids)


def _prog_gain_reward(
    *,
    previous_state: GameStateSnapshot,
    current_state: GameStateSnapshot,
    config: RewardConfig,
    weights: RewardWeights,
) -> float:
    previous_counts = _inventory_counts(previous_state)
    current_counts = _inventory_counts(current_state)
    if not previous_counts and not current_counts:
        return 0.0

    total_reward = 0.0
    for prog_id, current_count in current_counts.items():
        previous_count = previous_counts.get(prog_id, 0)
        if current_count <= previous_count:
            continue
        gained = current_count - previous_count
        priority_bonus = float(
            config.prog_priority_bonus_by_id.get(
                int(prog_id),
                config.prog_priority_fallback_bonus,
            )
        )
        total_reward += float(gained) * (abs(weights.prog_collected_base) + priority_bonus)
    return total_reward


def _available_points_total(state: GameStateSnapshot) -> int | None:
    if state.map.status != "ok":
        return None
    # Use decoded cells as authoritative total available board points to avoid double counting.
    return sum(max(int(cell.points), 0) for cell in state.map.cells)


def _harvest_targets(state: GameStateSnapshot) -> tuple[GridPosition, ...]:
    if state.map.status != "ok":
        return ()

    targets: set[GridPosition] = set()
    width = int(state.map.width)
    height = int(state.map.height)
    walls = _wall_positions(state)

    def _add_adjacent_walkable_positions(position: GridPosition) -> None:
        for dx, dy in _MOVE_VECTORS:
            candidate = GridPosition(x=position.x + dx, y=position.y + dy)
            if not _is_in_bounds(candidate, width=width, height=height):
                continue
            if candidate in walls:
                continue
            targets.add(candidate)

    for resource in state.map.resource_cells:
        if resource.credits > 0 or resource.energy > 0 or resource.points > 0:
            targets.add(resource.position)
    for cell in state.map.cells:
        if cell.is_wall:
            if cell.prog_id is not None or cell.points > 0:
                _add_adjacent_walkable_positions(cell.position)
            continue
        if cell.credits > 0 or cell.energy > 0 or cell.points > 0 or cell.prog_id is not None:
            targets.add(cell.position)
    for wall in state.map.walls:
        if wall.prog_id is not None or wall.points > 0:
            _add_adjacent_walkable_positions(wall.position)
    return tuple(targets)


def _nearest_harvest_target_distance(state: GameStateSnapshot) -> int | None:
    if state.map.status != "ok" or state.map.player_position is None:
        return None
    targets = _harvest_targets(state)
    if not targets:
        return None
    return _nearest_path_distance_to_targets(state=state, targets=targets)


def _enemy_can_attack_position(
    *,
    enemy: EnemyState,
    player_position: GridPosition,
) -> bool:
    if enemy.type_id == _ENEMY_TYPE_VIRUS:
        return _chebyshev(enemy.position, player_position) <= 1
    return _manhattan(enemy.position, player_position) <= 1


def _player_tile_threatened(state: GameStateSnapshot) -> bool | None:
    if state.map.status != "ok" or state.map.player_position is None:
        return None
    player = state.map.player_position
    enemies = tuple(enemy for enemy in state.map.enemies if enemy.in_bounds)
    if not enemies:
        return False
    return any(_enemy_can_attack_position(enemy=enemy, player_position=player) for enemy in enemies)


def compute_reward(
    *,
    previous_state: GameStateSnapshot,
    current_state: GameStateSnapshot,
    done: bool,
    config: RewardConfig | None = None,
    logger: logging.Logger | None = None,
    info: Mapping[str, Any] | None = None,
) -> RewardResult:
    """Compute deterministic reward for one state transition."""
    active_config = config or RewardConfig()
    weights = active_config.weights
    active_logger = logger or LOGGER
    info_payload: Mapping[str, Any] = info or {}

    previous_health = _numeric_field_value(previous_state.health)
    current_health = _numeric_field_value(current_state.health)
    health_delta = (
        0.0
        if previous_health is None or current_health is None
        else (current_health - previous_health)
    )

    previous_energy = _numeric_field_value(previous_state.energy)
    current_energy = _numeric_field_value(current_state.energy)
    energy_delta = (
        0.0
        if previous_energy is None or current_energy is None
        else (current_energy - previous_energy)
    )

    previous_currency = _numeric_field_value(previous_state.currency)
    current_currency = _numeric_field_value(current_state.currency)
    currency_delta = (
        0.0
        if previous_currency is None or current_currency is None
        else (current_currency - previous_currency)
    )

    previous_score = _state_extra_numeric(previous_state, key="score")
    current_score = _state_extra_numeric(current_state, key="score")
    score_delta = (
        0.0
        if previous_score is None or current_score is None
        else (current_score - previous_score)
    )

    previous_siphons = _count_siphons(previous_state)
    current_siphons = _count_siphons(current_state)
    siphons_collected = 0.0
    if previous_siphons is not None and current_siphons is not None:
        siphons_collected = float(max(previous_siphons - current_siphons, 0))

    previous_enemies = _count_live_enemies(previous_state)
    current_enemies = _count_live_enemies(current_state)
    enemy_damage = _enemy_damage_delta(
        previous_state=previous_state,
        current_state=current_state,
    )
    enemies_cleared = _enemy_cleared_delta(
        previous_state=previous_state,
        current_state=current_state,
    )

    previous_available_points = _available_points_total(previous_state)
    current_available_points = _available_points_total(current_state)
    points_collected_delta = 0.0
    if previous_available_points is not None and current_available_points is not None:
        points_collected_delta = float(max(previous_available_points - current_available_points, 0))

    harvest_progress = 0.0
    previous_harvest_distance = _nearest_harvest_target_distance(previous_state)
    current_harvest_distance = _nearest_harvest_target_distance(current_state)
    if previous_harvest_distance is not None and current_harvest_distance is not None:
        harvest_progress = float(max(previous_harvest_distance - current_harvest_distance, 0))

    tile_threat = _player_tile_threatened(current_state)
    safe_tile_component = abs(weights.safe_tile_bonus) if tile_threat is False else 0.0
    danger_tile_component = -abs(weights.danger_tile_penalty) if tile_threat is True else 0.0

    survival_component = (
        weights.survival if (not done or active_config.survival_when_done) else 0.0
    )
    step_penalty_component = -abs(weights.step_penalty)
    health_component = health_delta * weights.health_delta
    currency_component = currency_delta * weights.currency_delta
    energy_component = energy_delta * weights.energy_delta
    score_component = score_delta * weights.score_delta
    siphon_component = siphons_collected * abs(weights.siphon_collected)
    enemy_damage_component = enemy_damage * abs(weights.enemy_damaged)
    enemy_component = enemies_cleared * abs(weights.enemy_cleared)
    phase_progress_component = (
        _phase_progress_delta(
            previous_state=previous_state,
            current_state=current_state,
        )
        * weights.phase_progress
    )
    proximity_component = harvest_progress * abs(weights.resource_proximity)
    prog_component = _prog_gain_reward(
        previous_state=previous_state,
        current_state=current_state,
        config=active_config,
        weights=weights,
    )
    points_collected_component = points_collected_delta * abs(weights.points_collected)
    damage_taken_component = (
        -abs(health_delta) * abs(weights.damage_taken_penalty) if health_delta < 0 else 0.0
    )

    on_exit_now = _player_on_exit(current_state)
    map_cleared = (
        on_exit_now
        and current_siphons is not None
        and current_siphons == 0
        and current_enemies is not None
        and current_enemies == 0
    )
    map_clear_component = abs(weights.map_clear_bonus) if map_cleared else 0.0

    premature_exit_attempt = bool(info_payload.get("premature_exit_attempt", False))
    premature_exit_component = (
        -abs(weights.premature_exit_penalty) if premature_exit_attempt else 0.0
    )

    action_effective = info_payload.get("action_effective", True)
    invalid_action_component = (
        -abs(weights.invalid_action_penalty) if action_effective is False else 0.0
    )

    fail_state_value = _bool_field_value(current_state.fail_state)
    is_terminal_fail = done or fail_state_value is True
    fail_penalty_component = -abs(weights.fail_penalty) if is_terminal_fail else 0.0

    breakdown = RewardBreakdown(
        survival=survival_component,
        step_penalty=step_penalty_component,
        health_change=health_component,
        currency_change=currency_component,
        energy_change=energy_component,
        score_change=score_component,
        siphon_collected=siphon_component,
        enemy_damaged=enemy_damage_component,
        enemy_cleared=enemy_component,
        phase_progress=phase_progress_component,
        map_clear_bonus=map_clear_component,
        premature_exit_penalty=premature_exit_component,
        invalid_action_penalty=invalid_action_component,
        fail_penalty=fail_penalty_component,
        safe_tile_bonus=safe_tile_component,
        danger_tile_penalty=danger_tile_component,
        resource_proximity=proximity_component,
        prog_collected=prog_component,
        points_collected=points_collected_component,
        damage_taken_penalty=damage_taken_component,
    )
    unclipped_total = breakdown.total
    total = _clip(unclipped_total, clip_abs=float(active_config.reward_clip_abs))

    active_logger.debug(
        "Reward breakdown survival=%.4f step_penalty=%.4f health_change=%.4f currency_change=%.4f "
        "energy_change=%.4f score_change=%.4f siphon_collected=%.4f enemy_damaged=%.4f enemy_cleared=%.4f "
        "phase_progress=%.4f map_clear_bonus=%.4f premature_exit_penalty=%.4f "
        "invalid_action_penalty=%.4f fail_penalty=%.4f safe_tile_bonus=%.4f danger_tile_penalty=%.4f "
        "resource_proximity=%.4f prog_collected=%.4f points_collected=%.4f damage_taken_penalty=%.4f "
        "total=%.4f unclipped_total=%.4f done=%s health_delta=%.4f energy_delta=%.4f "
        "currency_delta=%.4f score_delta=%.4f",
        breakdown.survival,
        breakdown.step_penalty,
        breakdown.health_change,
        breakdown.currency_change,
        breakdown.energy_change,
        breakdown.score_change,
        breakdown.siphon_collected,
        breakdown.enemy_damaged,
        breakdown.enemy_cleared,
        breakdown.phase_progress,
        breakdown.map_clear_bonus,
        breakdown.premature_exit_penalty,
        breakdown.invalid_action_penalty,
        breakdown.fail_penalty,
        breakdown.safe_tile_bonus,
        breakdown.danger_tile_penalty,
        breakdown.resource_proximity,
        breakdown.prog_collected,
        breakdown.points_collected,
        breakdown.damage_taken_penalty,
        total,
        unclipped_total,
        done,
        health_delta,
        energy_delta,
        currency_delta,
        score_delta,
    )

    return RewardResult(total=total, breakdown=breakdown)
