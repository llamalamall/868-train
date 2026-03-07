"""Objective-driven reward shaping with deterministic component breakdowns."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

from src.state.schema import FieldState, GameStateSnapshot

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RewardWeights:
    """Configurable weights for each reward component."""

    survival: float = 0.02
    step_penalty: float = 0.01
    health_delta: float = 0.40
    currency_delta: float = 0.03
    siphon_collected: float = 2.50
    enemy_cleared: float = 1.50
    phase_progress: float = 0.25
    map_clear_bonus: float = 8.0
    premature_exit_penalty: float = 2.5
    invalid_action_penalty: float = 0.75
    fail_penalty: float = 12.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> RewardWeights:
        """Build weights from a partial config mapping."""
        return cls(
            survival=float(values.get("survival", cls.survival)),
            step_penalty=float(values.get("step_penalty", cls.step_penalty)),
            health_delta=float(values.get("health_delta", cls.health_delta)),
            currency_delta=float(values.get("currency_delta", cls.currency_delta)),
            siphon_collected=float(values.get("siphon_collected", cls.siphon_collected)),
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
        )


@dataclass(frozen=True)
class RewardConfig:
    """Reward configuration container for training settings."""

    weights: RewardWeights = field(default_factory=RewardWeights)
    survival_when_done: bool = False
    reward_clip_abs: float = 5.0


@dataclass(frozen=True)
class RewardBreakdown:
    """Per-component reward contributions for debugging."""

    survival: float
    step_penalty: float
    health_change: float
    currency_change: float
    siphon_collected: float
    enemy_cleared: float
    phase_progress: float
    map_clear_bonus: float
    premature_exit_penalty: float
    invalid_action_penalty: float
    fail_penalty: float

    @property
    def total(self) -> float:
        """Return the unclipped total reward across all components."""
        return (
            self.survival
            + self.step_penalty
            + self.health_change
            + self.currency_change
            + self.siphon_collected
            + self.enemy_cleared
            + self.phase_progress
            + self.map_clear_bonus
            + self.premature_exit_penalty
            + self.invalid_action_penalty
            + self.fail_penalty
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


def _count_siphons(state: GameStateSnapshot) -> int | None:
    if state.map.status != "ok":
        return None
    return len(state.map.siphons)


def _count_live_enemies(state: GameStateSnapshot) -> int | None:
    if state.map.status != "ok":
        return None
    return sum(1 for enemy in state.map.enemies if enemy.in_bounds and enemy.type_id > 0)


def _player_on_exit(state: GameStateSnapshot) -> bool:
    if state.map.status != "ok":
        return False
    player = state.map.player_position
    exit_position = state.map.exit_position
    return player is not None and exit_position is not None and player == exit_position


def _nearest_siphon_distance(state: GameStateSnapshot) -> int | None:
    if state.map.status != "ok":
        return None
    player = state.map.player_position
    if player is None or not state.map.siphons:
        return None
    return min(abs(player.x - target.x) + abs(player.y - target.y) for target in state.map.siphons)


def _nearest_enemy_distance(state: GameStateSnapshot) -> int | None:
    if state.map.status != "ok":
        return None
    player = state.map.player_position
    if player is None:
        return None
    enemies = tuple(enemy.position for enemy in state.map.enemies if enemy.in_bounds and enemy.type_id > 0)
    if not enemies:
        return None
    return min(abs(player.x - target.x) + abs(player.y - target.y) for target in enemies)


def _exit_distance(state: GameStateSnapshot) -> int | None:
    if state.map.status != "ok":
        return None
    player = state.map.player_position
    exit_position = state.map.exit_position
    if player is None or exit_position is None:
        return None
    return abs(player.x - exit_position.x) + abs(player.y - exit_position.y)


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
    health_delta = 0.0 if previous_health is None or current_health is None else (current_health - previous_health)

    previous_currency = _numeric_field_value(previous_state.currency)
    current_currency = _numeric_field_value(current_state.currency)
    currency_delta = 0.0 if previous_currency is None or current_currency is None else (current_currency - previous_currency)

    previous_siphons = _count_siphons(previous_state)
    current_siphons = _count_siphons(current_state)
    siphons_collected = 0.0
    if previous_siphons is not None and current_siphons is not None:
        siphons_collected = float(max(previous_siphons - current_siphons, 0))

    previous_enemies = _count_live_enemies(previous_state)
    current_enemies = _count_live_enemies(current_state)
    enemies_cleared = 0.0
    if previous_enemies is not None and current_enemies is not None:
        enemies_cleared = float(max(previous_enemies - current_enemies, 0))

    survival_component = weights.survival if (not done or active_config.survival_when_done) else 0.0
    step_penalty_component = -abs(weights.step_penalty)
    health_component = health_delta * weights.health_delta
    currency_component = currency_delta * weights.currency_delta
    siphon_component = siphons_collected * abs(weights.siphon_collected)
    enemy_component = enemies_cleared * abs(weights.enemy_cleared)
    phase_progress_component = _phase_progress_delta(
        previous_state=previous_state,
        current_state=current_state,
    ) * weights.phase_progress

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
        siphon_collected=siphon_component,
        enemy_cleared=enemy_component,
        phase_progress=phase_progress_component,
        map_clear_bonus=map_clear_component,
        premature_exit_penalty=premature_exit_component,
        invalid_action_penalty=invalid_action_component,
        fail_penalty=fail_penalty_component,
    )
    unclipped_total = breakdown.total
    total = _clip(unclipped_total, clip_abs=float(active_config.reward_clip_abs))

    active_logger.debug(
        "Reward breakdown survival=%.4f step_penalty=%.4f health_change=%.4f currency_change=%.4f "
        "siphon_collected=%.4f enemy_cleared=%.4f phase_progress=%.4f map_clear_bonus=%.4f "
        "premature_exit_penalty=%.4f invalid_action_penalty=%.4f fail_penalty=%.4f total=%.4f "
        "unclipped_total=%.4f done=%s health_delta=%.4f currency_delta=%.4f",
        breakdown.survival,
        breakdown.step_penalty,
        breakdown.health_change,
        breakdown.currency_change,
        breakdown.siphon_collected,
        breakdown.enemy_cleared,
        breakdown.phase_progress,
        breakdown.map_clear_bonus,
        breakdown.premature_exit_penalty,
        breakdown.invalid_action_penalty,
        breakdown.fail_penalty,
        total,
        unclipped_total,
        done,
        health_delta,
        currency_delta,
    )

    return RewardResult(total=total, breakdown=breakdown)
