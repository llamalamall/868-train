"""Reward shaping v1 with deterministic component breakdowns."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping

from src.state.schema import FieldState, GameStateSnapshot

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RewardWeights:
    """Configurable weights for each reward component."""

    survival: float = 0.1
    health_delta: float = 0.5
    currency_delta: float = 0.05
    fail_penalty: float = 10.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> RewardWeights:
        """Build weights from a partial config mapping."""
        return cls(
            survival=float(values.get("survival", cls.survival)),
            health_delta=float(values.get("health_delta", cls.health_delta)),
            currency_delta=float(values.get("currency_delta", cls.currency_delta)),
            fail_penalty=float(values.get("fail_penalty", cls.fail_penalty)),
        )


@dataclass(frozen=True)
class RewardConfig:
    """Reward configuration container for training settings."""

    weights: RewardWeights = field(default_factory=RewardWeights)
    survival_when_done: bool = False


@dataclass(frozen=True)
class RewardBreakdown:
    """Per-component reward contributions for debugging."""

    survival: float
    health_change: float
    currency_change: float
    fail_penalty: float

    @property
    def total(self) -> float:
        """Return the total reward across all components."""
        return self.survival + self.health_change + self.currency_change + self.fail_penalty


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


def compute_reward(
    *,
    previous_state: GameStateSnapshot,
    current_state: GameStateSnapshot,
    done: bool,
    config: RewardConfig | None = None,
    logger: logging.Logger | None = None,
) -> RewardResult:
    """Compute deterministic reward for one state transition."""
    active_config = config or RewardConfig()
    weights = active_config.weights
    active_logger = logger or LOGGER

    previous_health = _numeric_field_value(previous_state.health)
    current_health = _numeric_field_value(current_state.health)
    if previous_health is None or current_health is None:
        health_delta = 0.0
    else:
        health_delta = current_health - previous_health

    previous_currency = _numeric_field_value(previous_state.currency)
    current_currency = _numeric_field_value(current_state.currency)
    if previous_currency is None or current_currency is None:
        currency_delta = 0.0
    else:
        currency_delta = current_currency - previous_currency

    survival_component = weights.survival if (not done or active_config.survival_when_done) else 0.0
    health_component = health_delta * weights.health_delta
    currency_component = currency_delta * weights.currency_delta

    fail_state_value = _bool_field_value(current_state.fail_state)
    is_terminal_fail = done or fail_state_value is True
    fail_penalty_component = -abs(weights.fail_penalty) if is_terminal_fail else 0.0

    breakdown = RewardBreakdown(
        survival=survival_component,
        health_change=health_component,
        currency_change=currency_component,
        fail_penalty=fail_penalty_component,
    )
    total = breakdown.total

    active_logger.debug(
        "Reward breakdown survival=%.4f health_change=%.4f currency_change=%.4f "
        "fail_penalty=%.4f total=%.4f done=%s health_delta=%.4f currency_delta=%.4f",
        breakdown.survival,
        breakdown.health_change,
        breakdown.currency_change,
        breakdown.fail_penalty,
        total,
        done,
        health_delta,
        currency_delta,
    )

    return RewardResult(total=total, breakdown=breakdown)
