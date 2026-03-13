"""Shared contracts for the hybrid hierarchical agent stack."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from src.state.schema import GameStateSnapshot, GridPosition


class ObjectivePhase(str, Enum):
    """Top-level objective phases selected by the meta-controller."""

    COLLECT_SIPHONS = "collect_siphons"
    COLLECT_RESOURCES_PROGS_POINTS = "collect_resources_progs_points"
    EXIT_SECTOR = "exit_sector"


class ThreatOverride(str, Enum):
    """Low-level tactical override emitted by the threat DRQN."""

    ROUTE_DEFAULT = "route_default"
    EVADE = "evade"
    ENGAGE = "engage"
    WAIT = "wait"
    USE_PROG = "use_prog"


@dataclass(frozen=True)
class MetaObjectiveChoice:
    """Meta-controller objective decision for the current step."""

    phase: ObjectivePhase
    target_position: GridPosition | None
    reason: str
    q_value: float | None = None


@dataclass(frozen=True)
class HybridDecision:
    """Final action decision after objective routing + threat override."""

    objective: MetaObjectiveChoice
    threat_override: ThreatOverride
    action: str
    used_fallback: bool
    reason: str


@dataclass(frozen=True)
class HybridDecisionTrace:
    """Decision metadata required for training updates and reward shaping."""

    decision: HybridDecision
    meta_features: tuple[float, ...]
    threat_features: tuple[float, ...]
    objective_distance_before: int | None
    threat_active: bool
    available_actions: tuple[str, ...]


class HybridEpisodeEnv(Protocol):
    """Episode environment contract consumed by hybrid rollouts."""

    action_space: tuple[str, ...]
    current_episode_id: str | None

    def reset(self) -> GameStateSnapshot:
        """Reset and return initial snapshot."""

    def step(self, action: str) -> tuple[GameStateSnapshot, float, bool, dict[str, Any]]:
        """Apply one action and return `(state, reward, done, info)`."""

    def available_actions(self, state: GameStateSnapshot | None = None) -> tuple[str, ...]:
        """Return valid actions for current snapshot."""

